"""Fetch one month of Smogon usage statistics and write the input record (ADR-020 section 4, #975).

Run by hand, never in CI. Standard library only. One sequential client with a User-Agent that
names Areneon and its repository; ``robots.txt`` is read first, in the same run, and every URL
must be allowed for the agent before anything is downloaded (a robots.txt with wildcard rules
for the agent is refused, since urllib.robotparser does not implement RFC 9309 wildcards, and
any robots.txt answer other than 200 stops the run: conservative, RFC 9309 would allow a 4xx);
errors that can pass (HTTP 429, 5xx, a dropped connection) are retried with exponential
back-off up to a limit, any other non-200 answer (a redirect included) stops the run; no
credentials; nothing else is requested.

    PYTHONPATH=.:packages/schema/python python -m scripts.fetch_usage_stats \\
        --month 2026-09 --cutoff 1630 [--format gen9championsvgc2026regmc] [--out-dir .tmp/chaos]

The raw files go to the gitignored ``--out-dir`` (never committed, never overwritten: a second
month needs its own directory). The input record is the one #1120 committed, read by
``derive_usage_prior``: ``config/opponent-model/<format>/smogon-chaos.<month>-<cutoff>.inputs.json``
(month, cutoff, format_id, basis, ``robots_sha256`` of the robots.txt read in this run, and one
entry per file with its name, source id, URL, retrieval time in UTC, size and SHA-256), written as
RFC 8785 bytes and never over an existing record. Nothing is written unless every file was
fetched and checked (chaos is strict JSON, the summary starts with a battle count, leads has the
table header); files are staged beside their places and linked in, and removed again on failure.
Basis: owner-accepted risk, ADR-009 erratum 1 (Smogon states no licence).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from areneon_contracts.contracts import jcs

from scripts.usage_common import DerivationError, parse_source

AGENT = "Areneon-usage-ingest"
USER_AGENT = f"{AGENT}/1 (+https://github.com/piglor/areneon)"
BASIS = (
    "owner-accepted-risk: ADR-009 erratum 1, extended to the opponent prior by ADR-025 "
    "erratum E-9.2; aggregate statistics only"
)
DEFAULT_FORMAT = "gen9championsvgc2026regmc"
DEFAULT_PREVIEW_MAP = "generator-matched.uniform-v1.json"
BASE_URL = "https://www.smogon.com"
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 120.0
MAX_BYTES = 64 * 1024 * 1024
MONTH = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
FORMAT = re.compile(r"^[a-z0-9]{1,64}$")
LEADS_HEADER = re.compile(rb"\|\s*Rank\s*\|\s*Pokemon\s*\|")
# (source id, local file name, path under /stats/<month>/ with {stem} = "<format>-<cutoff>")
FILES = (
    ("chaos", "chaos.json", "chaos/{stem}.json"),
    ("summary", "summary.txt", "{stem}.txt"),
    ("leads", "leads.txt", "leads/{stem}.txt"),
)


class IngestError(RuntimeError):
    """The run stops; nothing is written."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def get(
    url: str,
    *,
    retries: int,
    backoff_s: float,
    timeout_s: float,
    sleep: Callable[[float], None],
) -> bytes:
    """GET a URL: 200 returns the body; retryable failures are retried, anything else stops."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise IngestError("only http and https URLs are fetched")
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        wait = backoff_s * 2**attempt
        try:
            with opener.open(request, timeout=timeout_s) as response:
                if response.status != HTTPStatus.OK:
                    raise IngestError(f"{url} answered HTTP {response.status}")
                body = bytes(response.read(MAX_BYTES + 1))
                if len(body) > MAX_BYTES:
                    raise IngestError(f"{urlsplit(url).path} is larger than {MAX_BYTES} bytes")
                return body
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE:
                raise IngestError(f"{url} answered HTTP {exc.code}") from None
            retry_after = exc.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = min(float(retry_after), MAX_RETRY_AFTER_S)
            failure = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            failure = type(exc).__name__
        if attempt == retries:
            raise IngestError(f"{url} failed after {retries + 1} attempts ({failure})")
        sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def wildcard_rules(robots: str) -> bool:
    """True if a group that applies to the agent has an Allow/Disallow path with ``*`` or ``$``.

    ``urllib.robotparser`` does not implement RFC 9309 wildcards, so such a file cannot be
    honoured by it and the run refuses it. The agent's own group applies if there is one,
    else the ``*`` groups.
    """
    groups: list[tuple[set[str], list[str]]] = []
    agents: set[str] = set()
    paths: list[str] = []
    in_rules = False
    for raw in robots.splitlines():
        line = raw.split("#", 1)[0].strip()
        key, sep, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep:
            continue
        if key == "user-agent":
            if in_rules:
                groups.append((agents, paths))
                agents, paths, in_rules = set(), [], False
            agents.add(value.lower())
        elif key in ("allow", "disallow"):
            in_rules = True
            paths.append(value)
    groups.append((agents, paths))
    specific = [g for g in groups if any(a != "*" and AGENT.lower().startswith(a) for a in g[0])]
    applicable = specific or [g for g in groups if "*" in g[0]]
    return any("*" in p or "$" in p for _, ps in applicable for p in ps)


def check_robots(base: str, urls: Sequence[str], **net: Any) -> str:
    """Read robots.txt, require every URL allowed for the agent, return its SHA-256."""
    body = get(f"{base}/robots.txt", **net)
    text = body.decode("utf-8", errors="replace")
    if wildcard_rules(text):
        raise IngestError("robots.txt has wildcard rules, which this client cannot honour")
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    for url in urls:
        if not parser.can_fetch(AGENT, url):
            raise IngestError(f"robots.txt disallows {urlsplit(url).path} for {AGENT}")
    return hashlib.sha256(body).hexdigest()


def check_content(files: dict[str, bytes]) -> None:
    """Refuse a body that is not the kind of file it was asked for."""
    try:
        parse_source(files["chaos"].decode("utf-8"))
    except (DerivationError, ValueError):
        raise IngestError("the chaos file is not strict JSON") from None
    if not files["summary"].startswith(b"Total battles: "):
        raise IngestError("the summary file does not start with a battle count")
    if LEADS_HEADER.search(files["leads"]) is None:
        raise IngestError("the leads file does not look like the leads table")


def place(staged: Path, final: Path, placed: list[Path]) -> None:
    """Move a staged file into place without replacing anything."""
    final.parent.mkdir(parents=True, exist_ok=True)
    os.link(staged, final)
    placed.append(final)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingest(  # noqa: PLR0913 - the run's inputs, all keyword
    *,
    base: str,
    month: str,
    cutoff: int,
    format_id: str,
    out_dir: Path,
    record_path: Path,
    preview_map: str,
    retries: int = 3,
    backoff_s: float = 5.0,
    delay_s: float = 5.0,
    timeout_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    if not MONTH.match(month) or cutoff < 0 or not FORMAT.match(format_id):
        raise IngestError("month, cutoff or format is malformed")
    stem = f"{format_id}-{cutoff}"
    targets = [
        (sid, name, f"{base}/stats/{month}/{path.format(stem=stem)}") for sid, name, path in FILES
    ]
    if record_path.exists():
        raise IngestError("the input record exists; a record is never overwritten")
    for _, name, _ in targets:
        if (out_dir / name).exists():
            raise IngestError("a raw file exists; use an empty directory for each month")
    net: dict[str, Any] = {
        "retries": retries,
        "backoff_s": backoff_s,
        "timeout_s": timeout_s,
        "sleep": sleep,
    }
    robots_sha256 = check_robots(base, [url for _, _, url in targets], **net)
    fetched = []
    for sid, name, url in targets:
        sleep(delay_s)
        body = get(url, **net)
        fetched.append((sid, name, url, now(), body))
    check_content({sid: body for sid, _, _, _, body in fetched})
    record = {
        "basis": BASIS,
        "cutoff": cutoff,
        "files": [
            {
                "file": name,
                "retrieved_at": at,
                "sha256": hashlib.sha256(body).hexdigest(),
                "size": len(body),
                "source_id": sid,
                "url": url,
            }
            for sid, name, url, at, body in sorted(fetched, key=lambda f: f[0])
        ],
        "format_id": format_id,
        "month": month,
        "preview_map": preview_map,
        "robots_sha256": robots_sha256,
    }
    # Stage everything beside the final places, then link each file in; a failure removes what
    # was linked, so a failed run leaves nothing that blocks a rerun.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    placed: list[Path] = []
    with (
        tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=".fetch-raw-") as raw_stage,
        tempfile.TemporaryDirectory(dir=record_path.parent, prefix=".fetch-rec-") as rec_stage,
    ):
        for _, name, _, _, body in fetched:
            (Path(raw_stage) / name).write_bytes(body)
        (Path(rec_stage) / "record").write_bytes(jcs(record) + b"\n")
        try:
            for _, name, _, _, _ in fetched:
                place(Path(raw_stage) / name, out_dir / name, placed)
            place(Path(rec_stage) / "record", record_path, placed)
        except OSError:
            for path in placed:
                path.unlink(missing_ok=True)
            raise IngestError("a file could not be put in place; nothing was kept") from None
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--cutoff", type=int, default=1630)
    parser.add_argument("--format", dest="format_id", default=DEFAULT_FORMAT)
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/chaos"))
    parser.add_argument("--record", type=Path, help="default: the config/opponent-model path")
    parser.add_argument("--preview-map", default=DEFAULT_PREVIEW_MAP)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=5.0)
    args = parser.parse_args(argv)
    record = args.record or Path(
        f"config/opponent-model/{args.format_id}/smogon-chaos.{args.month}-{args.cutoff}.inputs.json"
    )
    try:
        done = ingest(
            base=args.base_url.rstrip("/"),
            month=args.month,
            cutoff=args.cutoff,
            format_id=args.format_id,
            out_dir=args.out_dir,
            record_path=record,
            preview_map=args.preview_map,
            retries=args.retries,
            backoff_s=args.backoff,
            delay_s=args.delay,
        )
    except IngestError as exc:
        print(f"fetch_usage_stats: {exc}", file=sys.stderr)
        return 2
    for f in done["files"]:
        print(f"{f['source_id']}: {f['size']} bytes sha256 {f['sha256']}")
    print(f"record: {os.fspath(record)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
