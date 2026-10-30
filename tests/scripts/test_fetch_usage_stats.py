"""The usage-statistics fetcher against a local fake HTTP server (#975); no network."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_usage_stats as fetch

FMT = "gen9championsvgc2026regmc"
BODIES = {
    f"/stats/2026-09/chaos/{FMT}-1630.json": b'{"info": {}, "data": {}}',
    f"/stats/2026-09/{FMT}-1630.txt": b"Total battles: 12\n rest\n",
    f"/stats/2026-09/leads/{FMT}-1630.txt": b" | Rank | Pokemon | Usage % |\n",
}
ROBOTS_OK = b"User-agent: *\nDisallow: /private/\n"


class Fake:
    def __init__(self) -> None:
        self.robots = ROBOTS_OK
        self.status: dict[str, list[int]] = {}
        self.hits: list[str] = []
        self.agents: list[str] = []


@pytest.fixture
def server() -> Iterator[tuple[Fake, str]]:
    fake = Fake()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            fake.hits.append(self.path)
            fake.agents.append(self.headers.get("User-Agent", ""))
            queue = fake.status.get(self.path)
            code = queue.pop(0) if queue else 200
            body = fake.robots if self.path == "/robots.txt" else BODIES.get(self.path, b"")
            if code == 200 and self.path not in BODIES and self.path != "/robots.txt":
                code = 404
            self.send_response(code)
            if code == 301:
                self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", str(len(body) if code == 200 else 0))
            self.end_headers()
            if code == 200:
                self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield fake, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def run(base: str, tmp: Path, **kw: Any) -> dict[str, object]:
    sleeps: list[float] = []
    return fetch.ingest(
        base=base,
        month="2026-09",
        cutoff=1630,
        format_id=FMT,
        out_dir=tmp / "raw",
        record_path=tmp / "rec" / "inputs.json",
        preview_map="map.json",
        sleep=sleeps.append,
        now=lambda: "2026-10-29T00:00:00Z",
        **kw,
    )


def test_record_matches_served_bytes(server: tuple[Fake, str], tmp_path: Path) -> None:
    fake, base = server
    record = run(base, tmp_path)
    files = {f["source_id"]: f for f in record["files"]}  # type: ignore[attr-defined]
    for sid, name, path in fetch.FILES:
        served = BODIES[f"/stats/2026-09/{path.format(stem=f'{FMT}-1630')}"]
        on_disk = (tmp_path / "raw" / name).read_bytes()
        assert on_disk == served
        assert files[sid]["sha256"] == hashlib.sha256(served).hexdigest()
        assert files[sid]["size"] == len(served)
        assert files[sid]["url"] == base + "/stats/2026-09/" + path.format(stem=f"{FMT}-1630")
    assert record["robots_sha256"] == hashlib.sha256(ROBOTS_OK).hexdigest()
    assert record["basis"].startswith("owner-accepted-risk: ADR-009 erratum 1")  # type: ignore[attr-defined]
    stored = (tmp_path / "rec" / "inputs.json").read_bytes()
    assert stored.endswith(b"\n")
    assert json.loads(stored) == record
    assert fake.hits[0] == "/robots.txt"
    assert set(fake.agents) == {fetch.USER_AGENT}
    assert "areneon" in fetch.USER_AGENT.lower()


def test_robots_disallow_stops_before_any_download(
    server: tuple[Fake, str], tmp_path: Path
) -> None:
    fake, base = server
    fake.robots = b"User-agent: *\nDisallow: /stats/\n"
    with pytest.raises(fetch.IngestError, match=r"robots\.txt disallows"):
        run(base, tmp_path)
    assert fake.hits == ["/robots.txt"]
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "rec").exists()


def test_non_200_stops_after_the_retry_limit(server: tuple[Fake, str], tmp_path: Path) -> None:
    fake, base = server
    path = f"/stats/2026-09/chaos/{FMT}-1630.json"
    fake.status[path] = [503] * 10
    with pytest.raises(fetch.IngestError, match="failed after 3 attempts"):
        run(base, tmp_path, retries=2)
    assert fake.hits.count(path) == 3
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "rec").exists()


def test_a_transient_failure_is_retried(server: tuple[Fake, str], tmp_path: Path) -> None:
    fake, base = server
    fake.status[f"/stats/2026-09/{FMT}-1630.txt"] = [429, 503]
    run(base, tmp_path, retries=3)
    assert fake.hits.count(f"/stats/2026-09/{FMT}-1630.txt") == 3


@pytest.mark.parametrize("code", [404, 301, 403])
def test_other_statuses_stop_at_once(server: tuple[Fake, str], tmp_path: Path, code: int) -> None:
    fake, base = server
    path = f"/stats/2026-09/leads/{FMT}-1630.txt"
    fake.status[path] = [code]
    with pytest.raises(fetch.IngestError, match=f"HTTP {code}"):
        run(base, tmp_path, retries=5)
    assert fake.hits.count(path) == 1


def test_robots_unavailable_stops(server: tuple[Fake, str], tmp_path: Path) -> None:
    fake, base = server
    fake.status["/robots.txt"] = [404]
    with pytest.raises(fetch.IngestError, match="HTTP 404"):
        run(base, tmp_path)
    assert fake.hits == ["/robots.txt"]


def test_bad_summary_and_existing_outputs_are_refused(
    server: tuple[Fake, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, base = server
    run(base, tmp_path)
    with pytest.raises(fetch.IngestError, match="record exists"):
        run(base, tmp_path)
    (tmp_path / "rec" / "inputs.json").unlink()
    with pytest.raises(fetch.IngestError, match="raw file exists"):
        run(base, tmp_path)
    BODIES_KEY = f"/stats/2026-09/{FMT}-1630.txt"
    monkeypatch.setitem(BODIES, BODIES_KEY, b"not a summary")
    with pytest.raises(fetch.IngestError, match="battle count"):
        run(base, tmp_path / "other")


def test_only_http_is_fetched() -> None:
    with pytest.raises(fetch.IngestError):
        fetch.get("file:///etc/passwd", retries=0, backoff_s=0, timeout_s=1, sleep=lambda _: None)


def test_connection_failure_is_a_bounded_error() -> None:
    with pytest.raises(fetch.IngestError, match="failed after 2 attempts"):
        fetch.get("http://127.0.0.1:1/x", retries=1, backoff_s=0, timeout_s=1, sleep=lambda _: None)


def test_main_prints_and_returns_codes(
    server: tuple[Fake, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, base = server
    args = [
        "--month", "2026-09", "--base-url", base, "--out-dir", str(tmp_path / "r"),
        "--record", str(tmp_path / "rec.json"), "--delay", "0", "--backoff", "0",
    ]  # fmt: skip
    assert fetch.main(args) == 0
    assert "chaos:" in capsys.readouterr().out
    assert fetch.main(args) == 2
    assert "record exists" in capsys.readouterr().err


def test_wildcard_robots_is_refused(server: tuple[Fake, str], tmp_path: Path) -> None:
    fake, base = server
    fake.robots = b"User-agent: *\nDisallow: /a*b\n"
    with pytest.raises(fetch.IngestError, match="wildcard"):
        run(base, tmp_path)
    assert fake.hits == ["/robots.txt"]
    fake.robots = b"User-agent: *\nDisallow: /a*b\n\nUser-agent: Areneon\nDisallow: /x/\n"
    run(base, tmp_path)  # the agent's own group has no wildcard


def test_wildcard_detection() -> None:
    assert fetch.wildcard_rules("User-agent: *\nAllow: /x$ # c\n")
    assert not fetch.wildcard_rules("User-agent: other\nDisallow: /a*\n")
    assert not fetch.wildcard_rules("# nothing\n")


@pytest.mark.parametrize(
    ("key", "body", "message"),
    [
        (f"/stats/2026-09/chaos/{FMT}-1630.json", b"not json", "strict JSON"),
        (f"/stats/2026-09/chaos/{FMT}-1630.json", b'{"a": 1, "a": 2}', "strict JSON"),
        (f"/stats/2026-09/leads/{FMT}-1630.txt", b"hello", "leads table"),
        (f"/stats/2026-09/{FMT}-1630.txt", b"not a summary", "battle count"),
    ],
)
def test_content_checks(  # noqa: PLR0913, PLR0917 - fixtures plus the parametrised case
    server: tuple[Fake, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    body: bytes,
    message: str,
) -> None:
    _, base = server
    monkeypatch.setitem(BODIES, key, body)
    with pytest.raises(fetch.IngestError, match=message):
        run(base, tmp_path)
    assert not (tmp_path / "raw").exists()


def test_response_size_is_capped(
    server: tuple[Fake, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, base = server
    monkeypatch.setattr(fetch, "MAX_BYTES", 10)
    with pytest.raises(fetch.IngestError, match="larger than"):
        run(base, tmp_path)


def test_failed_placement_leaves_nothing(
    server: tuple[Fake, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, base = server
    real = os.link

    def link(src: str | Path, dst: str | Path) -> None:
        if Path(dst).name == "inputs.json":
            raise OSError("boom")
        real(src, dst)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(fetch.IngestError, match="nothing was kept"):
        run(base, tmp_path)
    assert list((tmp_path / "raw").iterdir()) == []
    monkeypatch.setattr(os, "link", real)
    run(base, tmp_path)  # a rerun is not blocked


def test_delay_follows_robots_and_each_file(server: tuple[Fake, str], tmp_path: Path) -> None:
    _, base = server
    sleeps: list[float] = []
    fetch.ingest(
        base=base, month="2026-09", cutoff=1630, format_id=FMT, out_dir=tmp_path / "r",
        record_path=tmp_path / "x.json", preview_map="m", delay_s=7.0, sleep=sleeps.append,
    )  # fmt: skip
    assert sleeps == [7.0, 7.0, 7.0]


@pytest.mark.parametrize(
    ("month", "cutoff", "fmt"),
    [("2026-13", 1630, FMT), ("../x", 1630, FMT), ("2026-09", -1, FMT), ("2026-09", 1630, "a/b")],
)
def test_arguments_are_validated(tmp_path: Path, month: str, cutoff: int, fmt: str) -> None:
    with pytest.raises(fetch.IngestError, match="malformed"):
        fetch.ingest(
            base="http://127.0.0.1:1", month=month, cutoff=cutoff, format_id=fmt,
            out_dir=tmp_path / "r", record_path=tmp_path / "x.json", preview_map="m",
        )  # fmt: skip
