"""The validation look log (ADR-017 erratum E2, item 5).

Validation teams are used again and again, so every look at a validation result is recorded. The
log is the sibling file ``<progress root>.looks.jsonl`` (``<run-dir>.progress.looks.jsonl`` for the
default root), never inside the progress root or the run directory, whose generations are atomic
directories with manifests and which the report, replay and inspect commands only read.

Each line is the JCS bytes of one entry and the entry carries the SHA-256 of the previous line (the
first entry carries ``GENESIS``), as the evaluation register does (ADR-010 section 9, O1), so a
dropped, changed or reordered line is found by any reader. ``append_look`` takes an exclusive
``flock`` on the log for the whole read, check and write; ``read_looks`` takes a shared one.

A look is progress-view bookkeeping: it says who looked at what, never what the results were, and
a validation result is never cited as held-out strength.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

SCHEMA_VERSION = 1
#: The ``prev`` of the first entry: a fixed value that no entry hash can be chosen to equal.
GENESIS = sha256_hex(b"areneon:progress-looks:genesis:v1")
COMMANDS = ("progress", "report", "replay", "private-test", "scripted-h2h")
SUFFIX = ".looks.jsonl"
_KEYS = frozenset(
    {"schema_version", "seq", "prev", "at", "command", "run_id", "generations", "matches", "games"}
)
_MATCH_KEYS = frozenset({"generation", "match_id", "games"})
_HEX = re.compile(r"[0-9a-f]{64}")
_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class LookError(ValueError):
    """The look log cannot be read or extended: the message says which rule it breaks."""


@dataclass(frozen=True)
class Look:
    """What one look names: a generation's match and how many games of it were shown."""

    generation: int
    match_id: str
    games: int


@dataclass(frozen=True)
class LookState:
    """The verified log: the number of looks, the head (hash of the last line) and the file hash."""

    count: int
    head: str
    file_sha256: str


def looks_path(root: Path, run_dir: Path | None = None) -> Path:
    """The log of a progress root: its sibling ``<name>.looks.jsonl``."""
    root = root.resolve()
    path = root.with_name(root.name + SUFFIX)
    if run_dir is not None:
        run = run_dir.resolve()
        if path == run or run in path.parents:
            raise LookError("the look log must not be inside the run directory")
    return path


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_entry(entry: object, index: int, prev: str, last_at: str, run_id: str | None) -> None:
    where = f"look {index}"
    if not isinstance(entry, dict) or set(entry) != _KEYS:
        raise LookError(f"{where}: not an entry of the look log")
    if entry["schema_version"] != SCHEMA_VERSION:
        raise LookError(f"{where}: an unknown schema_version")
    if entry["seq"] != index or not _int(entry["seq"]):
        raise LookError(f"{where}: seq is not its position (a line was dropped or reordered)")
    if entry["prev"] != prev:
        raise LookError(f"{where}: prev is not the hash of the line before (the chain is broken)")
    at = entry["at"]
    if not isinstance(at, str) or not _TIME.fullmatch(at) or at < last_at:
        raise LookError(f"{where}: the time is malformed or goes back")
    if entry["command"] not in COMMANDS:
        raise LookError(f"{where}: an unknown command")
    if not isinstance(entry["run_id"], str) or not _ID.fullmatch(entry["run_id"]):
        raise LookError(f"{where}: run_id is malformed")
    if run_id is not None and entry["run_id"] != run_id:
        raise LookError(f"{where}: the log names more than one run")
    _check_matches(entry, where)


def _check_matches(entry: dict[str, Any], where: str) -> None:
    matches = entry["matches"]
    if not isinstance(matches, list) or not matches:
        raise LookError(f"{where}: a look names at least one match")
    keys = []
    for m in matches:
        if (
            not isinstance(m, dict)
            or set(m) != _MATCH_KEYS
            or not _int(m["generation"])
            or not isinstance(m["match_id"], str)
            or not _ID.fullmatch(m["match_id"])
            or not _int(m["games"])
        ):
            raise LookError(f"{where}: a match entry is malformed")
        keys.append((m["generation"], m["match_id"]))
    if keys != sorted(set(keys)):
        raise LookError(f"{where}: matches are not sorted and unique")
    if entry["generations"] != sorted({k[0] for k in keys}):
        raise LookError(f"{where}: generations differ from the matches")
    if entry["games"] != sum(m["games"] for m in matches):
        raise LookError(f"{where}: games differ from the matches")


def verify_looks(data: bytes) -> LookState:
    """Check the whole log: canonical lines, a chain from ``GENESIS``, positions, times, one run."""
    prev, last_at, run_id, count = GENESIS, "", None, 0
    if data and not data.endswith(b"\n"):
        raise LookError("the look log does not end with a newline (a torn or edited last line)")
    for index, line in enumerate(data.split(b"\n")[:-1] if data else []):
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise LookError(f"look {index}: not JSON") from exc
        try:
            canonical = jcs(entry) == line
        except ValueError as exc:
            raise LookError(f"look {index}: not canonical JSON") from exc
        if not canonical:
            raise LookError(f"look {index}: the line is not the JCS bytes of its entry")
        _check_entry(entry, index, prev, last_at, run_id)
        prev, last_at, run_id, count = sha256_hex(line), entry["at"], entry["run_id"], index + 1
    return LookState(count, prev, sha256_hex(data))


def read_looks(path: Path) -> LookState:
    """The verified state of the log at ``path`` (an absent log has no looks)."""
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return LookState(0, GENESIS, sha256_hex(b""))
    with handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return verify_looks(handle.read())


def append_look(
    path: Path,
    command: str,
    run_id: str,
    looks: Iterable[Look],
    clock: Callable[[], str] = _now,
) -> LookState:
    """Append one line under an exclusive lock, after verifying the log as it stands."""
    matches = sorted(
        ({"generation": k.generation, "match_id": k.match_id, "games": k.games} for k in looks),
        key=lambda m: (m["generation"], m["match_id"]),
    )
    created = not path.exists()
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = handle.read()
        state = verify_looks(existing)
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seq": state.count,
            "prev": state.head,
            "at": clock(),
            "command": command,
            "run_id": run_id,
            "generations": sorted({m["generation"] for m in matches}),
            "matches": matches,
            "games": sum(m["games"] for m in matches),
        }
        # Check the new line by the reader's own rules before a byte of it is written.
        last: dict[str, Any] = json.loads(existing.split(b"\n")[-2]) if existing else {}
        _check_entry(entry, state.count, state.head, last.get("at", ""), last.get("run_id"))
        line = jcs(entry)
        handle.write(line + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        data = existing + line + b"\n"
    if created:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return LookState(state.count + 1, sha256_hex(line), sha256_hex(data))


__all__ = [
    "COMMANDS",
    "GENESIS",
    "Look",
    "LookError",
    "LookState",
    "append_look",
    "looks_path",
    "read_looks",
    "verify_looks",
]
