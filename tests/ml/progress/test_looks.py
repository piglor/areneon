"""The validation look log: chained, locked, a sibling of the progress root (ADR-017 E2 item 5)."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.progress.looks import (
    GENESIS,
    Look,
    LookError,
    append_look,
    looks_path,
    read_looks,
    verify_looks,
)

LOOKS = [Look(1, "actor-rule-sample", 10), Look(0, "random-legal-argmax", 4)]


def three(path: Path) -> bytes:
    times = iter(["2026-10-22T00:00:01Z", "2026-10-22T00:00:02Z", "2026-10-22T00:00:03Z"])
    for command in ("progress", "report", "replay"):
        append_look(path, command, "run-1", LOOKS, lambda: next(times))
    return path.read_bytes()


def lines(data: bytes) -> list[dict[str, object]]:
    return [json.loads(x) for x in data.split(b"\n")[:-1]]


def put(path: Path, entries: list[dict[str, object]]) -> bytes:
    data = b"".join(jcs(e) + b"\n" for e in entries)
    path.write_bytes(data)
    return data


def test_a_fresh_log_starts_at_the_genesis_value(tmp_path: Path) -> None:
    path = tmp_path / "r.progress.looks.jsonl"
    empty = read_looks(path)
    assert (empty.count, empty.head) == (0, GENESIS)
    state = append_look(path, "progress", "run-1", LOOKS)
    (entry,) = lines(path.read_bytes())
    assert entry["prev"] == GENESIS
    assert entry["seq"] == 0
    assert entry["generations"] == [0, 1]
    assert entry["games"] == 14
    assert [m["generation"] for m in entry["matches"]] == [0, 1]  # type: ignore[attr-defined]
    assert state.count == 1
    assert state == read_looks(path)
    assert state.file_sha256 == sha256_hex(path.read_bytes())
    assert path.read_bytes() == jcs(entry) + b"\n"


def test_lines_chain_by_the_hash_of_the_line_before(tmp_path: Path) -> None:
    data = three(tmp_path / "l.jsonl")
    entries = lines(data)
    raw = data.split(b"\n")
    assert entries[1]["prev"] == sha256_hex(raw[0])
    assert entries[2]["prev"] == sha256_hex(raw[1])
    state = verify_looks(data)
    assert (state.count, state.head) == (3, sha256_hex(raw[2]))


def test_the_log_is_a_sibling_never_inside_the_root_or_the_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert looks_path(tmp_path / "run.progress") == tmp_path.resolve() / "run.progress.looks.jsonl"
    with pytest.raises(LookError, match="inside the run"):
        looks_path(run / "p", run)
    assert looks_path(tmp_path / "x.progress", run).parent == tmp_path.resolve()


def tamper_cases(data: bytes) -> dict[str, bytes]:
    raw = data.split(b"\n")[:-1]
    e = lines(data)

    def with_(i: int, **change: object) -> bytes:
        entries = [dict(x) for x in e]
        entries[i].update(change)
        return b"".join(jcs(x) + b"\n" for x in entries)

    return {
        "dropped first": b"\n".join(raw[1:]) + b"\n",
        "dropped middle": b"\n".join([raw[0], raw[2]]) + b"\n",
        "reordered": b"\n".join([raw[0], raw[2], raw[1]]) + b"\n",
        "changed games": with_(1, games=999),
        "changed prev": with_(1, prev="0" * 64),
        "changed run": with_(2, run_id="other"),
        "time goes back": with_(2, at="2026-10-22T00:00:00Z"),
        "bad time": with_(0, at="yesterday"),
        "unknown command": with_(0, command="shout"),
        "bad version": with_(0, schema_version=2),
        "extra key": with_(0, note="x"),
        "no matches": with_(0, matches=[]),
        "generations differ": with_(0, generations=[7]),
        "unsorted matches": with_(0, matches=list(reversed(e[0]["matches"]))),  # type: ignore[call-overload]
        "malformed match": with_(0, matches=[{"generation": 0}]),
        "negative games": with_(0, matches=[{"generation": 0, "match_id": "a", "games": -1}]),
        "bad run id": with_(0, run_id="a b"),
        "truncated": data[:-1],
        "not json": b"nope\n",
        "not an object": b"[]\n",
        "not canonical": json.dumps(e[0], indent=1).replace("\n", " ").encode() + b"\n",
        "float": b'{"a":1.5}\n',
        "nan": b"NaN\n",
        "last line chopped": data[:-20] + b"\n",
    }


def test_every_tamper_kind_is_refused(tmp_path: Path) -> None:
    data = three(tmp_path / "l.jsonl")
    verify_looks(data)
    for name, bad in tamper_cases(data).items():
        with pytest.raises(LookError):
            verify_looks(bad)
        path = tmp_path / f"{name.replace(' ', '-')}.jsonl"
        path.write_bytes(bad)
        with pytest.raises(LookError):
            read_looks(path)
        with pytest.raises(LookError):
            append_look(path, "report", "run-1", LOOKS)
        assert path.read_bytes() == bad  # a refused append writes nothing


def test_an_append_refuses_a_line_the_reader_would_refuse(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    three(path)
    with pytest.raises(LookError, match="goes back"):
        append_look(path, "report", "run-1", LOOKS, lambda: "2026-10-22T00:00:00Z")
    with pytest.raises(LookError, match="more than one run"):
        append_look(path, "report", "run-2", LOOKS)
    with pytest.raises(LookError, match="at least one match"):
        append_look(path, "report", "run-1", [])
    with pytest.raises(LookError, match="unknown command"):
        append_look(path, "inspect", "run-1", LOOKS)
    assert verify_looks(path.read_bytes()).count == 3


def _worker(args: tuple[str, int]) -> None:
    path, n = args
    for i in range(n):
        append_look(Path(path), "report", "run-1", [Look(i, "m", 1)])


def test_concurrent_appends_all_land_in_one_unbroken_chain(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    with multiprocessing.get_context("fork").Pool(4) as pool:
        pool.map(_worker, [(str(path), 6)] * 4)
    state = read_looks(path)
    assert state.count == 24
    assert [e["seq"] for e in lines(path.read_bytes())] == list(range(24))


def test_the_first_append_syncs_a_new_file_and_a_second_does_not_recreate_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "n.jsonl"
    assert not path.exists()
    append_look(path, "progress", "run-1", LOOKS)
    inode = path.stat().st_ino
    append_look(path, "report", "run-1", LOOKS)
    assert path.stat().st_ino == inode
