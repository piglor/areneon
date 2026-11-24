"""The verified marker beside the teacher stream (#1501).

``open_teacher`` pays a full pass over the teacher stream (the digest, every line against the
contracts, every example encoded once) at every driver process start. After a pass that succeeded
it now writes a marker next to the stream; a later process that finds a marker for the same
everything skips the line checks and the encode pass and keeps only the digest check. These tests
hold: the marker is written only by a full pass that succeeded (an exception or a SIGKILL leaves
none), it is honoured only when every part of its key is the same (one test per part), a stream or
a marker that changed in any way falls back to the full pass with its own refusals, the offsets it
gives equal the full pass's, and two processes writing at once leave a marker that reads whole.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, sha256_hex
from areneon_contracts.training import check_stream_digest, check_stream_file
from areneon_ml.learner import imitation, teacher_marker
from areneon_ml.learner.data import LearnerDataError, _Collector
from areneon_ml.learner.imitation import open_teacher
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import make_stream, stream_of, tiny_config
from tests.ml.model.support import game

ROOT = Path(__file__).resolve().parents[3]
TEACHER_ID = "damage-aware@v3"
BASE = {"training_set_hash": "a" * 64, "stream_sha256": "b" * 64, "stream_bytes": 1000}
OFFSETS = [0, 10, 25, 400, 999]


def encoders() -> dict[str, Encoder]:
    return {role: Encoder(game(), tiny_config().model.encoder) for role in ROLES}


def fake_encoders(version: str = "enc-v", data: str = "e" * 64) -> dict[str, Any]:
    return {
        role: SimpleNamespace(version=version, game=SimpleNamespace(engine_data_hash=data))
        for role in ROLES
    }


def teacher_dir(tmp_path: Path, seed: int = 0) -> tuple[Path, dict[str, Any], str]:
    """A teacher directory (``stream.jsonl``, ``training-set.json``): path, manifest, its hash."""
    text, manifest = make_stream(policy=TEACHER_ID, seed=seed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "stream.jsonl").write_text(text)
    (tmp_path / "training-set.json").write_text(jcs(manifest).decode() + "\n")
    return tmp_path, manifest, sha256_hex(jcs(manifest))


def markers(directory: Path) -> list[Path]:
    return sorted(directory.glob("verified-*.json"))


def independent_offsets(stream: Path) -> list[int]:
    """The byte offsets of the Battle examples, found with nothing of the module under test."""
    out, position = [], 0
    for raw in stream.read_bytes().splitlines(keepends=True):
        if json.loads(raw)["observation"]["kind"] == "battle":
            out.append(position)
        position += len(raw)
    return out


class Spies:
    """Counts of the work a full pass does: the line checks and the encode of every example."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.checked = 0
        self.digests = 0
        self.encoded = 0
        real_file, real_digest = check_stream_file, check_stream_digest
        real_add = _Collector.add

        def file(*args: Any, **kwargs: Any) -> Any:
            self.checked += 1
            return real_file(*args, **kwargs)

        def digest(*args: Any, **kwargs: Any) -> Any:
            self.digests += 1
            return real_digest(*args, **kwargs)

        def add(collector: Any, example: dict[str, Any]) -> None:
            self.encoded += 1
            real_add(collector, example)

        monkeypatch.setattr(imitation, "check_stream_file", file)
        monkeypatch.setattr(imitation, "check_stream_digest", digest)
        monkeypatch.setattr(_Collector, "add", add)

    def reset(self) -> None:
        self.checked = self.digests = self.encoded = 0


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Spies:
    return Spies(monkeypatch)


def child(mode: str, directory: Path, *extra: str) -> subprocess.Popen[str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.ml.learner.teacher_marker_child",
            mode,
            str(directory),
            *extra,
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# -- the marker is written by a full pass and honoured by the next open -------------------------


def test_a_full_pass_writes_a_marker_and_the_next_open_honours_it(
    tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    first = open_teacher(stream, manifest, digest, encoders())
    assert (spies.checked, spies.encoded > 0) == (1, True)
    assert len(markers(directory)) == 1
    assert not list(directory.glob(".*"))  # no temporary file is left

    spies.reset()
    second = open_teacher(stream, manifest, digest, encoders())
    # nothing but the digest check ran: no line check, no encode of any example
    assert (spies.checked, spies.encoded, spies.digests) == (0, 0, 1)
    assert len(second) == len(first) == 55
    assert second._offsets == first._offsets
    assert [s.chosen for s in second.fetch(range(55))] == [s.chosen for s in first.fetch(range(55))]


def test_the_offsets_equal_the_full_passes_and_an_independent_scan(
    tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    full = open_teacher(stream, manifest, digest, encoders())
    honoured = open_teacher(stream, manifest, digest, encoders())
    assert spies.checked == 1  # the second open used the marker
    expected = independent_offsets(stream)
    assert list(full._offsets) == list(honoured._offsets) == expected
    assert 0 < len(expected) < sum(1 for _ in stream.open("rb"))  # previews are not in it
    assert teacher_marker.read_marker(directory, key_of(manifest, digest, stream)) == expected


def key_of(manifest: dict[str, Any], digest: str, stream: Path) -> dict[str, Any]:
    return teacher_marker.marker_key(
        training_set_hash=digest,
        stream_sha256=manifest["stream_sha256"],
        stream_bytes=stream.stat().st_size,
        encoders=encoders(),
    )


def test_a_marker_written_by_the_public_path_is_whole_and_canonical(tmp_path: Path) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    open_teacher(directory / "stream.jsonl", manifest, digest, encoders())
    (path,) = markers(directory)
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == jcs(value) + b"\n"
    assert value["format"] == teacher_marker.FORMAT
    assert value["key"]["training_set_hash"] == digest
    assert value["key"]["stream_sha256"] == manifest["stream_sha256"]
    assert value["key"]["stream_bytes"] == (directory / "stream.jsonl").stat().st_size
    assert path.name == f"verified-{sha256_hex(jcs(value['key']))}.json"
    assert value["offsets"] == independent_offsets(directory / "stream.jsonl")


# -- every part of the key invalidates the marker ------------------------------------------------


def with_marker(directory: Path, key: dict[str, Any]) -> None:
    assert teacher_marker.write_marker(directory, key, OFFSETS)
    assert teacher_marker.read_marker(directory, key) == OFFSETS


def build(**change: Any) -> dict[str, Any]:
    args = {**BASE, "encoders": fake_encoders(), **change}
    return teacher_marker.marker_key(**args)


def wrong_role(version: str = "enc-v", data: str = "e" * 64) -> dict[str, Any]:
    out = fake_encoders()
    out["battle"] = SimpleNamespace(version=version, game=SimpleNamespace(engine_data_hash=data))
    return out


def other_libraries(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    real = teacher_marker.library_versions
    monkeypatch.setattr(teacher_marker, "library_versions", lambda: {**real(), "jsonschema": "0"})
    return build()


PARTS: dict[str, Callable[[pytest.MonkeyPatch], dict[str, Any]]] = {
    "the training set hash": lambda mp: build(training_set_hash="c" * 64),
    "the stream sha256": lambda mp: build(stream_sha256="d" * 64),
    "the stream size": lambda mp: build(stream_bytes=1001),
    "the encoder version of one role": lambda mp: build(encoders=wrong_role(version="enc-w")),
    "the engine-data hash of one role": lambda mp: build(encoders=wrong_role(data="f" * 64)),
    "the validating code": lambda mp: (
        mp.setattr(teacher_marker, "code_identity", lambda roots=None: "0" * 64) or build()
    ),
    "a library the validator uses": other_libraries,
}


@pytest.mark.parametrize("part", list(PARTS))
def test_every_part_of_the_key_invalidates_the_marker(
    part: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = build()
    with_marker(tmp_path, key)
    changed = PARTS[part](monkeypatch)
    assert changed != key
    assert teacher_marker.marker_path(tmp_path, changed) != teacher_marker.marker_path(
        tmp_path, key
    )
    assert teacher_marker.read_marker(tmp_path, changed) is None
    # even the old marker's bytes under the new key's name are refused: the key is inside the file
    (tmp_path / teacher_marker.marker_path(tmp_path, changed).name).write_bytes(
        teacher_marker.marker_path(tmp_path, key).read_bytes()
    )
    assert teacher_marker.read_marker(tmp_path, changed) is None


def test_the_key_holds_the_encoder_of_every_role_and_the_libraries(tmp_path: Path) -> None:
    key = build()
    assert set(key["encoders"]) == set(ROLES)
    assert key["encoders"]["battle"] == {"engine_data_hash": "e" * 64, "version": "enc-v"}
    assert {"jsonschema", "pydantic", "torch"} <= set(key["libraries"])
    assert key["code_sha256"] == teacher_marker.code_identity()


def test_a_key_in_a_changed_open_runs_the_full_pass_again(
    tmp_path: Path, spies: Spies, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    open_teacher(stream, manifest, digest, encoders())
    spies.reset()
    monkeypatch.setattr(teacher_marker, "code_identity", lambda roots=None: "9" * 64)
    open_teacher(stream, manifest, digest, encoders())  # the validating code changed
    assert (spies.checked, spies.encoded > 0) == (1, True)
    assert len(markers(directory)) == 2  # the new key has its own marker, the old one stays
    spies.reset()
    open_teacher(stream, manifest, digest, encoders())  # and now that one is honoured
    assert (spies.checked, spies.encoded) == (0, 0)


def test_another_encoder_runs_the_full_pass(tmp_path: Path, spies: Spies) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    open_teacher(stream, manifest, digest, encoders())
    spies.reset()
    wider = tiny_config().model.encoder
    wider = dataclasses.replace(wider, volatile_buckets=wider.volatile_buckets * 2)
    other = {role: Encoder(game(), wider) for role in ROLES}
    assert other["battle"].version != encoders()["battle"].version
    open_teacher(stream, manifest, digest, other)
    assert (spies.checked, spies.encoded > 0) == (1, True)


# -- a stream or a marker that changed is refused by the full path -------------------------------


def test_a_stream_with_one_byte_flipped_at_the_same_size_is_refused_by_the_full_path(
    tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    open_teacher(stream, manifest, digest, encoders())
    raw = bytearray(stream.read_bytes())
    at = raw.index(b'"choice"') + 11
    raw[at] = raw[at] ^ 1
    assert len(raw) == stream.stat().st_size
    stream.write_bytes(bytes(raw))
    spies.reset()
    with pytest.raises(LearnerDataError, match="does not hash to its TrainingSet's stream_sha256"):
        open_teacher(stream, manifest, digest, encoders())
    assert spies.checked == 1  # the full path refused it, not the marker
    assert spies.encoded == 0


def test_a_stream_of_another_size_is_not_trusted(tmp_path: Path, spies: Spies) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    open_teacher(stream, manifest, digest, encoders())
    stream.write_bytes(stream.read_bytes() + b"\n")
    with pytest.raises(LearnerDataError):
        open_teacher(stream, manifest, digest, encoders())


def test_a_restored_stream_is_trusted_again_without_a_new_pass(
    tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    original = stream.read_bytes()
    open_teacher(stream, manifest, digest, encoders())
    stream.write_bytes(original[:-1] + b"X")
    with pytest.raises(LearnerDataError):
        open_teacher(stream, manifest, digest, encoders())
    stream.write_bytes(original)
    spies.reset()
    open_teacher(stream, manifest, digest, encoders())
    assert (spies.checked, spies.encoded) == (0, 0)


def damage_truncated(path: Path) -> None:
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def damage_flipped_offset(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["offsets"][1] += 1  # still ascending and in range, but the hash no longer holds
    path.write_bytes(jcs(value) + b"\n")


def damage_not_json(path: Path) -> None:
    path.write_bytes(b"\x00\xff not json")


def damage_empty(path: Path) -> None:
    path.write_bytes(b"")


def damage_format(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["format"] = "teacher-marker/2"
    value["sha256"] = sha256_hex(jcs({k: v for k, v in value.items() if k != "sha256"}))
    path.write_bytes(jcs(value) + b"\n")


def reseal(value: dict[str, Any]) -> bytes:
    value["sha256"] = sha256_hex(jcs({k: v for k, v in value.items() if k != "sha256"}))
    return jcs(value) + b"\n"


def damage_descending(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["offsets"].reverse()
    path.write_bytes(reseal(value))


def damage_beyond_the_stream(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["offsets"][-1] = value["key"]["stream_bytes"]
    path.write_bytes(reseal(value))


def damage_not_integers(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["offsets"][0] = True
    path.write_bytes(reseal(value))


def damage_extra_key(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["note"] = "x"
    path.write_bytes(reseal(value))


def damage_a_float(path: Path) -> None:
    value = json.loads(path.read_bytes())
    value["offsets"][0] = 0.5
    path.write_bytes(json.dumps(value).encode())


DAMAGE: dict[str, Callable[[Path], None]] = {
    "truncated": damage_truncated,
    "a changed offset": damage_flipped_offset,
    "not json": damage_not_json,
    "empty": damage_empty,
    "another format version": damage_format,
    "descending offsets": damage_descending,
    "an offset past the stream": damage_beyond_the_stream,
    "an offset that is no integer": damage_not_integers,
    "an extra key": damage_extra_key,
    "a float": damage_a_float,
}


@pytest.mark.parametrize("damage", list(DAMAGE))
def test_a_damaged_marker_falls_back_to_the_full_pass_and_is_rewritten(
    damage: str, tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    first = open_teacher(stream, manifest, digest, encoders())
    (path,) = markers(directory)
    DAMAGE[damage](path)
    spies.reset()
    again = open_teacher(stream, manifest, digest, encoders())
    assert (spies.checked, spies.encoded > 0) == (1, True)
    assert list(again._offsets) == list(first._offsets)
    spies.reset()
    open_teacher(stream, manifest, digest, encoders())  # the rewrite is honoured
    assert (spies.checked, spies.encoded) == (0, 0)
    assert len(markers(directory)) == 1


def test_a_marker_of_another_teacher_directory_is_not_used(tmp_path: Path, spies: Spies) -> None:
    a, manifest_a, digest_a = teacher_dir(tmp_path / "a", seed=0)
    b, manifest_b, digest_b = teacher_dir(tmp_path / "b", seed=1)
    assert manifest_a["stream_sha256"] != manifest_b["stream_sha256"]
    open_teacher(a / "stream.jsonl", manifest_a, digest_a, encoders())
    (found,) = markers(a)
    # copied under its own name: no marker of b's key exists, so b runs the full pass
    (b / found.name).write_bytes(found.read_bytes())
    spies.reset()
    open_teacher(b / "stream.jsonl", manifest_b, digest_b, encoders())
    assert (spies.checked, spies.encoded > 0) == (1, True)
    # and under b's own name it is refused by the key inside it
    own = teacher_marker.marker_path(b, key_of(manifest_b, digest_b, b / "stream.jsonl"))
    own.write_bytes(found.read_bytes())
    spies.reset()
    open_teacher(b / "stream.jsonl", manifest_b, digest_b, encoders())
    assert (spies.checked, spies.encoded > 0) == (1, True)


def test_the_refusals_of_the_full_path_are_unchanged_with_a_marker_around(
    tmp_path: Path, spies: Spies
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    open_teacher(stream, manifest, digest, encoders())
    with pytest.raises(LearnerDataError, match=r"does not hash to ppo\.imitation"):
        open_teacher(stream, manifest, "0" * 64, encoders())
    labelled = {**manifest, "search_labels": {"search_identity_hash": "c" * 64, "labelled": 1}}
    with pytest.raises(LearnerDataError, match="search_labels"):
        open_teacher(stream, labelled, sha256_hex(jcs(labelled)), encoders())
    private = {**manifest, "data_class": "private"}
    with pytest.raises(LearnerDataError, match="not synthetic"):
        open_teacher(stream, private, sha256_hex(jcs(private)), encoders())


def test_a_manifest_without_a_digest_reaches_the_full_paths_refusal(tmp_path: Path) -> None:
    directory, manifest, _ = teacher_dir(tmp_path)
    broken = {k: v for k, v in manifest.items() if k != "stream_sha256"}
    with pytest.raises(ContractError, match="stream_sha256"):  # as the pass raised it before
        open_teacher(directory / "stream.jsonl", broken, sha256_hex(jcs(broken)), encoders())
    assert not markers(directory)


def test_a_missing_stream_keeps_the_full_paths_error(tmp_path: Path) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    (directory / "stream.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        open_teacher(directory / "stream.jsonl", manifest, digest, encoders())


# -- no marker without a full pass that succeeded ------------------------------------------------


def refused_stream(tmp_path: Path) -> tuple[Path, dict[str, Any], str]:
    """A stream whose examples carry rows the run's encoder (no feature set) refuses."""
    text, _ = make_stream(policy=TEACHER_ID)
    lines = [json.loads(line) for line in text.splitlines()]
    for example in lines:
        if example["observation"]["kind"] == "battle":
            example["candidate_features"] = {
                "feature_set": "probe-damage/1",
                "feature_set_hash": "c" * 64,
                "rows": [],
            }
    refused, manifest = stream_of(lines, TEACHER_ID)
    (tmp_path / "stream.jsonl").write_text(refused)
    return tmp_path / "stream.jsonl", manifest, sha256_hex(jcs(manifest))


def test_an_encoder_refusal_leaves_no_marker(tmp_path: Path) -> None:
    stream, manifest, digest = refused_stream(tmp_path)
    with pytest.raises(LearnerDataError):
        open_teacher(stream, manifest, digest, encoders())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stream.jsonl"]


def test_an_exception_in_the_pass_leaves_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    real = _Collector.add
    seen = 0

    def add(self: Any, example: dict[str, Any]) -> None:
        nonlocal seen
        seen += 1
        if seen == 7:
            raise RuntimeError("interrupted")
        real(self, example)

    monkeypatch.setattr(_Collector, "add", add)
    with pytest.raises(RuntimeError, match="interrupted"):
        open_teacher(directory / "stream.jsonl", manifest, digest, encoders())
    assert not markers(directory)
    assert not list(directory.glob(".*"))


def test_a_sigkill_in_the_full_pass_leaves_no_marker(tmp_path: Path) -> None:
    directory, _, _ = teacher_dir(tmp_path)
    process = child("kill", directory, "9")
    process.communicate(timeout=300)
    assert process.returncode == -9
    assert sorted(p.name for p in directory.iterdir()) == ["stream.jsonl", "training-set.json"]
    # the next process, never killed, pays the pass and leaves a marker that a third one honours
    first = child("open", directory)
    out, err = first.communicate(timeout=300)
    assert first.returncode == 0, err
    assert len(markers(directory)) == 1
    assert out.strip() == "55"


def test_a_marker_that_cannot_be_written_does_not_fail_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("read-only")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    opened = open_teacher(directory / "stream.jsonl", manifest, digest, encoders())
    assert len(opened) == 55
    assert not markers(directory)
    assert teacher_marker.write_marker(directory, build(), OFFSETS) is False


def test_a_stream_that_changed_during_the_pass_leaves_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    stream = directory / "stream.jsonl"
    real = imitation._full_pass

    def grown(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        with stream.open("ab") as handle:
            handle.write(b"\n")
        return found

    monkeypatch.setattr(imitation, "_full_pass", grown)
    assert len(open_teacher(stream, manifest, digest, encoders())) == 55
    assert not markers(directory)


# -- two processes ------------------------------------------------------------------------------


def test_two_processes_opening_at_once_leave_one_whole_marker(tmp_path: Path, spies: Spies) -> None:
    directory, manifest, digest = teacher_dir(tmp_path)
    first, second = child("open", directory), child("open", directory)
    results = [(*p.communicate(timeout=300), p.returncode) for p in (first, second)]
    assert [r[2] for r in results] == [0, 0], results
    assert [r[0].strip() for r in results] == ["55", "55"]
    assert len(markers(directory)) == 1
    assert not list(directory.glob(".*"))
    open_teacher(directory / "stream.jsonl", manifest, digest, encoders())
    assert (spies.checked, spies.encoded) == (0, 0)  # the marker the two raced to write is whole


def test_concurrent_writers_never_leave_a_torn_marker(tmp_path: Path) -> None:
    key = build()
    path = teacher_marker.marker_path(tmp_path, key)
    wide = list(range(900))
    done = False

    def write(n: int) -> bool:
        return all(
            teacher_marker.write_marker(tmp_path, key, wide[: 100 + n * 50]) for _ in range(25)
        )

    def read() -> Iterator[bool]:
        while not done:
            if path.exists():
                # a rename is atomic, so whatever is there is a whole file of one writer
                value = json.loads(path.read_bytes())
                yield value["sha256"] == sha256_hex(
                    jcs({k: v for k, v in value.items() if k != "sha256"})
                )

    with ThreadPoolExecutor(max_workers=6) as pool:
        writers = [pool.submit(write, n) for n in range(5)]
        reads = pool.submit(lambda: list(read()))
        assert all(w.result(timeout=120) for w in writers)
        done = True
        checked = reads.result(timeout=120)
    assert checked
    assert all(checked)
    assert teacher_marker.read_marker(tmp_path, key) is not None
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


# -- the validating code --------------------------------------------------------------------------


def tree(base: Path) -> list[tuple[Path, tuple[str, ...]]]:
    return [(base / "pkg", ("*.py", "*.json"))]


def test_the_code_identity_follows_every_source_file_of_its_roots(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__pycache__").mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    (pkg / "sub" / "data.json").write_text("[1]\n")

    def identity() -> str:
        return teacher_marker.code_identity(tree(tmp_path))

    def changes(edit: Callable[[], object], undo: Callable[[], object]) -> bool:
        before = identity()
        edit()
        moved = identity() != before
        undo()
        return moved and identity() == before  # and back to the same value when it is undone

    base = identity()
    assert base == identity()  # stable
    (pkg / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00junk")
    (pkg / "notes.txt").write_text("not source\n")
    assert identity() == base  # not source: not identity
    assert changes(
        lambda: (pkg / "a.py").write_text("x = 2\n"), lambda: (pkg / "a.py").write_text("x = 1\n")
    )
    assert changes(
        lambda: (pkg / "sub" / "b.py").write_text("y = 2  # edited\n"),
        lambda: (pkg / "sub" / "b.py").write_text("y = 2\n"),
    )
    assert changes(
        lambda: (pkg / "sub" / "data.json").write_text("[2]\n"),
        lambda: (pkg / "sub" / "data.json").write_text("[1]\n"),
    )
    assert changes(lambda: (pkg / "c.py").write_text(""), (pkg / "c.py").unlink)
    assert changes(  # a moved file
        lambda: (pkg / "a.py").rename(pkg / "sub" / "a.py"),
        lambda: (pkg / "sub" / "a.py").rename(pkg / "a.py"),
    )
    removed = (pkg / "a.py").read_bytes()
    assert changes((pkg / "a.py").unlink, lambda: (pkg / "a.py").write_bytes(removed))
    assert identity() == base


def test_the_code_identity_is_the_same_whichever_directory_holds_the_tree(tmp_path: Path) -> None:
    for name in ("one", "two"):
        (tmp_path / name / "pkg").mkdir(parents=True)
        (tmp_path / name / "pkg" / "a.py").write_text("x = 1\n")
    assert teacher_marker.code_identity(tree(tmp_path / "one")) == teacher_marker.code_identity(
        tree(tmp_path / "two")
    )


def test_the_default_identity_covers_what_the_validation_runs() -> None:
    files = {p.relative_to(ROOT).as_posix() for p in teacher_marker.code_files()}
    for needed in (
        "ml/areneon_ml/learner/imitation.py",
        "ml/areneon_ml/learner/teacher_marker.py",
        "ml/areneon_ml/learner/data.py",
        "ml/areneon_ml/model/encoder.py",
        "ml/areneon_ml/reducer/reducer.py",
        "ml/areneon_ml/worker/view.py",
        "ml/areneon_ml/reducer/data/stall-moves.gen9championsvgc2026regmc.json",
        "packages/schema/python/areneon_contracts/training.py",
        "packages/schema/python/areneon_contracts/contracts.py",
        "packages/schema/contracts/v1/training-example.schema.json",
        "packages/schema/contracts/v1/training-set.schema.json",
    ):
        assert needed in files, needed
    assert not any("__pycache__" in f for f in files)
    assert teacher_marker.code_identity() == teacher_marker.code_identity()
    assert len(teacher_marker.code_identity()) == 64


def test_every_module_the_validation_imports_is_in_the_identity() -> None:
    probe = (
        "import json, sys; import areneon_ml.learner.imitation; "
        "print(json.dumps(sorted(m.__file__ for m in sys.modules.values() "
        "if getattr(m, '__file__', None))))"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    run = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    covered = {p.resolve() for p in teacher_marker.code_files()}
    ours = [
        Path(f).resolve()
        for f in json.loads(run.stdout)
        if os.path.isabs(f)  # torch.ops and torch.classes name a bare file
        and f.endswith(".py")
        and Path(f).resolve().is_relative_to(ROOT)
        and ".venv" not in Path(f).parts
        and "node_modules" not in Path(f).parts
        and "vendor" not in Path(f).parts
    ]
    assert ours  # areneon's own modules were found
    assert [p for p in ours if p not in covered] == []


# -- the small branches ---------------------------------------------------------------------------


def test_a_library_that_is_not_installed_is_named_with_no_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert teacher_marker.library_versions() == {"jsonschema": "", "pydantic": "", "torch": ""}


def test_a_marker_larger_than_any_marker_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = build()
    with_marker(tmp_path, key)
    monkeypatch.setattr(teacher_marker, "MAX_MARKER_BYTES", 10)
    assert teacher_marker.read_marker(tmp_path, key) is None


def test_offsets_that_are_no_list_are_refused_even_when_sealed(tmp_path: Path) -> None:
    key = build()
    with_marker(tmp_path, key)
    path = teacher_marker.marker_path(tmp_path, key)
    value = json.loads(path.read_bytes())
    value["offsets"] = "0,10"
    path.write_bytes(reseal(value))
    assert teacher_marker.read_marker(tmp_path, key) is None


def test_a_failed_rename_that_cannot_clean_up_still_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("refused")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(os, "unlink", refuse)
    assert teacher_marker.write_marker(tmp_path, build(), OFFSETS) is False
    assert not markers(tmp_path)
