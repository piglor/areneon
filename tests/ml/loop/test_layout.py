"""The run directory: atomic generation commits, manifests, and what is refused (ADR-010 §1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from areneon_ml.loop import layout


def build(run_dir: Path, generation: int, files: dict[str, bytes]) -> Path:
    work = layout.partial_dir(run_dir, generation)
    for name, data in files.items():
        layout.write_file(work / name, data)
    return work


def test_a_generation_is_committed_by_renaming_its_finished_partial(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a.txt": b"one", "sub/b.txt": b"two"})
    target = layout.commit_generation(tmp_path, 0, {"note": "x"})
    assert target == layout.generation_dir(tmp_path, 0)
    assert not layout.partial_dir(tmp_path, 0).exists()
    manifest = layout.verify_generation(target, 0)
    assert manifest["note"] == "x"
    assert sorted(manifest["files"]) == ["a.txt", "sub/b.txt"]
    found = layout.latest_complete(tmp_path)
    assert found is not None
    assert (found.generation, found.path) == (0, target)


def test_a_partial_is_not_a_generation_and_is_discarded(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a": b"1"})
    layout.commit_generation(tmp_path, 0, {})
    build(tmp_path, 1, {"a": b"1"})
    found = layout.latest_complete(tmp_path)
    assert found is not None
    assert found.generation == 0
    assert layout.discard_partials(tmp_path) == [".g000001.partial"]
    assert not layout.partial_dir(tmp_path, 1).exists()
    assert layout.discard_partials(tmp_path) == []


def test_an_empty_run_directory_has_no_generation(tmp_path: Path) -> None:
    assert layout.latest_complete(tmp_path) is None
    assert layout.discard_partials(tmp_path) == []


def test_a_generation_cannot_be_committed_twice(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a": b"1"})
    layout.commit_generation(tmp_path, 0, {})
    build(tmp_path, 0, {"a": b"1"})
    with pytest.raises(layout.RunDirError, match="already committed"):
        layout.commit_generation(tmp_path, 0, {})


def test_a_changed_added_or_removed_file_fails_verification(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a": b"1", "b": b"2"})
    path = layout.commit_generation(tmp_path, 0, {})
    (path / "a").write_bytes(b"changed")
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0)
    (path / "a").write_bytes(b"1")
    layout.verify_generation(path, 0)
    (path / "extra").write_bytes(b"x")
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0)
    (path / "extra").unlink()
    (path / "b").unlink()
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0)


def test_a_manifest_of_another_generation_is_refused(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a": b"1"})
    path = layout.commit_generation(tmp_path, 0, {})
    with pytest.raises(layout.RunDirError, match="not generation 1's"):
        layout.verify_generation(path, 1)
    layout.write_json(path / layout.MANIFEST, [])
    with pytest.raises(layout.RunDirError, match="not generation 0's"):
        layout.verify_generation(path, 0)


def test_generations_must_run_from_zero_without_a_gap(tmp_path: Path) -> None:
    for generation in (0, 2):
        build(tmp_path, generation, {"a": b"1"})
        layout.commit_generation(tmp_path, generation, {})
    with pytest.raises(layout.RunDirError, match=r"\[0, 2\], not 0..1"):
        layout.latest_complete(tmp_path)


def test_a_stray_entry_among_the_generations_is_refused(tmp_path: Path) -> None:
    build(tmp_path, 0, {"a": b"1"})
    layout.commit_generation(tmp_path, 0, {})
    (layout.generations_dir(tmp_path) / "notes.txt").write_text("x")
    with pytest.raises(layout.RunDirError, match="not a generation of this run"):
        layout.latest_complete(tmp_path)


def test_json_files_are_canonical_with_one_newline_and_read_back_strictly(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    assert layout.write_json(path, {"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}\n'
    assert layout.read_json(path) == {"a": [1, 2], "b": 1}
    for raw, message in [
        (b'{"a": 1}\n', "RFC 8785"),
        (b'{"a":1}', "RFC 8785"),
        (b"\xff\n", "not canonical JSON"),
        (b"{\n", "not canonical JSON"),
    ]:
        path.write_bytes(raw)
        with pytest.raises(layout.RunDirError, match=message):
            layout.read_json(path)


def test_names_are_zero_padded(tmp_path: Path) -> None:
    assert layout.generation_name(7) == "g000007"
    assert layout.partial_dir(tmp_path, 7).name == ".g000007.partial"
    assert json.loads(layout.write_json(tmp_path / "n.json", {"n": 1})) == {"n": 1}


def fsync_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every path os.fsync is called on, and every os.replace as ``rename:<target>``."""
    log: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(descriptor: int) -> None:
        log.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        real_fsync(descriptor)

    def replace(source: Path, target: Path) -> None:
        log.append(f"rename:{target}")
        real_replace(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    return log


def test_a_commit_flushes_every_file_and_directory_before_the_one_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = layout.partial_dir(tmp_path, 0)
    # a file made without fsync, as the Node side does
    (work / "records").mkdir(parents=True)
    (work / "records" / "a.json").write_bytes(b"1")
    (work / "episodes").mkdir()
    (work / "episodes" / "e.json").write_bytes(b"2")
    log = fsync_log(monkeypatch)
    layout.commit_generation(tmp_path, 0, {})
    rename = log.index(f"rename:{layout.generation_dir(tmp_path, 0)}")
    before = set(log[:rename])
    for path in [
        work / "records" / "a.json",
        work / "episodes" / "e.json",
        work / "records",
        work / "episodes",
        work,
        work / layout.MANIFEST,
    ]:
        assert str(path) in before, path
    # the manifest is itself flushed, and the directory entry that names it, before the rename
    manifest_at = log.index(str(work / layout.MANIFEST))
    assert str(work) in log[manifest_at + 1 : rename]
    # the manifest was hashed from flushed bytes: the files were flushed before it was written
    assert log.index(str(work / "records" / "a.json")) < log.index(str(work / layout.MANIFEST))
    # and the parent of the renamed directory is flushed after the rename
    assert str(layout.generations_dir(tmp_path)) in log[rename + 1 :]


def test_an_atomic_write_leaves_all_or_nothing_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run-config.json"
    log = fsync_log(monkeypatch)
    layout.write_atomic(path, b"one")
    layout.write_atomic(path, b"two")
    assert path.read_bytes() == b"two"
    assert [p.name for p in tmp_path.iterdir()] == ["run-config.json"]
    assert str(tmp_path) in log  # the directory holding the rename


def test_a_second_driver_on_a_run_directory_fails_fast_until_the_first_is_gone(
    tmp_path: Path,
) -> None:
    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            "import sys; from pathlib import Path; from areneon_ml.loop.layout import run_lock\n"
            "with run_lock(Path(sys.argv[1])):\n"
            "    print('locked', flush=True); sys.stdin.read()",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with (
            pytest.raises(layout.RunDirError, match="another process is running"),
            layout.run_lock(tmp_path),
        ):
            pass
        holder.kill()  # a killed holder releases the lock: it is the operating system's
        holder.wait()
        with layout.run_lock(tmp_path):
            pass
    finally:
        holder.kill()
        holder.wait()


def test_stale_checkpoint_temp_files_are_discarded(tmp_path: Path) -> None:
    store = tmp_path / "checkpoints"
    store.mkdir()
    (store / f".{'a' * 64}.partial").write_bytes(b"half")
    (store / f"{'b' * 64}.safetensors").write_bytes(b"whole")
    assert layout.discard_stale_checkpoint_files(tmp_path) == [f".{'a' * 64}.partial"]
    assert [p.name for p in store.iterdir()] == [f"{'b' * 64}.safetensors"]
    assert layout.discard_stale_checkpoint_files(tmp_path / "none") == []


def test_the_run_directory_and_its_entries_are_flushed_when_first_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = fsync_log(monkeypatch)
    layout.ensure_run_dirs(run_dir)
    assert {str(run_dir / "generations"), str(run_dir / "checkpoints")} <= set(log)
    assert log[-2:] == [str(run_dir), str(tmp_path)]
    log.clear()
    layout.ensure_run_dirs(run_dir)  # nothing new to flush but the parents
    assert log == [str(run_dir), str(tmp_path)]
