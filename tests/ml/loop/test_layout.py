"""The run directory: atomic generation commits, manifests, and what is refused (ADR-010 §1)."""

from __future__ import annotations

import json
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
