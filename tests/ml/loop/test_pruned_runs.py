"""Evaluating a run whose older generations had their bulk files pruned (``--allow-pruned``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import EXIT_REFUSED, main
from areneon_ml.loop.driver import LoopError, provenance_inputs
from areneon_ml.loop.population_provenance import PopulationProvenanceError, verified_manifests
from areneon_ml.report.model import ReportError, read_report_model

from tests.ml.loop.test_population_copy import GENERATIONS, run_to

KEPT = {
    "generation.json": b"g",
    "records/battle.json": b"r",
    "train-report.json": b"t",
    "learner-1.json": b"l",
    "members/m0/records/battle.json": b"mr",
    "members/m0/train-report.json": b"mt",
    "members/m0/learner-1.json": b"ml",
}
BULK = {
    "stream.jsonl": b"s",
    "learner-1.safetensors": b"w",
    "episodes/a.json": b"e",
    "episodes/deep/b.json": b"e2",
    "members/m0/stream.jsonl": b"ms",
    "members/m0/learner-1.safetensors": b"mw",
    "members/m0/episodes/c.json": b"me",
}


def committed(run_dir: Path, extra: dict[str, bytes] | None = None) -> Path:
    work = layout.partial_dir(run_dir, 0)
    for name, data in {**KEPT, **BULK, **(extra or {})}.items():
        layout.write_file(work / name, data)
    return layout.commit_generation(run_dir, 0, {})


def prune(path: Path) -> None:
    """What the storage cleaner deletes from a committed generation."""
    for name in BULK:
        (path / name).unlink()
    shutil.rmtree(path / "episodes")


def test_a_pruned_generation_verifies_only_with_the_flag(tmp_path: Path) -> None:
    path = committed(tmp_path)
    prune(path)
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0)
    manifest = layout.verify_generation(path, 0, allow_pruned=True)
    assert set(manifest["files"]) == set(KEPT) | set(BULK)
    # Nothing pruned also passes with the flag.
    assert layout.verify_generation(committed(tmp_path / "whole"), 0, allow_pruned=True)


@pytest.mark.parametrize("name", sorted(KEPT))
def test_a_missing_file_outside_the_bulk_set_fails_with_the_flag(tmp_path: Path, name: str) -> None:
    path = committed(tmp_path)
    prune(path)
    (path / name).unlink()
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)


@pytest.mark.parametrize(
    "name",
    [
        "members/m0/sub/stream.jsonl",  # not at a generation or member root
        "deep/learner-1.safetensors",
        "learner-g.safetensors",  # not a generation number
        "learner-1.safetensors.bak",
        "members/stream.jsonl",
        "episodes-old/a.json",  # not the episodes directory
    ],
)
def test_a_lookalike_of_a_bulk_file_may_not_be_missing(tmp_path: Path, name: str) -> None:
    path = committed(tmp_path, {name: b"x"})
    (path / name).unlink()
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)


def test_a_changed_or_extra_file_fails_with_the_flag(tmp_path: Path) -> None:
    path = committed(tmp_path)
    prune(path)
    (path / "train-report.json").write_bytes(b"changed")
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)
    (path / "train-report.json").write_bytes(b"t")
    layout.verify_generation(path, 0, allow_pruned=True)
    (path / "members/m0/stream.jsonl").write_bytes(b"not the stream")  # present: must match
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)
    (path / "members/m0/stream.jsonl").unlink()
    (path / "notes.txt").write_bytes(b"x")
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)


def test_a_manifest_without_a_file_table_fails_with_the_flag(tmp_path: Path) -> None:
    path = committed(tmp_path)
    layout.write_json(path / layout.MANIFEST, {"generation": 0, "files": []})
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        layout.verify_generation(path, 0, allow_pruned=True)


# -- a population run whose old generations were pruned ---


def prune_run(run_dir: Path, generation: int) -> None:
    path = layout.generation_dir(run_dir, generation)
    pruned = 0
    for name in layout.read_json(path / layout.MANIFEST)["files"]:
        if layout.prunable(name):
            (path / name).unlink()
            pruned += 1
    assert pruned > 0  # the fixture's generation holds bulk files


@pytest.fixture
def pruned_population(tmp_path: Path) -> tuple[Path, object]:
    run_dir = tmp_path / "run"
    run_to(run_dir, GENERATIONS)
    whole = provenance_inputs(run_dir)
    for generation in range(1, GENERATIONS):  # every generation older than the newest
        prune_run(run_dir, generation)
    return run_dir, whole


def test_population_provenance_of_a_pruned_run_needs_the_flag(
    pruned_population: tuple[Path, object],
) -> None:
    run_dir, whole = pruned_population
    with pytest.raises(LoopError, match="population: g000001: its files differ"):
        provenance_inputs(run_dir)
    assert provenance_inputs(run_dir, allow_pruned=True) == whole
    newest = layout.latest_complete(run_dir)
    assert newest is not None
    assert len(verified_manifests(run_dir, newest, allow_pruned=True)) == GENERATIONS + 1


def test_the_newest_generation_must_be_whole_even_with_the_flag(
    pruned_population: tuple[Path, object],
) -> None:
    run_dir, _ = pruned_population
    newest = layout.latest_complete(run_dir)
    assert newest is not None
    prune_run(run_dir, GENERATIONS)
    with pytest.raises(PopulationProvenanceError, match=f"g{GENERATIONS:06d}"):
        verified_manifests(run_dir, newest, allow_pruned=True)
    # The newest generation is found by verifying it whole, whatever the flag says.
    with pytest.raises(layout.RunDirError, match=f"g{GENERATIONS:06d}: its files differ"):
        provenance_inputs(run_dir, allow_pruned=True)


def test_a_missing_record_of_a_pruned_generation_fails_with_the_flag(
    pruned_population: tuple[Path, object],
) -> None:
    run_dir, _ = pruned_population
    (layout.generation_dir(run_dir, 1) / "members" / "m0" / "records" / "battle.json").unlink()
    with pytest.raises(LoopError, match="g000001: its files differ"):
        provenance_inputs(run_dir, allow_pruned=True)


def test_the_report_of_a_pruned_population_run_needs_the_flag(
    pruned_population: tuple[Path, object],
) -> None:
    run_dir, _ = pruned_population
    with pytest.raises(ReportError, match="differ from its manifest"):
        read_report_model(run_dir, None)
    model = read_report_model(run_dir, None, allow_pruned=True)
    assert [g.number for g in model.generations] == list(range(GENERATIONS + 1))
    prune_run(run_dir, GENERATIONS)
    with pytest.raises(ReportError, match="differ from its manifest"):
        read_report_model(run_dir, None, allow_pruned=True)


def test_the_report_command_takes_the_flag(
    pruned_population: tuple[Path, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = pruned_population
    out = tmp_path / "report.html"
    assert main(["report", "--run-dir", str(run_dir), "--out", str(out)]) == EXIT_REFUSED
    assert "g000001: its files differ from its manifest" in capsys.readouterr().err
    assert not out.exists()
    assert main(["report", "--run-dir", str(run_dir), "--out", str(out), "--allow-pruned"]) == 0
    assert out.is_file()
