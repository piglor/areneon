"""The provenance export of a population run (ADR-021 slice 12): C1 and C6 over members."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import main
from areneon_ml.loop.driver import (
    LoopError,
    export_provenance,
    population_learner_steps,
    provenance_graph,
    sources_registry,
)
from areneon_ml.loop.warm_start import import_clone
from areneon_ml.model.network import ROLES

from tests.ml.loop.test_population import (
    LABELS,
    gen_path,
    make_population,
    population_config,
    reseal,
)
from tests.ml.loop.test_population_copy import GENERATIONS, run_to
from tests.ml.loop.test_warm_start import clone_ids, clone_output


@pytest.fixture(scope="module")
def copying(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("copying-provenance")
    run_to(run_dir, GENERATIONS)
    return run_dir


def manifest_members(run_dir: Path, generation: int) -> dict[str, Any]:
    raw = layout.read_json(gen_path(run_dir, generation) / "manifest.json")
    members: dict[str, Any] = raw["members"]
    return members


def pair(run_dir: Path, generation: int, member: str) -> list[str]:
    return list(manifest_members(run_dir, generation)[member]["checkpoints"].values())


def test_every_member_and_snapshot_of_one_run_is_mutually_seen(copying: Path) -> None:
    graph = provenance_graph(copying)
    for g in range(1, GENERATIONS + 1):
        for a in LABELS:
            for b in LABELS:
                if a == b:
                    continue
                # a's pair of generation g against b's pair of generation g: seen, in both ways.
                assert graph.classify(pair(copying, g, a), pair(copying, g, b)).label == "seen"
                assert not graph.independent(pair(copying, g, a), pair(copying, g, b)).independent
    # An earlier snapshot of another member is in the ancestry of a later pair (league_opponents).
    later = graph.ancestry(pair(copying, GENERATIONS, "m0")).checkpoints
    assert set(pair(copying, GENERATIONS - 1, "m1")) <= later


def test_a_separate_run_is_independent(copying: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    make_population(other, population_config(key="pop-other")).run(1)
    records = [read_record(json.loads(p.read_text())) for p in _records(copying, tmp_path / "a")]
    records += [read_record(json.loads(p.read_text())) for p in _records(other, tmp_path / "b")]
    graph = ProvenanceGraph(records, {**sources_registry(copying), **sources_registry(other)})
    for member in LABELS:
        a, b = pair(copying, GENERATIONS, member), pair(other, 1, member)
        assert graph.independent(a, b).independent
        assert graph.classify(a, b).label == "held_out"


def _records(run_dir: Path, out: Path) -> list[Path]:
    export_provenance(run_dir, out)
    return sorted((out / "records").glob("*.json"))


def test_the_copy_enters_through_the_parent_and_cumulative_steps_follow_it(
    copying: Path,
) -> None:
    steps = population_learner_steps(copying)
    donor_then = manifest_members(copying, 1)["m0"]["checkpoints"]
    loser_then = manifest_members(copying, 1)["m1"]["checkpoints"]
    loser_now = manifest_members(copying, 2)["m1"]["checkpoints"]
    for role in ROLES:
        assert (
            read_record(
                layout.read_json(gen_path(copying, 2) / "members/m1/records" / f"{role}.json")
            ).parent
            == donor_then[role]
        )
        report = layout.read_json(gen_path(copying, 2) / "members/m1/train-report.json")[role]
        # The slot inherits the donor's count and adds its own (a value-only update here).
        assert steps[loser_now[role]] == steps[donor_then[role]] + report["optimiser_steps"]
        assert steps[donor_then[role]] > 0
        assert steps[loser_then[role]] > 0  # the loser's own history is not the donor's
    # A member that did not copy adds its own steps to its own count.
    for role in ROLES:
        own = manifest_members(copying, 2)["m0"]["checkpoints"][role]
        before = manifest_members(copying, 1)["m0"]["checkpoints"][role]
        report = layout.read_json(gen_path(copying, 2) / "members/m0/train-report.json")[role]
        assert steps[own] == steps[before] + report["optimiser_steps"]


def test_the_export_writes_one_record_per_checkpoint_and_the_registry(
    copying: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    digest = export_provenance(copying, out)
    written = {p.stem for p in (out / "records").glob("*.json")}
    expected = {c for g in range(GENERATIONS + 1) for m in LABELS for c in pair(copying, g, m)}
    assert written == expected
    registry = (out / "sources.json").read_bytes()
    assert sha256_hex(registry) == digest
    sources = json.loads(registry)
    # Generation g's set of a member was played by that member's pair frozen at g - 1.
    for g in range(1, GENERATIONS + 1):
        for m in LABELS:
            held = manifest_members(copying, g)[m]["training_set_hash"]
            assert sources[held] == sorted(pair(copying, g - 1, m))
    assert registry == jcs(sources_registry(copying))


def test_the_command_prints_the_cumulative_steps(
    copying: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["provenance", "--run-dir", str(copying), "--out", str(tmp_path / "o")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["learner_steps"] == population_learner_steps(copying)
    assert set(printed) >= {"sources_hash", "learner_steps"}


def test_a_warm_population_shares_one_clone_record_and_its_members_are_not_independent(
    tmp_path: Path,
) -> None:
    (tmp_path / "clone").mkdir()
    clone = clone_output(tmp_path / "clone")
    ids = clone_ids(clone)
    config = population_config(
        key="pop-warm-prov", edit=lambda raw: raw.update(warm_start={"battle": ids["battle"]})
    )
    run_dir = tmp_path / "run"
    import_clone(clone, run_dir, ("battle",))
    make_population(run_dir, config).run(2)
    out = tmp_path / "out"
    export_provenance(run_dir, out)
    files = {p.stem for p in (out / "records").glob("*.json")}
    assert ids["battle"] in files  # written once for every member
    unique = {c for g in range(3) for m in LABELS for c in pair(run_dir, g, m)}
    assert files == unique
    sources = json.loads((out / "sources.json").read_text())
    teacher = read_record(json.loads((out / "records" / f"{ids['battle']}.json").read_text()))
    assert all(sources[h] == [] for h in teacher.training_set_hashes)
    graph = provenance_graph(run_dir)
    a, b = pair(run_dir, 0, "m0"), pair(run_dir, 0, "m1")
    # The shared clone is a common ancestor: the members' generation-0 pairs are related.
    assert ids["battle"] in graph.ancestry(a).checkpoints & graph.ancestry(b).checkpoints
    assert not graph.independent(a, b).independent
    assert graph.classify(pair(run_dir, 2, "m0"), pair(run_dir, 2, "m1")).label == "seen"
    # Two records that differ for the shared clone checkpoint are refused.
    path = gen_path(run_dir, 0) / "members/m1/records/battle.json"
    forged = layout.read_json(path)
    forged["lineage_id"] = "elsewhere"
    layout.write_json(path, forged)
    reseal(run_dir, 0)
    with pytest.raises(LoopError, match="two different records"):
        export_provenance(run_dir, tmp_path / "o2")
