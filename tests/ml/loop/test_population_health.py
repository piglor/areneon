"""A population run with a diverging member (ADR-021 sections 6 and 12, slice 9).

The learner of ``m1`` is forced to a non-finite loss at one generation. Its update is discarded
(its pair stays the one it had), the ladder records the ``health`` outcome, and the next
generation gives its slot a copy of the top live member.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.trainer import Learner, TrainReport
from areneon_ml.loop import layout, selection
from areneon_ml.loop.health import HealthIncident
from areneon_ml.loop.ladder import read_ladder_record
from areneon_ml.loop.population import PopulationLoop
from areneon_ml.model.network import ROLES

from tests.ml.loop.support import Killed, killer
from tests.ml.loop.test_population import (
    FakePopulationPlayer,
    gen_path,
    ladder_of,
    make_population,
    population_config,
    tree_bytes,
)
from tests.ml.model.support import game

GENERATIONS = 3
BAD_GENERATION = 2
MEMBERS = 4


def config_four() -> Any:
    return population_config(members=MEMBERS)


STATE: dict[str, Any] = {}


def reset() -> dict[str, Any]:
    STATE.clear()
    STATE.update(loop=None, members={"m1"}, generation=BAD_GENERATION)
    return STATE


def forced_train(self: Learner, trajectories: Any) -> TrainReport:
    """The real update, then (for the named members at the named generation) a NaN loss.

    The weights are filled with NaN too, so a pair frozen from this update would be unusable.
    """
    report = REAL_TRAIN(self, trajectories)
    loop = STATE["loop"]
    names = [m for m, learner in loop._learners.items() if learner is self]
    if names and names[0] in STATE["members"] and self.generation == STATE["generation"]:
        with torch.no_grad():
            for role in ROLES:
                for parameter in self.networks[role].parameters():
                    parameter.fill_(float("nan"))
        roles = dict(report.roles)
        roles["battle"] = dataclasses.replace(roles["battle"], policy_loss=float("nan"))
        return TrainReport(roles)
    return report


REAL_TRAIN = Learner.train


@pytest.fixture(scope="module", autouse=True)
def forced() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Learner, "train", forced_train)
        yield


@pytest.fixture
def diverge() -> dict[str, Any]:
    """The knobs of the forced divergence, reset for each test: members and generation."""
    return reset()


def run_to(
    run_dir: Path, state: dict[str, Any], generations: int = GENERATIONS, on_stage: Any = None
) -> PopulationLoop:
    config = config_four()
    loop = make_population(run_dir, config, FakePopulationPlayer(config), on_stage)
    state["loop"] = loop
    loop.run(generations)
    return loop


def manifest_of(run_dir: Path, generation: int) -> dict[str, Any]:
    raw = layout.read_json(gen_path(run_dir, generation) / "manifest.json")
    assert isinstance(raw, dict)
    return raw


def frozen_of(run_dir: Path, generation: int, member: str) -> dict[str, str]:
    found = manifest_of(run_dir, generation)["members"][member]["checkpoints"]
    assert isinstance(found, dict)
    return found


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("health")
    run_to(run_dir, reset())
    return run_dir


def test_the_ladder_records_the_health_outcome_beside_the_copy_test(finished: Path) -> None:
    config = config_four()
    before = None
    for g in range(1, GENERATIONS + 1):
        record = ladder_of(finished, g)
        read_ladder_record(record, before, config)
        before = record
        expected = [{"kind": "health", "member": "m1", "cause": "non_finite_loss"}]
        assert record["selections"][1:] == (expected if g == BAD_GENERATION else [])
    # The next record starts from the replacement: m1 descends from the top live member.
    donor = selection.health_donor(ladder_of(finished, BAD_GENERATION))
    assert donor != "m1"
    third = {m["member"]: m for m in ladder_of(finished, BAD_GENERATION + 1)["members"]}
    assert (
        third["m1"]["root"]
        == ladder_of(finished, BAD_GENERATION)["members"][int(donor[1:])]["root"]
    )
    (found,) = selection.replacements(config, ladder_of(finished, BAD_GENERATION))
    assert third["m1"]["settings"] == found.settings


def test_the_discarded_update_leaves_the_pair_and_the_records_it_had(finished: Path) -> None:
    assert frozen_of(finished, BAD_GENERATION, "m1") == frozen_of(
        finished, BAD_GENERATION - 1, "m1"
    )
    for role in ROLES:
        now = gen_path(finished, BAD_GENERATION) / "members" / "m1" / "records" / f"{role}.json"
        then = (
            gen_path(finished, BAD_GENERATION - 1) / "members" / "m1" / "records" / f"{role}.json"
        )
        assert now.read_bytes() == then.read_bytes()
    # The other members trained on and froze new pairs.
    for member in ("m0", "m2", "m3"):
        assert frozen_of(finished, BAD_GENERATION, member) != frozen_of(
            finished, BAD_GENERATION - 1, member
        )
    assert manifest_of(finished, BAD_GENERATION)["members"]["m1"]["training_set_hash"] is None
    # The stored report carries no NaN (null where the loss was not a number).
    report = layout.read_json(
        gen_path(finished, BAD_GENERATION) / "members" / "m1" / "train-report.json"
    )
    assert report["battle"]["policy_loss"] is None
    assert isinstance(report["battle"]["value_loss"], float)


def test_the_slot_takes_a_copy_of_the_top_member_at_the_next_generation(
    tmp_path: Path, diverge: dict[str, Any]
) -> None:
    seen: dict[str, Any] = {}
    box: dict[str, PopulationLoop] = {}

    def hook(generation: int, stage: str) -> None:
        loop = box["loop"]
        if (generation, stage) != (BAD_GENERATION + 1, "copied"):
            return
        loser = loop._learners["m1"]
        seen["ids"] = {r: checkpoint_id(checkpoint_bytes(loser.networks[r], game())) for r in ROLES}
        seen["flag"] = loser.value_only
        seen["finite"] = all(
            bool(torch.isfinite(p).all()) for r in ROLES for p in loser.networks[r].parameters()
        )

    config = config_four()
    loop = make_population(tmp_path, config, FakePopulationPlayer(config), hook)
    box["loop"] = loop
    diverge["loop"] = loop
    loop.run(GENERATIONS)
    donor = selection.health_donor(ladder_of(tmp_path, BAD_GENERATION))
    assert seen["ids"] == frozen_of(tmp_path, BAD_GENERATION, donor)
    assert seen["flag"] is True
    assert seen["finite"] is True
    # Its records name the donor's frozen checkpoints as parent; the first update is value-only.
    for role in ROLES:
        path = gen_path(tmp_path, GENERATIONS) / "members" / "m1" / "records" / f"{role}.json"
        assert (
            read_record(layout.read_json(path)).parent
            == (frozen_of(tmp_path, BAD_GENERATION, donor)[role])
        )
    report = layout.read_json(
        gen_path(tmp_path, GENERATIONS) / "members" / "m1" / "train-report.json"
    )
    assert all(report[role].get("value_warmup") is True for role in report)
    # The weights of the following generations are finite.
    assert not loop._learners["m1"].value_only


def test_classify_accepts_the_records_and_sees_the_donor_in_the_ancestry(finished: Path) -> None:
    records: dict[str, Any] = {}
    sources: dict[str, list[str]] = {}
    for g in range(GENERATIONS + 1):
        facts = manifest_of(finished, g)["members"]
        for member, entry in facts.items():
            for role in ROLES:
                path = gen_path(finished, g) / "members" / member / "records" / f"{role}.json"
                record = read_record(layout.read_json(path))
                records[record.checkpoint_id] = record  # the frozen-again pair has one record
            if g >= 1 and entry["training_set_hash"] is not None:
                played = frozen_of(finished, g - 1, member)
                sources[entry["training_set_hash"]] = sorted(played.values())
    graph = ProvenanceGraph(list(records.values()), sources)
    donor = selection.health_donor(ladder_of(finished, BAD_GENERATION))
    now = list(frozen_of(finished, GENERATIONS, "m1").values())
    then = list(frozen_of(finished, BAD_GENERATION, donor).values())
    assert set(then) <= graph.ancestry(now).checkpoints


def test_an_independent_rerun_is_byte_identical(
    finished: Path, tmp_path: Path, diverge: dict[str, Any]
) -> None:
    run_to(tmp_path, diverge)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


@pytest.mark.parametrize(
    ("generation", "stage"),
    [
        (BAD_GENERATION, "trained:m1"),
        (BAD_GENERATION, "trained"),
        (BAD_GENERATION, "rated"),
        (BAD_GENERATION, "written"),
        (BAD_GENERATION, "committed"),  # the quarantine is committed; the resume applies it
        (BAD_GENERATION + 1, "played"),
        (BAD_GENERATION + 1, "copied"),
    ],
)
def test_a_kill_at_any_stage_around_a_quarantine_resumes_to_the_same_bytes(
    finished: Path, tmp_path: Path, diverge: dict[str, Any], generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        run_to(tmp_path, diverge, on_stage=killer(generation, stage))
    run_to(tmp_path, diverge)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


def test_half_the_members_quarantined_commits_nothing_and_the_partial_is_discarded(
    tmp_path: Path, diverge: dict[str, Any]
) -> None:
    diverge["members"] = {"m1", "m2"}
    with pytest.raises(HealthIncident, match="half or more"):
        run_to(tmp_path, diverge)
    newest = layout.latest_complete(tmp_path)
    assert newest is not None
    assert newest.generation == BAD_GENERATION - 1
    assert layout.partial_dir(tmp_path, BAD_GENERATION).is_dir()
    assert not gen_path(tmp_path, BAD_GENERATION).exists()
    # The next start deletes the partial and plays the generation again; nothing diverges now.
    diverge["members"] = set()
    result = run_to(tmp_path, diverge, BAD_GENERATION)
    assert result.frozen.generation == BAD_GENERATION
    assert not layout.partial_dir(tmp_path, BAD_GENERATION).exists()
    assert ladder_of(tmp_path, BAD_GENERATION)["selections"][1:] == []


def test_a_single_quarantine_below_half_is_not_an_incident(
    tmp_path: Path, diverge: dict[str, Any]
) -> None:
    diverge["members"] = {"m2"}
    run_to(tmp_path, diverge, BAD_GENERATION)
    assert ladder_of(tmp_path, BAD_GENERATION)["selections"][1] == {
        "kind": "health",
        "member": "m2",
        "cause": "non_finite_loss",
    }
