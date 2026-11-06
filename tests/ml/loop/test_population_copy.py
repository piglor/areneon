"""A population run that copies (ADR-021 section 6, slice 8): the copy, explore and records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.state import LearnerStateError
from areneon_ml.loop import ladder_write, layout, selection
from areneon_ml.loop.ladder import read_ladder_record
from areneon_ml.loop.population import PopulationLoop
from areneon_ml.loop.population_seed import population_draw
from areneon_ml.model.network import MAX_SEED, ROLES

from tests.ml.loop.support import Killed, killer
from tests.ml.loop.test_population import (
    FakePopulationPlayer,
    gen_path,
    ladder_of,
    make_population,
    population_config,
    reseal,
    tree_bytes,
)
from tests.ml.model.support import game

GENERATIONS = 3


def config_with_copies() -> Any:
    """Two members; a copy is decided as soon as the strong one has two pairs on the weak one."""

    def edit(raw: dict[str, Any]) -> None:
        raw["population"]["selection"].update(
            {"ready_generations": 1, "n_min": 2, "margin_ppm": 50_000, "min_roots": 1}
        )

    return population_config(edit=edit)


class Dominant(FakePopulationPlayer):
    """m0 beats every opponent, m1 loses to the anchors: m0 is the donor, m1 the loser."""

    def row(self, entry: dict[str, Any]) -> dict[str, Any]:
        row = super().row(entry)
        seats = row["seats"]
        if entry["kind"] == "self":
            return row
        for name, win in (("m0", True), ("m1", False)):
            if name in seats.values() and (win or "m0" not in seats.values()):
                mine = "p1" if seats["p1"] == name else "p2"
                other = "p2" if mine == "p1" else "p1"
                row["winner"] = mine if win else other
                return row
        return row


def run_to(run_dir: Path, generations: int, on_stage: Any = None) -> PopulationLoop:
    config = config_with_copies()
    loop = make_population(run_dir, config, Dominant(config), on_stage)
    loop.run(generations)
    return loop


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("copying")
    run_to(run_dir, GENERATIONS)
    return run_dir


def manifest_of(run_dir: Path, generation: int) -> dict[str, Any]:
    raw = layout.read_json(gen_path(run_dir, generation) / "manifest.json")
    assert isinstance(raw, dict)
    return raw


def sidecar_of(run_dir: Path, generation: int, member: str) -> dict[str, Any]:
    path = gen_path(run_dir, generation) / "members" / member / f"learner-{generation}.json"
    raw = json.loads(path.read_text())
    assert isinstance(raw, dict)
    return raw


def test_the_ladder_decides_the_copy_and_carries_its_evidence(finished: Path) -> None:
    config = config_with_copies()
    first = ladder_of(finished, 1)
    decision = first["selections"][0]
    assert (decision["kind"], decision["loser"], decision["donor"]) == ("copy", "m1", "m0")
    assert decision["margin_ppm"] == 50_000
    assert decision["lower_bound_ppm"] > 500_000 + decision["margin_ppm"]
    assert decision["pairs"] >= 2
    assert decision["learner_seed_index"] == 0
    assert (decision["roots_before"], decision["roots_after"]) == (2, 1)
    assert decision["settings"] == selection.explore(
        config, 1, "m1", first["members"][0]["settings"]
    )
    before = None
    for g in range(1, GENERATIONS + 1):
        record = ladder_of(finished, g)
        read_ladder_record(record, before, config)
        before = record
    # The next record starts from the copy: m1 descends from m0 and trains under the new settings.
    second = {m["member"]: m for m in ladder_of(finished, 2)["members"]}
    assert second["m1"]["root"] == "m0"
    assert second["m1"]["settings"] == decision["settings"]
    assert second["m0"]["settings"] == first["members"][0]["settings"]
    # Only the loser's results of generation 2 stand in its window now.
    cells = {(c["member"], c["opponent"]["id"]): c for c in ladder_of(finished, 2)["cells"]}
    assert cells[("m1", "actor-rule@v1")]["pairs"] == 1


def test_the_losers_slot_takes_the_donors_frozen_weights_and_adam_moments(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    box: dict[str, PopulationLoop] = {}

    def hook(generation: int, stage: str) -> None:
        if (generation, stage) != (2, "copied"):
            return
        loop = box["loop"]
        donor, loser = loop._learners["m0"], loop._learners["m1"]
        seen["ids"] = {r: checkpoint_id(checkpoint_bytes(loser.networks[r], game())) for r in ROLES}
        pairs = [
            (opt_d.state.get(pd), opt_l.state.get(pl))
            for (opt_d, entries_d), (opt_l, entries_l) in zip(
                donor.optimiser_groups(), loser.optimiser_groups(), strict=True
            )
            for (_, _, pd), (_, _, pl) in zip(entries_d, entries_l, strict=True)
        ]
        held = [(a, b) for a, b in pairs if a]
        seen["moments"] = len(held) > 10 and all(
            b is not None
            and set(a) == set(b)
            and all(torch.equal(a[k], b[k]) and a[k].data_ptr() != b[k].data_ptr() for k in a)
            for a, b in held
        )
        seen["seed"] = loser.generator.initial_seed()
        seen["config_seed"] = loser.config.learner_seed
        seen["settings"] = loser.population_settings
        seen["ppo"] = loser.config.ppo
        seen["lr"] = [g["lr"] for o, _ in loser.optimiser_groups() for g in o.param_groups]

    config = config_with_copies()
    loop = make_population(tmp_path, config, Dominant(config), hook)
    box["loop"] = loop
    loop.run(2)
    donor_frozen = manifest_of(tmp_path, 1)["members"]["m0"]["checkpoints"]
    assert seen["ids"] == donor_frozen  # the donor's newest frozen weights, byte for byte
    assert seen["moments"] is True
    decision = ladder_of(tmp_path, 1)["selections"][0]
    expected = population_draw(config.run_key, 1, "m1", "member-learner", 0) & MAX_SEED
    assert seen["seed"] == seen["config_seed"] == expected
    assert seen["settings"] == selection.ppo_part(decision["settings"])
    assert seen["ppo"] == selection.ppo_with(config.ppo, decision["settings"])
    assert set(seen["lr"]) == {decision["settings"]["learning_rate_micro"] / 1_000_000}


def test_the_losers_next_records_name_the_donors_checkpoints_as_parent(finished: Path) -> None:
    donor_then = manifest_of(finished, 1)["members"]["m0"]["checkpoints"]
    loser_then = manifest_of(finished, 1)["members"]["m1"]["checkpoints"]
    for role in ROLES:
        path = gen_path(finished, 2) / "members"
        assert (
            read_record(layout.read_json(path / "m1" / "records" / f"{role}.json")).parent
            == (donor_then[role])
        )
        # The donor is not touched: it descends from itself.
        assert (
            read_record(layout.read_json(path / "m0" / "records" / f"{role}.json")).parent
            == (manifest_of(finished, 1)["members"]["m0"]["checkpoints"][role])
        )
        assert loser_then[role] != donor_then[role]  # generation 1 froze the loser's own pair


def test_the_copy_inherits_the_donors_step_counts_and_names_its_settings(finished: Path) -> None:
    decision = ladder_of(finished, 1)["selections"][0]
    donor_then = sidecar_of(finished, 1, "m0")["optimizer_steps"]
    loser_now = sidecar_of(finished, 2, "m1")
    report = layout.read_json(gen_path(finished, 2) / "members" / "m1" / "train-report.json")
    for role in ROLES:
        for name, steps in donor_then[role].items():
            assert loser_now["optimizer_steps"][role][name] == (
                steps + report[role]["optimiser_steps"]
            )
    assert loser_now["population_settings"] == selection.ppo_part(decision["settings"])
    # The donor trained under its own, unexplored settings.
    assert sidecar_of(finished, 2, "m0")["population_settings"] == selection.ppo_part(
        ladder_of(finished, 1)["members"][0]["settings"]
    )
    # Every member of every generation names its settings (population runs only).
    for g in range(GENERATIONS + 1):
        for member in ("m0", "m1"):
            assert "population_settings" in sidecar_of(finished, g, member)


def test_classify_accepts_the_records_and_sees_the_donor_in_the_losers_ancestry(
    finished: Path,
) -> None:
    records = []
    sources: dict[str, list[str]] = {}
    for g in range(GENERATIONS + 1):
        facts = manifest_of(finished, g)["members"]
        for member, entry in facts.items():
            for role in ROLES:
                path = gen_path(finished, g) / "members" / member / "records" / f"{role}.json"
                records.append(read_record(layout.read_json(path)))
            if g >= 1:
                # The set trained on in generation g was played by the pair frozen at g - 1.
                played = manifest_of(finished, g - 1)["members"][member]["checkpoints"]
                sources[entry["training_set_hash"]] = sorted(played.values())
    graph = ProvenanceGraph(records, sources)  # parents agree in role and encoder version
    loser_now = list(manifest_of(finished, 2)["members"]["m1"]["checkpoints"].values())
    donor_then = list(manifest_of(finished, 1)["members"]["m0"]["checkpoints"].values())
    assert set(donor_then) <= graph.ancestry(loser_now).checkpoints
    assert not graph.independent(loser_now, donor_then).independent
    assert graph.classify(loser_now, "random-legal@v1").label == "seen"


def test_an_independent_rerun_is_byte_identical(finished: Path, tmp_path: Path) -> None:
    run_to(tmp_path, GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


@pytest.mark.parametrize(
    ("generation", "stage"),
    [
        (1, "committed"),  # the copy is decided and committed; the resume applies it
        (2, "played"),
        (2, "copied"),
        (2, "trained:m0"),
        (2, "trained"),
        (2, "rated"),
        (2, "written"),
        (3, "copied"),
    ],
)
def test_a_kill_at_any_stage_around_a_copy_resumes_to_the_same_bytes(
    finished: Path, tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        run_to(tmp_path, GENERATIONS, killer(generation, stage))
    run_to(tmp_path, GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


def test_a_ladder_record_whose_settings_the_sidecar_does_not_state_is_refused(
    tmp_path: Path,
) -> None:
    run_to(tmp_path, 2)
    record = ladder_of(tmp_path, 2)
    record["members"][1]["settings"]["epochs"] += 1
    layout.write_json(gen_path(tmp_path, 2) / "ladder.json", record)
    reseal(tmp_path, 2, ladder_hash=layout_hash(record))
    with pytest.raises(LearnerStateError, match="PPO settings"):
        run_to(tmp_path, 3)


def layout_hash(raw: object) -> str:
    return ladder_write.json_hash(raw)


def test_a_generation_read_for_the_window_must_name_the_frozen_anchors(tmp_path: Path) -> None:
    run_to(tmp_path, 2)
    path = gen_path(tmp_path, 1) / "manifest.json"
    manifest = layout.read_json(path)
    manifest["anchors_hash"] = "0" * 64
    layout.write_json(path, manifest)
    with pytest.raises(Exception, match="names other anchor strengths than generation 0 froze"):
        run_to(tmp_path, 3)
