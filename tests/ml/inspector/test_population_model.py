"""The population model (ADR-021 section 8, ADR-017 4a): joined, checked, refusing on mismatch."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector.pool import read_pool
from areneon_ml.inspector.population_model import (
    build_population_model,
    is_population,
    read_population_config,
    run_is_population,
    seat_class,
)
from areneon_ml.inspector.readers import InspectError
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import layout

from tests.ml.inspector.population_support import (
    POP_FIXTURE,
    copy_pop_run,
    edit,
    gen_dir,
    pop_model,
    pop_pool,
    rename_episode,
    reseal,
)
from tests.ml.inspector.support import FIXTURE


def build(run: Path) -> Any:
    return build_population_model(run, pop_pool(), read_population_config(run))


def test_the_fixture_run_is_read_and_joined() -> None:
    model = pop_model()
    assert model.members == ("m0", "m1")
    assert [g.number for g in model.generations] == [1, 2]
    for gen in model.generations:
        kinds = Counter(g.kind for g in gen.games)
        assert kinds == {"ladder": 8, "self": 4, "past": 4, "anchor": 8}
        assert gen.forfeits == 0
        assert gen.episodes == 48
        assert {m.member for m in gen.members} == {"m0", "m1"}
        for game in gen.games:
            assert set(game.policies) == {"p1", "p2"}
            assert game.hashes["p1"] in model.use
            assert game.hashes["p2"] in model.use
    assert sum(model.cumulative_drawn.values()) == 96
    assert len(model.manifest_hashes) == 3


def test_the_copy_is_read_from_the_ladder_and_the_records() -> None:
    model = pop_model()
    one, two = model.generations
    assert one.selection is not None
    assert one.selection["kind"] == "copy"
    assert (one.selection["donor"], one.selection["loser"]) == ("m0", "m1")
    assert two.selection == {"failed": "lower_bound", "kind": "none"}
    assert [m.copied_from for m in one.members] == [None, None]
    assert [m.copied_from for m in two.members] == [None, "m0"]
    assert [m.copied_from for m in model.initial] == [None, None]
    assert all(
        m.strength is not None and m.strength[0] <= m.strength[1] <= m.strength[2]
        for m in two.members
    )
    assert two.members[1].root == "m0"


def test_the_past_pool_games_name_the_snapshot_they_met() -> None:
    model = pop_model()
    for gen in model.generations:
        past = [g for g in gen.games if g.kind == "past"]
        assert past
        for game in past:
            snapshots = [s for s in game.plan.seats.values() if s.kind == "snapshot"]
            assert len(snapshots) == 1
            snap = gen.snapshot(snapshots[0].label)
            assert (
                snap.policy_id
                == game.policies["p1" if game.plan.seats["p1"].kind == "snapshot" else "p2"]
            )
    assert [s.id for s in model.generations[0].snapshots] == ["m0@g0", "m1@g0"]
    assert model.generations[0].committed_snapshots == ("m0@g0", "m1@g0", "m0@g1", "m1@g1")
    first = next(g for g in model.generations[0].games if g.kind == "past")
    assert first.played_by("p1") != first.played_by("p2")
    assert {seat_class(s) for g in model.generations[0].games for s in g.plan.seats.values()} == {
        "member",
        "past",
        "baseline",
    }


def test_a_team_results_are_tallied_by_kind_and_holder() -> None:
    model = pop_model()
    kinds = {kind for use in model.use.values() for kind, _ in use.results}
    assert kinds == {"ladder", "self", "past", "anchor"}
    holders = {by for use in model.use.values() for _, by in use.results}
    assert "m0" in holders
    assert "baseline actor-rule@v1" in holders
    assert any(h.startswith("m") and "@g" in h for h in holders)


def test_the_two_config_readers_know_their_own_runs() -> None:
    assert is_population(layout.read_json(POP_FIXTURE / "run-config.json"))
    assert not is_population({"a": 1})
    assert not is_population([])
    assert run_is_population(POP_FIXTURE)
    assert not run_is_population(FIXTURE)
    assert not run_is_population(POP_FIXTURE / "generations")  # no readable config: not population
    with pytest.raises(InspectError, match=r"no population section"):
        read_population_config(FIXTURE)
    with pytest.raises(InspectError, match=r"run-config.json"):
        read_population_config(POP_FIXTURE / "generations")


def test_a_private_lineage_or_an_exploiter_seat_is_refused() -> None:
    raw = layout.read_json(POP_FIXTURE / "run-config.json")
    raw["lineage_id"] = "private-teams"
    with pytest.raises(InspectError, match=r"not a public one"):
        build_population_model(POP_FIXTURE, pop_pool(), read_run_config(raw))
    raw = layout.read_json(POP_FIXTURE / "run-config.json")
    raw["population"]["exploiters"] = 1
    with pytest.raises(InspectError, match=r"exploiter seat"):
        build_population_model(POP_FIXTURE, pop_pool(), read_run_config(raw))
    raw = layout.read_json(POP_FIXTURE / "run-config.json")
    raw["generation"]["excluded_abilities"] = []
    with pytest.raises(InspectError, match=r"excluded abilities"):
        build_population_model(POP_FIXTURE, pop_pool(), read_run_config(raw))


def test_a_pool_of_another_source_is_refused() -> None:
    raw = json.loads((POP_FIXTURE / "pool.json").read_text())
    raw["source"] = "dataset"
    raw["dataset_hash"] = "a" * 64
    with pytest.raises(InspectError, match=r"pool's source does not match"):
        build_population_model(POP_FIXTURE, read_pool(raw), read_population_config(POP_FIXTURE))


def test_the_manifest_must_match_the_files(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    path = gen_dir(run, 1) / "plan.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(InspectError, match=r"differ from its manifest|not in RFC 8785"):
        build(run)


def test_plan_and_generation_must_agree(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 1, "generation.json", lambda g: g["games"][3].update(member="m1"))
    with pytest.raises(InspectError, match=r"game 3: plan.json and generation.json differ"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    edit(run, 1, "generation.json", lambda g: g["games"][0].update(opponent="m0"))
    with pytest.raises(InspectError, match=r"initiating seat or opponent"):
        build(run)
    run = copy_pop_run(tmp_path / "c")
    edit(run, 1, "generation.json", lambda g: g["games"].__delitem__(-1))
    with pytest.raises(InspectError, match=r"different numbers of games"):
        build(run)


def test_an_episode_must_agree_with_its_game(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(
        run, 2, "generation.json", lambda g: g["games"][0].update(turns=g["games"][0]["turns"] + 1)
    )
    with pytest.raises(InspectError, match=r"result differs from generation.json"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    rename_episode(run, 1, "m0", 0, lambda e: e.update(rights="all-rights"))
    with pytest.raises(InspectError, match=r"not a synthetic cc0-synthetic Episode"):
        build(run)
    run = copy_pop_run(tmp_path / "c")
    rename_episode(
        run, 1, "m0", 0, lambda e: e["provenance"]["collection"].update(run_id="another-run")
    )
    with pytest.raises(InspectError, match=r"not a league Episode of this run"):
        build(run)
    run = copy_pop_run(tmp_path / "d")
    rename_episode(
        run,
        1,
        "m0",
        0,
        lambda e: e["provenance"]["collection"].update(opponent_policy_id="actor-rule@v1x"),
    )
    with pytest.raises(InspectError, match=r"names another opponent|no Episode"):
        build(run)


def test_a_forfeited_game_may_not_have_an_episode_and_an_orphan_is_refused(tmp_path: Path) -> None:
    def forfeit(generation: dict[str, Any]) -> None:
        row = generation["games"][0]
        row.update(
            outcome="forfeit",
            winner="p2",
            forfeit={"loser": "p1", "cause": "crash", "request_seq": 1, "detail": "x"},
        )
        generation["forfeits"] = 1
        generation["forfeit_causes"] = {"crash": 1}

    run = copy_pop_run(tmp_path)
    edit(run, 1, "generation.json", forfeit)
    with pytest.raises(InspectError, match=r"a forfeited game has an Episode"):
        build(run)


def test_an_episode_in_another_members_directory_is_refused(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    gen = gen_dir(run, 1)
    source = sorted((gen / "members" / "m0" / "episodes").glob("*.json"))[0]
    shutil.move(source, gen / "members" / "m1" / "episodes" / source.name)

    def move(generation: dict[str, Any]) -> None:
        a, b = generation["members"]
        a["episode_hashes"].remove(source.stem)
        b["episode_hashes"] = sorted([*b["episode_hashes"], source.stem])

    edit(run, 1, "generation.json", move)
    with pytest.raises(InspectError, match=r"not in the directory of"):
        build(run)


def test_episode_files_must_be_the_listed_hashes(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    gen = gen_dir(run, 2)
    sorted((gen / "members" / "m0" / "episodes").glob("*.json"))[0].unlink()
    reseal(run, 2)
    with pytest.raises(InspectError, match=r"Episode files are not its episode_hashes"):
        build(run)


def test_counts_and_causes_must_agree(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 1, "generation.json", lambda g: g.update(forfeit_causes={"crash": 1}))
    with pytest.raises(InspectError, match=r"forfeit_causes are not the causes"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    edit(run, 1, "generation.json", lambda g: g.update(forfeits=1))
    with pytest.raises(InspectError, match=r"forfeit and played counts"):
        build(run)


def test_team_hashes_and_team_set_must_be_the_plans(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 1, "generation.json", lambda g: g["team_hashes"].__delitem__(-1))
    with pytest.raises(InspectError, match=r"team_hashes are not the teams its plan draws"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    edit(run, 2, "team-set.json", lambda t: t["teams"].__delitem__(-1))
    with pytest.raises(InspectError, match=r"team-set.json is not the union"):
        build(run)
    run = copy_pop_run(tmp_path / "c")
    edit(run, 0, "team-set.json", lambda t: t["teams"].append("f" * 64))
    with pytest.raises(InspectError, match=r"not in the pool"):
        build(run)


def test_the_ladder_and_the_past_pool_must_be_the_ones_the_manifest_names(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 2, "ladder.json", lambda ladder: ladder["members"][0].update(root="m1"))
    with pytest.raises(InspectError, match=r"ladder.json is not the one its manifest names"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    edit(run, 1, "past-pool.json", lambda p: p.update(generation=7))
    with pytest.raises(InspectError, match=r"past-pool.json is not the one its manifest names"):
        build(run)

    run = copy_pop_run(tmp_path / "c")
    path = gen_dir(run, 1)
    manifest = layout.read_json(path / layout.MANIFEST)
    del manifest["past_pool_hash"]
    layout.write_json(path / layout.MANIFEST, manifest)
    edit(run, 1, "past-pool.json", lambda p: p.update(generation=7))
    with pytest.raises(InspectError, match=r"past-pool.json: "):
        build(run)


def test_the_records_must_be_the_sealed_checkpoints_of_this_run(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 1, "members/m0/records/battle.json", lambda r: r.update(lineage_id="other.m0"))
    with pytest.raises(InspectError, match=r"names another lineage"):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    edit(run, 1, "members/m0/records/battle.json", lambda r: r.update(role="preview"))
    with pytest.raises(InspectError, match=r"is not this run's battle record"):
        build(run)
    run = copy_pop_run(tmp_path / "c")
    edit(run, 1, "members/m0/records/battle.json", lambda r: r.update(data_class="private"))
    with pytest.raises(InspectError, match=r"record"):
        build(run)
    run = copy_pop_run(tmp_path / "d")
    edit(run, 1, "members/m0/records/battle.json", lambda r: r.update(checkpoint_id="1" * 64))
    with pytest.raises(InspectError, match=r"are not the checkpoints sealed"):
        build(run)
    run = copy_pop_run(tmp_path / "e")
    shutil.rmtree(gen_dir(run, 1) / "members" / "m0" / "records")
    reseal(run, 1)
    with pytest.raises(InspectError, match=r"records"):
        build(run)


def test_a_parent_no_member_froze_is_refused(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 2, "members/m1/records/battle.json", lambda r: r.update(parent="2" * 64))
    with pytest.raises(InspectError, match=r"names a parent no member froze"):
        build(run)
    run = copy_pop_run(tmp_path / "b")

    def other(record: dict[str, Any]) -> None:
        # the Preview record names the donor's parent while the Battle record names its own
        record["parent"] = layout.read_json(
            gen_dir(run, 1) / "members" / "m0" / "records" / "preview.json"
        )["checkpoint_id"]

    edit(run, 2, "members/m1/records/preview.json", other)
    # m1's Battle parent is m0's frozen Battle (a copy) and its Preview parent is m0's frozen
    # Preview too, so this edit changes nothing: both name the donor.
    assert build(run).generations[1].members[1].copied_from == "m0"
    edit(
        run,
        2,
        "members/m0/records/preview.json",
        lambda r: r.update(
            parent=layout.read_json(
                gen_dir(run, 1) / "members" / "m1" / "records" / "preview.json"
            )["checkpoint_id"]
        ),
    )
    with pytest.raises(InspectError, match=r"different parents"):
        build(run)


def test_the_sealed_training_set_must_be_the_files(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)

    def seal(manifest: dict[str, Any]) -> None:
        manifest["members"]["m0"]["training_set_hash"] = "3" * 64

    edit(run, 1, "manifest.json", seal)
    with pytest.raises(InspectError):
        build(run)
    run = copy_pop_run(tmp_path / "b")
    path = gen_dir(run, 1)
    manifest = layout.read_json(path / layout.MANIFEST)
    manifest["members"]["m0"]["training_set_hash"] = "3" * 64
    layout.write_json(path / layout.MANIFEST, manifest)
    with pytest.raises(InspectError, match=r"training set is not the one sealed"):
        build(run)


def test_the_generations_must_be_numbered_without_gaps(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    shutil.rmtree(gen_dir(run, 1))
    with pytest.raises(InspectError, match=r"no gap"):
        build(run)
