"""The population loop driver (ADR-021 slice 5): members train in turn, one commit per round."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.state import pack_state
from areneon_ml.learner.trainer import Learner
from areneon_ml.loop import layout, population
from areneon_ml.loop.__main__ import EXIT_REFUSED, main
from areneon_ml.loop.driver import LoopError, export_provenance
from areneon_ml.loop.league import league_game_seed
from areneon_ml.loop.metrics import metrics_path, read_run_metrics
from areneon_ml.loop.play import PlayError, PopulationPlayRequest
from areneon_ml.loop.population import (
    PopulationLoop,
    member_config,
    member_ppo,
)
from areneon_ml.loop.population_seed import population_draw
from areneon_ml.loop.warm_start import import_clone
from areneon_ml.model.network import MAX_SEED, ROLES, init_network

from tests.ml.learner.support import edited_config, episode_lines, stream_of, tiny_config
from tests.ml.loop.support import MANIFEST, Killed, killer, make_loop
from tests.ml.loop.test_warm_start import clone_ids, clone_output
from tests.ml.model.support import RANDOM, SAMPLES, game

GENERATIONS = 3
LABELS = ("m0", "m1")


def population_config(
    *, members: int = 2, key: str = "pop-tiny", edit: Callable[[dict[str, Any]], None] | None = None
) -> RunConfig:
    sample = json.loads((SAMPLES / "training-run-config/valid/population.json").read_text())
    block = sample["population"]

    def change(raw: dict[str, Any]) -> None:
        block["members"] = members
        block["seats_per_member"] = 2
        block["shares"] = {"ladder_ppm": 500_000, "self_play_ppm": 500_000, "past_pool_ppm": 0}
        block["selection"]["min_roots"] = min(2, members)
        bounds = block["explore"]["bounds"]
        bounds["self_play_ppm"] = {"min": 450_000, "max": 550_000}
        bounds["past_pool_ppm"] = {"min": 0, "max": 0}
        raw["population"] = copy.deepcopy(block)
        raw["generation"]["baseline_games"] = []
        raw["seeds"]["run_key"] = key
        if edit is not None:
            edit(raw)

    return edited_config(change)


class FakePopulationPlayer:
    """The simulator side of a population run without a simulator (slice 4's files, faked).

    Each member plays one ladder pair against the next member and one self-play pair; the
    files are what ``play-generation.ts`` writes: ``plan.json``, ``league-run.json`` (version 2),
    ``generation.json`` (version 2) and per member ``members/m<k>/stream.jsonl`` and
    ``training-set.json``.
    """

    def __init__(
        self,
        config: RunConfig,
        change: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.change = change
        self.requests: list[PopulationPlayRequest] = []

    def manifest(self, format_id: str) -> dict[str, Any]:
        return dict(MANIFEST)

    def play_population(self, request: PopulationPlayRequest) -> None:
        self.requests.append(request)
        generation, run_key = request.generation, self.config.run_key
        labels = [m.member for m in request.members]
        ids = {m.member: m.policy_id for m in request.members}
        pairs: list[tuple[str, str, str]] = []
        for k, label in enumerate(labels):
            if len(labels) > 1:
                pairs.append(("ladder", label, labels[(k + 1) % len(labels)]))
            pairs.append(("self", label, label))
        plan: list[dict[str, Any]] = []
        for pair, (kind, member, other) in enumerate(pairs):
            for half in (0, 1):
                first, second = (member, other) if half == 0 or kind == "self" else (other, member)
                plan.append(
                    {
                        "game": 2 * pair + half,
                        "pair": pair,
                        "half": half,
                        "seed": league_game_seed(run_key, generation, pair),
                        "teams": {"p1": 0, "p2": 1},
                        "baseline_seed": 0,
                        "kind": kind,
                        "member": member,
                        "seats": {"p1": {"member": first}, "p2": {"member": second}},
                    }
                )
        tallies: dict[str, dict[str, Any]] = {
            m: {"games": 0, "played": 0, "episodes": 0, "opponents": {}} for m in labels
        }
        for entry in plan:
            seats = [entry["seats"]["p1"]["member"], entry["seats"]["p2"]["member"]]
            for label in sorted(set(seats)):
                tally = tallies[label]
                tally["games"] += 1
                tally["played"] += 1
                tally["episodes"] += seats.count(label)
                other = seats[1] if seats[0] == label else seats[0]
                held = tally["opponents"].setdefault(
                    other,
                    {"opponent_policy_id": ids[other], "member": other, "games": 0, "played": 0},
                )
                held["games"] += 1
                held["played"] += 1
        record = {
            "schema_version": 2,
            "run_id": run_key,
            "run_config_hash": self.config.hash,
            "generation": generation,
            "seed_derivation": "league-seed/1",
            "plan_hash": sha256_hex(jcs(plan)),
            "games": len(plan),
            "played": len(plan),
            "forfeits": 0,
            "episodes": 2 * len(plan),
            "copy_decision_hash": None,
            "members": [
                {
                    "member": m.member,
                    "pair_policy_id": m.policy_id,
                    "checkpoints": {"preview": m.preview, "battle": m.battle},
                    "games": tallies[m.member]["games"],
                    "played": tallies[m.member]["played"],
                    "forfeits": 0,
                    "episodes": tallies[m.member]["episodes"],
                    "opponents": sorted(
                        tallies[m.member]["opponents"].values(),
                        key=lambda o: (o["opponent_policy_id"], o["member"]),
                    ),
                }
                for m in request.members
            ],
        }
        summary: dict[str, Any] = {
            "schema_version": 2,
            "generation": generation,
            "games": [
                {"game": e["game"], "pair": e["pair"], "seed": e["seed"], "outcome": "played"}
                for e in plan
            ],
            "forfeits": 0,
            "members": [],
            "manifest": dict(MANIFEST),
            "team_hashes": sorted(
                hashlib.sha256(f"team-{generation}-{n}".encode()).hexdigest() for n in range(2)
            ),
        }
        out = request.out_dir
        for k, m in enumerate(request.members):
            picker = random.Random(generation * 100 + k)
            examples = [
                e
                for n, player in enumerate(("p1", "p2"))
                for e in episode_lines((generation + k + n) % len(RANDOM), player, picker)
            ]
            text, training_set = stream_of(examples, m.policy_id)
            path = out / "members" / m.member
            path.mkdir(parents=True)
            (path / "stream.jsonl").write_text(text, encoding="utf-8")
            layout.write_json(path / "training-set.json", training_set)
            summary["members"].append(
                {
                    "member": m.member,
                    "episode_hashes": training_set["episode_hashes"],
                    "training_set_hash": sha256_hex(jcs(training_set)),
                    "examples": training_set["examples"],
                }
            )
        if self.change is not None:
            self.change(summary, record)
        layout.write_json(out / "plan.json", plan)
        layout.write_json(out / "league-run.json", record)
        layout.write_json(out / "generation.json", summary)


def make_population(
    run_dir: Path,
    config: RunConfig | None = None,
    player: FakePopulationPlayer | None = None,
    on_stage: Callable[[int, str], None] | None = None,
) -> PopulationLoop:
    config = config or population_config()
    return PopulationLoop(config, run_dir, game(), player or FakePopulationPlayer(config), on_stage)


def gen_path(run_dir: Path, generation: int) -> Path:
    return layout.generation_dir(run_dir, generation)


def tree_bytes(run_dir: Path) -> dict[str, str]:
    """SHA-256 of every file the run's generations, checkpoints and config hold."""
    out: dict[str, str] = {}
    for sub in ("generations", "checkpoints"):
        base = run_dir / sub
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            out[path.relative_to(run_dir).as_posix()] = layout.sha256_file(path)
    out["run-config.json"] = layout.sha256_file(run_dir / "run-config.json")
    return out


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("population")
    make_population(run_dir).run(GENERATIONS)
    return run_dir


def member_record(run_dir: Path, generation: int, member: str, role: str) -> Any:
    path = gen_path(run_dir, generation) / "members" / member / "records" / f"{role}.json"
    return read_record(layout.read_json(path))


# -- the seeds -----------------------------------------------------------------------------------


def test_each_member_is_seeded_from_its_own_population_draws() -> None:
    config = population_config(members=3)
    configs = [member_config(config, k) for k in range(3)]
    for cfg in configs:
        assert 0 <= cfg.init_preview <= MAX_SEED
        assert 0 <= cfg.init_battle <= MAX_SEED
        assert 0 <= cfg.learner_seed <= MAX_SEED
        assert cfg.hash == config.hash  # records and sidecars name the run's config
    seeds = {(c.init_preview, c.init_battle, c.learner_seed) for c in configs}
    assert len(seeds) == 3
    assert len({s for c in configs for s in (c.init_preview, c.init_battle, c.learner_seed)}) == 9
    # The seeds are the draws of generation 0: member-init 0 Preview, 1 Battle, member-learner 0.
    assert (
        configs[1].init_preview == population_draw("pop-tiny", 0, "m1", "member-init", 0) & MAX_SEED
    )
    assert (
        configs[1].init_battle == population_draw("pop-tiny", 0, "m1", "member-init", 1) & MAX_SEED
    )
    assert (
        configs[1].learner_seed
        == population_draw("pop-tiny", 0, "m1", "member-learner", 0) & MAX_SEED
    )


def test_a_member_ppo_override_changes_only_that_members_start() -> None:
    config = population_config(
        edit=lambda raw: raw["population"].update(
            member_ppo=[
                {"member": 1, "learning_rate_micro": 700, "epochs": 3, "clip_micro": 250_000}
            ]
        )
    )
    assert member_ppo(config, 0) == config.ppo
    changed = member_ppo(config, 1)
    assert (changed.learning_rate, changed.epochs, changed.clip) == (0.0007, 3, 0.25)
    assert changed.value_coef == config.ppo.value_coef


def test_a_member_override_moves_the_schedule_starts() -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["learning_rate_schedule"] = {
            "kind": "linear",
            "final_micro": 100,
            "generations": 10,
        }
        raw["ppo"]["entropy_schedule"] = {"kind": "linear", "final_micro": 0, "generations": 10}
        raw["population"]["member_ppo"] = [
            {"member": 0, "learning_rate_micro": 900, "entropy_coef_micro": 12_000}
        ]

    config = population_config(edit=edit)
    changed = member_ppo(config, 0)
    assert changed.schedule is not None
    assert changed.schedule.start_micro == 900
    assert changed.entropy_schedule is not None
    assert changed.entropy_schedule.start_micro == 12_000


def test_target_kl_override() -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["target_kl_micro"] = 20_000
        raw["population"]["explore"]["bounds"]["target_kl_micro"] = {"min": 10_000, "max": 40_000}
        raw["population"]["member_ppo"] = [{"member": 1, "target_kl_micro": 30_000}]

    config = population_config(edit=edit)
    assert member_ppo(config, 1).target_kl == 0.03
    assert member_ppo(config, 0).target_kl == 0.02


# -- the run -------------------------------------------------------------------------------------


def test_every_generation_is_one_committed_directory_with_a_directory_per_member(
    finished: Path,
) -> None:
    names = sorted(p.name for p in layout.generations_dir(finished).iterdir())
    assert names == ["g000000", "g000001", "g000002", "g000003"]
    for g in range(GENERATIONS + 1):
        path = gen_path(finished, g)
        facts = layout.verify_generation(path, g)
        assert sorted(facts["members"]) == list(LABELS)
        assert (path / "team-set.json").is_file()
        for label in LABELS:
            member = path / "members" / label
            assert (member / f"learner-{g}.safetensors").is_file()
            assert (member / f"learner-{g}.json").is_file()
            assert sorted(p.name for p in (member / "records").iterdir()) == [
                "battle.json",
                "preview.json",
            ]
            assert (member / "train-report.json").is_file() == (g > 0)
            assert (member / "stream.jsonl").is_file() == (g > 0)
        assert (path / "league-run.json").is_file() == (g > 0)
        assert (path / "manifest.json").is_file()
    # The manifest covers the member files and the simulator side's files.
    files = layout.read_json(gen_path(finished, 2) / "manifest.json")["files"]
    assert "members/m1/train-report.json" in files
    assert "members/m0/stream.jsonl" in files
    assert "league-run.json" in files


def test_the_members_start_from_different_pairs_and_the_checkpoints_are_in_the_store(
    finished: Path,
) -> None:
    ids: dict[int, set[str]] = {}
    for g in range(GENERATIONS + 1):
        facts = layout.read_json(gen_path(finished, g) / "manifest.json")["members"]
        for label in LABELS:
            for role in ROLES:
                found = facts[label]["checkpoints"][role]
                stored = (finished / "checkpoints" / f"{found}.safetensors").read_bytes()
                assert checkpoint_id(stored) == found
                ids.setdefault(g, set()).add(found)
        assert len(ids[g]) == 4  # four distinct networks every generation
    zero = layout.read_json(gen_path(finished, 0) / "manifest.json")["members"]
    assert zero["m0"]["checkpoints"] != zero["m1"]["checkpoints"]


def test_member_records_name_their_lineage_parent_and_league_opponents(finished: Path) -> None:
    config = population_config()
    for g in range(GENERATIONS + 1):
        for label in LABELS:
            for role in ROLES:
                record = member_record(finished, g, label, role)
                assert record.lineage_id == f"{config.lineage_id}.{label}"
                assert record.role == role
                assert record.run_config_hash == config.hash
                assert record.data_class == "synthetic"
                if g == 0:
                    assert record.parent is None
                    assert record.training_set_hashes == ()
                    assert record.league_opponents == ()
                    continue
                before = member_record(finished, g - 1, label, role)
                assert record.parent == before.checkpoint_id
                other = LABELS[1 - LABELS.index(label)]
                expected = sorted(
                    member_record(finished, g - 1, other, r).checkpoint_id for r in ROLES
                )
                assert record.league_opponents == tuple(expected)
                facts = layout.read_json(gen_path(finished, g) / "manifest.json")
                assert record.training_set_hashes == (facts["members"][label]["training_set_hash"],)
                assert record.learner_state_hash == facts["members"][label]["learner_state_hash"]
    a = member_record(finished, 2, "m0", "battle")
    b = member_record(finished, 2, "m1", "battle")
    assert a.training_set_hashes != b.training_set_hashes
    assert a.learner_state_hash != b.learner_state_hash


def test_the_team_set_grows_and_the_records_name_it(finished: Path) -> None:
    first = layout.read_json(gen_path(finished, 1) / "team-set.json")
    second = layout.read_json(gen_path(finished, 2) / "team-set.json")
    assert layout.read_json(gen_path(finished, 0) / "team-set.json")["teams"] == []
    assert len(first["teams"]) == 2
    assert set(first["teams"]) < set(second["teams"])
    assert member_record(finished, 2, "m0", "preview").team_set_manifest_hash == sha256_hex(
        jcs(second)
    )


def test_run_metrics_sum_the_members_and_per_member_counts_sit_beside_them(finished: Path) -> None:
    config = population_config()
    for g in range(GENERATIONS + 1):
        record = read_run_metrics(layout.read_json(metrics_path(finished, g)), config, g)
        assert record["generation"] == g
    totals = layout.read_json(metrics_path(finished, 2))["counts"]
    members = layout.read_json(finished / "metrics" / "members" / "g000002.json")["members"]
    assert sorted(members) == list(LABELS)
    assert totals["decisions"] == sum(m["decisions"] for m in members.values())
    assert totals["optimiser_steps"] == sum(m["optimiser_steps"] for m in members.values())
    assert totals["games"] == 8
    assert not any(
        "metrics" in name
        for name in layout.read_json(gen_path(finished, 2) / "manifest.json")["files"]
    )


def test_an_independent_rerun_is_byte_identical(finished: Path, tmp_path: Path) -> None:
    make_population(tmp_path).run(GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


def test_a_finished_run_is_a_no_op_and_another_config_is_refused(finished: Path) -> None:
    result = make_population(finished).run(GENERATIONS)
    assert result.resumed_from == GENERATIONS
    assert result.generations == []
    other = population_config(key="other-run")
    with pytest.raises(LoopError, match="another TrainingRunConfig"):
        make_population(finished, other).run(1)


STAGES: list[tuple[int, str]] = [
    (1, "played"),
    (2, "trained:m0"),
    (2, "trained:m1"),
    (2, "trained"),
    (2, "written"),
    (2, "committed"),
    (3, "played"),
]


@pytest.mark.parametrize(("generation", "stage"), STAGES)
def test_a_run_killed_at_any_stage_resumes_to_the_same_bytes(
    finished: Path, tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        make_population(tmp_path, on_stage=killer(generation, stage)).run(GENERATIONS)
    resumed = make_population(tmp_path).run(GENERATIONS)
    committed_before = generation - (1 if stage != "committed" else 0)
    assert resumed.resumed_from == committed_before
    assert resumed.generations[-1].generation == GENERATIONS
    assert tree_bytes(tmp_path) == tree_bytes(finished)
    final = resumed.generations[-1]
    uninterrupted = layout.read_json(gen_path(finished, GENERATIONS) / "manifest.json")["members"]
    assert {m.member: m.learner_state_hash for m in final.members} == {
        label: uninterrupted[label]["learner_state_hash"] for label in LABELS
    }


def test_a_resume_refuses_a_member_whose_state_or_weights_changed(
    finished: Path, tmp_path: Path
) -> None:
    make_population(tmp_path).run(1)
    path = gen_path(tmp_path, 1) / "members" / "m1" / "records" / "battle.json"
    raw = layout.read_json(path)
    raw["lineage_id"] = "tiny-main.m0"
    layout.write_json(path, raw)
    manifest = layout.read_json(gen_path(tmp_path, 1) / "manifest.json")
    manifest["files"] = layout.hash_tree(gen_path(tmp_path, 1))
    layout.write_json(gen_path(tmp_path, 1) / "manifest.json", manifest)
    with pytest.raises(LoopError, match="disagrees with the state"):
        make_population(tmp_path).run(2)


def test_a_resume_refuses_a_manifest_of_other_members(tmp_path: Path) -> None:
    make_population(tmp_path).run(1)
    path = gen_path(tmp_path, 1) / "manifest.json"
    manifest = layout.read_json(path)
    del manifest["members"]["m1"]
    layout.write_json(path, manifest)
    with pytest.raises(LoopError, match="not the config's population"):
        make_population(tmp_path).run(2)


# -- the checks on the simulator side's files -----------------------------------------------------


def break_record(
    change: Callable[[dict[str, Any]], None],
) -> Callable[[dict[str, Any], dict[str, Any]], None]:
    return lambda _summary, record: change(record)


def refuses(
    tmp_path: Path, change: Callable[[dict[str, Any], dict[str, Any]], None], match: str
) -> None:
    config = population_config()
    with pytest.raises(LoopError, match=match):
        make_population(tmp_path, config, FakePopulationPlayer(config, change)).run(1)
    assert not any(p.name == "g000001" for p in layout.generations_dir(tmp_path).iterdir())


def test_a_record_of_another_pair_is_refused(tmp_path: Path) -> None:
    def other(record: dict[str, Any]) -> None:
        record["members"][0]["pair_policy_id"] = "policy:" + "e" * 64
        for entry in record["members"]:
            for o in entry["opponents"]:
                if o["member"] == "m0":
                    o["opponent_policy_id"] = "policy:" + "e" * 64

    refuses(tmp_path, break_record(other), "other pairs than the ones played")


def test_a_record_of_another_generation_or_with_other_forfeits_is_refused(tmp_path: Path) -> None:
    refuses(
        tmp_path / "a",
        lambda summary, record: summary.update(forfeits=1),
        "other forfeits",
    )
    refuses(
        tmp_path / "b",
        lambda summary, record: summary.update(games=summary["games"][:-1]),
        "another number of games",
    )
    refuses(
        tmp_path / "c", lambda summary, record: summary.update(generation=9), "another generation"
    )


def test_a_plan_seed_must_be_the_pairs(tmp_path: Path) -> None:
    config = population_config()

    class Wrong(FakePopulationPlayer):
        def play_population(self, request: PopulationPlayRequest) -> None:
            super().play_population(request)
            plan = layout.read_json(request.out_dir / "plan.json")
            plan[1]["seed"] = league_game_seed(config.run_key, request.generation, 1)
            layout.write_json(request.out_dir / "plan.json", plan)
            record = layout.read_json(request.out_dir / "league-run.json")
            record["plan_hash"] = sha256_hex(jcs(plan))
            layout.write_json(request.out_dir / "league-run.json", record)

    with pytest.raises(LoopError, match="seed the run key gives its pair"):
        make_population(tmp_path, config, Wrong(config)).run(1)


def test_a_training_set_that_is_not_the_reported_one_is_refused(tmp_path: Path) -> None:
    def tamper(summary: dict[str, Any], record: dict[str, Any]) -> None:
        summary["members"][1]["training_set_hash"] = "0" * 64

    refuses(tmp_path, tamper, "differs from the hash the simulator side reported")


def test_a_missing_member_entry_in_generation_json_is_refused(tmp_path: Path) -> None:
    refuses(
        tmp_path,
        lambda summary, record: summary.update(members=summary["members"][:1]),
        "exactly one entry for m1",
    )


def test_a_single_member_population_runs(tmp_path: Path) -> None:
    config = population_config(members=1, key="pop-one")
    result = make_population(tmp_path, config).run(2)
    final = result.generations[-1]
    assert [m.member for m in final.members] == ["m0"]
    record = member_record(tmp_path, 2, "m0", "battle")
    assert record.league_opponents == ()
    assert record.lineage_id == f"{config.lineage_id}.m0"


# -- refusals ------------------------------------------------------------------------------------


def test_the_single_lineage_driver_refuses_a_population_config(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match="PopulationLoop"):
        make_loop(tmp_path, population_config())


def test_the_population_driver_refuses_a_single_lineage_config(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match="use Loop"):
        PopulationLoop(tiny_config(), tmp_path, game(), FakePopulationPlayer(tiny_config()))
    assert list(tmp_path.iterdir()) == []


def test_exploiters_and_a_past_pool_are_refused_as_the_play_side_refuses_them(
    tmp_path: Path,
) -> None:
    def exploiters(raw: dict[str, Any]) -> None:
        raw["population"]["exploiters"] = 1

    with pytest.raises(LoopError, match="exploiter seats"):
        make_population(tmp_path / "x", population_config(edit=exploiters))

    def past(raw: dict[str, Any]) -> None:
        raw["population"]["shares"] = {
            "ladder_ppm": 500_000,
            "self_play_ppm": 250_000,
            "past_pool_ppm": 250_000,
        }
        bounds = raw["population"]["explore"]["bounds"]
        bounds["self_play_ppm"] = {"min": 200_000, "max": 300_000}
        bounds["past_pool_ppm"] = {"min": 200_000, "max": 300_000}

    with pytest.raises(LoopError, match="past_pool_ppm"):
        make_population(tmp_path / "p", population_config(edit=past))


def test_the_provenance_export_of_a_population_run_is_not_built_yet(
    finished: Path, tmp_path: Path
) -> None:
    with pytest.raises(LoopError, match="slice 12"):
        export_provenance(finished, tmp_path / "out")


def test_a_two_role_warm_start_is_refused_with_population() -> None:
    def both(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}

    with pytest.raises(RunConfigError, match="would give every member the same pair"):
        population_config(edit=both)


def test_the_corpus_samples_for_a_warm_population() -> None:
    root = SAMPLES / "training-run-config"
    valid = json.loads((root / "valid/population-warm-start-battle-only.json").read_text())
    assert read_run_config(valid).warm_start == {"battle": "b" * 64}
    bad = json.loads((root / "reader-invalid/population-warm-start-both.json").read_text())
    with pytest.raises(RunConfigError, match="would give every member the same pair"):
        read_run_config(bad)


# -- a Battle-only warm start ---------------------------------------------------------------------


def test_a_battle_only_warm_start_shares_the_clone_and_seeds_each_preview(
    tmp_path: Path,
) -> None:
    (tmp_path / "clone").mkdir()
    clone = clone_output(tmp_path / "clone")
    ids = clone_ids(clone)
    config = population_config(
        key="pop-warm", edit=lambda raw: raw.update(warm_start={"battle": ids["battle"]})
    )
    run_dir = tmp_path / "run"
    import_clone(clone, run_dir, ("battle",))
    make_population(run_dir, config).run(2)
    zero = layout.read_json(gen_path(run_dir, 0) / "manifest.json")["members"]
    assert (
        zero["m0"]["checkpoints"]["battle"] == zero["m1"]["checkpoints"]["battle"] == ids["battle"]
    )
    assert zero["m0"]["checkpoints"]["preview"] != zero["m1"]["checkpoints"]["preview"]
    # The clone record is each member's Battle record at generation 0, byte for byte.
    clone_record = (run_dir / "warm-start" / "records" / "battle.json").read_bytes()
    for label in LABELS:
        path = gen_path(run_dir, 0) / "members" / label / "records" / "battle.json"
        assert path.read_bytes() == clone_record
        preview = member_record(run_dir, 0, label, "preview")
        assert preview.lineage_id == f"{config.lineage_id}.{label}"
        assert preview.training_set_hashes == ()
    after = member_record(run_dir, 1, "m0", "battle")
    assert after.parent == ids["battle"]
    # Each member started from its own copy, so the trained Battle networks differ.
    one = layout.read_json(gen_path(run_dir, 1) / "manifest.json")["members"]
    assert one["m0"]["checkpoints"]["battle"] != one["m1"]["checkpoints"]["battle"]
    # A resume of generation 0 restores the clone records.
    again = tmp_path / "again"
    import_clone(clone, again, ("battle",))
    with pytest.raises(Killed):
        make_population(again, config, on_stage=killer(1, "played")).run(2)
    make_population(again, config).run(2)
    assert tree_bytes(again) == tree_bytes(run_dir)


def test_a_warm_run_refuses_a_generation_zero_whose_records_are_not_the_clone_records(
    tmp_path: Path,
) -> None:
    (tmp_path / "clone").mkdir()
    clone = clone_output(tmp_path / "clone")
    ids = clone_ids(clone)
    config = population_config(
        key="pop-warm2", edit=lambda raw: raw.update(warm_start={"battle": ids["battle"]})
    )
    run_dir = tmp_path / "run"
    import_clone(clone, run_dir, ("battle",))
    make_population(run_dir, config).run(0)
    path = gen_path(run_dir, 0) / "members" / "m0" / "records" / "battle.json"
    raw = layout.read_json(path)
    raw["lineage_id"] = "something-else"
    layout.write_json(path, raw)
    manifest = layout.read_json(gen_path(run_dir, 0) / "manifest.json")
    manifest["files"] = layout.hash_tree(gen_path(run_dir, 0))
    layout.write_json(gen_path(run_dir, 0) / "manifest.json", manifest)
    with pytest.raises(LoopError, match="not the warm-start clone records"):
        make_population(run_dir, config).run(1)


def test_initial_networks_differ_per_member_but_not_per_rerun() -> None:
    config = population_config()
    nets = [
        checkpoint_id(
            checkpoint_bytes(
                init_network("battle", game(), config.model, member_config(config, k).init_battle),
                game(),
            )
        )
        for k in (0, 1, 0)
    ]
    assert nets[0] == nets[2]
    assert nets[0] != nets[1]


def test_a_negative_generation_count_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match="cannot be negative"):
        make_population(tmp_path).run(-1)


# -- device residency ------------------------------------------------------------------------------


def test_one_learner_at_a_time_is_brought_to_the_device_and_the_bytes_do_not_change(
    finished: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_device: set[int] = set()
    seen: list[tuple[str, int]] = []
    most = [0]
    real_up, real_park = population.bring_up, population.park

    def up(learner: Any, device: Any) -> None:
        assert not on_device, "a second learner was brought to the device"
        real_up(learner, device)
        on_device.add(id(learner))
        seen.append(("up", id(learner)))
        most[0] = max(most[0], len(on_device))

    def down(learner: Any) -> None:
        real_park(learner)
        on_device.discard(id(learner))
        seen.append(("park", id(learner)))

    monkeypatch.setattr(population, "bring_up", up)
    monkeypatch.setattr(population, "park", down)
    make_population(tmp_path).run(GENERATIONS)
    assert most[0] == 1
    assert not on_device
    assert len(seen) == 2 * 2 * GENERATIONS  # two members, up and park, every generation
    assert [kind for kind, _ in seen] == ["up", "park"] * (2 * GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


def test_a_learner_is_parked_when_its_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_up, real_park = population.bring_up, population.park
    monkeypatch.setattr(
        population,
        "bring_up",
        lambda learner, device: (events.append("up"), real_up(learner, device)),
    )
    monkeypatch.setattr(
        population, "park", lambda learner: (events.append("park"), real_park(learner))
    )

    def boom(self: Any, trajectories: Any) -> Any:
        raise RuntimeError("train failed")

    monkeypatch.setattr(Learner, "train", boom)
    with pytest.raises(RuntimeError, match="train failed"):
        make_population(tmp_path).run(1)
    assert events == ["up", "park"]


def test_moving_a_learner_keeps_its_state_bytes_and_moves_the_moments() -> None:
    config = population_config()
    loop = PopulationLoop(config, Path("unused"), game(), FakePopulationPlayer(config))
    learner = loop._new_learner(0, None)
    assert learner.device == population.HOST
    before = pack_state(learner).hash
    population.move_learner(learner, torch.device("cpu"))
    assert pack_state(learner).hash == before
    # Give Adam moments, then move: the moment tensors are the ones the optimiser holds.
    for network in learner.networks.values():
        for parameter in network.parameters():
            parameter.grad = torch.zeros_like(parameter)
    for optimiser in learner.optimisers.values():
        optimiser.step()
    after_step = pack_state(learner).hash
    population.move_learner(learner, torch.device("cpu"))
    assert pack_state(learner).hash == after_step
    states = [s for o in learner.optimisers.values() for s in o.state.values()]
    assert states
    assert all(s["exp_avg"].device.type == "cpu" for s in states)


# -- the nits -----------------------------------------------------------------------


def test_two_members_with_one_pair_are_refused_at_generation_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = population.member_config
    monkeypatch.setattr(population, "member_config", lambda config, index: real(config, 0))
    with pytest.raises(LoopError, match="two members start from the same frozen pair"):
        make_population(tmp_path).run(1)
    assert not any(p.name == "g000000" for p in layout.generations_dir(tmp_path).iterdir())


def test_a_plan_game_must_carry_its_half(tmp_path: Path) -> None:
    config = population_config()

    class Wrong(FakePopulationPlayer):
        def play_population(self, request: PopulationPlayRequest) -> None:
            super().play_population(request)
            plan = layout.read_json(request.out_dir / "plan.json")
            plan[0]["half"] = 1
            layout.write_json(request.out_dir / "plan.json", plan)
            record = layout.read_json(request.out_dir / "league-run.json")
            record["plan_hash"] = sha256_hex(jcs(plan))
            layout.write_json(request.out_dir / "league-run.json", record)

    with pytest.raises(LoopError, match="seed the run key gives its pair"):
        make_population(tmp_path, config, Wrong(config)).run(1)


def test_a_member_whose_games_all_forfeited_fails_the_generation(tmp_path: Path) -> None:
    config = population_config()

    class Forfeited(FakePopulationPlayer):
        def play_population(self, request: PopulationPlayRequest) -> None:
            if request.generation == 2:
                # What the play side does when a member has no Episode of its own pair.
                raise PlayError(
                    "no Episode of m1's frozen pair: every game it played was forfeited"
                )
            super().play_population(request)

    with pytest.raises(PlayError, match="no Episode of m1's frozen pair"):
        make_population(tmp_path, config, Forfeited(config)).run(GENERATIONS)
    names = sorted(p.name for p in layout.generations_dir(tmp_path).iterdir())
    assert "g000002" not in names  # nothing of the failed generation is committed
    assert layout.latest_complete(tmp_path) is not None
    assert layout.latest_complete(tmp_path).generation == 1  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "command",
    [
        ["inspect"],
        ["report"],
        ["progress", "--generation", "1", "--engine-data", "x"],
        ["replay", "--generation", "1"],
    ],
)
def test_the_single_lineage_views_refuse_a_population_run_by_name(
    finished: Path, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    code = main([command[0], "--run-dir", str(finished), *command[1:]])
    assert code == EXIT_REFUSED
    assert "population run is not supported" in capsys.readouterr().err
