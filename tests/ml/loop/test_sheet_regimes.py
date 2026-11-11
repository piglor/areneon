"""Sheet regimes of a mixed-format run (ADR-027 sections 5 and 5a, #1153)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.record import RecordError, read_record
from areneon_ml.inspector.readers import read_generation, read_plan
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Loop as Driver
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.league import (
    LeagueRecordError,
    check_generation_plan,
    league_game_seed,
    read_league_run_record,
)
from areneon_ml.loop.play import PlayRequest
from areneon_ml.loop.records import RecordFacts, checkpoint_record
from areneon_ml.loop.sheet_regimes import (
    SheetRegimeError,
    check_plan_formats,
    closed_game_count,
    manifests_problem,
    plan_formats,
    regime_digest,
    self_play_regimes,
)

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import MANIFEST, FakePlayer, make_loop
from tests.ml.model.support import SAMPLES

OPEN = "gen9championsvgc2026regmcbo3"


def raw_config(**sheet: Any) -> dict[str, Any]:
    body: dict[str, Any] = json.loads((SAMPLES / "training-run-config/valid/tiny.json").read_text())
    body["sheet_regimes"] = {
        "open_format": OPEN,
        "closed_share_ppm": 500000,
        "open_only_generations": 1,
        "closed_sheet_source": "revealed-only/1",
        **sheet,
    }
    return body


def sheet_config(**sheet: Any) -> RunConfig:
    return read_run_config(raw_config(**sheet))


def test_the_sample_config_is_read_and_hashes_apart_from_the_closed_only_run() -> None:
    sample = read_run_config(
        json.loads((SAMPLES / "training-run-config/valid/sheet-regimes.json").read_text())
    )
    assert sample.sheet_regimes is not None
    assert sample.sheet_regimes.open_format == OPEN
    assert sample.sheet_regimes.closed_share_ppm == 500000
    assert sample.sheet_regimes.open_only_generations == 10
    assert sample.hash != tiny_config().hash
    assert tiny_config().sheet_regimes is None


def test_search_labels_are_refused_beside_sheet_regimes() -> None:
    body = raw_config()
    body["generation"]["search_labels"] = {"search_config_hash": "a" * 64, "start_generation": 3}
    body["ppo"]["search_distill"] = {
        "coef_micro": 1000000,
        "c_scale_micro": 4000000,
        "min_spread_ppm": 20000,
        "aggregate_generations": 4,
    }
    body["warm_start"] = {"battle": "b" * 64}
    body["ppo"]["value_warmup"] = {"generations": 2}
    body["team_dataset"] = {"path": "x", "dataset_hash": "5" * 64}
    with pytest.raises(Exception, match="search_labels"):
        read_run_config(body)


# Vectors computed with the Python reference above and pinned in sheet-regimes.test.ts too.
def test_the_digest_and_the_quota_order_are_pinned() -> None:
    assert regime_digest("tiny-run", 2, 0) == (
        "3bd74e5c7d17e70ba2fef095a9079d48a146c125e84d47653108a04cf85847f8"
    )
    sheet = sheet_config().sheet_regimes
    assert sheet is not None
    assert self_play_regimes(sheet, "tiny-run", 1, 8) == ["open"] * 8
    assert self_play_regimes(sheet, "tiny-run", 2, 8) == [
        "closed" if g % 2 == 0 else "open" for g in range(8)
    ]
    assert self_play_regimes(sheet, "tiny-run", 3, 8) == [
        "open",
        "closed",
        "open",
        "closed",
        "closed",
        "open",
        "open",
        "closed",
    ]


@pytest.mark.parametrize(
    ("games", "share", "closed"),
    [(4, 500000, 2), (3, 500000, 2), (5, 500000, 3), (7, 250000, 2), (10, 1, 0), (10, 999999, 10)],
)
def test_the_closed_count_rounds_half_up(games: int, share: int, closed: int) -> None:
    assert closed_game_count(games, share) == closed


def plan_for(config: RunConfig, generation: int) -> list[dict[str, Any]]:
    formats = plan_formats(config, generation)
    plan = []
    for game, format_id in enumerate(formats):
        plan.append(
            {
                "game": game,
                "seed": league_game_seed(config.run_key, generation, game),
                "format_id": format_id,
                "opponent": "self" if game < config.generation.games else {"actor": "x"},
            }
        )
    return plan


def test_the_plan_formats_are_the_quota_and_baselines_are_closed() -> None:
    config = sheet_config()
    first = plan_formats(config, 1)
    assert first == [OPEN] * 4 + [config.format_id] * 2
    second = plan_formats(config, 2)
    assert second.count(OPEN) == 2
    assert second[4:] == [config.format_id] * 2
    assert check_plan_formats(plan_for(config, 2), config, 2) == {"open": 2, "closed": 4}


def test_a_plan_that_breaks_the_quota_is_refused() -> None:
    config = sheet_config()
    plan = plan_for(config, 2)
    plan[0]["format_id"] = OPEN if plan[0]["format_id"] != OPEN else config.format_id
    with pytest.raises(SheetRegimeError, match="quota"):
        check_plan_formats(plan, config, 2)
    plain = tiny_config()
    assert check_plan_formats([{"game": 0}], plain, 1) == {}
    with pytest.raises(SheetRegimeError, match="no sheet_regimes"):
        check_plan_formats([{"game": 0, "format_id": OPEN}], plain, 1)
    with pytest.raises(SheetRegimeError, match="no sheet_regimes"):
        plan_formats(plain, 1)


def manifest(format_id: str, **change: str) -> dict[str, Any]:
    body = {
        "showdown_commit": "7" * 40,
        "format_id": format_id,
        "regulation": "m-c",
        "ruleset_hash": "1" * 64,
        "format_rules_hash": "2" * 64,
    }
    return {**body, **change}


def listed(config: RunConfig) -> list[dict[str, Any]]:
    return sorted([manifest(config.format_id), manifest(OPEN, format_rules_hash="3" * 64)], key=jcs)


def test_the_game_manifests_rules() -> None:
    config = sheet_config()
    both = listed(config)
    closed = manifest(config.format_id)
    assert manifests_problem(both, config, closed) is None
    assert manifests_problem(both, config) is None
    assert "sorted" in str(manifests_problem(both[::-1], config))
    assert "sorted" in str(manifests_problem([both[0], both[0]], config))
    assert "one of" in str(manifests_problem([manifest(OPEN)], config, closed))
    assert "one manifest for each" in str(manifests_problem([manifest("other")], config))
    assert "one manifest for each" in str(manifests_problem(both[:1], config))
    third = sorted([*both, manifest("third")], key=jcs)
    assert "one manifest for each" in str(manifests_problem(third, config))
    wrong = sorted([both[0], manifest("other")], key=jcs)
    assert "one manifest for each" in str(manifests_problem(wrong, config))
    other_commit = [both[0], {**both[1], "showdown_commit": "8" * 40}]
    assert "share one engine commit" in str(
        manifests_problem(sorted(other_commit, key=jcs), config)
    )
    assert "no sheet_regimes" in str(manifests_problem(both, tiny_config()))


def sheet_record() -> dict[str, Any]:
    body: dict[str, Any] = json.loads(
        (SAMPLES / "league-run-record/valid/sheet-regimes.json").read_text()
    )
    return body


def test_the_league_record_sample_reads_with_the_sheet_config_only() -> None:
    config = sheet_config(open_only_generations=10)
    record = copy.deepcopy(sheet_record())
    record["run_config_hash"] = config.hash
    read = read_league_run_record(record, config)
    assert read.regimes == (2, 4)
    with pytest.raises(LeagueRecordError, match="exist exactly for a config with sheet_regimes"):
        read_league_run_record(record, tiny_config())
    plain = json.loads((SAMPLES / "league-run-record/valid/tiny.json").read_text())
    with pytest.raises(LeagueRecordError, match="exist exactly for a config with sheet_regimes"):
        read_league_run_record(plain, config)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("regimes-do-not-sum.json", "regimes' counts must sum to games"),
        ("regimes-without-manifests.json", "come together or not at all"),
        ("manifests-unsorted.json", "sorted by their JCS bytes"),
    ],
)
def test_the_mixed_format_reader_invalid_records(name: str, message: str) -> None:
    config = sheet_config(open_only_generations=10)
    record = json.loads((SAMPLES / f"league-run-record/reader-invalid/{name}").read_text())
    record["run_config_hash"] = config.hash
    with pytest.raises(LeagueRecordError, match=message):
        read_league_run_record(record, config)


def test_the_record_must_name_the_plans_regimes() -> None:
    config = sheet_config()
    plan = plan_for(config, 2)
    record = copy.deepcopy(sheet_record())
    record["generation"] = 2
    record["run_config_hash"] = config.hash
    record["games"] = len(plan)
    record["played"] = len(plan)
    record["episodes"] = 2 * len(plan)
    record["opponents"] = [
        {"opponent_policy_id": record["pair_policy_id"], "games": 4},
        {"opponent_policy_id": "random-legal@tiny", "games": 2},
    ]
    record["plan_hash"] = __import__("hashlib").sha256(jcs(plan)).hexdigest()
    record["regimes"] = {"open": 2, "closed": 4}
    record["game_manifests"] = listed(config)
    read = read_league_run_record(record, config)
    check_generation_plan(read, plan, config)
    record["regimes"] = {"open": 3, "closed": 3}
    with pytest.raises(LeagueRecordError, match="not the plan's formats"):
        check_generation_plan(read_league_run_record(record, config), plan, config)
    plan[0]["format_id"] = "other"
    record["regimes"] = {"open": 2, "closed": 4}
    record["plan_hash"] = __import__("hashlib").sha256(jcs(plan)).hexdigest()
    with pytest.raises(LeagueRecordError, match="quota"):
        check_generation_plan(read_league_run_record(record, config), plan, config)


def test_a_checkpoint_record_carries_game_manifests_only_when_given() -> None:
    config = sheet_config()
    both = listed(config)
    facts: dict[str, Any] = {
        "role": "battle",
        "checkpoint_id": "1" * 64,
        "parent": None,
        "training_set_hashes": ["5" * 64],
        "team_set_manifest_hash": "7" * 64,
        "learner_state_hash": "9" * 64,
        "encoder_version": "enc1:" + "e" * 64,
        "manifest": manifest(config.format_id),
    }
    mixed = checkpoint_record(config, RecordFacts(**facts, game_manifests=tuple(both)))
    assert mixed.game_manifests == tuple(both)
    assert mixed.to_json()["game_manifests"] == both
    assert read_record(mixed.to_json()).game_manifests == tuple(both)
    # No-change: without the field the record has no such key (the closed-only run's bytes).
    plain = checkpoint_record(config, RecordFacts(**facts))
    assert plain.game_manifests is None
    assert "game_manifests" not in plain.to_json()
    assert set(plain.to_json()) == {
        "schema_version",
        "checkpoint_id",
        "role",
        "lineage_id",
        "league_opponents",
        "training_set_hashes",
        "team_set_manifest_hash",
        "run_config_hash",
        "learner_state_hash",
        "encoder_version",
        "data_class",
        "manifest",
    }


def test_the_record_reader_refuses_a_foreign_commit() -> None:
    body = json.loads((SAMPLES / "checkpoint-record/valid/game-manifests.json").read_text())
    other = {**body["game_manifests"][1], "showdown_commit": "8" * 40}
    body["game_manifests"] = sorted([body["game_manifests"][0], other], key=jcs)
    body["manifest"] = body["game_manifests"][0]
    with pytest.raises(RecordError, match="share one engine commit"):
        read_record(body)


def _namespace(raw: dict[str, Any]) -> Any:
    return SimpleNamespace(raw=raw)


def test_the_driver_checks_a_generation_s_manifests() -> None:
    config = sheet_config()
    both = listed(config)
    closed = manifest(config.format_id)
    check: Any = Driver._check_game_manifests
    mixed: Any = SimpleNamespace(config=config)
    plain: Any = SimpleNamespace(config=tiny_config())
    league = _namespace({"game_manifests": both})
    summary: dict[str, Any] = {"manifest": closed, "game_manifests": both}
    assert check(mixed, summary, league) == tuple(both)
    with pytest.raises(LoopError, match="lacks game_manifests"):
        check(mixed, {"manifest": closed}, league)
    with pytest.raises(LoopError, match="sorted"):
        check(mixed, {"manifest": closed, "game_manifests": both[::-1]}, league)
    with pytest.raises(LoopError, match="other game_manifests"):
        check(mixed, summary, _namespace({"game_manifests": both[:1]}))
    assert check(plain, {"manifest": closed}, league) is None
    with pytest.raises(LoopError, match="no sheet_regimes"):
        check(plain, summary, league)


def test_the_inspector_readers_accept_a_mixed_run_s_optional_keys() -> None:
    entry = {
        "game": 0,
        "seed": "sodium," + "0" * 32,
        "teams": {"p1": 0, "p2": 1},
        "opponent": "self",
        "policy_seat": "both",
        "baseline_seed": 1,
        "format_id": OPEN,
    }
    assert len(read_plan([entry])) == 1
    with pytest.raises(Exception, match="format_id"):
        read_plan([{**entry, "format_id": 3}])
    generation = {
        "schema_version": 1,
        "generation": 1,
        "games": [],
        "forfeits": 0,
        "episode_hashes": [],
        "training_set_hash": "a" * 64,
        "examples": 0,
        "manifest": manifest("x"),
        "team_hashes": [],
        "game_manifests": [manifest("x"), manifest(OPEN)],
    }
    assert read_generation(generation, 1).examples == 0
    with pytest.raises(Exception, match="game_manifests"):
        read_generation({**generation, "game_manifests": [{"format_id": "x"}]}, 1)


# -- the loop with a stand-in simulator side ------------------------------------------------------


def mixed_config() -> RunConfig:
    body = raw_config()
    body["generation"]["games"] = 2
    body["generation"]["baseline_games"] = []
    return read_run_config(body)


class MixedPlayer(FakePlayer):
    """A stand-in that names each game's format, as the real simulator side does."""

    def __init__(self, config: RunConfig, *, report_manifests: bool = True) -> None:
        super().__init__(config=config)
        self.mixed = config
        self.report_manifests = report_manifests
        self.open = {**MANIFEST, "format_id": OPEN, "format_rules_hash": "3" * 64}

    def manifest(self, format_id: str) -> dict[str, Any]:
        return dict(self.open) if format_id == OPEN else dict(MANIFEST)

    def play(self, request: PlayRequest) -> None:
        super().play(request)
        out = request.out_dir
        formats = plan_formats(self.mixed, request.generation)
        plan = json.loads((out / "plan.json").read_text())
        for entry, format_id in zip(plan, formats, strict=True):
            entry["format_id"] = format_id
        record = json.loads((out / "league-run.json").read_text())
        record["plan_hash"] = sha256_hex(jcs(plan))
        if self.report_manifests:
            both = sorted([dict(MANIFEST), self.open], key=jcs)
            opened = formats.count(OPEN)
            record["regimes"] = {"open": opened, "closed": len(formats) - opened}
            record["game_manifests"] = both
            summary = json.loads((out / "generation.json").read_text())
            summary["game_manifests"] = both
            layout.write_json(out / "generation.json", summary)
        layout.write_json(out / "plan.json", plan)
        layout.write_json(out / "league-run.json", record)


def test_a_mixed_format_run_records_both_formats_in_every_checkpoint(tmp_path: Path) -> None:
    config = mixed_config()
    make_loop(tmp_path, config, MixedPlayer(config)).run(2)
    opened = {**MANIFEST, "format_id": OPEN, "format_rules_hash": "3" * 64}
    both = tuple(sorted([dict(MANIFEST), opened], key=jcs))
    for generation in range(3):
        for role in ("battle", "preview"):
            path = layout.generation_dir(tmp_path, generation) / "records" / f"{role}.json"
            record = read_record(json.loads(path.read_text()))
            assert record.game_manifests == both
            assert record.manifest == dict(MANIFEST)
    # A resumed run reads the same files and goes on.
    resumed = make_loop(tmp_path, config, MixedPlayer(config))
    resumed.run(3)
    assert layout.latest_complete(tmp_path) is not None


def test_a_generation_that_does_not_report_its_manifests_is_refused(tmp_path: Path) -> None:
    config = mixed_config()
    with pytest.raises(LoopError, match="game_manifests"):
        make_loop(tmp_path, config, MixedPlayer(config, report_manifests=False)).run(1)


def test_a_closed_only_run_rebuilds_a_committed_checkpoint_record_byte_for_byte() -> None:
    """No-change pin: the committed population fixture's records, rebuilt without game_manifests."""
    root = SAMPLES.parent.parent.parent / "fixtures" / "loop" / "population-run"
    config = read_run_config(json.loads((root / "run-config.json").read_text()))
    assert config.sheet_regimes is None
    paths = sorted(root.glob("generations/*/members/*/records/*.json"))
    assert len(paths) >= 8
    for path in paths:
        text = path.read_text()
        committed = json.loads(text)
        facts = RecordFacts(
            role=committed["role"],
            checkpoint_id=committed["checkpoint_id"],
            parent=committed.get("parent"),
            training_set_hashes=committed["training_set_hashes"],
            team_set_manifest_hash=committed["team_set_manifest_hash"],
            learner_state_hash=committed["learner_state_hash"],
            encoder_version=committed["encoder_version"],
            manifest=committed["manifest"],
            lineage_id=committed["lineage_id"],
            league_opponents=tuple(committed["league_opponents"]),
        )
        rebuilt = checkpoint_record(config, facts).to_json()
        assert "game_manifests" not in rebuilt
        assert rebuilt["run_config_hash"] == committed["run_config_hash"]
        assert jcs(rebuilt) == jcs(committed), path
