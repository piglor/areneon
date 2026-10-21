"""The LeagueRunRecord reader and the league seed check (ADR-009 §3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.loop.league import (
    LeagueRecordError,
    check_generation_plan,
    league_game_seed,
    read_league_run_record,
)

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import league_files
from tests.ml.model.support import SAMPLES

RECORDS = SAMPLES / "league-run-record"


def raw(name: str = "valid/tiny.json") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((RECORDS / name).read_text())
    return loaded


# The pinned vector of ``league-seed/1`` is the TS test's: the same numbers, so the two
# languages derive one seed from one run key, generation and game index.
@pytest.mark.parametrize(
    ("run_key", "generation", "game", "seed"),
    [
        ("tiny-run", 1, 0, "sodium,b73e6191281f92d07753d42296dfb7c0"),
        ("tiny-run", 1, 5, "sodium,cb9117ffd3513b4158002651bc9692bb"),
        ("tiny-run", 2, 0, "sodium,bbbfbe1e69918c198699250e57ea85cf"),
        ("league-main", 7, 31, "sodium,ae40cd0b66b292755d3fb44100de3cec"),
    ],
)
def test_the_seed_reproduces_the_pinned_vector(
    run_key: str, generation: int, game: int, seed: str
) -> None:
    assert league_game_seed(run_key, generation, game) == seed


@pytest.mark.parametrize(("generation", "game"), [(0, 0), (1, -1)])
def test_an_index_out_of_range_is_refused(generation: int, game: int) -> None:
    with pytest.raises(ValueError, match="counts from"):
        league_game_seed("a", generation, game)


def test_the_valid_samples_are_read_and_the_hash_is_the_configs() -> None:
    record = read_league_run_record(raw(), tiny_config())
    assert (record.run_id, record.generation, record.games, record.played) == ("tiny-run", 1, 6, 5)
    assert record.opponents[1] == ("random-legal@tiny", 2)
    assert read_league_run_record(raw("valid/self-play-only.json"), tiny_config()).forfeits == 0
    assert record.forfeit_causes == (("crash", 1),)
    assert (
        read_league_run_record(raw("valid/self-play-only.json"), tiny_config()).forfeit_causes == ()
    )
    mixed = read_league_run_record(raw("valid/mixed-causes.json"), tiny_config())
    assert mixed.forfeit_causes == (("crash", 1), ("latency_overrun", 2))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("counts-disagree.json", r"played \+ forfeits"),
        ("episodes-not-two-per-game.json", "two per played game"),
        ("unsorted-opponents.json", "sorted by opponent_policy_id"),
        ("duplicate-opponent.json", "sorted by opponent_policy_id"),
        ("opponents-do-not-sum.json", "must sum to games"),
        ("causes-do-not-sum.json", "causes' counts must sum to forfeits"),
        ("unsorted-causes.json", "forfeit_causes must be sorted"),
        ("duplicate-cause.json", "forfeit_causes must be sorted"),
    ],
)
def test_the_reader_rules_the_profile_cannot_say(name: str, message: str) -> None:
    assert name in {p.name for p in (RECORDS / "reader-invalid").iterdir()}
    with pytest.raises(LeagueRecordError, match=message):
        read_league_run_record(raw(f"reader-invalid/{name}"), tiny_config())


def test_every_reader_invalid_sample_is_checked_above() -> None:
    assert len(list((RECORDS / "reader-invalid").iterdir())) == 8


@pytest.mark.parametrize("name", sorted(p.name for p in (RECORDS / "invalid").iterdir()))
def test_a_record_the_schema_refuses_is_refused(name: str) -> None:
    with pytest.raises(LeagueRecordError, match="league-run-record"):
        read_league_run_record(raw(f"invalid/{name}"), tiny_config())


def test_a_record_of_another_run_or_config_is_refused() -> None:
    other_run = raw() | {"run_id": "other"}
    with pytest.raises(LeagueRecordError, match="run key"):
        read_league_run_record(other_run, tiny_config())
    other_config = raw() | {"run_config_hash": "0" * 64}
    with pytest.raises(LeagueRecordError, match="config's hash"):
        read_league_run_record(other_config, tiny_config())


def test_the_plan_of_a_generation_must_carry_the_run_keys_seeds(tmp_path: Path) -> None:
    config = tiny_config()
    plan, record_json = league_files(config, 3, "policy:" + "a" * 64)
    record = read_league_run_record(record_json, config)
    check_generation_plan(record, plan, config)

    wrong_seed = copy.deepcopy(plan)
    wrong_seed[1]["seed"] = league_game_seed(config.run_key, 3, 0)
    record_json["plan_hash"] = sha256_hex(jcs(wrong_seed))
    with pytest.raises(LeagueRecordError, match="game 1 does not carry the seed"):
        check_generation_plan(read_league_run_record(record_json, config), wrong_seed, config)

    with pytest.raises(LeagueRecordError, match="plan_hash"):
        check_generation_plan(record, wrong_seed, config)
    with pytest.raises(LeagueRecordError, match="not a list"):
        check_generation_plan(record, {}, config)

    short = plan[:1]
    short_json = record_json | {"plan_hash": sha256_hex(jcs(short))}
    with pytest.raises(LeagueRecordError, match="number of games"):
        check_generation_plan(read_league_run_record(short_json, config), short, config)

    not_entry = ["x", "y"]
    odd = record_json | {"plan_hash": sha256_hex(jcs(not_entry))}
    with pytest.raises(LeagueRecordError, match="game 0 does not carry"):
        check_generation_plan(read_league_run_record(odd, config), not_entry, config)
