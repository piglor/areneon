"""The LeagueRunRecord reader and the league seed check (ADR-009 §3)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop.league import (
    LeagueRecordError,
    calibration_game_seed,
    check_generation_plan,
    league_game_seed,
    read_any_league_run_record,
    read_league_run_record,
    read_league_run_record_v2,
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


@pytest.mark.parametrize(
    ("pair", "seed"),
    [
        (0, "sodium,9c05e4913a482ba96e65556eb8e8fbd6"),
        (5, "sodium,f8600e1f06967b6cbbaaf2f75c5d9471"),
    ],
)
def test_the_calibration_seed_is_league_seed_at_generation_zero(pair: int, seed: str) -> None:
    # The same vectors as the TS calibrationGameStreams test (generation 0 of league-seed/1).
    assert calibration_game_seed("tiny-run", pair) == seed
    with pytest.raises(ValueError, match="counts from"):
        calibration_game_seed("tiny-run", -1)


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


def damage_config() -> Any:
    return read_run_config(
        json.loads((SAMPLES / "training-run-config/valid/damage-aware.json").read_text())
    )


def damage_record() -> dict[str, Any]:
    """The damage-aware sample, tied to the damage-aware config sample (two baseline games)."""
    record = raw("valid/damage-aware.json")
    record["run_config_hash"] = damage_config().hash
    record["baselines"][0]["games"] = 2
    record["games"] = 8
    record["played"] = 7
    record["episodes"] = 14
    record["opponents"][0]["games"] = 2
    return record


def test_a_damage_aware_baseline_entry_is_read_and_others_have_none() -> None:
    record = read_league_run_record(damage_record(), damage_config())
    (entry,) = record.raw["baselines"]
    assert (entry["policy_id"], entry["games"], entry["counts"]["queries"]) == (
        "damage-aware@v1",
        2,
        410,
    )
    assert "baselines" not in raw()


def test_baselines_must_match_the_configs_damage_aware_entries() -> None:
    missing = damage_record()
    del missing["baselines"]
    with pytest.raises(LeagueRecordError, match="exactly the config's damage-aware"):
        read_league_run_record(missing, damage_config())
    # The reverse: a record that carries baselines for a config that lists none.
    with pytest.raises(LeagueRecordError, match="exactly the config's damage-aware"):
        read_league_run_record(raw("valid/damage-aware.json"), tiny_config())


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
        ("baselines-unsorted.json", "baselines must be sorted"),
        ("baseline-duplicate.json", "baselines must be sorted"),
        ("baseline-games-disagree.json", "baseline's games must equal"),
        ("baseline-not-an-opponent.json", "baseline's games must equal"),
        ("baseline-probe-runs-exceed-queries.json", "cannot exceed its queries"),
        ("baseline-unsupported-exceed-queries.json", "cannot exceed its queries"),
        ("baseline-refusals-exceed-queries.json", "cannot exceed its queries"),
        ("baseline-decisions-exceed.json", "cannot exceed its decisions"),
    ],
)
def test_the_reader_rules_the_profile_cannot_say(name: str, message: str) -> None:
    assert name in {p.name for p in (RECORDS / "reader-invalid").iterdir()}
    with pytest.raises(LeagueRecordError, match=message):
        read_league_run_record(raw(f"reader-invalid/{name}"), tiny_config())


def test_every_reader_invalid_sample_is_checked_above() -> None:
    # 16 here, and the three mixed-format samples in tests/ml/loop/test_sheet_regimes.py
    assert len(list((RECORDS / "reader-invalid").iterdir())) == 19


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


# ADR-021 section 8: LeagueRunRecord version 2, one entry per member and exploiter seat.
RECORDS_2 = SAMPLES / "league-run-record-2"


def raw2(name: str = "valid/population.json") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((RECORDS_2 / name).read_text())
    return loaded


def population_config(members: int = 2, exploiters: int = 1) -> Any:
    """The population sample config, edited to the record samples' members and exploiters."""
    sample = json.loads((SAMPLES / "training-run-config/valid/population.json").read_text())
    sample["population"]["members"] = members
    sample["population"]["exploiters"] = exploiters
    return read_run_config(sample)


def single_config() -> Any:
    return read_run_config(
        json.loads((SAMPLES / "training-run-config/valid/population-single.json").read_text())
    )


def test_the_valid_version_2_samples_are_read() -> None:
    record = read_league_run_record_v2(raw2(), population_config())
    assert (record.generation, record.games, record.played, record.episodes) == (3, 10, 8, 16)
    assert [m.member for m in record.members] == ["m0", "m1", "x0"]
    assert record.members[0].opponents[1] == ("policy:" + "a" * 64, 2)
    assert [m.episodes for m in record.members] == [7, 7, 1]
    assert record.copy_decision_hash == "e" * 64
    assert record.forfeit_causes == (("crash", 1), ("latency_overrun", 1))
    single = read_league_run_record_v2(raw2("valid/single-member.json"), single_config())
    assert single.copy_decision_hash is None
    assert single.forfeit_causes == ()


def test_the_dispatcher_reads_both_versions_and_refuses_another() -> None:
    assert read_any_league_run_record(raw(), tiny_config()).games == 6
    assert read_any_league_run_record(raw2(), population_config()).games == 10
    for bad in (None, 3, {}, {"schema_version": 3}):
        with pytest.raises(LeagueRecordError, match="schema_version 1 or 2"):
            read_any_league_run_record(bad, tiny_config())
    with pytest.raises(LeagueRecordError, match="league-run-record"):
        read_league_run_record(raw2(), tiny_config())
    with pytest.raises(LeagueRecordError, match="league-run-record-2"):
        read_league_run_record_v2(raw(), population_config())


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("duplicate-member", "members must be sorted"),
        ("member-baseline-decisions-exceed", "cannot exceed its decisions"),
        ("member-baseline-games-disagree", "baseline's games must equal"),
        ("member-baseline-not-an-opponent", "baseline's games must equal"),
        ("member-baseline-probe-runs-exceed-queries", "cannot exceed its queries"),
        ("member-baselines-unsorted", "baselines must be sorted"),
        ("member-causes-do-not-sum", "must sum to forfeits"),
        ("member-causes-missing", "must be present with forfeits"),
        ("member-causes-unsorted", "forfeit_causes must be sorted by cause"),
        ("member-causes-without-forfeits", "must be absent without forfeits"),
        ("member-counts-disagree", r"played \+ forfeits must equal games"),
        ("member-duplicate-opponent", "opponents must be sorted"),
        ("member-episodes-disagree", "played self-play games"),
        ("member-episodes-ignore-self-play", "played self-play games"),
        ("member-opponents-games-do-not-sum", "games must sum to its games"),
        ("member-opponents-played-do-not-sum", "played counts must sum"),
        ("member-opponents-unsorted", "opponents must be sorted"),
        ("members-unsorted", "members must be sorted"),
        ("members-unsorted-numeric", "members must be sorted"),
        ("opponent-member-unknown", "names a member the record does not hold"),
        ("opponent-played-exceeds-games", "played cannot exceed its games"),
        ("opponent-policy-is-not-the-members", "pair_policy_id"),
        ("self-entry-wrong-policy", "pair_policy_id"),
        ("symmetry-games-differ", "about each other must agree"),
        ("symmetry-missing", "about each other must agree"),
        ("symmetry-played-differ", "about each other must agree"),
        ("total-cause-above-members-sum", "between the largest member count"),
        ("total-cause-not-a-members", "between the largest member count"),
        ("total-causes-do-not-sum", "the record's forfeit causes' counts must sum"),
        ("total-causes-missing", "the record's forfeit_causes must be present"),
        ("total-causes-without-forfeits", "the record's forfeit_causes must be absent"),
        ("total-counts-disagree", r"the record's played \+ forfeits"),
        ("total-episodes-differs", "the record's episodes"),
        ("total-games-differs", "count distinct games"),
        ("total-games-is-the-members-sum", "count distinct games"),
        ("total-played-differs", "count distinct games"),
    ],
)
def test_the_version_2_reader_rules_the_profile_cannot_say(name: str, message: str) -> None:
    with pytest.raises(LeagueRecordError, match=message):
        read_league_run_record_v2(raw2(f"reader-invalid/{name}.json"), population_config())


def test_every_version_2_reader_invalid_sample_is_checked_above() -> None:
    assert len(list((RECORDS_2 / "reader-invalid").iterdir())) == 36


@pytest.mark.parametrize("name", sorted(p.name for p in (RECORDS_2 / "invalid").iterdir()))
def test_a_version_2_record_the_schema_refuses_is_refused(name: str) -> None:
    with pytest.raises(LeagueRecordError, match="league-run-record-2"):
        read_league_run_record_v2(raw2(f"invalid/{name}"), population_config())


def test_a_version_2_record_belongs_to_its_config() -> None:
    with pytest.raises(LeagueRecordError, match="run key"):
        read_league_run_record_v2(raw2() | {"run_id": "other"}, population_config())
    with pytest.raises(LeagueRecordError, match="config's hash"):
        read_league_run_record_v2(raw2() | {"run_config_hash": "0" * 64}, population_config())
    with pytest.raises(LeagueRecordError, match="has a population"):
        read_league_run_record_v2(
            raw2(), dataclasses.replace(tiny_config(), hash=population_config().hash)
        )
    fitted = population_config().hash
    more = dataclasses.replace(population_config(members=3), hash=fitted)
    with pytest.raises(LeagueRecordError, match=r"population\.members"):
        read_league_run_record_v2(raw2(), more)
    none = dataclasses.replace(population_config(exploiters=0), hash=fitted)
    with pytest.raises(LeagueRecordError, match=r"population\.exploiters"):
        read_league_run_record_v2(raw2(), none)
    gap = json.loads(json.dumps(raw2()).replace('"m1"', '"m3"'))
    with pytest.raises(LeagueRecordError, match="without gaps"):
        read_league_run_record_v2(gap, population_config())
