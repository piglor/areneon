"""The progress records, read in Python (ADR-017 section 3): the corpus and the reader rules."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop.records import policy_id
from areneon_ml.progress.records import (
    RANDOM_PREVIEW_POLICY,
    ProgressRecordError,
    log_hash,
    read_game_record,
    read_progress_record,
    record_hash,
)

from tests.ml.learner.support import tiny_config
from tests.ml.progress.support import SAMPLES, sample


def corpus(contract: str, kind: str) -> list[Path]:
    return sorted((SAMPLES / contract / kind).glob("*.json"))


EVALUATION = "progress-evaluation-record"
EVALUATION2 = "progress-evaluation-record-2"
GAME = "progress-game-record"


@pytest.mark.parametrize(
    "path",
    corpus(EVALUATION, "valid") + corpus(EVALUATION2, "valid"),
    ids=lambda p: p.parent.parent.name + "/" + p.stem,
)
def test_valid_evaluation_samples_are_read(path: Path) -> None:
    assert read_progress_record(json.loads(path.read_text()))["data_class"] == "synthetic"


@pytest.mark.parametrize("path", corpus(GAME, "valid"), ids=lambda p: p.stem)
def test_valid_game_samples_are_read(path: Path) -> None:
    assert read_game_record(json.loads(path.read_text()))["data_class"] == "synthetic"


@pytest.mark.parametrize(
    "path",
    corpus(EVALUATION, "invalid")
    + corpus(EVALUATION, "reader-invalid")
    + corpus(EVALUATION2, "invalid")
    + corpus(EVALUATION2, "reader-invalid"),
    ids=lambda p: p.parent.parent.name + "/" + p.stem,
)
def test_every_refused_evaluation_sample_is_refused(path: Path) -> None:
    with pytest.raises(ProgressRecordError):
        read_progress_record(json.loads(path.read_text()))


@pytest.mark.parametrize(
    "path", corpus(GAME, "invalid") + corpus(GAME, "reader-invalid"), ids=lambda p: p.stem
)
def test_every_refused_game_sample_is_refused(path: Path) -> None:
    with pytest.raises(ProgressRecordError):
        read_game_record(json.loads(path.read_text()))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("counts-disagree", "wins + ties + losses must equal games"),
        ("forfeits-disagree", "forfeits must equal the forfeited game rows"),
        ("hash-count-wrong", "recorded_games_per_match long"),
        ("hashes-unsorted", "sorted, unique"),
        ("pairs-used-wrong", "pairs_used must equal"),
        ("rows-incomplete", "one game row per game"),
        ("seat-wrong", "seats alternating"),
        ("turn-cap-ties-wrong", "turn_cap_ties"),
        ("unsorted-matches", "sorted by match_id"),
    ],
)
def test_each_evaluation_rule_has_its_own_message(name: str, message: str) -> None:
    raw = json.loads((SAMPLES / EVALUATION / "reader-invalid" / f"{name}.json").read_text())
    with pytest.raises(ProgressRecordError, match=re.escape(message)):
        read_progress_record(raw)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("counts-disagree", "wins + ties + losses must equal games"),
        ("pair-is-game", "pair = game // 2"),
        ("seat-wrong", "seats alternating"),
        ("odd-games", "must be even"),
        ("pairs-used-wrong", "pairs_used must be half"),
        ("rows-incomplete", "one game row per game"),
        ("anchor-games-missing", "present exactly when"),
        ("anchor-plan-without-match", "present exactly when"),
        ("anchor-games-odd", "must be even"),
        ("anchor-not-earlier", "plays an earlier generation"),
        ("anchor-misnamed", "is named anchor-g<generation>-<mode>"),
        ("run-games-missing", "present exactly when"),
        ("run-plan-without-match", "present exactly when"),
        ("run-games-odd", "must be even"),
        ("run-misnamed", "is named run-<mode>"),
        ("run-same-run-id", "is named run-<mode>"),
        ("ablation-vs-anchor", "an ablation match plays a scripted baseline"),
        ("ablation-misnamed", "named preview-<variant>-<baseline>-<mode>"),
        ("ablation-not-earlier", "comes from an earlier generation"),
    ],
)
def test_each_version_2_rule_has_its_own_message(name: str, message: str) -> None:
    raw = json.loads((SAMPLES / EVALUATION2 / "reader-invalid" / f"{name}.json").read_text())
    with pytest.raises(ProgressRecordError, match=re.escape(message)):
        read_progress_record(raw)


def test_a_version_1_record_still_reads_as_it_did_and_a_version_2_is_a_different_contract() -> None:
    v1 = sample(f"{EVALUATION}/valid/baseline-match")
    assert read_progress_record(v1)["schema_version"] == 1
    v2 = sample(f"{EVALUATION2}/valid/baseline-match")
    assert read_progress_record(v2)["schema_version"] == 2
    with pytest.raises(ProgressRecordError):
        read_progress_record({**v2, "schema_version": 3})


def test_a_private_class_is_refused_by_the_schema() -> None:
    for name, contract, reader in (
        ("baseline-match", EVALUATION, read_progress_record),
        ("played", GAME, read_game_record),
    ):
        raw = sample(f"{contract}/valid/{name}")
        raw["data_class"] = "private"
        with pytest.raises(ProgressRecordError):
            reader(raw)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("chosen-not-candidate", "not among its candidates"),
        ("log-hash-wrong", "omniscient_log_sha256"),
        ("seed-wrong", "progress-seed/1"),
        ("pair-seed-wrong", "progress-seed/1"),
        ("pair-wrong", "divided by two"),
        ("team-hash-wrong", "team hash of p1"),
    ],
)
def test_each_game_rule_has_its_own_message(name: str, message: str) -> None:
    raw = json.loads((SAMPLES / GAME / "reader-invalid" / f"{name}.json").read_text())
    with pytest.raises(ProgressRecordError, match=re.escape(message)):
        read_game_record(raw)


def edited(change: Any) -> dict[str, Any]:
    raw = copy.deepcopy(sample(f"{EVALUATION}/valid/baseline-match"))
    change(raw)
    return raw


def test_the_rules_the_samples_do_not_reach() -> None:
    with pytest.raises(ProgressRecordError, match="exceed games_per_match"):
        read_progress_record(edited(lambda r: r["plan"].update(recorded_games_per_match=3)))
    with pytest.raises(ProgressRecordError, match="unknown match"):
        read_progress_record(edited(lambda r: r["games"][0].update(match_id="zzz")))
    with pytest.raises(ProgressRecordError, match="seats alternating"):
        read_progress_record(edited(lambda r: r["games"].reverse()))
    with pytest.raises(ProgressRecordError, match="games must equal plan"):
        read_progress_record(edited(lambda r: r["matches"][0].update(games=3)))
    with pytest.raises(ProgressRecordError, match="turns_total"):
        read_progress_record(edited(lambda r: r["matches"][0].update(turns_total=1)))
    with pytest.raises(ProgressRecordError, match="forfeits must be sorted"):
        read_progress_record(
            edited(
                lambda r: r["matches"][0].update(
                    forfeits=[
                        {"side": "candidate", "cause": "x", "count": 1},
                        {"side": "candidate", "cause": "x", "count": 1},
                    ]
                )
            )
        )


def test_a_winner_by_forfeit_and_a_turn_cap_tie_are_counted() -> None:
    def change(raw: dict[str, Any]) -> None:
        match = raw["matches"][0]
        match.update(wins=1, ties=1, losses=0, forfeits=[], turn_cap_ties=1)
        match["turns_total"] = 120 + 7
        raw["games"][0]["outcome"] = {"kind": "played", "winner": "tie", "turns": 120}
        raw["games"][1]["outcome"] = {
            "kind": "forfeit",
            "loser": "opponent",
            "cause": "crash",
            "request_seq": 2,
            "turns": 7,
        }
        match["forfeits"] = [{"side": "opponent", "cause": "crash", "count": 1}]

    assert read_progress_record(edited(change))["matches"][0]["turn_cap_ties"] == 1


def test_the_record_is_tied_to_its_run_and_its_policy_ids() -> None:
    config = tiny_config()
    raw = sample(f"{EVALUATION}/valid/baseline-match")
    with pytest.raises(ProgressRecordError, match="another run or run config"):
        read_progress_record(raw, config)
    bound = copy.deepcopy(raw)
    bound["run_config_hash"] = config.hash
    with pytest.raises(ProgressRecordError, match="differ from the run config's"):
        read_progress_record(bound, config)  # the corpus sample's caps are not the tiny run's
    bound["plan"]["max_turns"] = config.generation.max_turns
    bound["plan"]["latency_budget_ms"] = {"preview_ms": 30000, "battle_ms": 30000}
    with pytest.raises(ProgressRecordError, match="not the frozen pair's"):
        read_progress_record(bound, config)  # the sample's placeholder policy id
    previous = copy.deepcopy(bound)
    previous["matches"][0]["policy_id"] = record_policy(bound)
    previous["matches"][0]["opponent"] = {
        "kind": "previous",
        "policy_id": "policy:" + "b" * 64,
        "checkpoints": {"preview": "6" * 64, "battle": "7" * 64},
    }
    with pytest.raises(ProgressRecordError, match="previous pair's policy_id"):
        read_progress_record(previous, config)


def test_an_anchor_pairs_policy_id_must_be_its_checkpoints() -> None:
    config = tiny_config()
    raw = copy.deepcopy(sample(f"{EVALUATION2}/valid/anchor-match"))
    raw["run_config_hash"] = config.hash
    raw["plan"]["max_turns"] = config.generation.max_turns
    raw["plan"]["latency_budget_ms"] = {"preview_ms": 30000, "battle_ms": 30000}
    raw["matches"][0]["policy_id"] = record_policy(raw)
    with pytest.raises(ProgressRecordError, match="anchor pair's policy_id"):
        read_progress_record(raw, config)
    opponent = raw["matches"][0]["opponent"]
    opponent["policy_id"] = policy_id(
        opponent["checkpoints"]["preview"], opponent["checkpoints"]["battle"], "sample"
    )
    assert read_progress_record(raw, config)["plan"]["anchor_games_per_match"] == 4


def test_a_cross_run_pairs_policy_id_must_be_its_checkpoints() -> None:
    config = tiny_config()
    raw = copy.deepcopy(sample(f"{EVALUATION2}/valid/run-match"))
    raw["run_id"] = config.run_key
    raw["run_config_hash"] = config.hash
    raw["plan"]["max_turns"] = config.generation.max_turns
    raw["plan"]["latency_budget_ms"] = {"preview_ms": 30000, "battle_ms": 30000}
    raw["matches"][0]["policy_id"] = record_policy(raw)
    with pytest.raises(ProgressRecordError, match="run pair's policy_id"):
        read_progress_record(raw, config)
    opponent = raw["matches"][0]["opponent"]
    opponent["policy_id"] = policy_id(
        opponent["checkpoints"]["preview"], opponent["checkpoints"]["battle"], "sample"
    )
    assert read_progress_record(raw, config)["plan"]["run_games_per_match"] == 4


def test_a_cross_run_match_has_the_run_games_of_the_plan() -> None:
    raw = copy.deepcopy(sample(f"{EVALUATION2}/valid/run-match"))
    raw["plan"]["run_games_per_match"] = 2
    raw["teams"]["pairs_used"] = 2  # the baselines' 4 games are still 2 pairs
    with pytest.raises(ProgressRecordError, match=r"plan\.run_games_per_match"):
        read_progress_record(raw)


def record_policy(raw: dict[str, Any]) -> str:
    pair = raw["checkpoints"]
    return policy_id(pair["preview"], pair["battle"], raw["matches"][0]["mode"])


def test_the_plan_must_name_this_seed_derivation_and_pairing() -> None:
    # The schema fixes both words; the reader repeats it for a record built by hand.
    raw = edited(lambda r: None)
    raw["plan"]["pairing"] = "progress-pairing/1"
    assert read_progress_record(raw)


def test_hashes_are_of_the_jcs_bytes() -> None:
    raw = sample(f"{GAME}/valid/played")
    assert log_hash(raw["omniscient_log"]) == raw["omniscient_log_sha256"]
    assert len(record_hash(raw)) == 64


def test_an_ablation_candidate_policy_id_is_built_from_its_preview() -> None:
    config = tiny_config()
    raw = copy.deepcopy(sample(f"{EVALUATION2}/valid/ablation-match"))
    raw["run_config_hash"] = config.hash
    raw["plan"]["max_turns"] = config.generation.max_turns
    raw["plan"]["latency_budget_ms"] = {"preview_ms": 30000, "battle_ms": 30000}
    battle = raw["checkpoints"]["battle"]
    with pytest.raises(ProgressRecordError, match="not the frozen pair's"):
        read_progress_record(raw, config)  # the sample's placeholder policy ids
    for match in raw["matches"]:
        preview = match["candidate_preview"]
        own = {"learned": raw["checkpoints"]["preview"], "random": RANDOM_PREVIEW_POLICY}.get(
            preview["kind"], preview.get("checkpoint")
        )
        match["policy_id"] = policy_id(own, battle, match["mode"])
    assert len(read_progress_record(raw, config)["matches"]) == 3
    raw["matches"][0]["policy_id"] = policy_id(raw["checkpoints"]["preview"], battle, "argmax")
    with pytest.raises(ProgressRecordError, match="not the frozen pair's"):
        read_progress_record(raw, config)  # generation 0's Preview with the learned one's id


def test_an_ablation_match_is_argmax_and_never_mixed_with_ordinary_matches() -> None:
    raw = copy.deepcopy(sample(f"{EVALUATION2}/valid/ablation-match"))
    raw["matches"][0]["mode"] = "sample"
    with pytest.raises(ProgressRecordError, match="played in argmax"):
        read_progress_record(raw)
    mixed = copy.deepcopy(sample(f"{EVALUATION2}/valid/ablation-match"))
    ordinary = copy.deepcopy(sample(f"{EVALUATION2}/valid/baseline-match"))
    mixed["matches"].insert(0, ordinary["matches"][0])
    mixed["games"] = ordinary["games"] + mixed["games"]
    with pytest.raises(ProgressRecordError, match="ablation matches only"):
        read_progress_record(mixed)
