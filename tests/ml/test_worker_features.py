"""candidate_features on the worker side (ADR-022 §3): the handshake statement, the reader rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.training import StreamChecker, TrainingStreamError
from areneon_ml.worker.features import (
    FOE_SPREAD_TARGET_TYPES,
    coverage_problem,
    damaging_moves,
    rows_problem,
)
from areneon_ml.worker.protocol import QueryError, check
from areneon_ml.worker.serve import EXIT_OK, FeatureSupport, Worker, read_query

from tests.ml.helpers import MAX_LINE, battle_query, game, hello, line, make_worker, sample, session

DAMAGING = damaging_moves(game())


def featured() -> dict[str, Any]:
    return sample("policy-query/valid/battle-features.json")


def stated(query: dict[str, Any]) -> dict[str, str]:
    features = query["candidate_features"]
    keys = ("feature_set", "feature_set_hash")
    return {key: features[key] for key in keys}


def support(query: dict[str, Any]) -> FeatureSupport:
    return FeatureSupport(stated(query), DAMAGING)


def feature_worker(tmp_path: Path, feature_set: dict[str, str] | None) -> Worker:
    plain = make_worker(tmp_path)
    return Worker(
        plain._scorer,
        plain._ready["message"]["engine_data_hash"],
        plain._ready["message"]["checkpoints"],
        MAX_LINE,
        None if feature_set is None else FeatureSupport(feature_set, DAMAGING),
    )


def test_the_ready_message_states_the_feature_set_only_when_given(tmp_path: Path) -> None:
    query = featured()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    code, frames = session(feature_worker(tmp_path / "a", stated(query)), line(hello()))
    assert code == EXIT_OK
    assert frames[0]["message"]["feature_set"] == stated(query)
    check("policy-handshake", frames[0])
    _, plain = session(make_worker(tmp_path / "b"), line(hello()))
    assert "feature_set" not in plain[0]["message"]
    committed = sample("policy-handshake/valid/ready-features.json")
    check("policy-handshake", committed)


def test_the_valid_samples_are_accepted_under_their_set() -> None:
    query = featured()
    assert read_query(query, support(query)) == query
    preview = sample("policy-query/valid/preview-features.json")
    assert read_query(preview, support(preview))["candidate_features"]["rows"] == []


def test_features_go_with_a_stated_set_only() -> None:
    query = featured()
    with pytest.raises(QueryError, match="stated no set"):
        read_query(query)
    with pytest.raises(QueryError, match="has no candidate_features"):
        read_query(battle_query(), support(query))
    other = {**stated(query), "feature_set_hash": "0" * 64}
    with pytest.raises(QueryError, match="another set"):
        read_query(query, FeatureSupport(other, DAMAGING))
    # no feature set anywhere: today's query reads as before
    assert read_query(battle_query())["query_id"] == 1


@pytest.mark.parametrize("name", ["features-unsorted", "features-duplicate-row"])
def test_rows_must_be_sorted_and_unique(name: str) -> None:
    query = sample(f"policy-query/reader-invalid/{name}.json")
    check("policy-query", query)  # schema-valid
    assert rows_problem(query["candidate_features"]) is not None
    with pytest.raises(QueryError, match="sorted and unique"):
        read_query(query, support(query))


def test_a_missing_row_is_an_invalid_query(tmp_path: Path) -> None:
    query = sample("policy-query/reader-invalid/features-missing-row.json")
    check("policy-query", query)
    with pytest.raises(QueryError, match="no candidate_features row"):
        read_query(query, support(query))
    code, frames = session(feature_worker(tmp_path, stated(query)), line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "error"
    assert frames[1]["reply"]["error"] == "invalid_query"
    assert "no candidate_features row" in frames[1]["reply"]["detail"]


def test_a_query_with_all_its_rows_is_answered(tmp_path: Path) -> None:
    query = featured()
    code, frames = session(feature_worker(tmp_path, stated(query)), line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"


def request_query(moves: list[dict[str, str]], choice: str) -> dict[str, Any]:
    request = "|request|" + json.dumps({"active": [{"moves": moves}]})
    return {"observation": {"log": [request]}, "candidates": [{"choice": choice}]}


def test_the_coverage_rule_is_the_typescript_one() -> None:
    moves = [
        {"id": "earthquake", "target": "allAdjacent"},
        {"id": "tackle", "target": "normal"},
        {"id": "protect", "target": "self"},
    ]

    def rows(*keys: tuple[int, int, int, bool]) -> dict[str, Any]:
        return {"rows": [{"slot": s, "move": m, "target": t, "mega": g} for s, m, t, g in keys]}

    def problem(choice: str, feature_rows: dict[str, Any]) -> str | None:
        return coverage_problem(request_query(moves, f"{choice}, pass"), feature_rows, DAMAGING)

    assert problem("move 2 1", rows()) is not None
    assert problem("move 2 1", rows((0, 2, 1, False))) is None
    # a fainted foe's location has no row of its own
    assert problem("move 2 1", rows((0, 2, 2, False))) is None
    assert problem("move 2 1", rows((1, 2, 1, False), (0, 1, 1, False))) is not None
    assert problem("move 2 1 mega", rows((0, 2, 1, False))) is not None
    assert problem("move 2 1 mega", rows((0, 2, 1, True))) is None
    assert problem("move 1", rows()) is not None
    assert problem("move 1", rows((0, 1, 2, False))) is None
    for free in ("move 3", "move 2 -2", "switch 3", "pass"):
        assert problem(free, rows()) is None, free
    off_request = {"observation": {"log": ["|turn|2"]}, "candidates": []}
    assert coverage_problem(off_request, rows(), DAMAGING) is not None
    team_preview = {"observation": {"log": ['|request|{"teamPreview":true}']}, "candidates": []}
    assert coverage_problem(team_preview, rows(), DAMAGING) is None


def test_an_unknown_move_id_counts_as_damaging() -> None:
    assert DAMAGING("nosuchmove") is True
    assert DAMAGING("protect") is False
    assert DAMAGING("earthquake") is True


def test_the_spread_target_types_are_typescript_s() -> None:
    assert sorted(FOE_SPREAD_TARGET_TYPES) == ["allAdjacent", "allAdjacentFoes", "randomNormal"]


def stream_line(example: dict[str, Any]) -> str:
    return json.dumps(example)


def checker_for(example: dict[str, Any]) -> StreamChecker:
    return StreamChecker(
        {
            "schema_version": 1,
            "data_class": "synthetic",
            "episode_hashes": [example["episode_hash"]],
            "evaluation_manifest_hashes": [],
            "examples": 1,
            "stream_sha256": "0" * 64,
        }
    )


def test_the_stream_reader_validates_example_rows() -> None:
    example = sample("training-example/valid/battle-features.json")
    checker = checker_for(example)
    raw, _ = checker.line(1, stream_line(example))
    assert raw["candidate_features"]["rows"]
    bad = sample("training-example/reader-invalid/features-unsorted.json")
    with pytest.raises(TrainingStreamError, match="sorted and unique"):
        checker_for(bad).line(1, stream_line(bad))
    plain = sample("training-example/valid/battle.json")
    assert "candidate_features" not in plain


def test_a_stream_carries_one_feature_set_or_none() -> None:
    with_rows = sample("training-example/valid/battle-features.json")
    without = sample("training-example/valid/battle.json")
    other = copy.deepcopy(with_rows)
    other["candidate_features"]["feature_set_hash"] = "1" * 64
    for pair in ((with_rows, without), (with_rows, other)):
        checker = checker_for(with_rows)
        for number, example in enumerate(pair, start=1):
            checker.line(number, stream_line(example))
        with pytest.raises(TrainingStreamError, match="same candidate feature set"):
            checker.finish()
    same = checker_for(with_rows)
    same.line(1, stream_line(with_rows))
    same.line(2, stream_line(with_rows))
    same.finish()
