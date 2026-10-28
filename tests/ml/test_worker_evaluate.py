"""The ``evaluate`` sampling kind (ADR-026 §5): prior and value as integer ppm, batch 1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.worker.protocol import check
from areneon_ml.worker.sampler import PPM_LIMIT, ScoreError, to_ppm
from areneon_ml.worker.serve import EXIT_OK, Worker
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import (
    ENGINE_DATA_HASH,
    MAX_LINE,
    battle_query,
    hello,
    line,
    make_worker,
    preview_query,
    sample,
    session,
)
from tests.ml.test_worker_features import feature_worker, featured, stated
from tests.ml.test_worker_session import BrokenScorer


def evaluating(query: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**query, "sampling": {"kind": "evaluate"}, **changes}


def evaluations(worker: Worker, *queries: dict[str, Any]) -> list[dict[str, Any]]:
    code, frames = session(worker, line(hello()), *(line(q) for q in queries))
    assert code == EXIT_OK
    replies = frames[1:]
    for query, reply in zip(queries, replies, strict=True):
        check(
            "evaluation-reply" if query["sampling"]["kind"] == "evaluate" else "policy-reply", reply
        )
    return replies


def test_an_evaluate_query_gets_prior_and_value_for_its_candidates(tmp_path: Path) -> None:
    query = evaluating(battle_query())
    [reply] = evaluations(make_worker(tmp_path), query)
    assert reply["query_id"] == query["query_id"]
    evaluation = reply["reply"]["evaluation"]
    assert evaluation["identity"] == query["identity"]
    assert len(evaluation["prior_ppm"]) == len(query["candidates"])
    # a softmax in ppm: each entry rounded alone, so the sum is within half a ppm per entry
    assert abs(sum(evaluation["prior_ppm"]) - 1_000_000) <= len(query["candidates"])
    assert isinstance(evaluation["value_ppm"], int)


def test_the_prior_is_the_probabilities_a_sample_query_draws_from(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    query = evaluating(battle_query())
    [reply] = evaluations(worker, query)
    probabilities, value = worker._scorer.evaluate(encoder_input(query))
    assert probabilities == worker._scorer.probabilities(encoder_input(query))
    assert reply["reply"]["evaluation"]["prior_ppm"] == [to_ppm(p) for p in probabilities]
    assert reply["reply"]["evaluation"]["value_ppm"] == to_ppm(value)


def test_an_evaluation_is_bit_identical_alone_repeated_and_after_others(tmp_path: Path) -> None:
    query = evaluating(battle_query(), query_id=5)
    alone = evaluations(make_worker(tmp_path), query)[0]
    others = [
        evaluating(preview_query(), query_id=1),
        battle_query(query_id=2),
        evaluating(battle_query(), query_id=3),
        query,
        query,
    ]
    after = evaluations(make_worker(tmp_path), *others)
    assert alone == after[-1] == after[-2]
    assert after[-2]["reply"]["kind"] == "evaluation"


def test_the_a1_and_a2_forms_of_a_decision_give_equal_evaluations(tmp_path: Path) -> None:
    a1 = evaluating(battle_query(), query_id=1)
    a2 = evaluating(sample("policy-query/valid/battle-a2.json"), query_id=1)
    first, second = evaluations(make_worker(tmp_path), a1, a2)
    one, two = first["reply"]["evaluation"], second["reply"]["evaluation"]
    assert one["prior_ppm"] == two["prior_ppm"]
    assert one["value_ppm"] == two["value_ppm"]
    assert two["identity"] == a2["identity"]


def test_an_evaluate_query_with_features_uses_them_like_any_query(tmp_path: Path) -> None:
    query = evaluating(featured())
    worker = feature_worker(tmp_path, stated(query))
    [reply] = evaluations(worker, query)
    assert len(reply["reply"]["evaluation"]["prior_ppm"]) == len(query["candidates"])
    # the pair's rows are part of the view: the plain query is refused for want of them
    [refused] = evaluations(worker, evaluating(battle_query()))
    assert refused["reply"]["kind"] == "error"
    assert refused["reply"]["error"] == "invalid_query"


def test_a_refused_evaluate_query_gets_the_error_reply(tmp_path: Path) -> None:
    query = evaluating(battle_query())
    query["candidates"] = [query["candidates"][0], query["candidates"][0]]
    [reply] = evaluations(make_worker(tmp_path), query)
    assert reply["reply"]["kind"] == "error"
    assert reply["reply"]["error"] == "invalid_query"
    unsupported = evaluating(battle_query())
    unsupported["candidates"] = [{"choice": "move 9 9 9", "pending": False}]
    [reply] = evaluations(make_worker(tmp_path), unsupported)
    assert reply["reply"]["error"] == "unsupported_candidate"


def test_an_unusable_prior_crashes_the_worker_as_for_a_sample(tmp_path: Path) -> None:
    worker = Worker(BrokenScorer([1.0]), ENGINE_DATA_HASH, [], MAX_LINE)
    with pytest.raises(ScoreError):
        session(worker, line(hello()), line(evaluating(battle_query())))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0),
        (0.5, 500_000),
        (1.0, 1_000_000),
        (-0.15, -150_000),
        (0.0000005, 1),  # a tie at half a ppm goes up
        (-0.0000005, 0),  # round half up: toward positive infinity
        (0.0000004999, 0),
        (2.5, 2_500_000),
    ],
)
def test_the_rounding_rule_is_floor_of_x_million_plus_half(value: float, expected: int) -> None:
    assert to_ppm(value) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1e300, -1e300, 9.1e9 * 1e6])
def test_a_value_the_contract_cannot_hold_is_refused(bad: float) -> None:
    with pytest.raises(ScoreError):
        to_ppm(bad)


def test_the_limit_is_the_contracts_integer_range() -> None:
    assert PPM_LIMIT == 2**53 - 1
