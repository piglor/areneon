"""The ``value`` sampling kind (ADR-026 erratum E2 decision 7): value-only leaf queries."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from areneon_ml.model.encoder import Encoder, EncodingError, Layout
from areneon_ml.model.network import ModelConfig
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    OPPONENT_SHEET,
    read_options,
)
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker.__main__ import build
from areneon_ml.worker.protocol import QueryError, check
from areneon_ml.worker.serve import EXIT_OK, FeatureSupport, Worker, read_query
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import (
    ENGINE_DATA,
    MAX_LINE,
    battle_query,
    game,
    hello,
    line,
    make_run_dir,
    make_worker,
    sample,
    session,
)
from tests.ml.model.support import TINY_ENCODER, TINY_MODEL
from tests.ml.test_worker_features import DAMAGING, feature_model, featured, stated

HASH = "a" * 64


def valued(query: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The value-only form of a query: no candidates, no rows, no sheet."""
    plain = {
        k: v
        for k, v in query.items()
        if k not in ("candidates", "candidate_features", "opponent_sheet")
    }
    return {**plain, "candidates": [], "sampling": {"kind": "value"}, **changes}


def evaluating(query: dict[str, Any]) -> dict[str, Any]:
    return {**query, "sampling": {"kind": "evaluate"}}


def encoder_with(**options: Any) -> ModelConfig:
    encoder = replace(TINY_ENCODER, options=read_options("encoder", options))
    return replace(TINY_MODEL, encoder=encoder)


def scorer_of(worker: Worker) -> NetworkScorer:
    return cast(NetworkScorer, worker._scorer)


def serving(plain: Worker, features: dict[str, str] | None = None) -> Worker:
    """``plain`` with the value scorer on, as ``__main__`` builds it for an eligible pair."""
    scorer = scorer_of(plain)
    message = plain._ready["message"]
    return Worker(
        scorer,
        message["engine_data_hash"],
        message["checkpoints"],
        MAX_LINE,
        None if features is None else FeatureSupport(features, DAMAGING),
        value_scorer=scorer if scorer.value_query else None,
    )


def answers(worker: Worker, *queries: dict[str, Any]) -> list[dict[str, Any]]:
    code, frames = session(worker, line(hello()), *(line(q) for q in queries))
    assert code == EXIT_OK
    for reply in frames[1:]:
        if reply["reply"]["kind"] != "recommendation":
            check("evaluation-reply", reply)
    return frames[1:]


# -- eligibility comes from the encoder's layout ----------------------------------------------


@pytest.mark.parametrize(
    ("options", "reads"),
    [
        ({}, False),
        ({CANDIDATE_MATCHUP: True}, False),
        ({CANDIDATE_TARGET: True}, False),
        ({CANDIDATE_FEATURES: {"feature_set": "probe-damage/1", "feature_set_hash": HASH}}, False),
        (
            {
                CANDIDATE_FEATURES: {
                    "feature_set": "probe-damage-belief/1",
                    "feature_set_hash": HASH,
                }
            },
            False,
        ),
        ({CANDIDATE_FEATURES: {"feature_set": "probe-damage/2", "feature_set_hash": HASH}}, True),
        ({CANDIDATE_FEATURES: {"feature_set": "probe-damage/3", "feature_set_hash": HASH}}, True),
        ({OPPONENT_SHEET: {"feature_set": "opponent-sheet/1", "feature_set_hash": HASH}}, True),
    ],
)
def test_the_layout_says_whether_the_value_reads_candidate_rows(
    options: dict[str, Any], reads: bool
) -> None:
    config = replace(TINY_ENCODER, options=read_options("encoder", options))
    layout = Layout.of(config, game())
    assert layout.value_reads_candidates is reads
    assert "value_reads_candidates" not in layout.as_dict()  # encoder versions are untouched


@pytest.mark.parametrize(
    "options",
    [
        {CANDIDATE_FEATURES: {"feature_set": "probe-damage/2", "feature_set_hash": HASH}},
        {OPPONENT_SHEET: {"feature_set": "opponent-sheet/1", "feature_set_hash": HASH}},
    ],
)
def test_the_encoder_refuses_a_value_only_query_when_the_value_reads_rows(
    options: dict[str, Any],
) -> None:
    encoder = Encoder(game(), encoder_with(**options).encoder)
    with pytest.raises(EncodingError, match="value-only"):
        encoder.encode(encoder_input(valued(battle_query())), value_only=True)
    with pytest.raises(EncodingError, match="value-only"):
        encoder.session().encode(encoder_input(valued(battle_query())), value_only=True)


def test_the_encoder_refuses_candidates_or_rows_in_a_value_only_query() -> None:
    encoder = Encoder(game(), TINY_ENCODER)
    with pytest.raises(EncodingError, match="no candidates"):
        encoder.encode(encoder_input(battle_query()), value_only=True)
    with pytest.raises(EncodingError, match="no candidates"):
        encoder.encode({**encoder_input(valued(battle_query())), "candidate_features": {}}, True)


@pytest.mark.parametrize("matchup_and_target", [False, True])
def test_a_value_only_encoding_has_no_candidate_parts(matchup_and_target: bool) -> None:
    options = {CANDIDATE_MATCHUP: True, CANDIDATE_TARGET: True} if matchup_and_target else {}
    config = replace(TINY_ENCODER, options=read_options("encoder", options))
    encoder = Encoder(game(), config)
    full = encoder.encode(encoder_input(battle_query()))
    bare = encoder.encode(encoder_input(valued(battle_query())), value_only=True)
    assert bare.cand_cat.shape[0] == bare.cand_tok.shape[0] == bare.pending.shape[0] == 0
    assert bare.cand_cat.shape[1:] == full.cand_cat.shape[1:]
    assert bare.cand_tok.shape[1:] == full.cand_tok.shape[1:]
    # the state is the same tensors; the session gives the same result as encode
    for name in ("mon_cat", "mon_num", "field_cat", "field_num"):
        assert getattr(bare, name).equal(getattr(full, name))
    session_bare = encoder.session().encode(encoder_input(valued(battle_query())), value_only=True)
    assert session_bare.mon_num.equal(bare.mon_num)


# -- the value is bit-identical with and without the rows -------------------------------------


@pytest.mark.parametrize("with_options", [False, True])
def test_the_value_is_bit_identical_with_and_without_the_rows_of_a_first_set_pair(
    tmp_path: Path, with_options: bool
) -> None:
    query = featured()
    model = feature_model(stated(query))
    if with_options:  # candidate_matchup and candidate_target add candidate columns, not rows
        options = read_options(
            "encoder",
            {
                CANDIDATE_FEATURES: stated(query),
                CANDIDATE_MATCHUP: True,
                CANDIDATE_TARGET: True,
            },
        )
        model = replace(model, encoder=replace(TINY_ENCODER, options=options))
    worker = make_worker(tmp_path, model=model)
    scorer = scorer_of(worker)
    assert scorer.value_query is True
    full = scorer.evaluate(encoder_input(evaluating(query)))[1]
    alone = scorer.value(encoder_input(valued(query)))
    assert alone == full  # bit for bit, not within a tolerance
    # ... and after other queries, and for the plain query without rows under a plain pair
    scorer.probabilities(encoder_input(query))
    assert scorer.value(encoder_input(valued(query))) == full


def test_the_value_of_a_plain_pair_equals_the_evaluation_value(tmp_path: Path) -> None:
    worker = make_worker(tmp_path)
    scorer = scorer_of(worker)
    assert (
        scorer.value(encoder_input(valued(battle_query())))
        == scorer.evaluate(encoder_input(evaluating(battle_query())))[1]
    )


# -- the worker --------------------------------------------------------------------------------


def test_the_worker_answers_a_value_query_with_the_evaluation_value(tmp_path: Path) -> None:
    worker = serving(make_worker(tmp_path))
    assert worker._ready["message"]["value_query"] is True
    query = valued(battle_query(), query_id=4)
    value, evaluation = answers(worker, query, evaluating(battle_query()))
    assert value["query_id"] == 4
    assert value["reply"]["kind"] == "value"
    assert value["reply"]["value"]["identity"] == query["identity"]
    assert value["reply"]["value"]["value_ppm"] == evaluation["reply"]["evaluation"]["value_ppm"]
    # repeated and after others: the same bytes
    again = answers(worker, evaluating(battle_query()), query, query)
    assert again[1] == again[2] == value


def test_the_a1_and_a2_forms_of_a_decision_give_equal_values(tmp_path: Path) -> None:
    worker = serving(make_worker(tmp_path))
    one = valued(battle_query(), query_id=1)
    two = valued(sample("policy-query/valid/battle-a2.json"), query_id=1)
    first, second = answers(worker, one, two)
    assert first["reply"]["value"]["value_ppm"] == second["reply"]["value"]["value_ppm"]


def test_a_pair_with_rows_in_the_trunk_does_not_state_value_query(tmp_path: Path) -> None:
    plain = make_worker(tmp_path)
    assert "value_query" not in plain._ready["message"]
    [refused] = answers(plain, valued(battle_query()))
    assert refused["reply"]["error"] == "invalid_query"
    for options in (
        {CANDIDATE_FEATURES: {"feature_set": "probe-damage/2", "feature_set_hash": HASH}},
        {CANDIDATE_FEATURES: {"feature_set": "probe-damage/3", "feature_set_hash": HASH}},
        {OPPONENT_SHEET: {"feature_set": "opponent-sheet/1", "feature_set_hash": HASH}},
    ):
        (tmp_path / "x").mkdir(exist_ok=True)
        worker = make_worker(tmp_path / "x", model=encoder_with(**options))
        assert scorer_of(worker).value_query is False
        assert "value_query" not in serving(worker)._ready["message"]


def test_the_real_worker_states_value_query_from_its_checkpoints(tmp_path: Path) -> None:
    query = featured()
    for name in ("on", "off"):
        (tmp_path / name).mkdir()
    run_dir, preview, battle_id = make_run_dir(tmp_path / "on", feature_model(stated(query)))
    argv = [
        "--run-dir", str(run_dir), "--preview", preview, "--battle", battle_id,
        "--engine-data", str(ENGINE_DATA), "--max-line-bytes", str(MAX_LINE),
    ]  # fmt: skip
    worker = build(argv)
    assert worker._ready["message"]["value_query"] is True
    code, frames = session(worker, line(hello()), line(valued(query)), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "value"
    # the root query keeps its rows and its candidates
    assert frames[2]["reply"]["kind"] == "recommendation"
    # a value query that carries the rows is refused: it is value-only
    [refused] = answers(
        worker, {**valued(query), "candidate_features": query["candidate_features"]}
    )
    assert refused["reply"]["error"] == "invalid_query"
    other = sample("policy-query/valid/battle-features-2.json")
    run_dir, preview, battle_id = make_run_dir(tmp_path / "off", feature_model(stated(other)))
    argv[1], argv[3], argv[5] = str(run_dir), preview, battle_id
    assert "value_query" not in build(argv)._ready["message"]


def test_a_failing_state_gets_the_error_reply(tmp_path: Path) -> None:
    worker = serving(make_worker(tmp_path))
    broken = valued(battle_query())
    broken["observation"] = copy.deepcopy(broken["observation"])
    broken["observation"]["log"] = [*broken["observation"]["log"], "|nonsense|"]
    [reply] = answers(worker, broken)
    assert reply["reply"]["error"] == "invalid_query"


# -- reader rules ------------------------------------------------------------------------------


def test_the_reader_rules_of_a_value_query() -> None:
    query = valued(battle_query())
    assert read_query(query, value_query=True) is query
    with pytest.raises(QueryError, match="no value_query"):
        read_query(query)
    with pytest.raises(QueryError, match="no candidates"):
        read_query({**query, "candidates": battle_query()["candidates"]}, value_query=True)
    rows = featured()["candidate_features"]
    with pytest.raises(QueryError, match="no candidate_features"):
        read_query({**query, "candidate_features": rows}, value_query=True)
    sheet = sample("policy-query/valid/battle-opponent-sheet.json")["opponent_sheet"]
    with pytest.raises(QueryError, match="no candidate_features"):
        read_query({**query, "opponent_sheet": sheet}, value_query=True)
    other = copy.deepcopy(query)
    other["identity"] = {**other["identity"], "request_seq": other["identity"]["request_seq"] + 1}
    with pytest.raises(QueryError, match="identity differs"):
        read_query(other, value_query=True)
    # every other kind needs a candidate
    with pytest.raises(QueryError, match="only a value query"):
        read_query({**battle_query(), "candidates": []})
