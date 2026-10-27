"""The worker's sampler and the encoder's view of a query (ADR-010 §2)."""

from __future__ import annotations

import struct
from typing import Any

import pytest
from areneon_ml.model.scorer import FULL_TEMPERATURE_PPM as scorer_full_temperature
from areneon_ml.worker.sampler import (
    FULL_TEMPERATURE_PPM,
    U_SCALE,
    ScoreError,
    check_probabilities,
    select,
    to_float32,
    truncate,
)
from areneon_ml.worker.view import encoder_input, view_digest

from tests.ml.helpers import battle_query, preview_query, sample

TOP = 2**53 - 1
LAST = 2
TEN = 10
FOUR = 4
TENTH = 0.1


def sample_of(u: int) -> dict[str, Any]:
    return {"kind": "sample", "u": u}


def test_argmax_takes_the_first_highest() -> None:
    assert select([0.2, 0.5, 0.5, 0.1], {"kind": "argmax"}) == 1


def test_sample_takes_the_first_candidate_whose_cumulative_sum_exceeds_u() -> None:
    probabilities = [0.25, 0.25, 0.5]
    assert select(probabilities, sample_of(0)) == 0
    assert select(probabilities, sample_of(2**51 - 1)) == 0
    # u / 2^53 = 0.25: not exceeded by the sum 0.25, so the next candidate.
    assert select(probabilities, sample_of(2**51)) == 1
    assert select(probabilities, sample_of(2**52)) == LAST
    assert select(probabilities, sample_of(TOP)) == LAST


def test_a_zero_probability_candidate_is_never_drawn() -> None:
    assert select([0.0, 1.0], sample_of(0)) == 1
    assert select([0.5, 0.5, 0.0], sample_of(TOP)) == 1


def test_a_sum_short_of_the_threshold_falls_back_to_the_last_candidate_with_mass() -> None:
    assert select([0.25, 0.25, 0.0], sample_of(TOP)) == 1


def test_probabilities_are_summed_as_float64_in_order() -> None:
    # float32 values summed in float64 reach exactly 1.0 only if added in order.
    probabilities = [to_float32(TENTH)] * TEN
    total = 0.0
    for p in probabilities:
        total += p
    assert total != 1.0
    assert select(probabilities, sample_of(TOP)) == TEN - 1


def test_to_float32_rounds_and_refuses_overflow() -> None:
    assert to_float32(TENTH) == struct.unpack("<f", struct.pack("<f", TENTH))[0] != TENTH
    with pytest.raises(ScoreError):
        to_float32(1e300)


@pytest.mark.parametrize(
    ("probabilities", "count"),
    [
        ([0.5], 2),
        ([0.5, float("nan")], 2),
        ([0.5, -0.1], 2),
        ([0.0, 0.0], 2),
        ([float("inf"), 1.0], 2),
    ],
)
def test_unusable_probabilities_are_refused(probabilities: list[float], count: int) -> None:
    with pytest.raises(ScoreError):
        check_probabilities(probabilities, count)


def test_the_a1_and_a2_forms_of_a_decision_have_one_encoder_view() -> None:
    a1, a2 = battle_query(), sample("policy-query/valid/battle-a2.json")
    # The two differ where the encoder must not look.
    assert a1["observation"]["identity"] != a2["observation"]["identity"]
    assert "decision_context" not in a1["observation"]
    assert "decision_context" in a2["observation"]
    assert a1["observation"]["log"] != a2["observation"]["log"]
    assert (a1["query_id"], a1["sampling"]) != (a2["query_id"], a2["sampling"])
    assert encoder_input(a1) == encoder_input(a2)
    assert view_digest(a1) == view_digest(a2)


def test_the_view_changes_with_what_the_encoder_reads() -> None:
    base = battle_query()
    other_log = battle_query(
        observation={**base["observation"], "log": base["observation"]["log"][:-1]}
    )
    other_candidates = battle_query(candidates=base["candidates"][:-1])
    other_seq = battle_query(
        observation={
            **base["observation"],
            "identity": {**base["observation"]["identity"], "request_seq": 99},
        }
    )
    assert len({view_digest(q) for q in (base, other_log, other_candidates, other_seq)}) == FOUR


def test_the_view_holds_no_sampling_policy_or_query_id() -> None:
    view = encoder_input(battle_query())
    assert sorted(view) == ["candidates", "observation"]
    text = repr(view)
    assert "sampling" not in text
    assert "policy:" not in text


def test_the_preview_view_is_built_the_same_way() -> None:
    view = encoder_input(preview_query())
    assert view["observation"]["kind"] == "preview"


def threshold(min_ppm: int, u: int) -> dict[str, Any]:
    return {"kind": "sample", "u": u, "truncation": {"kind": "threshold", "min_ppm": min_ppm}}


def test_a_threshold_zeroes_the_small_candidates_and_renormalises_the_rest() -> None:
    # 0.05 < 0.1 is dropped; 0.5, 0.3 and 0.15 keep their order and are divided by 0.95.
    out = truncate([0.5, 0.3, 0.15, 0.05], 100_000)
    assert out[3] == 0.0
    assert out == [to_float32(p / 0.95) for p in (0.5, 0.3, 0.15)] + [0.0]
    assert abs(sum(out) - 1.0) < 1e-6
    # A candidate exactly at the threshold is kept (the rule is "below min_ppm").
    assert truncate([0.75, 0.25], 250_000) == [0.75, 0.25]


def test_a_threshold_that_would_drop_every_candidate_keeps_the_first_highest() -> None:
    uniform = [to_float32(1 / 40)] * 40
    assert truncate(uniform, 999_999) == [1.0] + [0.0] * 39
    assert select(uniform, threshold(999_999, TOP)) == 0
    # The first highest, not the last, when the top is tied.
    assert truncate([0.1, 0.45, 0.45], 999_999) == [0.0, 1.0, 0.0]
    # A top below the threshold is kept while a candidate above it cannot exist.
    assert truncate([0.2, 0.3, 0.3, 0.2], 400_000) == [0.0, 1.0, 0.0, 0.0]


def test_a_truncated_draw_never_lands_on_a_dropped_candidate() -> None:
    probabilities = [0.6, 0.3, 0.1]
    seen = {select(probabilities, threshold(200_000, u)) for u in (0, 2**52, 2**53 - 1)}
    assert seen == {0, 1}
    # u / 2^53 at the renormalised boundary 2/3: the sum 2/3 (float32) does not exceed it.
    boundary = int(to_float32(0.6 / 0.9) * U_SCALE)
    assert select(probabilities, threshold(200_000, boundary - 1)) == 0
    assert select(probabilities, threshold(200_000, boundary + 2**30)) == 1


def test_argmax_ignores_a_truncation_and_the_default_draw_is_unchanged() -> None:
    assert select([0.2, 0.5, 0.3], {"kind": "argmax"}) == 1
    assert select([0.25, 0.25, 0.5], sample_of(2**52)) == LAST
    assert scorer_full_temperature == FULL_TEMPERATURE_PPM
