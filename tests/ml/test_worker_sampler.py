"""The worker's sampler and the encoder's view of a query (ADR-010 §2)."""

from __future__ import annotations

import struct
from typing import Any

import pytest
from areneon_ml.worker.sampler import ScoreError, check_probabilities, select, to_float32
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
