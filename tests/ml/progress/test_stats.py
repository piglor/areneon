"""The Wilson interval, derived on display in integers (ADR-017 section 2)."""

from __future__ import annotations

import hashlib
from math import sqrt

import pytest
from areneon_ml.progress.stats import (
    B_BOOT,
    CONDITIONAL_LABEL,
    PAIR_INTERVAL_LABEL,
    PPM,
    PairInterval,
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
    paired_differences,
    wilson,
)


def reference(wins: int, ties: int, games: int) -> tuple[float, float]:
    z = 1.959964
    p = (wins + ties / 2) / games
    centre = p + z * z / (2 * games)
    spread = z * sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    return (centre - spread) / (1 + z * z / games), (centre + spread) / (1 + z * z / games)


@pytest.mark.parametrize(
    ("wins", "ties", "games"),
    [
        (50, 0, 100),
        (85, 0, 100),
        (0, 0, 10),
        (10, 0, 10),
        (3, 1, 8),
        (1, 1, 2),
        (104, 3, 107),
        (0, 5, 5),
    ],
)
def test_it_agrees_with_the_closed_form_to_a_millionth_and_errs_outward(
    wins: int, ties: int, games: int
) -> None:
    got = wilson(wins, ties, games)
    low, high = reference(wins, ties, games)
    assert abs(got.low_ppm / PPM - low) < 2e-6
    assert abs(got.high_ppm / PPM - high) < 2e-6
    assert got.low_ppm <= low * PPM + 1e-6
    assert got.high_ppm >= high * PPM - 1e-6
    assert 0 <= got.low_ppm <= got.score_ppm <= got.high_ppm <= PPM
    assert got.score_ppm == (2 * wins + ties) * PPM // (2 * games)


def test_the_adr_examples() -> None:
    half = wilson(50, 0, 100)
    assert (half.low_ppm // 10_000, half.high_ppm // 10_000) == (40, 59)
    high = wilson(85, 0, 100)
    assert (high.low_ppm // 10_000, high.high_ppm // 10_000) == (76, 90)


@pytest.mark.parametrize(("wins", "ties", "games"), [(0, 0, 0), (-1, 0, 5), (3, 3, 5), (1, -1, 5)])
def test_impossible_counts_are_refused(wins: int, ties: int, games: int) -> None:
    with pytest.raises(ValueError, match="at least one game"):
        wilson(wins, ties, games)


def test_the_label_says_what_the_interval_is_conditional_on() -> None:
    assert CONDITIONAL_LABEL == "conditional on the fixed validation plan"


# --- the pair-level interval (ADR-017 erratum E2; ticket #948) ---------------------------------

PINNED = (552_083, 406_250, 697_917)
KEY = bootstrap_stream_key("run-a", 3, "match:x")


def independent(values: list[int], key: bytes) -> tuple[int, int, int]:
    """The documented rule written out again: SHA-256(key || counter) words, rejection, ranks."""
    n = len(values)
    limit = (1 << 32) // n * n
    words: list[int] = []
    counter = 0
    while len(words) < n * B_BOOT:
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        counter += 1
        words += [int.from_bytes(block[i : i + 4], "big") for i in range(0, 32, 4)]
    picks = [w % n for w in words if w < limit][: n * B_BOOT]
    sums = sorted(sum(values[i] for i in picks[b * n : (b + 1) * n]) for b in range(B_BOOT))
    return sum(values) * PPM // (4 * n), sums[100] * PPM // (4 * n), -(-sums[3899] * PPM // (4 * n))


def test_the_stream_key_is_sha256_of_the_jcs_of_run_generation_label_and_purpose() -> None:
    text = b'{"generation":3,"label":"match:x","purpose":"progress-bootstrap/1","run_key":"run-a"}'
    assert hashlib.sha256(text).digest() == KEY
    assert bootstrap_stream_key("run-a", 4, "match:x") != KEY
    assert bootstrap_stream_key("run-b", 3, "match:x") != KEY
    assert bootstrap_stream_key("run-a", 3, "match:y") != KEY


def test_one_pair_has_a_point_interval_and_equal_pairs_have_no_spread() -> None:
    assert pair_interval([3], KEY) == PairInterval(750_000, 750_000, 750_000, 1)
    flat = pair_interval([2, 2, 2], KEY)
    assert (flat.mean_ppm, flat.low_ppm, flat.high_ppm, flat.pairs) == (
        500_000,
        500_000,
        500_000,
        3,
    )


def test_two_pairs_of_a_sweep_and_a_loss_span_the_possible_means() -> None:
    # Pair scores 1 and 0 (4 and 0 half-games): the resample mean is 0, 1/2 or 1 with
    # probabilities 1/4, 1/2, 1/4, so the 2.5% and 97.5% order statistics are 0 and 1.
    got = pair_interval([4, 0], KEY)
    assert (got.mean_ppm, got.low_ppm, got.high_ppm) == (500_000, 0, PPM)


def test_it_equals_the_documented_rule_written_out_again_and_is_pinned() -> None:
    values = [0, 1, 2, 3, 4] * 4 + [2, 4, 4, 3]
    got = pair_interval(values, KEY)
    assert (got.mean_ppm, got.low_ppm, got.high_ppm) == independent(values, KEY)
    assert got.pairs == len(values)
    assert got.low_ppm <= got.mean_ppm <= got.high_ppm
    # pinned: a change of the stream, the ranks or the rounding changes these
    assert (got.mean_ppm, got.low_ppm, got.high_ppm) == PINNED


def test_it_is_identical_on_a_rerun_and_changes_with_the_run_key() -> None:
    values = [4, 3, 3, 2, 1, 0, 2, 2, 4, 1]
    assert pair_interval(values, KEY) == pair_interval(list(values), KEY)
    spread = [(i * 7 + i // 3) % 5 for i in range(101)]
    keys = {bootstrap_stream_key(f"run-{c}", 3, "match:x") for c in "bcdef"} | {KEY}
    # a different run key reseeds the resampling: the bounds are not all one value
    assert len({pair_interval(spread, k) for k in keys}) > 1


def test_differences_are_per_pair_and_a_negative_mean_rounds_down_with_outward_bounds() -> None:
    d = paired_differences([2, 3, 0], [4, 3, 1])
    assert d == [-2, 0, -1]
    got = pair_interval(d, KEY)
    assert got.mean_ppm == -3 * PPM // 12 == -250_000
    assert got.low_ppm <= got.mean_ppm <= got.high_ppm
    with pytest.raises(ValueError, match="same pairs"):
        paired_differences([1], [1, 2])


def test_it_refuses_no_pairs_and_too_few_resamples() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        pair_interval([], KEY)
    with pytest.raises(ValueError, match="two resamples"):
        pair_interval([1], KEY, b_boot=1)


def _record(outcomes: list[dict[str, str]]) -> dict[str, object]:
    games = [
        {"match_id": "m", "outcome": {"kind": "played", "winner": w["winner"]}}
        if w["kind"] == "played"
        else {"match_id": "m", "outcome": w}
        for w in outcomes
    ]
    return {
        "games": [*games, {"match_id": "other", "outcome": {"kind": "played", "winner": "tie"}}]
    }


def test_both_seats_of_a_pair_count_once_and_a_tie_is_half() -> None:
    record = _record(
        [
            {"kind": "played", "winner": "candidate"},  # pair 0: win + loss = 1/2
            {"kind": "played", "winner": "opponent"},
            {"kind": "played", "winner": "tie"},  # pair 1: tie + tie = 1/2
            {"kind": "played", "winner": "tie"},
            {"kind": "played", "winner": "candidate"},  # pair 2: win + tie = 3/4
            {"kind": "played", "winner": "tie"},
            {"kind": "forfeit", "loser": "candidate"},  # pair 3: forfeit loss + win = 1/2
            {"kind": "forfeit", "loser": "opponent"},
        ]
    )
    scores = match_pair_scores(record, "m")
    assert scores == [2, 2, 3, 2]
    assert len(scores) == 4  # eight games, four pairs: each pair is one cluster
    assert sum(scores) * PPM // (4 * len(scores)) == 562_500


def test_the_label_says_descriptive_and_conditional() -> None:
    assert "descriptive" in PAIR_INTERVAL_LABEL
    assert CONDITIONAL_LABEL in PAIR_INTERVAL_LABEL
