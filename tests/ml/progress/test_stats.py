"""The Wilson interval, derived on display in integers (ADR-017 section 2)."""

from __future__ import annotations

from math import sqrt

import pytest
from areneon_ml.progress.stats import CONDITIONAL_LABEL, PPM, wilson


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
