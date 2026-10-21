"""The score and its Wilson interval, derived on display and never stored (ADR-017 section 2).

A record stores integers (wins, ties, losses). The score counts a tie as half a win, so it is
``(2 * wins + ties) / (2 * games)``. The interval is the Wilson 95% interval for a proportion
with that score and ``games`` trials, which is approximate when there are ties (protocol 6.2).
It is computed with integer arithmetic only (a fixed ``z**2`` and an integer square root), so
it is the same on every machine, and returned in parts per million, rounded outward.

The interval is CONDITIONAL ON THE FIXED VALIDATION PLAN: its estimand is the frozen pair's
expected score on these validation teams under this pairing and these seeds, not over the
generator's distribution of teams. That is a labelled deviation from protocol 6.2 (which
allows Wilson only for fresh team pairs from the generator). The label travels with the
numbers: ``CONDITIONAL_LABEL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

PPM = 1_000_000
# z for a two-sided 95% interval is 1.959964: z**2 = 3.841459 (the usual 1.96**2 = 3.8416).
Z2_PPM = 3_841_459
CONDITIONAL_LABEL = "conditional on the fixed validation plan"


@dataclass(frozen=True)
class Interval:
    """A score and its Wilson interval, in parts per million."""

    score_ppm: int
    low_ppm: int
    high_ppm: int
    games: int


def wilson(wins: int, ties: int, games: int) -> Interval:
    """The score ``(wins + ties / 2) / games`` and its Wilson 95% interval, in integers."""
    if games < 1 or wins < 0 or ties < 0 or wins + ties > games:
        raise ValueError("a match needs at least one game, and wins + ties must not exceed it")
    n = games
    a = 2 * wins + ties  # the score is p = a / (2 n)
    # With z2 = Z2_PPM / PPM the bounds are
    #   (p + z2 / (2 n) -+ sqrt(z2 (p (1 - p) / n + z2 / (4 n^2)))) / (1 + z2 / n).
    # In units of 1 / (2 n PPM) the centre is a PPM + Z2_PPM and the root is
    # sqrt(Z2_PPM (a (2 n - a) PPM + Z2_PPM n) / n); the divisor 1 + z2 / n is
    # (n PPM + Z2_PPM) / (n PPM).
    centre = a * PPM + Z2_PPM
    inner = Z2_PPM * (a * (2 * n - a) * PPM + Z2_PPM * n)
    root = isqrt(inner // n) + 1  # an integer root, rounded up: the interval errs outward
    denominator = 2 * (n * PPM + Z2_PPM)
    low = max(0, (PPM * (centre - root)) // denominator)
    high = min(PPM, -((-PPM * (centre + root)) // denominator))
    score = (a * PPM) // (2 * n)
    return Interval(score_ppm=score, low_ppm=low, high_ppm=high, games=n)
