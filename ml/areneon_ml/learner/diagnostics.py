"""Value-head diagnostics of one role's update (ADR-010 erratum A4 item 1, #1001).

Diagnostic only: nothing reads these numbers to stop, schedule or gate. They are computed in
float64 from the frozen-weight values and the returns the learner trains on, so two CPU runs
give the same bytes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

CALIBRATION_BINS = 10
#: Explained variance of a return with zero variance: there is nothing to explain, so the fixed
#: value is 0 (the same as a constant predictor), never a division by zero.
ZERO_VARIANCE_EXPLAINED = 0.0


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of predicted value: how many decisions fell in it, and their mean value and score.

    An empty bin has count 0 and means of 0.0.
    """

    count: int
    mean_value: float
    mean_score: float


def _variance(xs: Sequence[float]) -> float:
    mean = math.fsum(xs) / len(xs)
    return math.fsum((x - mean) ** 2 for x in xs) / len(xs)


def explained_variance(returns: Sequence[float], values: Sequence[float]) -> float:
    """``1 - Var(return - V) / Var(return)`` (population variances)."""
    if not returns or len(returns) != len(values):
        raise ValueError("returns and values must be non-empty and of one length")
    total = _variance(returns)
    if total == 0.0:
        return ZERO_VARIANCE_EXPLAINED
    return 1.0 - _variance([r - v for r, v in zip(returns, values, strict=True)]) / total


def bin_of(value: float) -> int:
    """Bin ``floor(value * 10)`` of width 0.1, clamped to 0 and 9 at the ends."""
    return min(CALIBRATION_BINS - 1, max(0, math.floor(value * CALIBRATION_BINS)))


def calibration(values: Sequence[float], scores: Sequence[float]) -> tuple[CalibrationBin, ...]:
    """Ten fixed bins of predicted value against the final game score of each decision."""
    if len(values) != len(scores):
        raise ValueError("values and scores must be of one length")
    members: list[list[int]] = [[] for _ in range(CALIBRATION_BINS)]
    for i, value in enumerate(values):
        members[bin_of(value)].append(i)
    return tuple(
        CalibrationBin(
            len(m),
            math.fsum(values[i] for i in m) / len(m) if m else 0.0,
            math.fsum(scores[i] for i in m) / len(m) if m else 0.0,
        )
        for m in members
    )
