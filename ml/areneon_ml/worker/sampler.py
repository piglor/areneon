"""Choosing among candidates from their probabilities (ADR-010 §2, Sampling).

The worker is handed ``u``, an integer in ``[0, 2^53 - 1]``, never the seed.
It sums the float32 probabilities, cast to float64, sequentially in canonical
order and takes the first candidate whose cumulative sum exceeds ``u / 2^53``.

A sample query may carry a play temperature (applied by the scorer, to the logits) and a
``threshold`` truncation (ADR-023 §1): every candidate whose probability is below
``min_ppm / 10^6`` gets 0, except the first highest, which is always kept; the rest are
renormalised (float32 values divided by their float64 sum, rounded to float32).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Any

U_SCALE = float(2**53)
FULL_TEMPERATURE_PPM = 1_000_000


class ScoreError(ValueError):
    """A scorer returned probabilities that cannot be sampled from."""


def to_float32(value: float) -> float:
    """``value`` rounded to the nearest float32, as a float64."""
    try:
        packed = struct.pack("<f", value)
    except OverflowError as exc:
        raise ScoreError("a probability does not fit in float32") from exc
    unpacked: float = struct.unpack("<f", packed)[0]
    return unpacked


def check_probabilities(probabilities: Sequence[float], count: int) -> None:
    if len(probabilities) != count:
        raise ScoreError(f"{len(probabilities)} probabilities for {count} candidates")
    if not all(math.isfinite(p) and p >= 0.0 for p in probabilities) or sum(probabilities) <= 0.0:
        raise ScoreError("probabilities must be finite, non-negative and not all zero")


def truncate(probabilities: Sequence[float], min_ppm: int) -> list[float]:
    """The probabilities after a threshold truncation: small ones 0, the top one kept, renormalised.

    The comparison is ``probability * 10^6 < min_ppm`` in float64 (the float32 is exact there).
    """
    top = max(range(len(probabilities)), key=lambda i: (probabilities[i], -i))
    kept = [p if i == top or p * 1_000_000 >= min_ppm else 0.0 for i, p in enumerate(probabilities)]
    total = 0.0
    for p in kept:
        total += p
    return [to_float32(p / total) for p in kept]


def select(probabilities: Sequence[float], sampling: dict[str, Any]) -> int:
    """The chosen candidate's index: ``argmax`` the first highest, ``sample`` as ``u`` says."""
    check_probabilities(probabilities, len(probabilities))
    if sampling["kind"] == "argmax":
        return max(range(len(probabilities)), key=lambda i: (probabilities[i], -i))
    truncation = sampling.get("truncation")
    if truncation is not None:
        probabilities = truncate(probabilities, truncation["min_ppm"])
    threshold = sampling["u"] / U_SCALE
    total = 0.0
    for index, probability in enumerate(probabilities):
        total += probability
        if total > threshold:
            return index
    # The sum fell short of a threshold near 1: the last candidate that has any mass.
    return max(i for i, p in enumerate(probabilities) if p > 0.0)
