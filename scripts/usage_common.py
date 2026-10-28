"""Shared pieces of the offline usage-statistics derivations (ADR-025 section 2, ADR-020 section 4).

Standard library only, no network, no binary floats: the source's numbers are parsed as decimals and
handled as exact fractions. The pieces here are the ones every derivation from a Smogon monthly file
shares: the strict decimal reader, the name-to-id table, the truncation (top k, ties by id), the
largest-remainder scaling to an exact integer total, and the decimal-to-integer unit rule.

The unit rule (stated once, tested): a list's weights are rating weights in the source's own unit.
The list's integer denominator is its summed weight times ``UNIT`` (10^6), rounded half to even, and
each weight is then its exact share of that denominator, found by largest remainder, so the weights
of a list add up to the denominator exactly and no binary float is ever read.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

UNIT = 10**6


class DerivationError(ValueError):
    """An input the derivation refuses; the message names a rule, never a raw value."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DerivationError("the source has a duplicate object key")
        out[key] = value
    return out


def _no_constant(_: str) -> Any:
    raise DerivationError("the source holds NaN or Infinity")


def parse_source(text: str) -> Any:
    """Parse a source file: floats become exact ``Decimal``s, duplicate keys and NaN are refused."""
    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=_no_constant,
        object_pairs_hook=_no_duplicates,
    )


def fraction(value: Any) -> Fraction:
    """An exact fraction of a parsed source number (an ``int`` or a ``Decimal``, never a float)."""
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise DerivationError("a source weight is not a number")
    if not Fraction(value) >= 0:
        raise DerivationError("a source weight is negative")
    return Fraction(value)


def to_units(total: Fraction) -> int:
    """The integer denominator of a list: ``total * UNIT`` rounded half to even."""
    return round(total * UNIT)


class NameTable:
    """Engine ids of one vocabulary, found by exact id or exact name; anything else is unmapped."""

    def __init__(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self.ids = {e["id"] for e in entries}
        self.by_name = {e["name"]: e["id"] for e in entries}

    def resolve(self, key: str) -> str | None:
        if key in self.ids:
            return key
        return self.by_name.get(key)


def top_k(weights: Mapping[str, Fraction], k: int) -> list[str]:
    """The ids of the ``k`` largest weights, ties by id; the result is ordered by weight then id."""
    return sorted(weights, key=lambda key: (-weights[key], key))[:k]


def largest_remainder(
    total: int,
    weights: Mapping[str, Fraction],
    caps: Mapping[str, int] | None = None,
    order: Callable[[str], Any] = lambda key: key,
) -> dict[str, int]:
    """Integers in the proportion of ``weights`` that add up to exactly ``total``.

    Each key gets the floor of its exact quota ``total * w / sum(w)``; the units left over go one by
    one to the largest fractional remainders, ties by ``order(key)``. A key never exceeds its cap
    (a key at its cap passes its unit on to the next remainder). Refuses a zero weight sum and a
    total the caps cannot hold.
    """
    caps = caps or {}
    mass = sum(weights.values(), Fraction(0))
    if mass == 0:
        if total != 0:
            raise DerivationError("a list with no weight cannot be scaled to a total")
        return dict.fromkeys(weights, 0)
    quotas = {key: total * w / mass for key, w in weights.items()}
    result = {key: min(int(q), caps.get(key, total)) for key, q in quotas.items()}
    left = total - sum(result.values())
    ranked = sorted(weights, key=lambda key: (-(quotas[key] - int(quotas[key])), order(key)))
    while left > 0:
        moved = False
        for key in ranked:
            if left > 0 and result[key] < caps.get(key, total):
                result[key] += 1
                left -= 1
                moved = True
        if not moved:
            raise DerivationError("the caps cannot hold the total")
    return result
