"""Opponent sheet block counts and E4's validity rule (ADR-025 erratum E4 point 5).

A pair that reads ``opponent_sheet`` is given one block per request. A refused belief is the
``unknown`` block with a fixed code (counted, never a rule fallback), and "a refused share above
1% of decisions invalidates the experiment". The counts are integers: requests (blocks built),
refused, and the refusals by code. This module reads them from the closed ``sheet`` object a game
row (``generation.json``, a progress record) carries, sums them, and states the share in parts per
million and whether the run is still valid; it decides nothing else.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: 1% of the requests, in parts per million (ADR-025 erratum E4 point 5).
MAX_REFUSED_PPM = 10_000
PPM = 1_000_000


class SheetCountsError(ValueError):
    """A ``sheet`` object breaks the rules of its contract."""


@dataclass(frozen=True)
class SheetCounts:
    requests: int
    refused: int
    refusals: tuple[tuple[str, int], ...]  # sorted by code, unique

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "refused": self.refused,
            "refusals": [{"code": code, "count": count} for code, count in self.refusals],
        }

    @property
    def refused_share_ppm(self) -> int:
        """The refused share of the requests, rounded down; 0 for no request."""
        return 0 if self.requests == 0 else self.refused * PPM // self.requests

    @property
    def valid(self) -> bool:
        """E4 point 5: a refused share above 1% invalidates the experiment."""
        return self.refused_share_ppm <= MAX_REFUSED_PPM


def _count(value: object, what: str) -> int:
    if type(value) is not int or value < 0:
        raise SheetCountsError(f"{what} is not a non-negative integer")
    return value


def read_sheet_counts(raw: object, what: str = "sheet") -> SheetCounts:
    """A closed ``{requests, refused, refusals[{code, count}]}`` object: refusals sorted by code and
    unique, their counts summing to ``refused``, and ``refused`` at most ``requests``."""
    if not isinstance(raw, Mapping) or set(raw) != {"requests", "refused", "refusals"}:
        raise SheetCountsError(f"{what} must hold exactly requests, refused and refusals")
    requests, refused = _count(raw["requests"], f"{what} requests"), _count(raw["refused"], what)
    listed = raw["refusals"]
    if not isinstance(listed, list):
        raise SheetCountsError(f"{what} refusals is not a list")
    codes: list[tuple[str, int]] = []
    for item in listed:
        if not isinstance(item, Mapping) or set(item) != {"code", "count"}:
            raise SheetCountsError(f"{what} refusals must hold code and count")
        if not isinstance(item["code"], str) or not item["code"]:
            raise SheetCountsError(f"{what} refusal code is not text")
        codes.append((item["code"], _count(item["count"], f"{what} refusal count")))
    if any(a[0] >= b[0] for a, b in itertools.pairwise(codes)):
        raise SheetCountsError(f"{what} refusals must be sorted by code and unique")
    if refused > requests:
        raise SheetCountsError(f"{what} refused more than it built")
    if sum(count for _, count in codes) != refused:
        raise SheetCountsError(f"{what} refusals do not add up to refused")
    return SheetCounts(requests, refused, tuple(codes))


def total(counts: Iterable[SheetCounts]) -> SheetCounts:
    """The counts summed, refusals by code."""
    items = list(counts)
    by_code: dict[str, int] = {}
    for c in items:
        for code, count in c.refusals:
            by_code[code] = by_code.get(code, 0) + count
    return SheetCounts(
        sum(c.requests for c in items),
        sum(c.refused for c in items),
        tuple(sorted(by_code.items())),
    )
