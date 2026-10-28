"""The ``candidate_features`` rows as encoder inputs (ADR-022 tier A, slice D3).

A query under a ``probe-damage/1`` checkpoint carries, beside its candidates, the rows of the
engine's damage calculation on assumed opponent sets (``PolicyQuery.candidate_features``,
built on the Node side). This module turns the row of one move part into fixed-width columns:
per foe location (1 to ``SLOTS``) a learned category (no row, computed, or the probe's closed
reason for an unsupported row) and five numbers, each an integer from the row scaled by a
constant. A spread move fills one group per foe it hits; a single-target move fills the
group of its target. An unsupported row is its category with zero numbers: nothing is
filled in with a guess (ADR-016 §3). No game rule is written here: which move parts need a row
is decided by the engine-data table (a Status move needs none) and the request's own target
type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from areneon_ml.model.candidates import SLOTS
from areneon_ml.model.gamedata import NONE, RESERVED

PPM = 1_000_000  # mean_ppm, max_ppm and min_ppm are shares in parts per million
OHKO_ROLLS = 16  # the probe's rolls per damage vector (the contract bounds ohko_rolls by it)

# Row categories after the reserved indices: a computed row, then each closed reason of an
# unsupported row (the contract's enum, in its order). NONE is "no row for this foe".
REASONS = ("query_kind", "random_effect", "scenario_rule_table")
KINDS = ("computed", *(f"unsupported:{reason}" for reason in REASONS))
KIND_CATEGORIES = RESERVED + len(KINDS)
NUMBERS = ("mean", "max", "min", "ohko_known", "ohko")  # per foe location
NUMBER_COLUMNS = len(NUMBERS)

# The request target types that hit every foe or a random one. A damaging move part of such a
# type that names no target location needs at least one row. The TypeScript side holds the same
# list (feature-rows.ts); a test requires them equal.
FOE_SPREAD_TARGET_TYPES = ("allAdjacent", "allAdjacentFoes", "randomNormal")

Row = Mapping[str, Any]
RowKey = tuple[int, int, int, bool]  # slot, move position, target location, Mega flag


def row_index(features: Mapping[str, Any]) -> dict[RowKey, Row]:
    """The rows by (slot, move, target, mega); a row twice or a target past ``SLOTS`` is refused."""
    found: dict[RowKey, Row] = {}
    for row in features["rows"]:
        if row["target"] > SLOTS:
            raise ValueError("a candidate_features row aims at a foe location the format lacks")
        key = (row["slot"], row["move"], row["target"], row["mega"])
        if key in found:
            raise ValueError("a candidate_features row is listed twice")
        found[key] = row
    return found


def category(row: Row | None) -> int:
    if row is None:
        return NONE
    if row["kind"] == "computed":
        return RESERVED
    return RESERVED + 1 + REASONS.index(row["reason"])


def numbers(row: Row | None) -> list[float]:
    """The five numbers of a computed row; zeros for no row and for an unsupported one."""
    if row is None or row["kind"] != "computed":
        return [0.0] * NUMBER_COLUMNS
    ohko = row["ohko_rolls"]
    return [
        row["mean_ppm"] / PPM,
        row["max_ppm"] / PPM,
        row["min_ppm"] / PPM,
        0.0 if ohko is None else 1.0,
        0.0 if ohko is None else ohko / OHKO_ROLLS,
    ]


def part_columns(rows: Sequence[Row | None]) -> tuple[list[int], list[float]]:
    """The categorical and numeric columns of one move part, from its row per foe location."""
    cats = [category(row) for row in rows]
    nums = [x for row in rows for x in numbers(row)]
    return cats, nums


def no_columns() -> tuple[list[int], list[float]]:
    """The columns of a part with no row (a pass, a switch, a status move, an ally target)."""
    return part_columns([None] * SLOTS)
