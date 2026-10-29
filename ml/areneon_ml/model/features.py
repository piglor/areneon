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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


# -- probe-damage/2 (ADR-022 erratum E-22.3.3) ---------------------------------------------------
#
# The four arrays beside ``rows`` are facts about the state, so the encoder attaches them to the
# own and foe tokens (threats, speed relations, foe priorities) and to the slot parts (own
# priorities, the Mega variant's speed relations and threats). The categories below are the
# contract's closed enums in their order; an unsupported row is its category with zero numbers,
# never a guess.

FEATURE_SET_ARRAYS = "probe-damage/2"  # the set whose object carries the four arrays
ARRAY_NAMES = ("threats", "speeds", "own_priorities", "foe_priorities")
RELATIONS = ("own_first", "tie", "foe_first")
# The reasons of an unsupported row of any array: `/1`'s three, then the two that only speeds and
# priorities carry (the contract's enum order).
ALL_REASONS = (*REASONS, "unqueueable", "fractional_priority")
PROBE_KINDS = ("computed", *(f"unsupported:{reason}" for reason in ALL_REASONS))
PROBE_KIND_CATEGORIES = RESERVED + len(PROBE_KINDS)
SPEED_KINDS = (*RELATIONS, *(f"unsupported:{reason}" for reason in ALL_REASONS))
SPEED_CATEGORIES = RESERVED + len(SPEED_KINDS)
PRIORITY_SCALE = 8.0  # a priority is read as priority / 8 (the engine's keys run -7 to +5)
THREAT_NUMBERS = (*NUMBERS, "foe_priority")  # a threat row's numbers; the last is its move's key
FOE_NUMBERS = ("priority",)
OWN_NUMBERS = ("own_priority",)

ThreatKey = tuple[int, bool, int, str]  # own position, own Mega flag, foe location, foe move id
SpeedKey = tuple[int, bool, int]  # own position, own Mega flag, foe location


def reads_arrays(feature_set: str | None) -> bool:
    """Whether the feature set is the one that carries the four arrays."""
    return feature_set == FEATURE_SET_ARRAYS


def probe_category(row: Row) -> int:
    """The learned category of a threat or priority row: computed or its closed reason."""
    if row["kind"] == "computed":
        return RESERVED
    return RESERVED + 1 + ALL_REASONS.index(row["reason"])


def speed_category(row: Row) -> int:
    """The learned category of a speed row: one of the relations or its closed reason."""
    if row["kind"] == "computed":
        return RESERVED + RELATIONS.index(row["relation"])
    return RESERVED + len(RELATIONS) + ALL_REASONS.index(row["reason"])


def priority_number(row: Row) -> float:
    """A priority row's number: its key over ``PRIORITY_SCALE``; zero when unsupported."""
    if row["kind"] != "computed":
        return 0.0
    return float(row["priority"]) / PRIORITY_SCALE


@dataclass(frozen=True)
class Arrays:
    """The four arrays of a ``probe-damage/2`` query, indexed; a key twice is refused."""

    threats: list[Row]
    speeds: dict[SpeedKey, Row]
    own_priorities: dict[tuple[int, int, bool], Row]
    foe_priorities: dict[tuple[int, str], Row]
    locations: frozenset[int]  # the foe locations any array or row names (the reader's rule)

    @classmethod
    def of(cls, features: Mapping[str, Any]) -> Arrays:
        missing = [name for name in ARRAY_NAMES if name not in features]
        if missing:
            raise ValueError(f"a {FEATURE_SET_ARRAYS} query lacks the array {missing[0]}")
        speeds = _unique(features["speeds"], lambda r: (r["own"], r["mega"], r["foe"]))
        own = _unique(features["own_priorities"], lambda r: (r["slot"], r["move"], r["mega"]))
        foes = _unique(features["foe_priorities"], lambda r: (r["foe"], r["move"]))
        _unique(features["threats"], lambda r: (r["own"], r["mega"], r["foe"], r["move"]))
        where = (
            {r["target"] for r in features["rows"]}
            | {r["foe"] for r in features["threats"]}
            | {r["foe"] for r in features["speeds"]}
            | {r["foe"] for r in features["foe_priorities"]}
        )
        return cls(list(features["threats"]), speeds, own, foes, frozenset(where))


def _unique(rows: Sequence[Row], key: Callable[[Row], Any]) -> dict[Any, Row]:
    found: dict[Any, Row] = {}
    for row in rows:
        if key(row) in found:
            raise ValueError("a candidate_features array lists a row twice")
        found[key(row)] = row
    return found


def threat_numbers(row: Row, foe: Row) -> list[float]:
    """A threat row's numbers (as a damage row's) and its foe move's priority."""
    return [*numbers(row), priority_number(foe)]
