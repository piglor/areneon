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
from dataclasses import dataclass, field
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


class RowError(ValueError):
    """A row the set's categories cannot name (the encoder turns it into an ``EncodingError``)."""


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


# ``probe-outcome/1`` (ADR-022 erratum E-22.5.3) also has the reason ``history_unknown``, appended
# after every other category so that no other set's index moves.
FEATURE_SET_OUTCOME = "probe-outcome/1"
HISTORY_UNKNOWN = "history_unknown"
REASONS_OUTCOME = (*REASONS, HISTORY_UNKNOWN)
KINDS_OUTCOME = (*KINDS, f"unsupported:{HISTORY_UNKNOWN}")
KIND_CATEGORIES_OUTCOME = RESERVED + 1 + len(REASONS_OUTCOME)


def _index(reasons: tuple[str, ...], reason: str) -> int:
    if reason not in reasons:
        raise RowError(f"a candidate_features row has the reason {reason!r}, which this set lacks")
    return reasons.index(reason)


def category(row: Row | None, outcome: bool = False) -> int:
    if row is None:
        return NONE
    if row["kind"] == "computed":
        return RESERVED
    return RESERVED + 1 + _index(REASONS_OUTCOME if outcome else REASONS, row["reason"])


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


def part_columns(
    rows: Sequence[Row | None], outcome: bool = False
) -> tuple[list[int], list[float]]:
    """The categorical and numeric columns of one move part, from its row per foe location."""
    cats = [category(row, outcome) for row in rows]
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
FEATURE_SET_BELIEF_THREATS = "probe-damage/3"  # ... and belief_threats beside them
ARRAY_NAMES = ("threats", "speeds", "own_priorities", "foe_priorities")
BELIEF_ARRAY = "belief_threats"
OUTCOME_ARRAY = "outcomes"
BELIEF_REFUSED = "belief_threats_refused"
# The opponent model's fixed refusal codes (the contract's enum, in its order). A refused belief
# gives no belief rows and says so: the field token carries the code as a category, so the
# network can tell "refused" from "no likely threats".
BELIEF_REFUSALS = (
    "reader_flagged",
    "team_size",
    "species_not_in_prior",
    "forme_not_in_prior",
    "brought_exceeds_picked",
    "disguise_assignments_exceeded",
    "evidence_has_no_support",
    "legality_redraws_exhausted",
    "empty_distribution",
    "disguise_unsettled",
)
BELIEF_REFUSAL_CATEGORIES = RESERVED + len(BELIEF_REFUSALS)


def belief_refusal_category(code: str | None) -> int:
    """NONE when the belief answered (or none was asked), else the code's learned category."""
    return NONE if code is None else RESERVED + BELIEF_REFUSALS.index(code)


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
# ``probe-damage/3`` folds the belief's threats into the same rows (ADR-022): two more columns
# say whether the row is the belief's (1) or a revealed move's (0) and the probability the foe
# holds the move (1 for a revealed one).
THREAT_NUMBERS_3 = (*THREAT_NUMBERS, "belief", "inclusion")
FOE_NUMBERS = ("priority",)
OWN_NUMBERS = ("own_priority",)

ThreatKey = tuple[int, bool, int, str]  # own position, own Mega flag, foe location, foe move id
SpeedKey = tuple[int, bool, int]  # own position, own Mega flag, foe location


PROBE_KINDS_OUTCOME = (*PROBE_KINDS, f"unsupported:{HISTORY_UNKNOWN}")
PROBE_KIND_CATEGORIES_OUTCOME = RESERVED + len(PROBE_KINDS_OUTCOME)
ALL_REASONS_OUTCOME = (*ALL_REASONS, HISTORY_UNKNOWN)
SPEED_KINDS_OUTCOME = (*SPEED_KINDS, f"unsupported:{HISTORY_UNKNOWN}")
SPEED_CATEGORIES_OUTCOME = RESERVED + len(SPEED_KINDS_OUTCOME)


def reads_arrays(feature_set: str | None) -> bool:
    """Whether the feature set carries the four arrays (``/2``, ``/3`` and ``probe-outcome/1``)."""
    return feature_set in (FEATURE_SET_ARRAYS, FEATURE_SET_BELIEF_THREATS, FEATURE_SET_OUTCOME)


def reads_outcomes(feature_set: str | None) -> bool:
    """Whether the feature set also carries ``outcomes`` (``probe-outcome/1``)."""
    return feature_set == FEATURE_SET_OUTCOME


def reads_belief_threats(feature_set: str | None) -> bool:
    """Whether the feature set also carries ``belief_threats`` (``/3``)."""
    return feature_set == FEATURE_SET_BELIEF_THREATS


def array_names(feature_set: str | None) -> tuple[str, ...]:
    """The arrays a feature set carries beside ``rows`` (none for ``/1`` and the belief set)."""
    if reads_belief_threats(feature_set):
        return (*ARRAY_NAMES, BELIEF_ARRAY)
    if reads_outcomes(feature_set):
        return (*ARRAY_NAMES, OUTCOME_ARRAY)
    return ARRAY_NAMES if reads_arrays(feature_set) else ()


def probe_category(row: Row, outcome: bool = False) -> int:
    """The learned category of a threat or priority row: computed or its closed reason."""
    if row["kind"] == "computed":
        return RESERVED
    return RESERVED + 1 + _index(ALL_REASONS_OUTCOME if outcome else ALL_REASONS, row["reason"])


def speed_category(row: Row, outcome: bool = False) -> int:
    """The learned category of a speed row: one of the relations or its closed reason."""
    if row["kind"] == "computed":
        return RESERVED + RELATIONS.index(row["relation"])
    reasons = ALL_REASONS_OUTCOME if outcome else ALL_REASONS
    return RESERVED + len(RELATIONS) + _index(reasons, row["reason"])


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
    belief_threats: list[Row] = field(default_factory=list)  # ``probe-damage/3`` only
    outcomes: list[Row] = field(default_factory=list)  # ``probe-outcome/1`` only
    refused: str | None = None  # the model's code when the belief was refused (then no belief rows)

    @classmethod
    def of(cls, features: Mapping[str, Any]) -> Arrays:
        names = array_names(features.get("feature_set"))
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"a {features.get('feature_set')} query lacks the array {missing[0]}")
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
        belief = list(features.get(BELIEF_ARRAY, []))
        _unique(belief, lambda r: (r["own"], r["mega"], r["foe"], r["move"]))
        refused = features.get(BELIEF_REFUSED)
        if refused is not None and (
            belief or not reads_belief_threats(features.get("feature_set"))
        ):
            raise ValueError("a belief_threats_refused decision has no belief_threats")
        revealed = {(r["foe"], r["move"]) for r in features["threats"]}
        if any((r["foe"], r["move"]) in revealed for r in belief):
            raise ValueError("a belief_threats row repeats a move already in threats for that foe")
        outcomes = list(features.get(OUTCOME_ARRAY, []))
        _unique(outcomes, _outcome_key)
        return cls(
            list(features["threats"]),
            speeds,
            own,
            foes,
            frozenset(where),
            belief_threats=belief,
            refused=refused,
            outcomes=outcomes,
        )


def _unique(rows: Sequence[Row], key: Callable[[Row], Any]) -> dict[Any, Row]:
    found: dict[Any, Row] = {}
    for row in rows:
        if key(row) in found:
            raise ValueError("a candidate_features array lists a row twice")
        found[key(row)] = row
    return found


def threat_numbers(row: Row, foe: Row | None) -> list[float]:
    """A threat row's numbers (as a damage row's) and its foe move's priority."""
    return [*numbers(row), 0.0 if foe is None else priority_number(foe)]


def belief_threat_numbers(row: Row) -> list[float]:
    """A belief threat row's numbers: damage, no foe priority, the belief flag, the inclusion."""
    return [*numbers(row), 0.0, 1.0, row["inclusion_ppm"] / PPM]


def revealed_threat_numbers_3(row: Row, foe: Row | None) -> list[float]:
    """A revealed threat's numbers under ``/3``: ``/2``'s, not the belief's, inclusion 1."""
    return [*threat_numbers(row, foe), 0.0, 1.0]


# -- probe-outcome/1 (ADR-022 erratum E-22.5.3, encoder reading E-22.5.4) ------------------------
#
# An ``outcomes`` row is one own move part (slot, move position, target, Mega flag) against one
# opposing-action branch: the switch branch first, then one branch per revealed foe move in move-id
# order (a foe with no revealed move has one assumed branch), at most ``MAX_BRANCHES`` in all. The
# part is about ONE foe (the foe at its target location, else the lowest living foe), so the
# encoder puts a pooled summary of the part's branches in that foe's location group of the move
# part and, because that foe is the part's own, the per-branch values in a block of their own.
# Pooling rule: over the COMPUTED branches the mean and the minimum of ``success_ppm / 10^6``; the
# pooled precondition is ``passes`` when every computed branch passes, ``fails`` when every one
# fails and ``chance`` otherwise (none when no branch is computed); the pooled kind is ``computed``
# when every branch is, else the first unsupported branch's reason. Nothing is guessed: an
# unsupported branch adds no number.

MAX_BRANCHES = 5  # the reviewed record's ceiling (``max_branches``): switch plus four moves
OUTCOME_REASONS = (*REASONS, HISTORY_UNKNOWN)
OUTCOME_KINDS = ("computed", *(f"unsupported:{reason}" for reason in OUTCOME_REASONS))
OUTCOME_KIND_CATEGORIES = RESERVED + len(OUTCOME_KINDS)
PRECONDITIONS = ("passes", "fails", "chance")
OUTCOME_PRE_CATEGORIES = RESERVED + len(PRECONDITIONS)
BRANCH_KINDS = (*PRECONDITIONS, *(f"unsupported:{reason}" for reason in OUTCOME_REASONS))
BRANCH_KIND_CATEGORIES = RESERVED + len(BRANCH_KINDS)
OUTCOME_POOLED = ("mean", "min", "known")  # numbers per foe location
OUTCOME_BRANCH_NUMBERS = ("success", "assumed")  # numbers per branch
OutcomeKey = tuple[int, int, int, bool]  # slot, move position, target, Mega flag


def _outcome_key(row: Row) -> tuple[int, int, int, bool, str, str]:
    return (
        row["slot"],
        row["move"],
        row["target"],
        row["mega"],
        row["action"],
        row["foe_move"] or "",
    )


def outcome_blocks(rows: Sequence[Row]) -> dict[OutcomeKey, list[Row]]:
    """The rows by part, each part's branches in the wire's order; a malformed block is refused.

    The wire lists a part's switch branch first and then its move branches (the readers hold
    this); the encoder re-checks it, since it indexes the per-branch columns by position.
    """
    blocks: dict[OutcomeKey, list[Row]] = {}
    for row in rows:
        blocks.setdefault((row["slot"], row["move"], row["target"], row["mega"]), []).append(row)
    for group in blocks.values():
        if group[0]["action"] != "switch" or any(r["action"] != "move" for r in group[1:]):
            raise RowError("an outcomes block must start with its switch branch")
        if not 1 + 1 <= len(group) <= MAX_BRANCHES:
            raise RowError("an outcomes block has a switch branch and one to four move branches")
    return blocks


def _outcome_kind(group: Sequence[Row]) -> int:
    for row in group:
        if row["kind"] != "computed":
            return RESERVED + 1 + OUTCOME_REASONS.index(row["reason"])
    return RESERVED


def _pooled_precondition(computed: Sequence[Row]) -> int:
    if not computed:
        return NONE
    found = {row["precondition"] for row in computed}
    if found == {"passes"}:
        return RESERVED + PRECONDITIONS.index("passes")
    if found == {"fails"}:
        return RESERVED + PRECONDITIONS.index("fails")
    return RESERVED + PRECONDITIONS.index("chance")


def outcome_pooled(group: Sequence[Row]) -> tuple[list[int], list[float]]:
    """The pooled categories (kind, precondition) and numbers (mean, min, known) of one part."""
    computed = [row for row in group if row["kind"] == "computed"]
    shares = [row["success_ppm"] / PPM for row in computed]
    cats = [_outcome_kind(group), _pooled_precondition(computed)]
    nums = [sum(shares) / len(shares), min(shares), 1.0] if shares else [0.0] * len(OUTCOME_POOLED)
    return cats, nums


def _branch_kind(row: Row) -> int:
    if row["kind"] == "computed":
        return RESERVED + PRECONDITIONS.index(row["precondition"])
    return RESERVED + len(PRECONDITIONS) + OUTCOME_REASONS.index(row["reason"])


def outcome_branches(
    group: Sequence[Row], move_index: Callable[[str], int]
) -> tuple[list[int], list[float]]:
    """The per-branch categories (kind, foe move) and numbers (success, assumed) of one part,
    ``MAX_BRANCHES`` blocks, an absent branch NONE and zero. The switch branch has no foe move."""
    cats: list[int] = []
    nums: list[float] = []
    for index in range(MAX_BRANCHES):
        if index >= len(group):
            cats += [NONE, NONE]
            nums += [0.0, 0.0]
            continue
        row = group[index]
        move = row["foe_move"]
        cats += [_branch_kind(row), NONE if move is None else move_index(move)]
        success = row["success_ppm"] / PPM if row["kind"] == "computed" else 0.0
        nums += [success, 1.0 if row["assumed"] else 0.0]
    return cats, nums


def outcome_no_columns() -> tuple[list[int], list[float]]:
    """The columns of a part with no outcome block, in the layout's order."""
    pooled = SLOTS * 2 + MAX_BRANCHES * 2
    return [NONE] * pooled, [0.0] * (SLOTS * len(OUTCOME_POOLED) + MAX_BRANCHES * 2)


def outcome_columns(
    group: Sequence[Row], location: int, move_index: Callable[[str], int]
) -> tuple[list[int], list[float]]:
    """The columns of a move part's block, in the layout's order: the pooled categories and
    numbers of every foe location (only ``location``, the part's foe, is filled) and then the
    per-branch ones."""
    if not 1 <= location <= SLOTS:
        raise RowError("an outcomes block is about a foe location the format lacks")
    cats, nums = outcome_no_columns()
    pooled_cats, pooled_nums = outcome_pooled(group)
    branch_cats, branch_nums = outcome_branches(group, move_index)
    at = (location - 1) * 2
    cats[at : at + 2] = pooled_cats
    at = (location - 1) * len(OUTCOME_POOLED)
    nums[at : at + len(OUTCOME_POOLED)] = pooled_nums
    cats[SLOTS * 2 :] = branch_cats
    nums[SLOTS * len(OUTCOME_POOLED) :] = branch_nums
    return cats, nums
