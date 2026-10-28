"""The reader rules of ``candidate_features`` that the contract profile cannot say (ADR-022 §3).

The same rules as ``packages/showdown-bridge/src/feature-rows.ts``: rows sorted and unique by
(slot, move, target, mega), and a row for every damaging move part of every candidate. The worker
holds no engine, so "damaging" comes from the engine-data table the encoder already loads (a move
whose category is not Status). A problem is returned as fixed text naming no value; ``None`` means
the rules hold.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from areneon_ml.model.candidates import parse_choice
from areneon_ml.model.gamedata import GameData

# Request target types that hit every foe or a random one; a damaging move part of such a type that
# names no target location needs at least one row. A test holds this equal to
# FOE_SPREAD_TARGET_TYPES in feature-rows.ts.
FOE_SPREAD_TARGET_TYPES = ("allAdjacent", "allAdjacentFoes", "randomNormal")

_REQUEST = "|request|"


def damaging_moves(game: GameData) -> Callable[[str], bool]:
    """Tells a damaging move from a status move by its id, from the worker's engine-data table.

    An id the table does not list counts as damaging: the encoder refuses such a query anyway, and
    the coverage rule then asks for the rows rather than letting a move through unscored.
    """
    return lambda move_id: not game.move_status[game.moves.lookup_or_unknown(move_id)]


def _key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (row["slot"], row["move"], row["target"], int(row["mega"]))


def rows_problem(features: dict[str, Any]) -> str | None:
    """None when the rows are strictly ascending by (slot, move, target, mega false first)."""
    rows = features["rows"]
    for before, now in itertools.pairwise(rows):
        if not _key(before) < _key(now):
            return (
                "the candidate_features rows must be sorted and unique by "
                "(slot, move, target, mega)"
            )
    return None


def coverage_problem(
    query: dict[str, Any], features: dict[str, Any], damaging: Callable[[str], bool]
) -> str | None:
    """None when every damaging move part of every candidate has its row, as far as the query says.

    A part that aims at a foe location, and a part of a spread or random-target move that names
    none, need at least one row of (slot, move, mega). The row of the exact location is not
    required: the engine lists a candidate that aims at a fainted foe's location, which the builder
    rightly gives no row, and the worker holds no battle state to tell an absent foe from a
    missing row.
    """
    last = query["observation"]["log"][-1]
    if not last.startswith(_REQUEST):
        return "the observation's log does not end in a request"
    active = json.loads(last[len(_REQUEST) :]).get("active")
    if active is None:  # Team Preview, a forced switch or a wait: no move to score
        return None
    some = {(row["slot"], row["move"], int(row["mega"])) for row in features["rows"]}
    for candidate in query["candidates"]:
        for slot, action in enumerate(parse_choice(candidate["choice"]).slots):
            if action.kind != "move":
                continue
            move = active[slot]["moves"][(action.index or 1) - 1]
            if not damaging(move["id"]):
                continue
            if action.target is not None:
                aims_at_foe = action.target > 0
            else:
                aims_at_foe = move.get("target") in FOE_SPREAD_TARGET_TYPES
            if aims_at_foe and (slot, action.index, int(action.event == "mega")) not in some:
                return "a damaging move part of a candidate has no candidate_features row"
    return None
