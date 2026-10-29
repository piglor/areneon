"""The reader rules of ``candidate_features`` that the contract profile cannot say (ADR-022 §3).

The same rules as ``packages/showdown-bridge/src/feature-rows.ts``: rows sorted and unique by
(slot, move, target, mega), and a row for every damaging move part of every candidate. The worker
holds no engine, so "damaging" comes from the engine-data table the encoder already loads (a move
whose category is not Status). A problem is returned as fixed text naming no value; ``None`` means
the rules hold.

Under ``probe-damage/2`` (ADR-022 erratum E-22.3.2) the object also carries ``threats``, ``speeds``,
``own_priorities`` and ``foe_priorities``: present exactly then, each sorted and unique by its key
(``rows_problem``); and, for a query, the own positions, Mega flags, own-priority rows and speed
rows the request and the candidates imply (``coverage_problem``).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from areneon_ml.model.candidates import parse_choice
from areneon_ml.model.features import FOE_SPREAD_TARGET_TYPES
from areneon_ml.model.gamedata import GameData

# Request target types that hit every foe or a random one (defined beside the encoder, which reads
# the same rule); a test holds it equal to FOE_SPREAD_TARGET_TYPES in feature-rows.ts.

_REQUEST = "|request|"

__all__ = ["FOE_SPREAD_TARGET_TYPES", "coverage_problem", "damaging_moves", "rows_problem"]


def damaging_moves(game: GameData) -> Callable[[str], bool]:
    """Tells a damaging move from a status move by its id, from the worker's engine-data table.

    An id the table does not list counts as damaging: the encoder refuses such a query anyway, and
    the coverage rule then asks for the rows rather than letting a move through unscored.
    """
    return lambda move_id: not game.move_status[game.moves.lookup_or_unknown(move_id)]


def _key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (row["slot"], row["move"], row["target"], int(row["mega"]))


#: The feature set whose object carries the four extra arrays, and their names.
FEATURE_SET_WITH_ARRAYS = "probe-damage/2"
_ARRAYS = ("threats", "speeds", "own_priorities", "foe_priorities")

# The sort key of each array's rows (a Mega flag sorts false first).
_ARRAY_KEYS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
    "threats": lambda r: (r["own"], int(r["mega"]), r["foe"], r["move"]),
    "speeds": lambda r: (r["own"], int(r["mega"]), r["foe"]),
    "own_priorities": lambda r: (r["slot"], r["move"], int(r["mega"])),
    "foe_priorities": lambda r: (r["foe"], r["move"]),
}


def rows_problem(features: dict[str, Any]) -> str | None:
    """None when the rows are strictly ascending by (slot, move, target, mega false first).

    Under ``probe-damage/2`` the four arrays must be present and strictly ascending by their keys
    (threats by (own, mega, foe, move id), speeds by (own, mega, foe), own priorities by
    (slot, move, mega), foe priorities by (foe, move id)); under any other set none may be present.
    """
    rows = features["rows"]
    for before, now in itertools.pairwise(rows):
        if not _key(before) < _key(now):
            return (
                "the candidate_features rows must be sorted and unique by "
                "(slot, move, target, mega)"
            )
    present = [name for name in _ARRAYS if name in features]
    if features.get("feature_set") == FEATURE_SET_WITH_ARRAYS:
        wrong = len(present) != len(_ARRAYS)
    else:
        wrong = bool(present)
    if wrong:
        return (
            f"the candidate_features arrays {', '.join(_ARRAYS)} "
            f"are present exactly under {FEATURE_SET_WITH_ARRAYS}"
        )
    for name in present:
        for before, now in itertools.pairwise(features[name]):
            if not _ARRAY_KEYS[name](before) < _ARRAY_KEYS[name](now):
                return f"the candidate_features {name} must be sorted and unique by their key"
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
    request = json.loads(last[len(_REQUEST) :])
    active = request.get("active")
    if active is not None:  # else Team Preview, a forced switch or a wait: no move to score
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
    if features.get("feature_set") == FEATURE_SET_WITH_ARRAYS:
        return _array_problem(request, query["candidates"], features)
    return None


def _fixed(text: str) -> str:
    return f"the candidate_features {text}"


def _implied(
    request: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]]] | None:
    """The (position, Mega flag) variants and the (slot, move, Mega) move parts a request implies.

    The own Pokemon the arrays are about are the living actives of a move request, every living
    benched Pokemon a ``switch N`` candidate names, and the own Mega variant of an active slot where
    a candidate Mega Evolves. None when a switch candidate names a Pokemon the request lacks.
    """
    active = request.get("active")
    slots = active if active is not None else request.get("forceSwitch")
    pokemon = request.get("side", {}).get("pokemon", [])

    def alive(position: int) -> bool:
        condition = pokemon[position].get("condition")
        return isinstance(condition, str) and not condition.endswith(" fnt")

    variants: set[tuple[int, int]] = set()
    mega_slots: set[int] = set()
    parts: set[tuple[int, int, int]] = set()
    parsed = [parse_choice(c["choice"]).slots for c in candidates] if slots is not None else []
    for slot_actions in parsed:
        for slot, action in enumerate(slot_actions):
            if action.kind == "move":
                mega = int(action.event == "mega")
                parts.add((slot, action.index or 1, mega))
                if mega:
                    mega_slots.add(slot)
            elif action.kind == "switch":
                position = (action.index or 1) - 1
                if position >= len(pokemon):
                    return None
                if alive(position):
                    variants.add((position, 0))
    for slot in range(len(active or [])):
        if alive(slot):
            variants.add((slot, 0))
            if slot in mega_slots:
                variants.add((slot, 1))
    return variants, (parts if active is not None else set())


def _array_problem(  # noqa: PLR0911
    request: dict[str, Any], candidates: list[dict[str, Any]], features: dict[str, Any]
) -> str | None:
    """The ``probe-damage/2`` rules that need the request (as in feature-rows.ts).

    ``threats`` name only the implied own variants; ``speeds`` hold exactly one row per
    (own position, own Mega flag, foe location) for each of them against every foe location the
    arrays show; ``own_priorities`` hold exactly one row per own move part of the candidates; a
    request with no move to score has no foe priority row.
    """
    implied = _implied(request, candidates)
    if implied is None:
        return _fixed("arrays name a Pokemon the request does not list")
    variants, parts = implied
    threats, speeds = features["threats"], features["speeds"]
    foes = features["foe_priorities"]
    locations = (
        {row["target"] for row in features["rows"]}
        | {row["foe"] for row in threats}
        | {row["foe"] for row in speeds}
        | {row["foe"] for row in foes}
    )
    if any((row["own"], int(row["mega"])) not in variants for row in threats):
        return _fixed("threats name an own Pokemon or Mega variant the request does not imply")
    expected_speeds = sorted((o, m, f) for o, m in variants for f in locations)
    if [(r["own"], int(r["mega"]), r["foe"]) for r in speeds] != expected_speeds:
        return _fixed("speeds must hold exactly one row per own position, own Mega flag and foe")
    own = [(r["slot"], r["move"], int(r["mega"])) for r in features["own_priorities"]]
    if own != sorted(parts):
        return _fixed("own_priorities must hold one row per own move part of the candidates")
    if request.get("active") is None and foes:
        return _fixed("foe_priorities are empty when the request has no move to score")
    if request.get("active") is not None:
        priced = {(r["foe"], r["move"]) for r in foes}
        if any((r["foe"], r["move"]) not in priced for r in threats):
            return _fixed(
                "foe_priorities must hold a row for the foe move of every threat in a move request"
            )
    return None
