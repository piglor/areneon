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

Under ``probe-outcome/1`` (ADR-022 erratum E-22.5.3) the object carries those four arrays and
``outcomes`` (one row per own move part of the candidates and opposing-action branch); the reason
``history_unknown`` of an unsupported row is that set's alone.
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
#: ``probe-damage/3`` carries these and ``belief_threats``.
FEATURE_SET_WITH_BELIEF_THREATS = "probe-damage/3"
#: ``probe-outcome/1`` carries the four arrays and ``outcomes``; its reason alone is
#: ``history_unknown``.
FEATURE_SET_WITH_OUTCOMES = "probe-outcome/1"
HISTORY_UNKNOWN = "history_unknown"
#: The most branches one own move part has in the ``outcomes`` (the reviewed record's ceiling).
MAX_OUTCOME_BRANCHES = 5
_MILLION = 1_000_000
_ARRAYS = ("threats", "speeds", "own_priorities", "foe_priorities")
_BELIEF_ARRAYS = (*_ARRAYS, "belief_threats")
_OUTCOME_ARRAYS = (*_ARRAYS, "outcomes")
_EVERY_ARRAY = (*_ARRAYS, "belief_threats", "outcomes")

# The sort key of each array's rows (a Mega flag sorts false first).
_ARRAY_KEYS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
    "threats": lambda r: (r["own"], int(r["mega"]), r["foe"], r["move"]),
    "speeds": lambda r: (r["own"], int(r["mega"]), r["foe"]),
    "own_priorities": lambda r: (r["slot"], r["move"], int(r["mega"])),
    "foe_priorities": lambda r: (r["foe"], r["move"]),
    "belief_threats": lambda r: (r["own"], int(r["mega"]), r["foe"], r["move"]),
    "outcomes": lambda r: (
        r["slot"],
        r["move"],
        r["target"],
        int(r["mega"]),
        int(r["action"] != "switch"),
        r["foe_move"] or "",
    ),
}


def _wanted_arrays(feature_set: Any) -> tuple[str, ...]:
    if feature_set == FEATURE_SET_WITH_BELIEF_THREATS:
        return _BELIEF_ARRAYS
    if feature_set == FEATURE_SET_WITH_OUTCOMES:
        return _OUTCOME_ARRAYS
    return _ARRAYS if feature_set == FEATURE_SET_WITH_ARRAYS else ()


def _row_problem(r: dict[str, Any]) -> str | None:
    """The rule of one outcome row alone: branch shape, precondition against its ppm, success."""
    if r["action"] == "switch":
        if r["foe_move"] is not None or r["assumed"]:
            return _fixed("outcomes: a switch branch has no foe move and is not assumed")
    elif r["foe_move"] is None:
        return _fixed("outcomes: a move branch names its foe move")
    if r["kind"] != "computed":
        return None
    pre = r["precondition_ppm"]
    if r["precondition"] == "passes":
        fits = pre == _MILLION
    elif r["precondition"] == "fails":
        fits = pre == 0
    else:
        fits = 0 < pre < _MILLION
    if not fits:
        return _fixed("outcomes: the precondition does not fit its ppm")
    if r["success_ppm"] != pre * r["hit_ppm"] // _MILLION:
        return _fixed("outcomes: success_ppm must be floor(precondition_ppm x hit_ppm / 10^6)")
    return None


def _branches_problem(group: list[dict[str, Any]]) -> str | None:
    """The branches of one own move part: its switch branch, then one to four move branches."""
    moves = group[1:]
    if group[0]["action"] != "switch":
        return _fixed("outcomes: each own move part starts with its switch branch")
    if not 1 <= len(moves) <= MAX_OUTCOME_BRANCHES - 1 or any(r["action"] != "move" for r in moves):
        return _fixed(
            "outcomes: each own move part has a switch branch and one to four move branches"
        )
    if any(r["assumed"] for r in moves) and (len(moves) != 1 or not moves[0]["assumed"]):
        return _fixed("outcomes: an assumed branch is the only move branch of its part")
    return None


def _outcomes_problem(rows: list[dict[str, Any]]) -> str | None:
    """The ``outcomes`` rules that need no request (``outcomesProblem`` in feature-rows.ts)."""
    for r in rows:
        problem = _row_problem(r)
        if problem is not None:
            return problem
    parts = itertools.groupby(rows, key=lambda r: (r["slot"], r["move"], r["target"], r["mega"]))
    for _, group in parts:
        problem = _branches_problem(list(group))
        if problem is not None:
            return problem
    return None


def rows_problem(features: dict[str, Any]) -> str | None:  # noqa: PLR0911
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
    present = [name for name in _EVERY_ARRAY if name in features]
    if present != list(_wanted_arrays(features.get("feature_set"))):
        return (
            f"the candidate_features arrays {', '.join(_ARRAYS)} "
            f"are present exactly under {FEATURE_SET_WITH_ARRAYS}, and with belief_threats "
            f"under {FEATURE_SET_WITH_BELIEF_THREATS}, and with outcomes "
            f"under {FEATURE_SET_WITH_OUTCOMES}"
        )
    for name in present:
        for before, now in itertools.pairwise(features[name]):
            if not _ARRAY_KEYS[name](before) < _ARRAY_KEYS[name](now):
                return f"the candidate_features {name} must be sorted and unique by their key"
    if "belief_threats_refused" in features and (
        features.get("feature_set") != FEATURE_SET_WITH_BELIEF_THREATS
        or features.get("belief_threats")
    ):
        return (
            "a candidate_features belief_threats_refused is a probe-damage/3 field and means "
            "the belief_threats are empty"
        )
    revealed = {(r["foe"], r["move"]) for r in features.get("threats", [])}
    if any((r["foe"], r["move"]) in revealed for r in features.get("belief_threats", [])):
        return (
            "a candidate_features belief_threats row repeats a move already in threats for that foe"
        )
    if features.get("feature_set") != FEATURE_SET_WITH_OUTCOMES and any(
        r.get("reason") == HISTORY_UNKNOWN
        for rows in (features["rows"], *(features.get(name, []) for name in _ARRAYS))
        for r in rows
    ):
        return (
            f"a candidate_features reason {HISTORY_UNKNOWN} is a {FEATURE_SET_WITH_OUTCOMES} value"
        )
    return _outcomes_problem(features.get("outcomes", []))


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
    if _wanted_arrays(features.get("feature_set")):
        return _array_problem(request, query["candidates"], features)
    return None


def _fixed(text: str) -> str:
    return f"the candidate_features {text}"


def _alive(pokemon: list[dict[str, Any]], position: int) -> bool:
    """Whether the request's Pokemon at ``position`` lives; one the request lacks does not."""
    if not 0 <= position < len(pokemon):
        return False
    condition = pokemon[position].get("condition")
    return isinstance(condition, str) and not condition.endswith(" fnt")


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

    if len(active or []) > len(pokemon):
        return None  # an active slot with no Pokemon in the request: a malformed query

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
                if _alive(pokemon, position):
                    variants.add((position, 0))
    for slot in range(len(active or [])):
        if _alive(pokemon, slot):
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
    if any(
        (row["own"], int(row["mega"])) not in variants for row in features.get("belief_threats", [])
    ):
        return _fixed(
            "belief_threats name an own Pokemon or Mega variant the request does not imply"
        )
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
    if "outcomes" in _wanted_arrays(features.get("feature_set")):
        return _outcome_coverage(request, candidates, features, locations)
    return None


def _outcome_coverage(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    features: dict[str, Any],
    locations: set[int],
) -> str | None:
    """One block of ``outcomes`` per own move part of the candidates that has a foe to ask about.

    The part's Pokemon must live and its foe location hold a living foe (an untargeted or
    ally-targeted part is about the lowest foe, always present); a request with no move to score has
    none.
    """
    expected: set[tuple[int, int, int, int]] = set()
    pokemon = request.get("side", {}).get("pokemon", [])
    if request.get("active") is not None and locations:
        for candidate in candidates:
            for slot, action in enumerate(parse_choice(candidate["choice"]).slots):
                target = action.target or 0
                lives = _alive(pokemon, slot)
                if action.kind == "move" and lives and (target <= 0 or target in locations):
                    expected.add((slot, action.index or 1, target, int(action.event == "mega")))
    shown: list[tuple[int, int, int, int]] = []
    for row in features["outcomes"]:
        part = (row["slot"], row["move"], row["target"], int(row["mega"]))
        if not shown or shown[-1] != part:
            shown.append(part)
    if shown != sorted(expected):
        return _fixed(
            "outcomes must hold the rows of exactly the own move parts of the candidates that "
            "have a foe to ask about"
        )
    return None
