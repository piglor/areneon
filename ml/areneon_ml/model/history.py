"""The ``history_facts`` encoder option (ADR-022 erratum E-22.5.4, slice H5).

The Pokémon and field tokens also read the history facts of the reduced state
(``BattleState.history``, ``reducer/history.py``: the battle history the pinned engine reads from
fields of its own, derived from the player's own channel). The facts are position-keyed like the
engine's own state, so a Pokémon in play reads the facts of its position and a Pokémon on the bench
only its per-Pokémon facts (berry eaten, last item).

Unknowns are never filled in. A fact the channel cannot settle for a Pokémon (an Illusion until its
``|replace|``, a hit whose source is not shown, a roster entry the tracker cannot attribute) is its
own value: the Pokémon's ``history_state`` category is ``UNKNOWN``, a fact that is unknown alone has
its own flag, and every number of an unknown fact is zero. A tracker that refused a line (a typed
refusal of ``reducer/history.py``) leaves its facts unavailable for the rest of the channel, which
the encoder reads as every fact unknown (the field's ``history_available`` category says so).

Divisors that are not config fields (the count scale, the stall counter's maximum from the engine
data) are hashed into ``encoder_version`` by the encoder; counts of turns use ``turn_scale``.
No game rule is written here: the stall counter's maximum and the slot condition ids are the
hashed ``history-facts`` engine data the reducer already holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from areneon_ml.model.gamedata import NONE, RESERVED, UNKNOWN, GameData

SIDES = ("own", "opponent")
LETTERS = 3  # position letters a to c (the widest gametype the reducer reads)
COUNT_SCALE = 8.0  # hits taken and volatiles held, over a fixed divisor
STATE_CATEGORIES = RESERVED + 1  # history_state: none (not in play), unknown, known
KNOWN = RESERVED
UNSETTLED = "unsettled"  # the tracker's last_item of an item used up but not yet named


class HistoryError(ValueError):
    """The facts cannot be encoded (the state carries none, or a fact is of a shape not known)."""


def slot_ids(game: GameData) -> tuple[str, ...]:
    """The ids of the pending slot conditions (the engine data's ``slot_conditions``), sorted."""
    data = game.vocabulary.history_data
    if data is None:
        raise HistoryError("history_facts needs the history facts data of this pin")
    return tuple(sorted(str(entry["id"]) for entry in data["slot_conditions"]))


def stall_maximum(game: GameData) -> float:
    """The stall counter's maximum from the engine data: the divisor of a counter."""
    data = game.vocabulary.history_data
    if data is None:
        raise HistoryError("history_facts needs the history facts data of this pin")
    return float(data["stall_counter"]["max"])


def mon_cat_columns() -> tuple[tuple[str, str], ...]:
    """The categories of a Pokémon token: whether its position's facts are known, whether its
    per-Pokémon facts are, and the item it last used up."""
    return (
        ("history_state", "history_state"),
        ("history_roster", "history_state"),
        ("history_last_item", "item"),
    )


def mon_num_columns(max_moves: int) -> tuple[str, ...]:
    """The numbers of a Pokémon token (zero while the facts are unknown or it is out of play)."""
    return (
        "hist_move_actions",
        "hist_stall_active",
        "hist_stall_counter",
        "hist_times_attacked",
        "hist_hurt_this_turn",
        *(f"hist_attacked_by_{side}_{chr(97 + i)}" for side in SIDES for i in range(LETTERS)),
        "hist_attackers_unknown",
        "hist_last_move_failed",
        "hist_last_move_failed_unknown",
        "hist_stats_raised",
        "hist_stats_lowered",
        "hist_moves_used_known",
        *(f"hist_used_{i + 1}" for i in range(max_moves)),
        "hist_volatile_count",
        "hist_volatile_oldest_age",
        "hist_ate_berry",
    )


def field_cat_columns() -> tuple[tuple[str, str], ...]:
    """The categories of the field token: whether the facts are available and the last move."""
    return (("history_available", "history_state"), ("history_last_move", "move"))


def field_num_columns(game: GameData) -> tuple[str, ...]:
    """The numbers of the field token: each side's faints and the pending slot effects' remaining
    turns, one column per (side, position letter, slot condition id)."""
    return (
        *(f"hist_total_fainted_{side}" for side in SIDES),
        *(
            f"hist_pending_{side}_{chr(97 + i)}_{cid}"
            for side in SIDES
            for i in range(LETTERS)
            for cid in slot_ids(game)
        ),
    )


class HistoryView:
    """The history facts of one reduced state, read for the tokens (one per encode)."""

    def __init__(
        self,
        facts: Mapping[str, Any] | None,
        game: GameData,
        *,
        turn_scale: float,
        max_moves: int,
        max_team: int,
    ) -> None:
        if facts is None:
            raise HistoryError("history_facts needs the history facts of the reduced state")
        self.game = game
        self.turn_scale = turn_scale
        self.max_moves = max_moves
        self.max_team = max_team
        self.refused = "refused" in facts
        self.facts = facts
        self.turn = int(facts.get("turn", 0))
        self.stall_max = stall_maximum(game)
        self.slot_ids = slot_ids(game)

    # -- tokens ---------------------------------------------------------------------------------

    def mon(
        self, side_key: str, letters: Mapping[str, int | None], roster: list[Any], mon: Any
    ) -> tuple[list[int], list[float]]:
        """The categories and numbers of one Pokémon token (``side_key`` is ``own`` or
        ``opponent``; ``letters`` the side's position letter to roster index map)."""
        zero = [0.0] * len(mon_num_columns(self.max_moves))
        if self.refused:
            return [UNKNOWN, UNKNOWN, UNKNOWN], zero
        side = self.facts[side_key]
        entry = side["roster"].get(mon.name)
        ate = None if entry is None else entry["ate_berry"]
        roster_state = NONE if entry is None else (UNKNOWN if ate is None else KNOWN)
        last_item = self._last_item(entry)
        letter = next(
            (
                k
                for k, index in sorted(letters.items())
                if index is not None and roster[index] is mon
            ),
            None,
        )
        if letter is None:  # on the bench or fainted: no position, no per-position facts
            return [NONE, roster_state, last_item], self._with_ate(zero, ate)
        active = side["active"].get(letter)
        if active is None or active.get("unknown") or active.get("name") != mon.name:
            return [UNKNOWN, roster_state, last_item], self._with_ate(zero, ate)
        return [KNOWN, roster_state, last_item], self._with_ate(self._active(active, mon), ate)

    def _last_item(self, entry: Mapping[str, Any] | None) -> int:
        if entry is None or entry["ate_berry"] is None:
            return NONE if entry is None else UNKNOWN
        item = entry["last_item"]
        if item is None:
            return NONE
        if item == UNSETTLED:
            return UNKNOWN
        return self.game.items.lookup_or_unknown(item)

    def _with_ate(self, nums: list[float], ate: bool | None) -> list[float]:
        out = list(nums)
        out[-1] = 1.0 if ate else 0.0
        return out

    def _active(self, active: Mapping[str, Any], mon: Any) -> list[float]:
        counter = active["stall_counter"]
        attackers = active["attackers_this_turn"]
        by = [0.0] * (len(SIDES) * LETTERS)
        for name in attackers or ():
            side, _, letter = str(name).partition(":")
            place = ord(letter) - 97 if len(letter) == 1 else -1
            if side not in SIDES or not 0 <= place < LETTERS:
                raise HistoryError("a history attacker names no position")
            by[SIDES.index(side) * LETTERS + place] = 1.0
        failed = active["last_move_failed"]
        used = active["moves_used"]
        shown = mon.moves_own if mon.own and mon.moves_own is not None else mon.moves_revealed
        started = active["volatile_started"]
        oldest = max((self.turn - v["started_turn"] for v in started.values()), default=0)
        return [
            active["move_actions"] / self.turn_scale,
            0.0 if counter is None else 1.0,
            0.0 if counter is None else counter / self.stall_max,
            active["times_attacked"] / COUNT_SCALE,
            1.0 if active["hurt_this_turn"] else 0.0,
            *by,
            1.0 if attackers is None else 0.0,
            1.0 if failed else 0.0,
            1.0 if failed is None else 0.0,
            1.0 if active["stats_raised_this_turn"] else 0.0,
            1.0 if active["stats_lowered_this_turn"] else 0.0,
            0.0 if used is None else 1.0,
            *(
                1.0 if used is not None and i < len(shown) and shown[i] in used else 0.0
                for i in range(self.max_moves)
            ),
            len(started) / COUNT_SCALE,
            oldest / self.turn_scale,
            0.0,  # ate_berry, filled by the roster entry
        ]

    # -- the field -------------------------------------------------------------------------------

    def field(self) -> tuple[list[int], list[float]]:
        """The categories and numbers of the field token."""
        width = len(field_num_columns(self.game))
        if self.refused:
            return [UNKNOWN, NONE], [0.0] * width
        last = self.facts["field_last_move"]
        cat = [KNOWN, NONE if last is None else self.game.moves.lookup_or_unknown(last)]
        faints = [self.facts[side]["total_fainted"] / self.max_team for side in SIDES]
        pending: list[float] = []
        for side_key in SIDES:
            slots = self.facts[side_key]["pending_slot"]
            for i in range(LETTERS):
                live = slots.get(chr(97 + i), {})
                pending += [
                    live[cid]["remaining"] / self.turn_scale if cid in live else 0.0
                    for cid in self.slot_ids
                ]
        return cat, [*faints, *pending]
