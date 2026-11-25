"""The reducer-state view the encoder reads, rebuilt from one seat's projection (ADR-030 G4, #1464).

``reduced_view(index, view, team)`` reads nothing but the view (the fields the seat may read) and
the seat's own team (its own information: the names its own requests use), and returns the
``BattleState`` whose encoding the gate holds bit-equal to ``Encoder.encode`` of the seat's query.
Which field gives which reducer value is the record's ``encoder`` and ``reducer`` mapping:

- shared: the turn, weather, terrain and field conditions from the engine fields (public); the
  request kind, ``started`` and the elapsed turns from the seat's bookkeeping;
- each side: its conditions and layers from the engine (public), their elapsed turns and the team
  size from the bookkeeping; the active slots from the engine for the own side (a fainted Pokemon
  leaves its slot, as its faint line does) and from the entry the seat was shown for the other;
- each entry: its stages from the slot it is in (``active.boosts``), its faint and shiny flags from
  the engine (the other side's only where the seat has seen it), everything else from the seat's
  own bookkeeping (what its channel and its own requests showed), as the reducer holds it;
- the history facts as ``HistoryView`` reads them (synthetic names for the other side's entries:
  the facts and the roster only need to agree with each other).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from areneon_ml.model.encoder import STAT_NAMES
from areneon_ml.model.history import UNSETTLED
from areneon_ml.reducer.history import HISTORY_FACTS_FORMAT
from areneon_ml.reducer.state import STAT_KEYS, BattleState, PokemonState, Recent, SideState

from .packer import unpack_derived, unpack_reveal, unpack_volatiles
from .spec import SEATS, ProjectionError, SpecIndex

GENDERS = (None, "M", "F", "N")
REQUEST_KINDS = (None, "preview", "move", "forceSwitch")
SIDES = ("own", "opponent")
KNOWN = 2  # the bookkeeping's "known" (history_state, attackers): 0 no, 1 unknown, 2 known


class _Facts:
    """The history facts of a rebuilt state (``BattleState.history`` needs ``facts()``)."""

    def __init__(self, facts: dict[str, Any]) -> None:
        self._facts = facts

    def facts(self) -> dict[str, Any]:
        return self._facts


class _View:
    def __init__(self, index: SpecIndex, view: Mapping[str, Any]) -> None:
        if view.get("spec_hash") != index.spec_hash:
            raise ProjectionError("the view was projected with another record")
        self.index = index
        self.fields: Mapping[str, Any] = view["fields"]

    def get(self, field_id: str, half: str | None = None) -> Any:
        key = field_id if half is None else f"{field_id}@{half}"
        if key not in self.fields:
            raise ProjectionError(f"the view holds no {key}: the seat may not read it")
        return self.fields[key]

    def id(self, field_id: str, value: int) -> str | None:
        return self.index.decode_id(field_id, value)


def _elapsed(turn: int, value: int) -> int | None:
    return None if value < 0 else turn - value


def _side(
    v: _View, half: str, player: str, team: Sequence[Mapping[str, Any]] | None, turn: int
) -> SideState:
    index = v.index
    side = SideState(player=player)
    side.teamsize = v.get("seat_side.team_size", half) or None
    present = v.get("side.condition_present", half)
    layers = v.get("side.condition_layers", half)
    elapsed = v.get("seat_side.condition_elapsed", half)
    for i, name in enumerate(index.ids("SIDE_CONDITION")):
        if present[i]:
            side.conditions[name] = layers[i] or 1
        started = _elapsed(turn, elapsed[i])
        if started is not None:
            side.condition_started[name] = started
    roster = _roster(v, half, team)
    side.roster = roster
    letters = [chr(97 + a) for a in range(index.size("ACTIVE"))]
    if half == "own":
        slots = v.get("active.roster_index", half)
        fainted = v.get("roster.fainted", half)
        side.slots = {
            letter: (slot if slot >= 0 and slot < len(roster) and not fainted[slot] else None)
            for letter, slot in zip(letters, slots, strict=True)
        }
    else:
        shown = v.get("seat_active.shown_roster_index", half)
        side.slots = {
            letter: (slot if slot >= 0 else None)
            for letter, slot in zip(letters, shown, strict=True)
        }
    _slot_values(v, half, side)
    return side


def _roster(  # noqa: PLR0915
    v: _View, half: str, team: Sequence[Mapping[str, Any]] | None
) -> list[PokemonState]:
    index = v.index
    species = v.get("seat_roster.preview_species", half)
    count = len(team) if team is not None else sum(1 for s in species if s)
    own = half == "own"
    roster: list[PokemonState] = []
    for r in range(count):

        def at(field_id: str, r: int = r) -> Any:
            return v.get(field_id, half)[r]

        name = None
        if own and team is not None:
            name = str(team[r].get("name") or team[r].get("species"))
        elif at("seat_roster.name_shown"):
            name = f"#{r}"
        mon = PokemonState(own=own, name=name)
        mon.base_species = (
            v.id("seat_roster.preview_species", at("seat_roster.preview_species")) or ""
        )
        mon.species = (
            v.id("seat_roster.shown_species", at("seat_roster.shown_species"))
            if at("seat_roster.species_shown")
            else None
        )
        mon.temp_species = v.id("seat_roster.temp_species", at("seat_roster.temp_species"))
        mon.level = at("seat_roster.preview_level") or None
        mon.gender = GENDERS[at("seat_roster.preview_gender")]
        shiny = at("roster.shiny")
        mon.shiny = bool(shiny)
        hp = at("seat_roster.hp_shown")
        mon.hp_num = None if hp < 0 else hp
        mon.hp_den = at("seat_roster.hp_den_shown") or None
        mon.status = v.id("seat_roster.status_shown", at("seat_roster.status_shown"))
        mon.fainted = bool(at("seat_roster.fainted_shown"))
        mon.brought = (None, False, True)[at("seat_roster.brought")]
        mon.item = v.id("seat_roster.shown_item", at("seat_roster.shown_item"))
        mon.item_known = bool(at("seat_roster.item_known"))
        mon.item_lost = bool(at("seat_roster.item_lost"))
        mon.ability = v.id("seat_roster.shown_ability", at("seat_roster.shown_ability"))
        mon.ability_suppressed = bool(at("seat_roster.ability_suppressed"))
        mon.mega = bool(at("seat_roster.mega_shown"))
        mon.mega_stone = v.id("seat_roster.mega_stone_shown", at("seat_roster.mega_stone_shown"))
        mon.moves_revealed = unpack_reveal(
            index,
            at("seat_roster.revealed_moves"),
            at("seat_roster.revealed_extra"),
            at("seat_roster.revealed_extra_at"),
        )
        if at("seat_roster.types_known"):
            types = [
                t
                for t, on in zip(index.ids("TYPE"), at("seat_roster.types_changed"), strict=True)
                if on
            ]
            mon.types = [*types, "???"] if at("seat_roster.types_typeless") else types
        if own:
            stats = at("seat_roster.own_stats")
            mon.stats = (
                {k: s for k, s in zip(STAT_NAMES, stats, strict=True)} if any(stats) else None
            )
            moves = [v.id("seat_roster.own_moves", m) for m in at("seat_roster.own_moves")]
            mon.moves_own = [m for m in moves if m is not None]
            mon.pp = {
                m: p
                for m, p in zip(moves, at("seat_roster.own_pp"), strict=True)
                if m is not None and p >= 0
            }
            mon.can_mega_evo = bool(at("seat_roster.own_can_mega_evo"))
        recent = Recent()
        recent.last_move = v.id("seat_roster.recent_last_move", at("seat_roster.recent_last_move"))
        turn = at("seat_roster.recent_last_move_turn")
        recent.last_move_turn = None if turn < 0 else turn
        acted = at("seat_roster.recent_last_action_turn")
        recent.last_action_turn = None if acted < 0 else acted
        recent.last_action_order = at("seat_roster.recent_last_action_order") or None
        recent.hp_taken = at("seat_roster.recent_hp_taken_this_turn")
        recent.hp_taken_last_turn = at("seat_roster.recent_hp_taken_last_turn")
        recent.hp_dealt = at("seat_roster.recent_hp_dealt_this_turn")
        recent.hp_dealt_last_turn = at("seat_roster.recent_hp_dealt_last_turn")
        recent.protect_streak = at("seat_roster.recent_protect_streak")
        target = at("seat_roster.recent_last_target")
        if target:
            place, letter = divmod(target - 1, 3)
            recent.last_target = f"{SIDES[place]}:{chr(97 + letter)}"
        mon.recent = recent
        roster.append(mon)
    return roster


def _slot_values(v: _View, half: str, side: SideState) -> None:
    """What each slot shows of its Pokemon: stages, volatiles, derived suffixes, active turns."""
    index = v.index
    boosts = v.get("active.boosts", half)
    turns = v.get("active.active_turns", half)
    shown = v.get("seat_active.volatile_shown", half)
    numbered = v.get("seat_active.volatile_numbered", half)
    stats = v.get("seat_active.volatile_stat_raw", half)
    protocol = v.get("seat_active.protocol_volatiles", half)
    derived = v.get("seat_active.volatile_derived", half)
    protocol_derived = v.get("seat_active.protocol_derived", half)
    occupant = v.get("seat_active.shown_roster_index", half)
    for a, letter in enumerate(sorted(side.slots)):
        slot = side.slots[letter]
        if slot is not None:
            mon = side.roster[slot]
            mon.active_slot = letter
            mon.boosts = {k: b for k, b in zip(STAT_KEYS, boosts[a], strict=True)}
            mon.recent.active_turns = turns[a]
        # The seat's bookkeeping of a slot is of the entry it holds there (its reducer's slots).
        held = occupant[a]
        if held < 0 or held >= len(side.roster):
            continue
        mon = side.roster[held]
        codes = {
            "shown": shown[a],
            "numbered": numbered[a],
            "stat": stats[a],
            "protocol": protocol[a],
        }
        mon.volatiles = unpack_volatiles(index, codes)
        mon.derived = unpack_derived(index, derived[a], protocol_derived[a])


def _history(v: _View, state: BattleState) -> dict[str, Any]:
    index = v.index
    if not v.get("seat_field.history_available"):
        return {"refused": "unavailable"}
    facts: dict[str, Any] = {
        "history_facts_format": HISTORY_FACTS_FORMAT,
        "turn": state.turn,
        "field_last_move": v.id(
            "seat_field.history_last_move", v.get("seat_field.history_last_move")
        ),
    }
    for half, side in (("own", state.own), ("opponent", state.opponent)):
        roster_state = v.get("seat_roster.history_roster_state", half)
        ate = v.get("seat_roster.ate_berry", half)
        last = v.get("seat_roster.last_item_used", half)
        entries: dict[str, Any] = {}
        for r, mon in enumerate(side.roster):
            if mon.name is None or not roster_state[r]:
                continue
            if roster_state[r] == 1:
                entries[mon.name] = {"ate_berry": None, "last_item": None}
            else:
                item = UNSETTLED if last[r] == 1 else v.id("seat_roster.last_item_used", last[r])
                entries[mon.name] = {"ate_berry": bool(ate[r]), "last_item": item}
        active: dict[str, Any] = {}
        state_of = v.get("seat_roster.history_state", half)
        for a, letter in enumerate(sorted(side.slots)):
            slot = side.slots[letter]
            if slot is None or state_of[slot] != KNOWN:
                active[letter] = None
                continue
            active[letter] = _active_facts(v, half, a, side.roster[slot])
        pending_rows = v.get("seat_active.slot_pending_remaining", half)
        pending: dict[str, Any] = {}
        for a, letter in enumerate(sorted(side.slots)):
            live = {
                cid: {"remaining": n}
                for cid, n in zip(index.ids("SLOT_CONDITION"), pending_rows[a], strict=True)
                if n > 0
            }
            if live:
                pending[letter] = live
        facts[half] = {
            "total_fainted": v.get("side.total_fainted", half),
            "active": active,
            "pending_slot": pending,
            "roster": entries,
        }
    return facts


def _active_facts(v: _View, half: str, a: int, mon: PokemonState) -> dict[str, Any]:
    def at(field_id: str) -> Any:
        return v.get(field_id, half)[a]

    grid = at("seat_active.attackers")
    attackers = (
        None
        if any(c == 1 for row in grid for c in row)
        else [
            f"{SIDES[s]}:{chr(97 + p)}"
            for s, row in enumerate(grid)
            for p, c in enumerate(row)
            if c == KNOWN
        ]
    )
    used = at("seat_active.own_moves_used")
    shown = mon.moves_own or []
    counter = at("seat_active.stall_counter")
    count = at("seat_active.volatile_started_count")
    oldest = at("seat_active.volatile_oldest_started")
    started = {
        f"#{i}": {"started_turn": oldest if i == 0 else v.get("battle.turn")} for i in range(count)
    }
    return {
        "name": mon.name,
        "unknown": False,
        "move_actions": at("seat_active.move_actions"),
        "stall_counter": counter or None,
        "times_attacked": at("seat_active.times_attacked"),
        "hurt_this_turn": bool(at("seat_active.hurt_this_turn")),
        "attackers_this_turn": attackers,
        "last_move_failed": (False, None, True)[at("seat_active.last_move_failed")],
        "stats_raised_this_turn": bool(at("seat_active.stats_raised")),
        "stats_lowered_this_turn": bool(at("seat_active.stats_lowered")),
        "moves_used": [m for m, u in zip(shown, used, strict=False) if u]
        if half == "own"
        else None,
        "volatile_started": started,
    }


def reduced_view(
    index: SpecIndex, view: Mapping[str, Any], team: Sequence[Mapping[str, Any]]
) -> BattleState:
    """The ``BattleState`` of one seat's view (see the module text)."""
    v = _View(index, view)
    seat = str(view["seat"])
    other = SEATS[1 - SEATS.index(seat)]
    turn = int(v.get("battle.turn"))
    state = BattleState(player=seat)
    state.turn = turn
    state.started = bool(v.get("seat_field.started"))
    state.request_kind = REQUEST_KINDS[v.get("seat_field.request_kind")]
    state.weather = v.id("field.weather", v.get("field.weather"))
    state.terrain = v.id("field.terrain", v.get("field.terrain"))
    state.field_conditions = {
        n
        for n, on in zip(index.ids("PSEUDO_WEATHER"), v.get("field.pseudo_present"), strict=True)
        if on
    }
    state.weather_started = _elapsed(turn, v.get("seat_field.weather_elapsed"))
    state.terrain_started = _elapsed(turn, v.get("seat_field.terrain_elapsed"))
    for name, e in zip(
        index.ids("PSEUDO_WEATHER"), v.get("seat_field.pseudo_elapsed"), strict=True
    ):
        started = _elapsed(turn, e)
        if started is not None:
            state.field_started[name] = started
    state.own = _side(v, "own", seat, team, turn)
    state.opponent = _side(v, "opponent", other, None, turn)
    state.history = _Facts(_history(v, state))
    return state
