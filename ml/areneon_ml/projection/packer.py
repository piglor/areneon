"""The CPU reference packer of the battle-state spec (ADR-030 G4, #1464). Not a simulator.

A point is the pinned engine's ``Battle.toJSON()`` at a step (trimmed or whole: only the keys the
spec reads are read) plus both seats' own channels up to it and both teams as brought. The engine
groups (``battle``, ``field``, ``side``, ``roster``, ``active``) are read from the state by each
field's engine path and read rule; the seat groups are each seat's own bookkeeping, read from the
reducer (and its history facts) run on that seat's channel with that seat's team, which is what the
shared reader, the world facts and the request derive and the parity tests hold equal. The result
is the spec's struct of arrays for one battle (integers only, nested lists in each field's full
shape), with the record's hash: ``{"spec_hash", "fields": {field id: values}}``.

Indices: a side and a seat by player (p1 first); a roster entry by the side's team as brought (the
Team Preview order; the engine holds the brought four after it, each joined to its set by name); an
active slot by its position. Every refusal is a ``PackError`` with a code; nothing is clamped.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from areneon_ml.model.encoder import STAT_NAMES
from areneon_ml.model.history import UNSETTLED
from areneon_ml.reducer import ChannelReducer, ReducerError, Vocabulary
from areneon_ml.reducer.state import STAT_KEYS, BattleState, PokemonState, SideState
from areneon_ml.reducer.vocab import PROTOCOL_VOLATILES

from .spec import SEATS, ProjectionError, SpecIndex

PROTOCOL = tuple(sorted(PROTOCOL_VOLATILES))
NUMBER_BITS = 15  # volatile_numbered holds numbers 0 to 14 (int16)
PROTOCOL_NUMBERS = 6  # protocol_volatiles holds numbers 0 to 5 per word
DERIVED_NUMBERS = 11  # volatile_derived: 1 + a number 0 to 10, then 12 + a stat index
DERIVED_STAT = 12
PROTOCOL_DERIVED_BITS = 5
GENDER = {None: 0, "M": 1, "F": 2, "N": 3}
REQUEST_KIND = {None: 0, "wait": 0, "preview": 1, "move": 2, "forceSwitch": 3}
ATTACKER = {False: 0, None: 1, True: 2}
_REF = re.compile(r"^\[Pokemon:(p[12])([a-z])\]$")
_KIND_REF = re.compile(r"^\[[A-Za-z]+:(.*)\]$")
_NON_ID = re.compile(r"[^a-z0-9]")
_NUMBERED = re.compile(r"^(.*\D)(\d+)$")


class PackError(ProjectionError):
    """A point the packer refuses, with a code (``phantom``, ``reducer_refused``, ...)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def to_id(value: Any) -> str | None:
    """An engine value as an id: none for an empty or missing value; ``[Kind:name]`` unwrapped."""
    if value is None or value is False or value == "":
        return None
    text = str(value)
    found = _KIND_REF.match(text)
    if found is not None:
        text = found.group(1)
    return _NON_ID.sub("", text.lower()) or None


def _name(mon: Mapping[str, Any]) -> str:
    held = mon.get("set") or {}
    return str(held.get("name") or held.get("species") or mon.get("name") or "")


def _set_name(pokemon_set: Mapping[str, Any]) -> str:
    return str(pokemon_set.get("name") or pokemon_set.get("species") or "")


def _filled(shape: Sequence[int], value: int) -> Any:
    if not shape:
        return value
    return [_filled(shape[1:], value) for _ in range(shape[0])]


# ---- the seat bookkeeping codecs ----------------------------------------------------------------


def pack_volatiles(index: SpecIndex, vocab: Vocabulary, raw: set[str]) -> dict[str, Any]:
    """The seat's raw volatile ids of one Pokemon as the four v2 fields (see their descriptions)."""
    del vocab  # the ids are the record's (the table's union); the reducer already checked each
    ids = index.ids("VOLATILE")
    where = {name: i for i, name in enumerate(ids)}
    shown = [0] * len(ids)
    numbered = [0] * len(ids)
    stats = [0] * len(ids)
    protocol = 0
    for name in sorted(raw):
        found = _NUMBERED.match(name)
        if name in where:
            shown[where[name]] = 1
        elif name in PROTOCOL:
            protocol |= 1 << PROTOCOL.index(name)
        elif found is not None and found.group(1) in where:
            number = int(found.group(2))
            if number >= NUMBER_BITS:
                raise PackError("volatile_number", name)
            numbered[where[found.group(1)]] |= 1 << number
        elif found is not None and found.group(1) in PROTOCOL:
            number = int(found.group(2))
            if number >= PROTOCOL_NUMBERS:
                raise PackError("volatile_number", name)
            word = PROTOCOL.index(found.group(1))
            protocol |= 1 << (len(PROTOCOL) + PROTOCOL_NUMBERS * word + number)
        else:
            stat = next(
                (s for s in STAT_NAMES if name.endswith(s) and name[: -len(s)] in where), None
            )
            if stat is None:
                raise PackError("unknown_volatile", name)
            stats[where[name[: -len(stat)]]] |= 1 << STAT_NAMES.index(stat)
    return {"shown": shown, "numbered": numbered, "stat": stats, "protocol": protocol}


def unpack_volatiles(index: SpecIndex, codes: Mapping[str, Any]) -> set[str]:
    """The raw volatile ids back from the four fields."""
    ids = index.ids("VOLATILE")
    out: set[str] = set()
    for i, name in enumerate(ids):
        if codes["shown"][i]:
            out.add(name)
        out.update(f"{name}{n}" for n in range(NUMBER_BITS) if codes["numbered"][i] >> n & 1)
        out.update(f"{name}{s}" for b, s in enumerate(STAT_NAMES) if codes["stat"][i] >> b & 1)
    protocol = int(codes["protocol"])
    for w, word in enumerate(PROTOCOL):
        if protocol >> w & 1:
            out.add(word)
        base = len(PROTOCOL) + PROTOCOL_NUMBERS * w
        out.update(f"{word}{n}" for n in range(PROTOCOL_NUMBERS) if protocol >> (base + n) & 1)
    return out


def pack_derived(index: SpecIndex, derived: Mapping[str, str]) -> tuple[list[int], int]:
    """``PokemonState.derived`` as volatile_derived (per VOLATILE id) and protocol_derived."""
    ids = index.ids("VOLATILE")
    where = {name: i for i, name in enumerate(ids)}
    per = [0] * len(ids)
    protocol = 0
    for base, suffix in sorted(derived.items()):
        if base in PROTOCOL and suffix.isdigit() and int(suffix) < 2**PROTOCOL_DERIVED_BITS - 1:
            protocol |= (1 + int(suffix)) << (PROTOCOL_DERIVED_BITS * PROTOCOL.index(base))
        elif base in where and suffix.isdigit() and int(suffix) < DERIVED_NUMBERS:
            per[where[base]] = 1 + int(suffix)
        elif base in where and suffix in STAT_NAMES:
            per[where[base]] = DERIVED_STAT + STAT_NAMES.index(suffix)
        else:
            raise PackError("unknown_derived", f"{base}{suffix}")
    return per, protocol


def unpack_derived(index: SpecIndex, per: Sequence[int], protocol: int) -> dict[str, str]:
    ids = index.ids("VOLATILE")
    out: dict[str, str] = {}
    for i, code in enumerate(per):
        if code >= DERIVED_STAT:
            out[ids[i]] = STAT_NAMES[code - DERIVED_STAT]
        elif code:
            out[ids[i]] = str(code - 1)
    mask = 2**PROTOCOL_DERIVED_BITS - 1
    for w, word in enumerate(PROTOCOL):
        code = protocol >> (PROTOCOL_DERIVED_BITS * w) & mask
        if code:
            out[word] = str(code - 1)
    return out


def pack_reveal(index: SpecIndex, moves: Sequence[str]) -> tuple[list[int], int, int]:
    """A reveal list as its MOVE slots, the fifth entry and where it stands (always last here)."""
    slots = index.size("MOVE")
    if len(moves) > slots + 1:
        raise PackError("reveal_overflow", str(len(moves)))
    field = "seat_roster.revealed_moves"
    head = [index.encode_id(field, m) for m in moves[:slots]]
    head += [0] * (slots - len(head))
    if len(moves) <= slots:
        return head, 0, 0
    return head, index.encode_id("seat_roster.revealed_extra", moves[slots]), slots + 1


def unpack_reveal(index: SpecIndex, slots: Sequence[int], extra: int, at: int) -> list[str]:
    moves = [index.decode_id("seat_roster.revealed_moves", s) for s in slots]
    out = [m for m in moves if m is not None]
    if at:
        decoded = index.decode_id("seat_roster.revealed_extra", extra)
        out.insert(at - 1, decoded if decoded is not None else "")
    return out


# ---- the engine groups ------------------------------------------------------------------------


class _Engine:
    """The engine state of one point, joined to the roster entries."""

    def __init__(
        self,
        state: Mapping[str, Any],
        teams: Mapping[str, Sequence[Mapping[str, Any]]],
        vocab: Vocabulary,
    ) -> None:
        self.state = state
        self.vocab = vocab
        self.sides: list[Mapping[str, Any]] = []
        self.entries: list[list[Mapping[str, Any] | None]] = []  # per side, per roster entry
        self.index_of: list[dict[int, int]] = []  # per side: engine index -> roster entry
        for seat in SEATS:
            side = next((s for s in state["sides"] if s.get("id") == seat), None)
            if side is None:
                raise PackError("state_side", seat)
            names = [_set_name(s) for s in teams[seat]]
            entries: list[Mapping[str, Any] | None] = [None] * len(names)
            where: dict[int, int] = {}
            for i, mon in enumerate(side.get("pokemon", [])):
                if _name(mon) not in names:
                    raise PackError("unmatched_pokemon", seat)
                r = names.index(_name(mon))
                entries[r] = mon
                where[i] = r
            self.sides.append(side)
            self.entries.append(entries)
            self.index_of.append(where)

    def is_mega(self, species: str | None) -> bool:
        """A Mega forme: a forme of its base species whose forme name starts with Mega (the
        engine-data table's base species; the forme part of the id)."""
        if species is None:
            return False
        try:
            base = self.vocab.base_species(species)
        except ReducerError:
            return False
        return (
            species != base and species.startswith(base) and species[len(base) :].startswith("mega")
        )

    def ref(self, value: Any) -> tuple[int, int] | None:
        """A Pokemon reference as (side, engine index)."""
        found = _REF.match(str(value)) if value else None
        if found is None:
            return None
        return SEATS.index(found.group(1)), ord(found.group(2)) - 97

    def roster_of(self, value: Any) -> int | None:
        place = self.ref(value)
        return None if place is None else self.index_of[place[0]].get(place[1])

    def active(self, side: int, slot: int) -> Mapping[str, Any] | None:
        refs = self.sides[side].get("active") or []
        place = self.ref(refs[slot]) if slot < len(refs) else None
        if place is None:
            return None
        pokemon = self.sides[side].get("pokemon", [])
        return pokemon[place[1]] if place[1] < len(pokemon) else None


def _segments(path: str) -> list[str]:
    return re.findall(r"^[a-z]+|\.[A-Za-z]+|\[\]|\{\}", path)


def _scalar(  # noqa: PLR0911, PLR0912
    index: SpecIndex, field: Mapping[str, Any], value: Any, eng: _Engine
) -> int:
    """One engine value by the field's read rule."""
    read = field["source"]["read"]
    fid = field["id"]
    if read == "present":
        return 0 if value is None else 1
    if read == "bool":
        return 1 if value else 0
    if read == "id":
        if fid == "roster.status" and value == "fnt":  # the engine's faint marker is roster.fainted
            return 0
        return index.encode_id(fid, to_id(value))
    if read == "length":
        return len(value) if isinstance(value, list) else 0
    if read == "details_level":
        if not value:
            return 0
        found = re.search(r", L(\d+)", str(value))
        return int(found.group(1)) if found else 100
    if read == "details_shiny":
        return 1 if ", shiny" in str(value or "") else 0
    if read == "species_mega":
        return 1 if eng.is_mega(to_id(value)) else 0
    if read == "pokemon_ref":
        entry = eng.roster_of(value)
        if fid == "active.roster_index":
            return -1 if entry is None else entry
        return 0 if entry is None else 1 + entry
    if fid in ("roster.move_last_turn_result", "roster.move_this_turn_result"):
        return 0 if value is None else (2 if value is True else 1 if value is False else 0)
    if fid == "roster.gender":
        return GENDER.get(value or None, 0)
    if value is None or value == "" or value is False:
        return 0
    if value is True:
        return 1
    return int(value)


def _leaf(
    index: SpecIndex, field: Mapping[str, Any], node: Any, extras: list[str], eng: _Engine
) -> Any:
    """A value with the extra dimensions the path did not walk (a dict or list read whole)."""
    if not extras:
        return _scalar(index, field, node, eng)
    symbol = extras[0]
    if symbol == "BOOST":
        held = node if isinstance(node, Mapping) else {}
        return [int(held.get(k, 0)) for k in STAT_KEYS]
    if symbol == "STAT":
        held = node if isinstance(node, Mapping) else {}
        return [int(held.get(k, 0)) for k in STAT_NAMES]
    if symbol == "TYPE":
        types = {to_id(t) for t in node or [] if t != "???"}
        return [1 if t in types else 0 for t in index.ids("TYPE")]
    if symbol == "PRNG":
        text = str(node or "").split(",")[-1]
        words = [int(text[i : i + 8] or "0", 16) for i in range(0, 8 * index.size("PRNG"), 8)]
        return [w - (1 << 32) if w >= 1 << 31 else w for w in words]
    if extras == ["SIDE", "ACTIVE"]:  # roster.attacked_by: the sources that hit it this turn
        out = _filled([index.size("SIDE"), index.size("ACTIVE")], 0)
        for hit in node or []:
            place = eng.ref(hit.get("source")) if isinstance(hit, Mapping) else None
            if place is not None and hit.get("thisTurn") and place[1] < index.size("ACTIVE"):
                out[place[0]][place[1]] = 1
        return out
    raise ProjectionError(f"{field['id']}: no rule for the dimension {symbol}")


def _walk(  # noqa: PLR0913, PLR0917
    index: SpecIndex,
    field: Mapping[str, Any],
    node: Any,
    segs: list[str],
    extras: list[str],
    eng: _Engine,
) -> Any:
    if not segs:
        return _leaf(index, field, node, extras, eng)
    seg, rest = segs[0], segs[1:]
    if seg == "[]":
        items = node if isinstance(node, list) else []
        size = index.size(extras[0])
        return [
            _walk(index, field, items[i] if i < len(items) else None, rest, extras[1:], eng)
            for i in range(size)
        ]
    if seg == "{}":
        held = node if isinstance(node, Mapping) else {}
        return [
            _walk(index, field, held.get(name), rest, extras[1:], eng)
            for name in index.ids(extras[0])
        ]
    inner = node.get(seg[1:]) if isinstance(node, Mapping) else None
    return _walk(index, field, inner, rest, extras, eng)


def _engine_field(index: SpecIndex, field: Mapping[str, Any], eng: _Engine) -> Any:
    segs = _segments(field["source"]["path"])
    root, rest = segs[0], segs[1:]
    extras = list(field["extra"])
    group = field["group"]
    if group == "battle":
        return _walk(index, field, eng.state, rest, extras, eng)
    if group == "field":
        return _walk(index, field, eng.state.get("field"), rest, extras, eng)
    if group == "side":
        return [_walk(index, field, side, rest, extras, eng) for side in eng.sides]
    if group == "roster":
        return [
            [
                _walk(index, field, mon, rest, extras, eng)
                for mon in _padded(entries, index.size("ROSTER"))
            ]
            for entries in eng.entries
        ]
    out = []  # active: the slot's Pokemon, or the side with its first list indexed by the slot
    for s, side in enumerate(eng.sides):
        row = []
        for a in range(index.size("ACTIVE")):
            if root == "active":
                row.append(_walk(index, field, eng.active(s, a), rest, extras, eng))
            else:
                first = rest.index("[]")
                node: Any = side
                for seg in rest[:first]:
                    node = node.get(seg[1:]) if isinstance(node, Mapping) else None
                items = node if isinstance(node, list) else []
                row.append(
                    _walk(
                        index,
                        field,
                        items[a] if a < len(items) else None,
                        rest[first + 1 :],
                        extras,
                        eng,
                    )
                )
        out.append(row)
    return out


def _padded(entries: list[Any], size: int) -> list[Any]:
    if len(entries) > size:
        raise PackError("roster_overflow", str(len(entries)))
    return [*entries, *([None] * (size - len(entries)))]


# ---- the seat groups --------------------------------------------------------------------------


def _letters(side: SideState) -> list[str]:
    return sorted(side.slots)


def _letter_of(side: SideState, mon: PokemonState) -> str | None:
    for letter in _letters(side):
        slot = side.slots[letter]
        if slot is not None and side.roster[slot] is mon:
            return letter
    return None


def _seat(
    index: SpecIndex, vocab: Vocabulary, state: BattleState, out: dict[str, Any], si: int
) -> None:
    """Write seat ``si``'s bookkeeping (its reducer state) into ``out``."""
    put = _Writer(index, out, si)
    facts: dict[str, Any] = (
        state.history.facts() if state.history is not None else {"refused": "none"}
    )
    known = "refused" not in facts
    turn = state.turn
    put.field("seat_field.started", 1 if state.started else 0)
    put.field("seat_field.request_kind", REQUEST_KIND.get(state.request_kind, 0))
    put.field("seat_field.history_available", 1 if known else 0)
    put.field(
        "seat_field.history_last_move",
        index.encode_id(
            "seat_field.history_last_move", facts.get("field_last_move") if known else None
        ),
    )
    put.field(
        "seat_field.weather_elapsed",
        -1 if state.weather_started is None else turn - state.weather_started,
    )
    put.field(
        "seat_field.terrain_elapsed",
        -1 if state.terrain_started is None else turn - state.terrain_started,
    )
    put.field(
        "seat_field.pseudo_elapsed",
        [
            turn - state.field_started[k] if k in state.field_started else -1
            for k in index.ids("PSEUDO_WEATHER")
        ],
    )
    for half, side in (("own", state.own), ("opponent", state.opponent)):
        d = si if half == "own" else 1 - si
        if any(mon.phantom for mon in side.roster):
            raise PackError("phantom", half)
        side_facts: Mapping[str, Any] | None = facts.get(half) if known else None
        put.side(d, "seat_side.team_size", side.teamsize or 0)
        put.side(
            d,
            "seat_side.condition_elapsed",
            [
                turn - side.condition_started[k] if k in side.condition_started else -1
                for k in index.ids("SIDE_CONDITION")
            ],
        )
        for r, mon in enumerate(_padded(list(side.roster), index.size("ROSTER"))):
            if mon is not None:
                _entry(index, put, d, r, mon, side, side_facts, half == "own")
        for a, letter in enumerate(_letters(side)):
            slot = side.slots[letter]
            put.active(d, a, "seat_active.shown_roster_index", -1 if slot is None else slot)
            pending = (side_facts or {}).get("pending_slot", {}).get(letter, {})
            put.active(
                d,
                a,
                "seat_active.slot_pending_remaining",
                [
                    int(pending[k]["remaining"]) if k in pending else 0
                    for k in index.ids("SLOT_CONDITION")
                ],
            )
            if slot is None:
                continue
            mon = side.roster[slot]
            codes = pack_volatiles(index, vocab, set(mon.volatiles))
            put.active(d, a, "seat_active.volatile_shown", codes["shown"])
            put.active(d, a, "seat_active.volatile_numbered", codes["numbered"])
            put.active(d, a, "seat_active.volatile_stat_raw", codes["stat"])
            put.active(d, a, "seat_active.protocol_volatiles", codes["protocol"])
            per, protocol = pack_derived(index, mon.derived)
            put.active(d, a, "seat_active.volatile_derived", per)
            put.active(d, a, "seat_active.protocol_derived", protocol)
            active = (side_facts or {}).get("active", {}).get(letter)
            if not active or active.get("unknown") or active.get("name") != mon.name:
                continue
            _history_active(index, put, d, a, active, mon, half == "own")


def _history_active(  # noqa: PLR0913, PLR0917
    index: SpecIndex,
    put: _Writer,
    d: int,
    a: int,
    active: Mapping[str, Any],
    mon: PokemonState,
    own: bool,
) -> None:
    put.active(d, a, "seat_active.move_actions", int(active["move_actions"]))
    put.active(d, a, "seat_active.stall_counter", int(active["stall_counter"] or 0))
    put.active(d, a, "seat_active.times_attacked", int(active["times_attacked"]))
    put.active(d, a, "seat_active.hurt_this_turn", 1 if active["hurt_this_turn"] else 0)
    attackers = active["attackers_this_turn"]
    grid = _filled([index.size("SIDE"), index.size("ACTIVE")], 1 if attackers is None else 0)
    for name in attackers or ():
        side, _, letter = str(name).partition(":")
        place = ord(letter) - 97 if len(letter) == 1 else -1
        if side not in ("own", "opponent") or not 0 <= place < index.size("ACTIVE"):
            raise PackError("history_attacker", str(name))
        grid[("own", "opponent").index(side)][place] = 2
    put.active(d, a, "seat_active.attackers", grid)
    put.active(d, a, "seat_active.last_move_failed", ATTACKER[active["last_move_failed"]])
    put.active(d, a, "seat_active.stats_raised", 1 if active["stats_raised_this_turn"] else 0)
    put.active(d, a, "seat_active.stats_lowered", 1 if active["stats_lowered_this_turn"] else 0)
    used = active["moves_used"]
    shown = mon.moves_own if own and mon.moves_own is not None else []
    put.active(
        d,
        a,
        "seat_active.own_moves_used",
        [
            1 if used is not None and i < len(shown) and shown[i] in used else 0
            for i in range(index.size("MOVE"))
        ],
    )
    started = active["volatile_started"]
    put.active(d, a, "seat_active.volatile_started_count", len(started))
    turns = [int(v["started_turn"]) for v in started.values()]
    put.active(d, a, "seat_active.volatile_oldest_started", min(turns) if turns else -1)
    ids = index.ids("VOLATILE")
    put.active(
        d,
        a,
        "seat_active.volatile_started",
        [int(started[k]["started_turn"]) if k in started else -1 for k in ids],
    )


def _entry(  # noqa: PLR0913, PLR0915, PLR0917
    index: SpecIndex,
    put: _Writer,
    d: int,
    r: int,
    mon: PokemonState,
    side: SideState,
    facts: Mapping[str, Any] | None,
    own: bool,
) -> None:
    put.roster(d, r, "seat_roster.seen", 1 if mon.hp_num is not None else 0)
    put.roster(
        d,
        r,
        "seat_roster.preview_species",
        index.encode_id("seat_roster.preview_species", mon.base_species or None),
    )
    put.roster(d, r, "seat_roster.preview_level", mon.level or 0)
    put.roster(d, r, "seat_roster.preview_gender", GENDER.get(mon.gender, 0))
    put.roster(d, r, "seat_roster.species_shown", 0 if mon.species is None else 1)
    put.roster(
        d, r, "seat_roster.shown_species", index.encode_id("seat_roster.shown_species", mon.species)
    )
    put.roster(
        d,
        r,
        "seat_roster.temp_species",
        index.encode_id("seat_roster.temp_species", mon.temp_species),
    )
    put.roster(d, r, "seat_roster.hp_shown", -1 if mon.hp_num is None else mon.hp_num)
    put.roster(d, r, "seat_roster.hp_den_shown", mon.hp_den or 0)
    put.roster(
        d, r, "seat_roster.status_shown", index.encode_id("seat_roster.status_shown", mon.status)
    )
    put.roster(d, r, "seat_roster.item_known", 1 if mon.item_known else 0)
    put.roster(d, r, "seat_roster.item_lost", 1 if mon.item_lost else 0)
    put.roster(d, r, "seat_roster.shown_item", index.encode_id("seat_roster.shown_item", mon.item))
    put.roster(
        d, r, "seat_roster.shown_ability", index.encode_id("seat_roster.shown_ability", mon.ability)
    )
    put.roster(d, r, "seat_roster.ability_suppressed", 1 if mon.ability_suppressed else 0)
    put.roster(d, r, "seat_roster.mega_shown", 1 if mon.mega else 0)
    put.roster(
        d,
        r,
        "seat_roster.mega_stone_shown",
        index.encode_id("seat_roster.mega_stone_shown", mon.mega_stone),
    )
    put.roster(d, r, "seat_roster.name_shown", 0 if mon.name is None else 1)
    put.roster(d, r, "seat_roster.phantom", 1 if mon.phantom else 0)
    put.roster(d, r, "seat_roster.brought", {None: 0, False: 1, True: 2}[mon.brought])
    head, extra, at = pack_reveal(index, mon.moves_revealed)
    put.roster(d, r, "seat_roster.revealed_moves", head)
    put.roster(d, r, "seat_roster.revealed_extra", extra)
    put.roster(d, r, "seat_roster.revealed_extra_at", at)
    if own:
        stats = mon.stats or {}
        put.roster(d, r, "seat_roster.own_stats", [int(stats.get(k, 0)) for k in STAT_NAMES])
        moves = list(mon.moves_own or [])
        if len(moves) > index.size("MOVE"):
            raise PackError("own_moves_overflow", str(len(moves)))
        put.roster(
            d,
            r,
            "seat_roster.own_moves",
            [index.encode_id("seat_roster.own_moves", m) for m in moves]
            + [0] * (index.size("MOVE") - len(moves)),
        )
        put.roster(
            d,
            r,
            "seat_roster.own_pp",
            [mon.pp.get(m, -1) for m in moves] + [-1] * (index.size("MOVE") - len(moves)),
        )
        put.roster(d, r, "seat_roster.own_can_mega_evo", 1 if mon.can_mega_evo else 0)
    types = mon.types
    put.roster(d, r, "seat_roster.types_known", 0 if types is None else 1)
    put.roster(d, r, "seat_roster.types_typeless", 1 if types and "???" in types else 0)
    put.roster(
        d,
        r,
        "seat_roster.types_changed",
        [1 if types and t in types else 0 for t in index.ids("TYPE")],
    )
    recent = mon.recent
    put.roster(
        d,
        r,
        "seat_roster.recent_last_move",
        index.encode_id("seat_roster.recent_last_move", recent.last_move),
    )
    put.roster(
        d,
        r,
        "seat_roster.recent_last_move_turn",
        -1 if recent.last_move_turn is None else recent.last_move_turn,
    )
    put.roster(
        d,
        r,
        "seat_roster.recent_last_action_turn",
        -1 if recent.last_action_turn is None else recent.last_action_turn,
    )
    put.roster(d, r, "seat_roster.recent_last_action_order", recent.last_action_order or 0)
    put.roster(d, r, "seat_roster.recent_hp_taken_this_turn", recent.hp_taken)
    put.roster(d, r, "seat_roster.recent_hp_taken_last_turn", recent.hp_taken_last_turn)
    put.roster(d, r, "seat_roster.recent_hp_dealt_this_turn", recent.hp_dealt)
    put.roster(d, r, "seat_roster.recent_hp_dealt_last_turn", recent.hp_dealt_last_turn)
    put.roster(d, r, "seat_roster.recent_protect_streak", recent.protect_streak)
    target = 0
    if recent.last_target is not None:
        place, _, letter = recent.last_target.partition(":")
        if place not in ("own", "opponent") or len(letter) != 1:
            raise PackError("recent_target", recent.last_target)
        target = 1 + ("own", "opponent").index(place) * 3 + ord(letter) - 97
    put.roster(d, r, "seat_roster.recent_last_target", target)
    # The history facts' roster entry and the position's state, as HistoryView reads them.
    entry = None if facts is None or mon.name is None else facts["roster"].get(mon.name)
    put.roster(
        d,
        r,
        "seat_roster.history_roster_state",
        0 if entry is None else 1 if entry["ate_berry"] is None else 2,
    )
    if entry is not None and entry["ate_berry"] is not None:
        put.roster(d, r, "seat_roster.ate_berry", 1 if entry["ate_berry"] else 0)
        item = entry["last_item"]
        code = 1 if item == UNSETTLED else index.encode_id("seat_roster.last_item_used", item)
        put.roster(d, r, "seat_roster.last_item_used", code)
    held_at = _letter_of(side, mon)
    if held_at is None:
        state = 0
    else:
        active = None if facts is None else facts["active"].get(held_at)
        state = 1 if not active or active.get("unknown") or active.get("name") != mon.name else 2
    put.roster(d, r, "seat_roster.history_state", state)


class _Writer:
    """Writes one seat's row of the seat groups, checking each value against its field's range."""

    def __init__(self, index: SpecIndex, out: dict[str, Any], si: int) -> None:
        self.index, self.out, self.si = index, out, si

    def _check(self, field_id: str, value: Any) -> Any:
        field = self.index.field(field_id)
        rule = field["value"]
        if isinstance(value, list):
            for v in value:
                self._check(field_id, v)
            return value
        low, high = (rule["min"], rule["max"]) if rule["kind"] == "range" else (0, 1 << 30)
        if not low <= value <= high:
            raise PackError("out_of_range", field_id)
        return value

    def field(self, field_id: str, value: Any) -> None:
        self.out[field_id][self.si] = self._check(field_id, value)

    def side(self, d: int, field_id: str, value: Any) -> None:
        self.out[field_id][self.si][d] = self._check(field_id, value)

    def roster(self, d: int, r: int, field_id: str, value: Any) -> None:
        self.out[field_id][self.si][d][r] = self._check(field_id, value)

    def active(self, d: int, a: int, field_id: str, value: Any) -> None:
        self.out[field_id][self.si][d][a] = self._check(field_id, value)


def _range_check(index: SpecIndex, field_id: str, value: Any) -> None:
    rule = index.field(field_id)["value"]
    if isinstance(value, list):
        for v in value:
            _range_check(index, field_id, v)
        return
    # 0 is the empty entry of every field (a Pokemon the engine no longer holds, a slot or an
    # effect that is not there); any other value outside the record's range is refused.
    if rule["kind"] == "range" and value != 0 and not rule["min"] <= value <= rule["max"]:
        raise PackError("out_of_range", field_id)


def _defaults(index: SpecIndex) -> dict[str, Any]:
    """Every field at its empty value: 0, and -1 where -1 means none (the record's descriptions)."""
    out: dict[str, Any] = {}
    for field_id, field in index.fields.items():
        empty = -1 if field["value"]["kind"] == "range" and field["value"]["min"] == -1 else 0
        out[field_id] = _filled(index.shape(field_id), empty)
    return out


# ---- packing ----------------------------------------------------------------------------------


def _pack(
    index: SpecIndex, vocab: Vocabulary, point: Mapping[str, Any], states: Sequence[BattleState]
) -> dict[str, Any]:
    eng = _Engine(point["state"], point["teams"], vocab)
    out = _defaults(index)
    for field_id, field in index.fields.items():
        if field["source"]["kind"] == "engine":
            value = _engine_field(index, field, eng)
            _range_check(index, field_id, value)
            out[field_id] = value
    for si, state in enumerate(states):
        try:
            _check_preview_order(vocab, state, point["teams"][SEATS[1 - si]])
        except ReducerError as error:
            raise PackError("preview_order", type(error).__name__) from error
        _seat(index, vocab, state, out, si)
    return {"spec_hash": index.spec_hash, "fields": out}


def _check_preview_order(
    vocab: Vocabulary, state: BattleState, team: Sequence[Mapping[str, Any]]
) -> None:
    """The opponent's roster entries are its team as brought, in order (the |poke| lines are)."""
    for mon, pokemon_set in zip(state.opponent.roster, team, strict=False):
        if mon.base_species != vocab.base_species(vocab.species(str(pokemon_set["species"]))):
            raise PackError("preview_order")


def _reduce(
    seat: str, lines: Sequence[str], team: Sequence[Mapping[str, Any]], vocab: Vocabulary
) -> ChannelReducer:
    reducer = ChannelReducer(seat, team, vocab)
    for line in lines:
        reducer.feed(line)
    return reducer


def _checked(point: Mapping[str, Any], index: SpecIndex) -> None:
    if point.get("format_id") != index.spec["format_id"]:
        raise PackError("format_mismatch")


def pack(index: SpecIndex, vocab: Vocabulary, point: Mapping[str, Any]) -> dict[str, Any]:
    """Pack one point: the engine state and both seats' bookkeeping."""
    _checked(point, index)
    try:
        states = [
            _reduce(seat, point["channels"][seat], point["teams"][seat], vocab).state
            for seat in SEATS
        ]
    except ReducerError as error:
        raise PackError("reducer_refused", type(error).__name__) from error
    return _pack(index, vocab, point, states)


class PackSession:
    """Packs the points of one battle in order, feeding each seat's reducer only its new lines.

    A point whose channel does not extend the lines fed so far restarts that seat's reducer, so
    every result equals ``pack`` of the same point."""

    def __init__(self, index: SpecIndex, vocab: Vocabulary) -> None:
        self.index, self.vocab = index, vocab
        self._reducers: dict[str, tuple[ChannelReducer, list[str], str]] = {}

    def _state(
        self, seat: str, lines: Sequence[str], team: Sequence[Mapping[str, Any]]
    ) -> BattleState:
        key = repr(sorted(_set_name(s) for s in team))
        held = self._reducers.get(seat)
        if held is None or held[2] != key or list(lines[: len(held[1])]) != held[1]:
            held = (ChannelReducer(seat, team, self.vocab), [], key)
        reducer, fed, _ = held
        for line in lines[len(fed) :]:
            reducer.feed(line)
        self._reducers[seat] = (reducer, list(lines), key)
        return reducer.state

    def pack(self, point: Mapping[str, Any]) -> dict[str, Any]:
        _checked(point, self.index)
        try:
            states = [
                self._state(seat, point["channels"][seat], point["teams"][seat]) for seat in SEATS
            ]
        except ReducerError as error:
            self._reducers.clear()
            raise PackError("reducer_refused", type(error).__name__) from error
        return _pack(self.index, self.vocab, point, states)


# ---- the canary's engine planting -------------------------------------------------------------


def _other(value: Any, low: int, high: int) -> int:
    number = int(value or 0)
    return number + 1 if number + 1 <= high else max(low, number - 1)


def plant_engine(  # noqa: PLR0912
    state: Mapping[str, Any], seat: str, moves: Callable[[], str] | None = None
) -> dict[str, Any]:
    """A copy of the engine state with planted values in what ``seat`` may not read: the other
    side's exact hit points, stats, PP, move slots, item, ability, status and its counters,
    volatile durations, counters and hit points, the Mega flag, counters of moves and hits, and
    the PRNG, the Quick Claw roll and every duration of the field and both sides. Public values
    (stages, faints, species, details, the turn, conditions present) are left as they are."""
    planted: dict[str, Any] = copy.deepcopy(dict(state))
    if isinstance(planted.get("prng"), str):
        head, _, words = planted["prng"].rpartition(",")
        planted["prng"] = f"{head},{''.join('7' if c != '7' else '8' for c in words)}"
    planted["quickClawRoll"] = not planted.get("quickClawRoll")
    field = planted.get("field") or {}
    for key in ("weatherState", "terrainState"):
        if isinstance(field.get(key), dict) and "duration" in field[key]:
            field[key]["duration"] = _other(field[key]["duration"], 0, 64)
    for effect in (field.get("pseudoWeather") or {}).values():
        if isinstance(effect, dict) and "duration" in effect:
            effect["duration"] = _other(effect["duration"], 0, 64)
    for side in planted["sides"]:
        for effect in (side.get("sideConditions") or {}).values():
            if isinstance(effect, dict) and "duration" in effect:
                effect["duration"] = _other(effect["duration"], 0, 64)
        for slot in side.get("slotConditions") or []:
            for effect in slot.values():
                if isinstance(effect, dict) and "duration" in effect:
                    effect["duration"] = _other(effect["duration"], 0, 64)
        if side.get("id") == seat:
            continue
        for mon in side.get("pokemon", []):
            if not mon.get("fainted"):
                mon["hp"] = _other(mon.get("hp"), 1, int(mon.get("maxhp") or 1))
            mon["maxhp"] = _other(mon.get("maxhp"), 1, 9999)
            if isinstance(mon.get("storedStats"), dict):
                mon["storedStats"] = {k: _other(v, 1, 9999) for k, v in mon["storedStats"].items()}
            for slot in mon.get("moveSlots") or []:
                slot["pp"] = _other(slot.get("pp"), 0, 64)
                slot["maxpp"] = _other(slot.get("maxpp"), 0, 64)
                slot["disabled"] = not slot.get("disabled")
                slot["used"] = not slot.get("used")
                if moves is not None:
                    slot["id"] = moves()
            mon["item"] = "" if mon.get("item") else "leftovers"
            mon["ability"] = "pressure" if to_id(mon.get("ability")) != "pressure" else "intimidate"
            mon["baseAbility"] = (
                "pressure" if to_id(mon.get("baseAbility")) != "pressure" else "intimidate"
            )
            mon["lastItem"] = "" if mon.get("lastItem") else "sitrusberry"
            if mon.get("status") not in ("fnt",):
                mon["status"] = "brn" if mon.get("status") != "brn" else "par"
            if isinstance(mon.get("statusState"), dict):
                for key in ("time", "startTime", "stage"):
                    mon["statusState"][key] = _other(mon["statusState"].get(key), 0, 64)
            for effect in (mon.get("volatiles") or {}).values():
                if isinstance(effect, dict):
                    for key, high in (
                        ("duration", 64),
                        ("counter", 127),
                        ("hp", 9999),
                        ("layers", 8),
                    ):
                        if key in effect:
                            effect[key] = _other(effect[key], 0, high)
            mon["canMegaEvo"] = not mon.get("canMegaEvo")
            mon["timesAttacked"] = _other(mon.get("timesAttacked"), 0, 127)
            mon["activeMoveActions"] = _other(mon.get("activeMoveActions"), 0, 127)
            mon["previouslySwitchedIn"] = _other(mon.get("previouslySwitchedIn"), 0, 127)
            mon["truantTurn"] = not mon.get("truantTurn")
            mon["usedItemThisTurn"] = not mon.get("usedItemThisTurn")
            mon["ateBerry"] = not mon.get("ateBerry")
    return planted
