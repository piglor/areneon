"""The history facts of the reduced state (ADR-022 erratum E-22.5.2, slice H1).

The Python mirror of `packages/public-state/src/history-facts.ts` (`history-facts/1`): the battle
history the pinned engine reads from fields of its own (`activeMoveActions`, `stall`'s counter,
`timesAttacked`, `hurtThisTurn`, `attackedBy`, `moveLastTurnResult`, `statsRaisedThisTurn`,
`ateBerry`, `lastItem`, `moveSlots[].used`, `side.totalFainted`, `battle.lastMove`, the pending
slot conditions) derived from ONE player's own channel, nothing else. The two readers are held
equal by `history-facts-python-parity.test.ts` and each against the live engine's fields by
`history-facts-parity.test.ts`; see the TS module for the definition of every fact.

The facts are position-keyed like the engine's own state (a Pokemon's counters belong to its
stint in a position), and an Illusion's `|replace|` makes the Pokemon at that position `unknown`
until it leaves play. A line the tracker cannot place makes its facts a typed refusal (a code, never
the line's text); facts from before stand in the TS module, here the state carries the refusal.
No game rule or list is written here: the data comes from `Vocabulary` and the hashed data file
`data/history-facts.<format>.json` (`export-history-facts-data.ts`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

HISTORY_FACTS_FORMAT = "history-facts/1"
UNSETTLED = "unsettled"

_TAG = re.compile(r"^\[([A-Za-z_]+)\]\s?(.*)$")
_IDENT = re.compile(r"^(p[12])([a-c])?: (.+)$")
_NON_ID = re.compile(r"[^a-z0-9]+")
_CALLED = re.compile(r"^(move|ability|item):")
_PREFIX = re.compile(r"^[a-z ]+: ")

_BOUNDARY = frozenset(
    {
        "move",
        "cant",
        "switch",
        "drag",
        "faint",
        "turn",
        "upkeep",
        "",
        "replace",
        "swap",
        "win",
        "tie",
    }
)
_ACTION_END = frozenset(
    {"move", "cant", "switch", "drag", "turn", "upkeep", "", "replace", "swap", "win", "tie"}
)
_EFFECT = frozenset(
    {
        "-damage", "-heal", "-boost", "-unboost", "-status", "-start", "-sidestart", "-fieldstart",
        "-weather", "-singleturn", "-singlemove", "-curestatus", "-setboost", "-item", "-ability",
        "-clearallboost", "-clearboost", "-clearnegativeboost", "-clearpositiveboost",
        "-copyboost", "-swapboost", "-invertboost", "-cureteam", "-sethp", "-transform",
        "-formechange", "-prepare", "-mustrecharge", "-sideend", "-fieldend", "-end", "-enditem",
    }
)  # fmt: skip


def _to_id(text: str) -> str:
    return _NON_ID.sub("", text.lower())


def _effect_id(text: str) -> str:
    return _to_id(_PREFIX.sub("", text, count=1))


class _Refusal(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _Line:
    kind: str
    args: list[str]
    tags: dict[str, str]


def _split(raw: str) -> _Line:
    if not raw.startswith("|"):
        raise _Refusal("malformed_line")
    parts = raw[1:].split("|")
    if parts[0] == "request":
        return _Line("request", ["|".join(parts[1:])], {})
    args: list[str] = []
    tags: dict[str, str] = {}
    for part in parts[1:]:
        match = _TAG.match(part)
        if match is None:
            args.append(part)
        else:
            tags[match.group(1)] = match.group(2)
    return _Line(parts[0], args, tags)


@dataclass
class _Occupant:
    name: str
    unknown: bool = False
    move_actions: int = 0
    stall_count: int = 0
    stall_turn: int = -1
    times_attacked: int = 0
    hurt: bool = False
    attackers: set[str] = field(default_factory=set)
    attackers_unsettled: bool = False
    result: str = "none"  # "ok" | "failed" | "unknown" | "none"
    last_failed: bool | None = False
    raised: bool = False
    lowered: bool = False
    used: set[str] = field(default_factory=set)
    volatiles: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _Slot:
    id: str
    started_turn: int
    after_residual: bool
    source: _Occupant
    source_side: str


@dataclass
class _Roster:
    ate: bool = False
    last_item: str = ""
    unknown: bool = False


@dataclass
class _Side:
    active: dict[str, _Occupant | None] = field(default_factory=dict)
    total_fainted: int = 0
    slots: dict[str, dict[str, _Slot]] = field(default_factory=dict)
    roster: dict[str, _Roster] = field(default_factory=dict)


@dataclass
class _Action:
    side: str
    letter: str
    occupant: _Occupant
    target: tuple[str, str] | None
    hit: bool = False
    misses: int = 0
    self_effect: bool = False
    other: bool = False
    delayed: bool = False
    nested: bool = False
    self_target: bool = False
    slot: tuple[str, str, str] | None = None  # side, letter, condition id


class HistoryTracker:
    """The history facts of one player's own channel, fed line by line."""

    def __init__(self, player: str, data: dict[str, Any]) -> None:
        self._player = player
        self._data = data
        self._stall = frozenset(data["stall_moves"])
        self._ally_side = frozenset(data["ally_side_moves"])
        self._action_sources = frozenset(_to_id(s) for s in data["action_sources"])
        self._external = frozenset(_to_id(s) for s in data["external_action_sources"])
        self._restorers = frozenset(_to_id(s) for s in data["last_item_restorers"])
        self._takers = frozenset(_to_id(s) for s in data["last_item_takers"])
        self._berries = frozenset(data["eaten_berries"])
        self._eat_tags = frozenset(data["eat_tags"])
        self._steal_eat = frozenset(data["steal_eat_tags"])
        self._gem = frozenset(data["gem_tags"])
        self._direct_end = frozenset(data["direct_end_items"])
        self._item_use = frozenset(_to_id(s) for s in data["item_use_sources"])
        self._null_cant = frozenset(data["null_cant_reasons"])
        self._breakers = frozenset(data["protect_breaker_moves"])
        self._not_copied = frozenset(data["volatile_not_copied"])
        self._shed_tail = frozenset(data["shed_tail_copies"])
        self._slot_conditions = list(data["slot_conditions"])
        self._sides: dict[str, _Side] = {"p1": _Side(), "p2": _Side()}
        self._request_moves: dict[str, set[str]] = {}
        self._guards: set[str] = set()
        self._last_letter: dict[str, str] = {}
        self._turn = 0
        self._upkeep_at = -1
        self._last_move: str | None = None
        self._action: _Action | None = None
        self._last_cant: _Occupant | None = None
        self._pending: tuple[str, str, str] | None = None
        self._failure: str | None = None
        self._handlers = self._build_handlers()

    # -- feeding -----------------------------------------------------------------

    def feed(self, raw: str) -> None:
        """Reads one line. A refused line keeps the module's own typed refusal (facts() reports
        it); any other exception is a defect and propagates."""
        if self._failure is not None:
            return
        try:
            self._line(_split(raw))
        except _Refusal as error:
            self._failure = error.code

    # -- output ------------------------------------------------------------------

    def facts(self) -> dict[str, Any]:
        """The facts as the TS module's `HistoryFacts`, or `{"refused": code}`."""
        if self._failure is not None:
            return {"refused": self._failure}
        if self._pending is not None:
            return {"refused": "stall_unresolved"}
        opposing = "p2" if self._player == "p1" else "p1"
        return {
            "history_facts_format": HISTORY_FACTS_FORMAT,
            "turn": self._turn,
            "field_last_move": self._last_move,
            "own": self._side_facts(self._player),
            "opponent": self._side_facts(opposing),
        }

    def _counter(self, occupant: _Occupant) -> int | None:
        horizon = self._turn - (0 if self._upkeep_at == self._turn else 1)
        if occupant.stall_count == 0 or occupant.stall_turn < horizon:
            return None
        spec = self._data["stall_counter"]
        counter = spec["start"]
        for _ in range(1, occupant.stall_count):
            if counter < spec["max"]:
                counter *= spec["factor"]
        return int(counter)

    def _remaining(self, slot: _Slot) -> int:
        span = next((c["residual_turns"] for c in self._slot_conditions if c["id"] == slot.id), 0)
        elapsed = (
            self._turn
            - slot.started_turn
            - (1 if slot.after_residual else 0)
            + (1 if self._upkeep_at == self._turn else 0)
        )
        return int(span - elapsed)

    def _side_facts(self, player: str) -> dict[str, Any]:
        side = self._sides[player]
        own = player == self._player
        active: dict[str, Any] = {}
        for letter, occupant in sorted(side.active.items()):
            if occupant is None:
                active[letter] = None
            elif occupant.unknown:
                active[letter] = {"name": occupant.name, "unknown": True}
            else:
                active[letter] = {
                    "name": occupant.name,
                    "unknown": False,
                    "move_actions": occupant.move_actions,
                    "stall_counter": self._counter(occupant),
                    "times_attacked": occupant.times_attacked,
                    "hurt_this_turn": occupant.hurt,
                    "attackers_this_turn": None
                    if occupant.attackers_unsettled
                    else sorted(occupant.attackers),
                    "last_move_failed": occupant.last_failed,
                    "stats_raised_this_turn": occupant.raised,
                    "stats_lowered_this_turn": occupant.lowered,
                    "moves_used": sorted(occupant.used) if own else None,
                    "volatile_started": {
                        key: dict(value) for key, value in sorted(occupant.volatiles.items())
                    },
                }
        pending: dict[str, Any] = {}
        for letter, slots in sorted(side.slots.items()):
            live = {
                key: {
                    "started_turn": slot.started_turn,
                    "after_residual": slot.after_residual,
                    "remaining": self._remaining(slot),
                }
                for key, slot in sorted(slots.items())
                if self._remaining(slot) >= 1
            }
            if live:
                pending[letter] = live
        roster: dict[str, Any] = {}
        for name, entry in sorted(side.roster.items()):
            if entry.unknown:
                roster[name] = {"ate_berry": None, "last_item": None}
            else:
                roster[name] = {
                    "ate_berry": entry.ate,
                    "last_item": None
                    if entry.last_item == ""
                    else UNSETTLED
                    if entry.last_item == "?"
                    else entry.last_item,
                }
        return {
            "total_fainted": side.total_fainted,
            "active": active,
            "pending_slot": pending,
            "roster": roster,
        }

    # -- references --------------------------------------------------------------

    @staticmethod
    def _ident(text: str | None) -> tuple[str, str | None, str]:
        match = None if text is None else _IDENT.match(text)
        if match is None:
            raise _Refusal("unattributable_line")
        return match.group(1), match.group(2), match.group(3)

    def _at(self, text: str | None) -> tuple[str, str, _Occupant] | None:
        match = None if text is None else _IDENT.match(text)
        if match is None or match.group(2) is None:
            return None
        occupant = self._sides[match.group(1)].active.get(match.group(2))
        return None if occupant is None else (match.group(1), match.group(2), occupant)

    def _occupant(self, text: str | None) -> tuple[str, str, _Occupant]:
        found = self._at(text)
        if found is None:
            raise _Refusal("unattributable_line")
        return found

    def _seen(self, side: str, letter: str) -> str:
        return f"{'own' if side == self._player else 'opponent'}:{letter}"

    def _residual_ran(self) -> bool:
        return self._turn in (0, self._upkeep_at)

    # -- the stall outcome -------------------------------------------------------

    def _resolve_stall(self, line: _Line) -> None:
        if line.kind == "-singleturn":
            match = _IDENT.match(line.args[0] if line.args else "")
            shown = _to_id((line.args[1] if len(line.args) > 1 else "").removeprefix("move: "))
            if match is not None and shown in self._ally_side:
                self._guards.add(f"{match.group(1)}:{shown}")
        pending = self._pending
        if pending is None:
            return
        side, letter, move = pending
        mine = bool(line.args) and line.args[0].startswith(f"{side}{letter}:")
        occupant = self._sides[side].active.get(letter)
        if line.kind in ("-singleturn", "-fail") and mine and occupant is not None:
            self._pending = None
            if line.kind == "-fail":
                occupant.stall_count = 0
                occupant.stall_turn = -1
            else:
                self._stall_succeeded(occupant)
        elif line.kind in _BOUNDARY:
            if (
                occupant is not None
                and move in self._ally_side
                and f"{side}:{move}" in self._guards
            ):
                self._pending = None
                self._stall_succeeded(occupant)
            else:
                raise _Refusal("stall_unresolved")

    def _stall_succeeded(self, occupant: _Occupant) -> None:
        occupant.stall_count = (
            occupant.stall_count + 1 if occupant.stall_turn >= self._turn - 1 else 1
        )
        occupant.stall_turn = self._turn

    # -- actions -----------------------------------------------------------------

    def _end_action(self) -> None:
        action = self._action
        if action is None:
            return
        self._action = None
        if action.delayed:
            return
        if action.nested:
            action.occupant.result = "unknown"
        elif action.hit or (
            action.self_target and action.self_effect and action.misses == 0 and not action.other
        ):
            action.occupant.result = "ok"
        elif action.misses > 0 and not action.self_effect and not action.other:
            action.occupant.result = "failed"
        else:
            action.occupant.result = "unknown"
        if action.slot is not None:
            self._start_slot(action.slot, action.occupant, action.side)

    def _start_slot(self, slot: tuple[str, str, str], source: _Occupant, user: str) -> None:
        side, letter, condition = slot
        slots = self._sides[side].slots.setdefault(letter, {})
        known = slots.get(condition)
        if known is None or self._remaining(known) < 1:
            slots[condition] = _Slot(condition, self._turn, self._residual_ran(), source, user)

    def _nested_move(self, line: _Line) -> bool:
        origin = line.tags.get("from")
        return (
            line.kind == "move"
            and origin is not None
            and _CALLED.match(origin) is not None
            and _to_id(origin) not in self._action_sources
        )

    def _line(self, line: _Line) -> None:
        just_cant = self._last_cant
        self._last_cant = None
        self._resolve_stall(line)
        if line.kind in _ACTION_END and self._action is not None and not self._nested_move(line):
            self._end_action()
        if line.kind == "move":
            self._move(line, just_cant)
        else:
            handler = self._handlers.get(line.kind)
            if handler is not None:
                handler(line)
        if self._action is not None and line.kind in _EFFECT and line.kind != "-damage":
            self._effect(line)

    def _build_handlers(self) -> dict[str, Callable[[_Line], None]]:
        return {
            "request": lambda line: self._request(_arg(line, 0)),
            "turn": self._turn_line,
            "upkeep": lambda line: setattr(self, "_upkeep_at", self._turn),
            "switch": lambda line: self._switch_in(_arg(line, 0), line),
            "drag": lambda line: self._switch_in(_arg(line, 0), line),
            "swap": lambda line: self._swap(_arg(line, 0), _arg(line, 1)),
            "replace": lambda line: self._replace(_arg(line, 0)),
            "faint": lambda line: self._faint(_arg(line, 0)),
            "cant": lambda line: self._cant(_arg(line, 0), _arg(line, 1)),
            "-fail": lambda line: self._fail(_arg(line, 0)),
            "-notarget": lambda line: self._fail(_arg(line, 0)),
            "-miss": self._miss,
            "-immune": self._miss,
            "-damage": self._damage,
            "-boost": lambda line: self._stat(_arg(line, 0), True),
            "-unboost": lambda line: self._stat(_arg(line, 0), False),
            "-enditem": self._end_item,
            "-item": self._item,
            "-transform": lambda line: self._transform(_arg(line, 0), _arg(line, 1)),
            "-start": lambda line: self._volatile_start(_arg(line, 0), _arg(line, 1)),
            "-end": self._end_line,
            "-activate": self._activate,
        }

    def _turn_line(self, line: _Line) -> None:
        turn = _int(_arg(line, 0))
        if turn < self._turn:
            raise _Refusal("malformed_line")
        self._turn = turn
        self._guards.clear()
        self._rollover(turn)

    def _miss(self, line: _Line) -> None:
        if self._action is not None:
            self._action.misses += 1

    def _end_line(self, line: _Line) -> None:
        self._volatile_end(_arg(line, 0), _arg(line, 1))
        self._slot_lands(_arg(line, 0), _arg(line, 1))

    def _effect(self, line: _Line) -> None:
        action = self._action
        assert action is not None
        zero = line.kind in ("-boost", "-unboost") and _number(line.args, 2) == 0
        if not zero and action.self_target and self._self_line(line.args[0] if line.args else None):
            action.self_effect = True
        else:
            action.other = True

    def _self_line(self, text: str | None) -> bool:
        action = self._action
        return (
            action is not None
            and text is not None
            and text.startswith(f"{action.side}{action.letter}:")
        )

    def _request(self, body: str | None) -> None:
        try:
            parsed = json.loads(body or "")
        except ValueError:
            return
        active = parsed.get("active") if isinstance(parsed, dict) else None
        if not isinstance(active, list):
            return
        for index, entry in enumerate(active):
            moves = entry.get("moves") if isinstance(entry, dict) else None
            if not isinstance(moves, list):
                continue
            self._request_moves[chr(97 + index)] = {
                m["id"]
                for m in moves
                if isinstance(m, dict)
                and isinstance(m.get("id"), str)
                and isinstance(m.get("maxpp"), (int, float))
                and not isinstance(m.get("maxpp"), bool)
            }

    def _rollover(self, turn: int) -> None:
        for side in self._sides.values():
            for occupant in side.active.values():
                if occupant is None:
                    continue
                occupant.last_failed = (
                    None if occupant.result == "unknown" else occupant.result == "failed"
                )
                occupant.result = "none"
                occupant.attackers.clear()
                occupant.attackers_unsettled = False
                if turn != 1:
                    occupant.hurt = False
                    occupant.raised = False
                    occupant.lowered = False

    def _switch_in(self, text: str | None, line: _Line) -> None:
        side_id, letter, name = self._ident(text)
        if letter is None:
            raise _Refusal("unattributable_line")
        side = self._sides[side_id]
        out = side.active.get(letter)
        occupant = _Occupant(name=name)
        origin = line.tags.get("from")
        if out is not None and not out.unknown and origin in ("Baton Pass", "Shed Tail"):
            copied = (
                "stall" not in self._not_copied
                if origin == "Baton Pass"
                else "stall" in self._shed_tail
            )
            if copied:
                occupant.stall_count = out.stall_count
                occupant.stall_turn = out.stall_turn
        side.active[letter] = occupant
        self._last_letter[f"{side_id}:{name}"] = letter
        if side_id == self._player:
            self._request_moves.pop(letter, None)
        side.roster.setdefault(name, _Roster())

    def _swap(self, text: str | None, target: str | None) -> None:
        side_id, letter, _ = self._ident(text)
        index = _int(target) if target is not None else -1
        to = chr(97 + index)
        active = self._sides[side_id].active
        if letter is None or letter not in active or to not in active:
            raise _Refusal("unattributable_line")
        one = active[letter]
        active[letter] = active[to]
        active[to] = one
        if one is not None:
            self._last_letter[f"{side_id}:{one.name}"] = to
        moved = active[letter]
        if moved is not None:
            self._last_letter[f"{side_id}:{moved.name}"] = letter
        if side_id == self._player:
            moves = self._request_moves.get(letter)
            other = self._request_moves.get(to)
            if other is None:
                self._request_moves.pop(letter, None)
            else:
                self._request_moves[letter] = other
            if moves is None:
                self._request_moves.pop(to, None)
            else:
                self._request_moves[to] = moves

    def _replace(self, text: str | None) -> None:
        side_id, letter, name = self._ident(text)
        occupant = None if letter is None else self._sides[side_id].active.get(letter)
        if occupant is None:
            raise _Refusal("unattributable_line")
        side = self._sides[side_id]
        shown = side.roster.get(occupant.name)
        if shown is not None:
            shown.unknown = True
        occupant.unknown = True
        occupant.name = name
        side.roster[name] = _Roster(unknown=True)

    def _faint(self, text: str | None) -> None:
        side_id, letter, _ = self._ident(text)
        side = self._sides[side_id]
        if letter is None or letter not in side.active:
            raise _Refusal("unattributable_line")
        side.active[letter] = None
        side.total_fainted += 1

    def _move(self, line: _Line, just_cant: _Occupant | None) -> None:
        side, letter, occupant = self._occupant(line.args[0] if line.args else None)
        move = _to_id(line.args[1] if len(line.args) > 1 else "")
        if move == "":
            raise _Refusal("malformed_line")
        self._last_move = move
        origin = line.tags.get("from")
        called = origin is not None and _CALLED.match(origin) is not None
        target = _IDENT.match(_arg(line, 2) or "")
        slot_letter = (
            None
            if target is None
            else target.group(2) or self._last_letter.get(f"{target.group(1)}:{target.group(3)}")
        )
        slot: tuple[str, str, str] | None = None
        creating = next((c for c in self._slot_conditions if move in c["moves"]), None)
        if creating is not None and not called and target is not None and slot_letter is not None:
            slot = (target.group(1), slot_letter, creating["id"])
        nested = called and origin is not None and _to_id(origin) not in self._action_sources
        if self._action is not None and nested:
            self._action.nested = True
            if self._action.occupant is not occupant:
                occupant.result = "unknown"
            if move in self._stall:
                self._pending = (side, letter, move)
            return
        self._action = _Action(
            side=side,
            letter=letter,
            occupant=occupant,
            target=(target.group(1), target.group(2))
            if target is not None and target.group(2) is not None
            else None,
            self_target=target is not None
            and target.group(1) == side
            and target.group(2) == letter,
            slot=slot,
        )
        if nested:
            if move in self._stall:
                self._pending = (side, letter, move)
            return
        if just_cant is not occupant:
            occupant.move_actions += 1
        if (
            side == self._player
            and not (called and origin is not None and _to_id(origin) in self._external)
            and move in self._request_moves.get(letter, set())
        ):
            occupant.used.add(move)
        if called:
            return
        if move in self._stall:
            self._pending = (side, letter, move)

    def _cant(self, text: str | None, reason: str | None) -> None:
        found = self._at(text)
        if found is None:
            return
        occupant = found[2]
        occupant.move_actions += 1
        occupant.result = "ok" if (reason or "") in self._null_cant else "failed"
        self._last_cant = occupant
        self._action = None

    def _fail(self, text: str | None) -> None:
        action = self._action
        if action is None:
            return
        if text is not None and text.startswith(f"{action.side}{action.letter}:"):
            action.slot = None
        action.other = True

    def _damage(self, line: _Line) -> None:
        target = self._at(line.args[0] if line.args else None)
        if target is None:
            return
        action = self._action
        origin = line.tags.get("from")
        own = action is not None and action.occupant is target[2]
        direct = origin == "recoil" or (origin is None and own)
        if origin == "confusion":
            target[2].move_actions += 1
            target[2].result = "failed"
        if not direct:
            target[2].hurt = True
        if origin is None and action is not None and not own:
            self._hit(action, target)
            action.hit = True

    def _hit(self, action: _Action, target: tuple[str, str, _Occupant]) -> None:
        target[2].times_attacked += 1
        if action.letter == "?":
            target[2].attackers_unsettled = True
        else:
            target[2].attackers.add(self._seen(action.side, action.letter))

    def _stat(self, text: str | None, raise_: bool) -> None:
        found = self._at(text)
        if found is None:
            return
        if raise_:
            found[2].raised = True
        else:
            found[2].lowered = True

    def _end_item(self, line: _Line) -> None:
        side_id, _, name = self._ident(line.args[0] if line.args else None)
        side = self._sides[side_id]
        item = _to_id(line.args[1] if len(line.args) > 1 else "")
        origin = line.tags.get("from")
        entry = side.roster.get(name)
        if any(tag in self._eat_tags for tag in line.tags):
            if entry is not None:
                entry.ate = True
                entry.last_item = item
        elif origin is not None and origin in self._steal_eat:
            of = line.tags.get("of")
            eater = None if of is None else _IDENT.match(of)
            roster = (
                None if eater is None else self._sides[eater.group(1)].roster.get(eater.group(3))
            )
            if roster is not None:
                roster.ate = True
        elif origin is None or origin in self._gem or _to_id(origin) in self._item_use:
            if entry is not None and item not in self._direct_end:
                entry.last_item = item
            target = self._action.target if self._action is not None else None
            if (
                origin is not None
                and origin not in self._gem
                and target is not None
                and item in self._berries
            ):
                eater_occ = self._sides[target[0]].active.get(target[1])
                roster = (
                    None if eater_occ is None else self._sides[target[0]].roster.get(eater_occ.name)
                )
                if roster is not None:
                    roster.ate = True

    def _item(self, line: _Line) -> None:
        origin = line.tags.get("from")
        if origin is None:
            return
        side_id, _, name = self._ident(line.args[0] if line.args else None)
        if _to_id(origin) in self._restorers:
            entry = self._sides[side_id].roster.get(name)
            if entry is not None:
                entry.last_item = ""
        elif _to_id(origin) in self._takers:
            item = _to_id(line.args[1] if len(line.args) > 1 else "")
            for side in self._sides.values():
                for entry in side.roster.values():
                    if entry.last_item == item:
                        entry.last_item = "?"

    def _transform(self, user: str | None, target: str | None) -> None:
        one = self._occupant(user)[2]
        two = self._at(target)
        if two is None:
            raise _Refusal("unattributable_line")
        one.times_attacked = two[2].times_attacked
        one.used.clear()

    def _slot_lands(self, text: str | None, name: str | None) -> None:
        side_id, letter, _ = self._ident(text)
        if letter is None or not (name or "").startswith("move: "):
            return
        move = _effect_id(name or "")
        condition = next((c for c in self._slot_conditions if move in c["moves"]), None)
        slot = (
            None
            if condition is None
            else self._sides[side_id].slots.get(letter, {}).get(condition["id"])
        )
        if slot is None:
            return
        found = (slot.source_side, "?")
        for side, state in self._sides.items():
            for key, occupant in state.active.items():
                if occupant is slot.source:
                    found = (side, key)
        self._action = _Action(
            side=found[0], letter=found[1], occupant=slot.source, target=None, delayed=True
        )

    def _activate(self, line: _Line) -> None:
        effect = line.args[1] if len(line.args) > 1 else ""
        if not effect.startswith("move: ") or _effect_id(effect) not in self._breakers:
            return
        found = self._at(line.args[0] if line.args else None)
        if found is not None:
            found[2].stall_count = 0
            found[2].stall_turn = -1

    def _volatile_start(self, text: str | None, name: str | None) -> None:
        found = self._at(text)
        volatile = _effect_id(name or "")
        if volatile == "":
            raise _Refusal("malformed_line")
        if found is not None and volatile not in found[2].volatiles:
            found[2].volatiles[volatile] = {
                "started_turn": self._turn,
                "after_residual": self._residual_ran(),
            }

    def _volatile_end(self, text: str | None, name: str | None) -> None:
        found = self._at(text)
        if found is not None:
            found[2].volatiles.pop(_effect_id(name or ""), None)


def _int(text: str | None) -> int:
    """An integer argument, or the typed refusal for a missing or non-numeric one."""
    try:
        return int(text if text is not None else "")
    except ValueError:
        raise _Refusal("malformed_line") from None


def _arg(line: _Line, index: int) -> str | None:
    return line.args[index] if index < len(line.args) else None


def _number(args: list[str], index: int) -> float | None:
    try:
        return float(args[index])
    except (IndexError, ValueError):
        return None
