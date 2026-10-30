"""The clean-room log-to-state reducer (ADR-010 §3, R10).

It turns ONE player's own channel (the player-perspective protocol lines up to
the request being answered, ADR-008 §5) plus that player's own team into a
`BattleState`. The grammar comes from this repository's channel fixtures, the
upstream wire-protocol document (`sim/SIM-PROTOCOL.md`) and ADR-006/008; no
client parser was consulted or translated.

Rules the reducer keeps:

* Fail closed. A line type, tag, positional field, request key or id it has no
  rule for raises a typed `ReducerError`; nothing is skipped or guessed.
* Player perspective only. It consumes nothing but the player's channel and
  team. A `|split|` marker, an opponent HP that is not a percentage, or a
  request for the other player raises `OpponentSecretError`.
* Conservative reveals. A fact is recorded only when a line states it about
  its owner: `[from] item:`/`[from] ability:` count only without an `[of]`
  tag (with `[of]`, which of the two Pokemon owns the effect depends on the
  message); a move counts as revealed when it was chosen, not when another
  effect called it (`[from]`).
* Illusion. A switch-in is bound to the Pokemon it is shown as. `|replace|` (and, for the
  player's own Pokemon, the player's next request) says who it was: everything seen under the
  disguise moves to the real Pokemon and the disguised one is put back as it was before the
  switch-in. What the channel cannot attribute without guessing raises `IllusionError`.
* Deterministic. The same lines and team always give the same state.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    IllusionError,
    MalformedLineError,
    OpponentSecretError,
    ReducerError,
    UnknownFieldError,
    UnknownLineError,
    VocabularyError,
)
from .state import STAT_KEYS, BattleState, PokemonState, Recent, SideState, zero_boosts
from .vocab import Vocabulary, to_id

# Display-only tags of the wire protocol: accepted, but only `from` and `of`
# ever move state. Any other tag raises `UnknownFieldError`.
KNOWN_TAGS = frozenset(
    {
        "identify",
        "from", "of", "spread", "miss", "still", "silent", "anim", "upkeep", "eat", "msg",
        "weak", "weaken", "wisher", "persistent", "fail", "notarget", "damage", "consumed",
        "interrupt", "partiallytrapped", "move", "ability", "forme", "ohko", "heavy",
        "broken", "fatigue", "premajor", "source", "fromitem", "unidentified", "ruleid",
    }
)  # fmt: skip

# Tags the engine appends to the move line once per call (`attrLastMove`), so a line a second
# call reaches carries one more, with the targets that call hit (Magic Bounce reflecting a
# spread move: the bounced move and the original both add theirs to the bounce's line).
# Display-only; the last value is kept.
_REPEATABLE_TAGS = frozenset({"spread"})

# Lines whose text may hold `[` or `|`-free prose: no tag parsing.
_FREE_TEXT = frozenset({"uhtml", "uhtmlchange", "-message", "-hint", "rule", "tier", "error"})

# The no-state lines below that still feed the recent history (`_observe`).
_OBSERVED = frozenset({"cant", "upkeep", "-singleturn", "-fail"})
# Lines that change no state, with the number of positional fields they carry.
_NO_STATE: dict[str, tuple[int, int | None]] = {
    "": (0, 0),
    "gen": (1, 1),
    "rule": (1, 1),
    "rated": (0, 1),
    "upkeep": (0, 0),
    "uhtml": (1, None),
    "uhtmlchange": (1, None),
    "cant": (2, 3),
    "-crit": (1, 1),
    "-supereffective": (1, 2),
    "-resisted": (1, 2),
    "-immune": (1, 3),
    "-fail": (1, 4),
    "-anim": (1, 4),
    "-block": (2, 4),
    "-notarget": (0, 1),
    "-miss": (1, 2),
    "-hint": (1, None),
    "-message": (1, None),
    "-center": (0, 0),
    "-combine": (0, 0),
    "-waiting": (2, 2),
    "-prepare": (2, 3),
    "-mustrecharge": (1, 1),
    "-nothing": (0, 0),
    "-hitcount": (2, 2),
    "-singlemove": (2, 2),
    "-singleturn": (2, 2),
    "-fieldactivate": (1, 1),
    "-ohko": (0, 0),
    "-zbroken": (1, 1),
}

# The refusal text the engine (or, on the A2 server path, the server layer) sends to the
# player alone for a choice it did not take. It carries only the player's own refusal, is
# never part of the room log, and is followed by a fresh `|request|` (an `[Unavailable
# choice]`) or by nothing (an `[Invalid choice]`): no state changes. Any other form is refused.
_ERROR_TEXT = re.compile(r"^\[(?:Unavailable|Invalid) choice\] \S[^\n\r]*$")
_TAG = re.compile(r"^\[([a-z]+)\]\s*(.*)$")
_REF = re.compile(r"^(p[1-4])([a-d])?: (.+)$")
_SIDE = re.compile(r"^(p[1-4]): ")
_LEVEL = re.compile(r"^L(\d{1,3})$")
_HP = re.compile(r"^(\d+)(?:/(\d+)([gyr])?)?$")
# The engine's marker for a Pokemon with no type left (`pokemon.ts`; Burn Up removes Fire); it is
# not an entry of the types vocabulary.
TYPELESS = "???"


def _kept_derived(derived: dict[str, str], kept: set[str]) -> dict[str, str]:
    """The derived suffixes whose raw volatile id (`stockpile2`) is among the ``kept`` ones."""
    return {base: suffix for base, suffix in derived.items() if f"{base}{suffix}" in kept}


MAX_VOLATILE_LENGTH = 64  # an opaque volatile id is short; a longer one is not an id
_VOLATILE = re.compile(r"^(?:(move|ability|item): )?([A-Za-z][A-Za-z0-9 '.-]*)$")
# Colour class the champions mod appends to a shared HP at exactly 50% or 20%.
_HP_COLOURS = {50: "gy", 20: "yr"}
# `[from]` effects after which the `[of]` Pokemon holds the new ability too.
_ABILITY_COPIES = frozenset({"abilitytrace", "moveroleplay"})
# The `[from]` tag of a Fling that ends no item (`|-enditem|POKEMON||[from] move: Fling`, #1162).
FLING_SOURCE = "move: Fling"
_PREFIXED = re.compile(r"^(move|ability|item): (.+)$")
HP_PERCENT = 100  # the denominator of every opponent HP
MAX_BOOST = 6
_SLOTS = {"singles": "a", "doubles": "ab", "triples": "abc"}

_REQUEST_KEYS = frozenset(
    {
        "wait",
        "teamPreview",
        "forceSwitch",
        "maxChosenTeamSize",
        "active",
        "side",
        "noCancel",
        "update",
        "rqid",
    }
)
_SIDE_KEYS = frozenset({"name", "id", "pokemon", "noCancel"})
_POKEMON_KEYS = frozenset(
    {"ident", "details", "condition", "active", "stats", "moves", "baseAbility", "item", "pokeball",
     "ability", "commanding", "reviving"}
)  # fmt: skip
_MEGA_FLAGS = ("canMegaEvo", "canMegaEvoX", "canMegaEvoY")
_ACTIVE_KEYS = frozenset(
    {"moves", "maybeDisabled", "maybeLocked", "trapped", "maybeTrapped", *_MEGA_FLAGS}
)
_MOVE_KEYS = frozenset({"move", "id", "pp", "maxpp", "target", "disabled", "disabledSource"})
ILLUSION = "Illusion"  # the ability whose end `|replace|` reports (protocol: "Illusion has ended")
_SET_KEYS = ("species", "name", "item", "ability", "moves", "level", "gender")


@dataclass
class _Line:
    raw: str
    kind: str
    args: list[str]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _Passed:
    """What a Baton Pass or Shed Tail hands to the Pokemon that comes in."""

    boosts: dict[str, int]
    volatiles: set[str]
    ability_suppressed: bool
    derived: dict[str, str]


@dataclass
class _Disguise:
    """A switch-in bound to the Pokemon it is shown as, until a `|replace|` or an own request
    says who it was."""

    index: int  # the roster entry the position is bound to
    before: PokemonState | None  # that entry before the switch-in (None: created by it)


@dataclass
class _Pair:
    """The same Pokemon shown in two positions: one of the two is an Illusion user, and the
    channel does not say which. `phantom` is the added entry of the second one shown."""

    owner: int  # the roster entry shown first
    phantom: int  # the added roster entry
    owner_before: PokemonState | None  # the owner before its switch-in (None: known to be real)
    phantom_before: PokemonState


@dataclass(frozen=True)
class _Details:
    species: str
    base_species: str
    hidden_forme: bool
    level: int
    gender: str
    shiny: bool


def _parse_line(raw: str) -> _Line:
    if not raw.startswith("|"):
        raise MalformedLineError("a protocol line starts with '|'", raw)
    kind, _, rest = raw[1:].partition("|")
    if kind == "request":
        return _Line(raw, kind, [rest])
    parts = raw[1:].split("|")[1:]
    if kind in _FREE_TEXT:
        return _Line(raw, kind, parts)
    args: list[str] = []
    tags: dict[str, str] = {}
    for part in parts:
        match = _TAG.match(part)
        if match is None:
            if tags:
                raise MalformedLineError("a positional field after a tag", raw)
            args.append(part)
            continue
        name, value = match.groups()
        if name not in KNOWN_TAGS:
            raise UnknownFieldError(f"unknown tag [{name}]", raw)
        if name in tags and name not in _REPEATABLE_TAGS and (tags[name] != value or value):
            # The engine repeats a bare tag ([still], [miss]) when two effects add it.
            raise MalformedLineError(f"duplicate tag [{name}]", raw)
        tags[name] = value
    return _Line(raw, kind, args, tags)


def _arity(line: _Line, low: int, high: int | None) -> None:
    if len(line.args) < low:
        raise MalformedLineError(f"{line.kind or 'blank'} needs at least {low} fields", line.raw)
    if high is not None and len(line.args) > high:
        raise UnknownFieldError(f"{line.kind or 'blank'} carries more than {high} fields", line.raw)


def _optional(line: _Line, index: int) -> str | None:
    return line.args[index] if index < len(line.args) else None


def _integer(text: str, line: _Line) -> int:
    if not re.fullmatch(r"-?\d+", text):
        raise MalformedLineError(f"{text!r} is not an integer", line.raw)
    return int(text)


class _Reducer:
    def __init__(self, player: str, team: Sequence[Mapping[str, Any]], vocab: Vocabulary) -> None:
        if player not in ("p1", "p2"):
            raise MalformedLineError(f"unsupported player {player!r}")
        self.vocab = vocab
        self.state = BattleState(player=player)
        self.state.own = SideState(player=player)
        self.state.opponent = SideState(player="p2" if player == "p1" else "p1")
        self._preview_own: set[int] = set()
        # By (player, position letter): the name the position shows, and its unsettled disguise.
        self._shown: dict[tuple[str, str], str] = {}
        self._disguise: dict[tuple[str, str], _Disguise] = {}
        self._pairs: dict[tuple[str, int], _Pair] = {}  # by (player, added entry)
        # Recent history (#1013): the place of the next action of the turn, the Pokemon whose move
        # is running (damage lines without a `[from]` tag are its), and a stalling move whose
        # `-singleturn` or `-fail` has not come yet.
        self._actions = 0
        self._actor: PokemonState | None = None
        self._stalling: PokemonState | None = None
        for pokemon_set in team:
            self.state.own.roster.append(self._own_mon(pokemon_set))
        names = [mon.name for mon in self.state.own.roster]
        if len(set(names)) != len(names):
            raise MalformedLineError(
                "own team has two Pokemon the protocol cannot tell apart by name"
            )
        self._handlers: dict[str, Callable[[_Line], None]] = {
            "player": self._player,
            "error": self._error,
            "tier": self._tier,
            "gametype": self._gametype,
            "clearpoke": self._clearpoke,
            "poke": self._poke,
            "teampreview": self._teampreview,
            "teamsize": self._teamsize,
            "start": self._start,
            "turn": self._turn,
            "win": self._win,
            "tie": self._tie,
            "request": self._request,
            "switch": self._switch,
            "drag": self._switch,
            "detailschange": self._detailschange,
            "-formechange": self._formechange,
            "swap": self._swap,
            "replace": self._replace,
            "-activate": self._activate,
            "move": self._move,
            "faint": self._faint,
            "-damage": self._hp_line,
            "-heal": self._hp_line,
            "-sethp": self._sethp,
            "-status": self._status,
            "-curestatus": self._curestatus,
            "-cureteam": self._cureteam,
            "-boost": self._boost,
            "-unboost": self._boost,
            "-setboost": self._boost,
            "-swapboost": self._swapboost,
            "-invertboost": self._invertboost,
            "-clearboost": self._clearboost,
            "-clearallboost": self._clearallboost,
            "-clearpositiveboost": self._clearpositiveboost,
            "-clearnegativeboost": self._clearnegativeboost,
            "-copyboost": self._copyboost,
            "-weather": self._weather,
            "-fieldstart": self._fieldstart,
            "-fieldend": self._fieldend,
            "-sidestart": self._sidestart,
            "-sideend": self._sideend,
            "-swapsideconditions": self._swapsideconditions,
            "-start": self._volatile_start,
            "-end": self._volatile_end,
            "-item": self._item,
            "-enditem": self._enditem,
            "-ability": self._ability,
            "-endability": self._endability,
            "-transform": self._transform,
            "-mega": self._mega,
        }

    # -- construction --------------------------------------------------------

    def _own_mon(self, pokemon_set: Mapping[str, Any]) -> PokemonState:
        for key in _SET_KEYS:
            if key not in pokemon_set:
                raise MalformedLineError(f"own team set lacks {key!r}")
        species = self.vocab.species(str(pokemon_set["species"]))
        item = str(pokemon_set["item"])
        gender = str(pokemon_set["gender"])
        return PokemonState(
            own=True,
            name=str(pokemon_set["name"]) or str(pokemon_set["species"]),
            base_species=self.vocab.base_species(species),
            species=species,
            level=int(pokemon_set["level"]),
            gender=gender or None,
            item=self.vocab.item(item) if item else None,
            item_known=True,
            ability=self.vocab.ability(str(pokemon_set["ability"])),
            moves_own=[self.vocab.move(str(m)) for m in pokemon_set["moves"]],
        )

    # -- driver --------------------------------------------------------------

    def feed(self, raw: str) -> None:
        try:
            if raw.startswith("|split|"):
                raise OpponentSecretError(
                    "an unresolved |split| holds the opponent's secret half", raw
                )
            line = _parse_line(raw)
            if line.kind in _NO_STATE:
                _arity(line, *_NO_STATE[line.kind])
                if line.kind in _OBSERVED:
                    self._observe(line)
                return
            handler = self._handlers.get(line.kind)
            if handler is None:
                raise UnknownLineError(f"no rule for line type {line.kind!r}", raw)
            handler(line)
        except ReducerError as error:
            if error.line is None:
                error.line = raw
            raise

    # -- references ----------------------------------------------------------

    def _side(self, player_id: str, line: _Line) -> SideState:
        if player_id == self.state.own.player:
            return self.state.own
        if player_id == self.state.opponent.player:
            return self.state.opponent
        raise UnknownFieldError(f"unsupported side {player_id!r}", line.raw)

    def _ref(self, text: str, line: _Line) -> tuple[SideState, str | None, str]:
        match = _REF.match(text)
        if match is None:
            raise MalformedLineError(f"{text!r} is not a Pokemon id", line.raw)
        player_id, letter, name = match.groups()
        return self._side(player_id, line), letter, name

    def _mon(self, text: str, line: _Line) -> PokemonState:
        side, letter, name = self._ref(text, line)
        if letter is not None:
            index = side.slots.get(letter)
            if index is not None and self._shown.get((side.player, letter)) == name:
                return side.roster[index]
        for mon in side.roster:
            if mon.name == name:
                return mon
        raise MalformedLineError(f"{text!r} names a Pokemon that has not been introduced", line.raw)

    # -- parsing helpers -----------------------------------------------------

    def _details(self, text: str, line: _Line) -> _Details:
        name, *rest = text.split(", ")
        hidden = name.endswith("-*")
        species = self.vocab.species(name[:-2] if hidden else name)
        level, gender, shiny = 100, "", False
        for part in rest:
            level_match = _LEVEL.match(part)
            if level_match:
                level = int(level_match.group(1))
            elif part in ("M", "F"):
                gender = part
            elif part == "shiny":
                shiny = True
            else:
                raise UnknownFieldError(f"unknown details field {part!r}", line.raw)
        return _Details(species, self.vocab.base_species(species), hidden, level, gender, shiny)

    def _parse_hp(
        self, text: str, mon: PokemonState, line: _Line
    ) -> tuple[int, int | None, str | None]:
        number, _, status = text.partition(" ")
        match = _HP.match(number)
        if match is None:
            raise MalformedLineError(f"{text!r} is not an HP", line.raw)
        num = int(match.group(1))
        den = int(match.group(2)) if match.group(2) else None
        colour = match.group(3)
        if den is None:
            if status != "fnt" or num != 0:
                raise MalformedLineError(f"{text!r} is not an HP", line.raw)
            return 0, None, None
        if not mon.own and den != HP_PERCENT:
            raise OpponentSecretError("an opponent HP that is not a percentage", line.raw)
        if num > den or den == 0:
            raise MalformedLineError(f"{text!r} is not an HP", line.raw)
        if colour is not None and (mon.own or colour not in _HP_COLOURS.get(num, "")):
            raise MalformedLineError(f"{text!r} has a colour class an HP cannot carry", line.raw)
        if status and status != "fnt":
            status = self.vocab.status(status)
        elif status == "fnt":
            if num != 0:
                raise MalformedLineError(f"{text!r} is not an HP", line.raw)
            status = ""
        return num, den, status or None

    def _set_hp(self, mon: PokemonState, text: str, line: _Line, *, status: bool = True) -> None:
        num, den, new_status = self._parse_hp(text, mon, line)
        mon.hp_num = num
        mon.hp_den = den if den is not None else (mon.hp_den or (None if mon.own else HP_PERCENT))
        if status:
            mon.status = new_status
        mon.fainted = num == 0

    def _resolve_effect(self, text: str, line: _Line) -> tuple[str, str]:
        """A field or side condition (a move or an ability), looked up."""
        prefixed = _PREFIXED.match(text)
        if prefixed:
            kind, name = prefixed.groups()
            known = {
                "move": self.vocab.moves,
                "ability": self.vocab.abilities,
                "item": self.vocab.items,
            }[kind]
            found = to_id(name)
            if found not in known:
                raise VocabularyError(f"unknown {kind} {name!r}", line.raw)
            return kind, found
        # Some conditions print bare (`Reflect`); a bare name is a move or an ability.
        found = to_id(text)
        for kind, known in (("move", self.vocab.moves), ("ability", self.vocab.abilities)):
            if found in known:
                return kind, found
        raise VocabularyError(f"unknown condition {text!r}", line.raw)

    def _volatile(self, text: str, line: _Line) -> tuple[str, str]:
        """A volatile's id. Its shape is checked; against an engine-data version 2 table the id
        must also resolve (`Vocabulary.volatile`), against version 1 (no condition list) it is
        kept as an opaque public string. `ability:` names are always looked up, because they also
        reveal the holder's ability."""
        match = _VOLATILE.match(text)
        if match is None or len(text) > MAX_VOLATILE_LENGTH:
            raise MalformedLineError(f"{text!r} is not a volatile id", line.raw)
        kind, name = match.groups()
        found = to_id(name)
        if kind == "ability":
            return kind, self.vocab.ability(name)
        self.vocab.volatile(found, kind)
        return kind or "volatile", found

    def _reveal_from(self, mon: PokemonState, line: _Line) -> None:
        """`[from] item:` / `[from] ability:` without `[of]` names the target's own."""
        source = line.tags.get("from")
        if source is None or "of" in line.tags:
            return
        prefixed = _PREFIXED.match(source)
        if prefixed is None:
            return
        kind, name = prefixed.groups()
        if kind == "item":
            mon.item, mon.item_known = self.vocab.item(name), True
        elif kind == "ability":
            mon.ability = self.vocab.ability(name)

    # -- header --------------------------------------------------------------

    def _player(self, line: _Line) -> None:
        _arity(line, 1, 4)
        side = self._side(line.args[0], line)
        if len(line.args) > 1:
            side.username = line.args[1] or None

    def _tier(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self.state.format_name = line.args[0]

    def _error(self, line: _Line) -> None:
        text = "|".join(line.args)
        if _ERROR_TEXT.match(text) is None:
            raise MalformedLineError("an |error| line that is not a choice refusal", line.raw)

    def _gametype(self, line: _Line) -> None:
        _arity(line, 1, 1)
        letters = _SLOTS.get(line.args[0])
        if letters is None:
            raise UnknownFieldError(f"unsupported gametype {line.args[0]!r}", line.raw)
        self.state.gametype = line.args[0]
        for side in (self.state.own, self.state.opponent):
            side.slots = {letter: None for letter in letters}

    def _clearpoke(self, line: _Line) -> None:
        _arity(line, 0, 0)
        self.state.opponent.roster.clear()
        self.state.opponent.slots = dict.fromkeys(self.state.opponent.slots)
        opponent = self.state.opponent.player
        self._shown = {key: v for key, v in self._shown.items() if key[0] != opponent}
        self._disguise = {key: v for key, v in self._disguise.items() if key[0] != opponent}
        self._pairs = {key: v for key, v in self._pairs.items() if key[0] != opponent}
        self._preview_own.clear()

    def _poke(self, line: _Line) -> None:
        _arity(line, 2, 3)
        flag = _optional(line, 2)
        if flag not in (None, "", "item"):
            raise UnknownFieldError(f"unknown item flag {flag!r}", line.raw)
        side = self._side(line.args[0], line)
        details = self._details(line.args[1], line)
        if side is self.state.own:
            for index, mon in enumerate(side.roster):
                if index not in self._preview_own and mon.base_species == details.base_species:
                    self._preview_own.add(index)
                    return
            raise MalformedLineError(
                "team preview shows a Pokemon that is not in the own team", line.raw
            )
        side.roster.append(
            PokemonState(
                own=False,
                base_species=details.base_species,
                species=None if details.hidden_forme else details.species,
                level=details.level,
                gender=details.gender or None,
                shiny=details.shiny,
            )
        )

    def _teampreview(self, line: _Line) -> None:
        _arity(line, 0, 1)
        if line.args:
            _integer(line.args[0], line)

    def _teamsize(self, line: _Line) -> None:
        _arity(line, 2, 2)
        self._side(line.args[0], line).teamsize = _integer(line.args[1], line)

    def _start(self, line: _Line) -> None:
        _arity(line, 0, 0)
        self.state.started = True

    def _turn(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self._require_stall_outcome(line)
        self.state.turn = _integer(line.args[0], line)
        self._actions, self._actor = 0, None
        for side in (self.state.own, self.state.opponent):
            for mon in side.roster:
                recent = mon.recent
                recent.hp_taken_last_turn, recent.hp_taken = recent.hp_taken, 0
                recent.hp_dealt_last_turn, recent.hp_dealt = recent.hp_dealt, 0
                if mon.active_slot is not None:
                    recent.active_turns += 1
                # A use two turns back no longer counts: the `stall` volatile ends after one.
                if recent.stall_turn is not None and recent.stall_turn < self.state.turn - 1:
                    recent.protect_streak, recent.stall_turn = 0, None

    def _require_stall_outcome(self, line: _Line) -> None:
        """The engine prints `-singleturn` (success) or `-fail` right after every stalling move,
        so another action, the turn's end or the game's end with neither is a log the reducer
        cannot read (the world facts' `stall_unresolved`)."""
        if self._stalling is not None:
            raise MalformedLineError("a stalling move with no -singleturn or -fail", line.raw)

    def _end_move_action(self, line: _Line) -> None:
        """A switch ends the running move: no stalling move may be waiting, no damage is its."""
        self._require_stall_outcome(line)
        self._actor = None

    def _observe(self, line: _Line) -> None:
        """Lines that carry no state of their own but feed the recent history."""
        if line.kind == "cant":
            self._require_stall_outcome(line)
            self._acted(self._mon(line.args[0], line))
        elif line.kind == "upkeep":
            self._actor = None
        elif self._stalling is not None and line.args:  # -singleturn, -fail
            if self._mon(line.args[0], line) is self._stalling:
                recent = self._stalling.recent
                if line.kind == "-fail":
                    recent.protect_streak, recent.stall_turn = 0, None
                else:
                    kept = (
                        recent.stall_turn is not None and recent.stall_turn >= self.state.turn - 1
                    )
                    recent.protect_streak = recent.protect_streak + 1 if kept else 1
                    recent.stall_turn = self.state.turn
                self._stalling = None

    def _acted(self, mon: PokemonState) -> None:
        self._actions += 1
        mon.recent.last_action_turn, mon.recent.last_action_order = self.state.turn, self._actions

    def _win(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self._require_stall_outcome(line)
        for label, side in (("own", self.state.own), ("opponent", self.state.opponent)):
            if side.username is not None and side.username == line.args[0]:
                self.state.winner = label
                return
        raise MalformedLineError(f"{line.args[0]!r} is neither player", line.raw)

    def _tie(self, line: _Line) -> None:
        _arity(line, 0, 0)
        self._require_stall_outcome(line)
        self.state.winner = "tie"

    # -- requests ------------------------------------------------------------

    def _request(self, line: _Line) -> None:
        try:
            body = json.loads(line.args[0])
        except json.JSONDecodeError as error:
            raise MalformedLineError(f"request is not JSON: {error.msg}", line.raw) from error
        self._keys(body, _REQUEST_KEYS, line, "request")
        side_data = body.get("side")
        if not isinstance(side_data, dict):
            raise MalformedLineError("request has no side", line.raw)
        self._keys(side_data, _SIDE_KEYS, line, "request side")
        if side_data.get("id") != self.state.own.player:
            raise OpponentSecretError("a request for the other player", line.raw)
        kind = self._request_kind(body, line)
        entries = side_data.get("pokemon", [])
        for entry in entries:
            self._keys(entry, _POKEMON_KEYS, line, "request pokemon")
        if kind != "preview":
            self._settle_own(entries, line)
        if any(mon.phantom for mon in self.state.own.roster):
            raise IllusionError("an own Pokemon the request cannot tell from its twin", line.raw)
        listed = [self._apply_request_mon(entry, line) for entry in entries]
        if kind != "preview":
            for mon in self.state.own.roster:
                mon.brought = mon in listed
        for index, active in enumerate(body.get("active", [])):
            self._keys(active, _ACTIVE_KEYS, line, "request active")
            if index >= len(listed):
                raise MalformedLineError("request has more active entries than Pokemon", line.raw)
            mon = listed[index]
            for move in active["moves"]:
                self._keys(move, _MOVE_KEYS, line, "request move")
                if isinstance(move.get("pp"), int):
                    mon.pp[self.vocab.move(move["id"])] = move["pp"]
            mon.can_mega_evo = any(
                active.get(k) for k in ("canMegaEvo", "canMegaEvoX", "canMegaEvoY")
            )
        self.state.request_kind = kind

    def _request_kind(self, body: dict[str, Any], line: _Line) -> str:
        if body.get("wait"):
            return "wait"
        if body.get("teamPreview"):
            return "preview"
        if "forceSwitch" in body:
            return "forceSwitch"
        if "active" in body:
            return "move"
        raise MalformedLineError("request has no phase", line.raw)

    def _keys(self, data: object, allowed: frozenset[str], line: _Line, what: str) -> None:
        if not isinstance(data, dict):
            raise MalformedLineError(f"{what} is not an object", line.raw)
        extra = sorted(set(data) - allowed)
        if extra:
            raise UnknownFieldError(f"{what} carries unknown keys {extra}", line.raw)

    def _own_by_ident(self, entry: Mapping[str, Any], line: _Line) -> PokemonState:
        match = _SIDE.match(str(entry["ident"]))
        name = str(entry["ident"])[match.end() :] if match else ""
        mon = next((m for m in self.state.own.roster if m.name == name), None)
        if mon is None:
            raise MalformedLineError(
                f"request lists {entry['ident']!r}, who is not in the own team", line.raw
            )
        return mon

    def _apply_request_mon(self, entry: object, line: _Line) -> PokemonState:
        self._keys(entry, _POKEMON_KEYS, line, "request pokemon")
        assert isinstance(entry, dict)
        mon = self._own_by_ident(entry, line)
        details = self._details(entry["details"], line)
        mon.species, mon.level = details.species, details.level
        mon.gender, mon.shiny = details.gender or None, details.shiny
        self._set_hp(mon, entry["condition"], line)
        item = entry["item"]
        mon.item, mon.item_known = (self.vocab.item(item) if item else None), True
        mon.ability = self.vocab.ability(entry.get("ability") or entry["baseAbility"])
        mon.stats = {stat: int(value) for stat, value in entry["stats"].items()}
        mon.moves_own = [self.vocab.move(move) for move in entry["moves"]]
        return mon

    # -- switching and forms -------------------------------------------------

    def _locate(self, side: SideState, name: str, details: _Details) -> int | None:
        """The roster entry a switch-in binds to, or None (nothing is changed)."""
        for index, mon in enumerate(side.roster):
            if mon.name == name:
                return index
        if side.roster and side.roster[0].own:
            return None
        for index, mon in enumerate(side.roster):
            if (
                mon.name is None
                and mon.base_species == details.base_species
                and mon.species in (None, details.species)
            ):
                return index
        return None

    def _bind(self, side: SideState, name: str, details: _Details, line: _Line) -> PokemonState:
        index = self._locate(side, name, details)
        if index is not None:
            mon = side.roster[index]
            if mon.name is None:
                mon.name = name
            return mon
        if side.roster and side.roster[0].own:
            raise MalformedLineError(f"{name!r} is not in the own team", line.raw)
        if side.teamsize is not None and len(side.roster) >= side.teamsize:
            raise MalformedLineError(f"{name!r} is more Pokemon than the team size", line.raw)
        mon = PokemonState(own=False, name=name, base_species=details.base_species)
        side.roster.append(mon)
        return mon

    def _passed(self, cause: str, out: PokemonState) -> _Passed:
        """What the outgoing Pokemon hands over on a Baton Pass or Shed Tail switch."""
        kept = self.vocab.passed_volatiles(cause, out.volatiles)
        return _Passed(
            dict(out.boosts) if cause == "batonpass" else zero_boosts(),
            kept,
            out.ability_suppressed
            and "gastroacid" in self.vocab.passed_volatiles(cause, {"gastroacid"}),
            _kept_derived(out.derived, kept),
        )

    def _switch(self, line: _Line) -> None:
        _arity(line, 3, 3)
        side, letter, name = self._ref(line.args[0], line)
        if letter is None or letter not in side.slots:
            raise MalformedLineError("a switch needs an active position", line.raw)
        details = self._details(line.args[1], line)
        located = self._locate(side, name, details)
        before = None if located is None else copy.deepcopy(side.roster[located])
        self._end_move_action(line)
        mon = self._bind(side, name, details, line)
        key = (side.player, letter)
        if mon.fainted and not mon.own:
            # A fainted Pokemon cannot come back: the one that fainted was an Illusion user
            # shown as this one (no `|replace|`). Its faint stays on an entry of its own, whose
            # identity the channel does not give, and this entry starts again.
            dead = copy.deepcopy(mon)
            dead.phantom = True
            side.roster.append(dead)
            slot = self._index_of(side, mon)
            mon = side.roster[slot] = PokemonState(
                own=False, name=mon.name, base_species=mon.base_species, species=mon.species
            )
            before = copy.deepcopy(mon)
        if mon.active_slot not in (None, letter):
            # Shown in two positions: one of the two is an Illusion user, and the channel does
            # not say which, so this one gets an entry of its own until a replace or a request.
            owner = self._index_of(side, mon)
            known = next(
                (
                    pair
                    for (player, _), pair in self._pairs.items()
                    if player == side.player and pair.owner == owner
                ),
                None,
            )
            if known is None:
                first = self._disguise.get((side.player, mon.active_slot))
                mon = self._phantom(side, mon)
                self._pairs[(side.player, len(side.roster) - 1)] = _Pair(
                    owner,
                    len(side.roster) - 1,
                    None if first is None else first.before,
                    copy.deepcopy(mon),
                )
            elif side.roster[known.phantom].active_slot is None:
                mon = side.roster[known.phantom]  # the added one is back
            else:
                raise IllusionError("a Pokemon is shown in three positions", line.raw)
            disguise = _Disguise(self._index_of(side, mon), copy.deepcopy(mon))
        else:
            disguise = _Disguise(self._index_of(side, mon), before)
        index = self._index_of(side, mon)
        previous = side.slots[letter]
        # Baton Pass (`selfSwitch: copyvolatile`) and Shed Tail: the engine's `copyVolatileFrom`
        # gives the incoming Pokemon the outgoing one's stat stages (Baton Pass only) and its
        # passable volatiles, and prints no line for them; only the switch line says so
        # (`[from] Baton Pass` / `[from] Shed Tail`). Which volatiles pass is the engine's own
        # rule, read from the committed `volatile-copy` table (#1102).
        cause = to_id(line.tags.get("from") or "")
        passed: _Passed | None = None
        if previous is not None and cause in ("batonpass", "shedtail"):
            out = side.roster[previous]
            passed = self._passed(cause, out)
        if previous is not None:
            self._switch_out(side.roster[previous])
        side.slots[letter] = index
        self._shown[key] = name
        self._disguise[key] = disguise
        self._switch_out(mon)
        if passed is not None:
            mon.boosts, mon.volatiles, mon.ability_suppressed, mon.derived = (
                passed.boosts,
                passed.volatiles,
                passed.ability_suppressed,
                passed.derived,
            )
        mon.active_slot, mon.brought = letter, True
        mon.species, mon.level = details.species, details.level
        mon.gender, mon.shiny = details.gender or None, details.shiny
        self._set_hp(mon, line.args[2], line)

    def _switch_out(self, mon: PokemonState) -> None:
        mon.active_slot = None
        mon.recent = Recent()
        mon.boosts = zero_boosts()
        mon.volatiles = set()
        mon.derived = {}
        mon.types = None
        mon.temp_species = None
        mon.ability_suppressed = False
        if mon.moves_base is not None:  # Transform ends with the stay on the field
            mon.moves_revealed, mon.moves_base = mon.moves_base, None

    def _detailschange(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        details = self._details(line.args[1], line)
        mon.species, mon.level = details.species, details.level
        mon.gender, mon.shiny = details.gender or None, details.shiny
        mon.types = None  # the engine's `setSpecies` puts the new species' own types back
        hp = _optional(line, 2)
        if hp is not None:
            self._set_hp(mon, hp, line)

    def _formechange(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        mon.temp_species = self.vocab.species(line.args[1])
        mon.types = None  # `formeChange` runs `setSpecies`, which resets the types

    def _swap(self, line: _Line) -> None:
        _arity(line, 2, 2)
        side, letter, name = self._ref(line.args[0], line)
        position = _integer(line.args[1], line)
        target = chr(ord("a") + position)
        if letter is None or letter not in side.slots or target not in side.slots:
            raise MalformedLineError("swap names an unknown position", line.raw)
        if side.slots[letter] is None or self._shown.get((side.player, letter)) != name:
            raise MalformedLineError("swap of a Pokemon that is not in that position", line.raw)
        side.slots[letter], side.slots[target] = side.slots[target], side.slots[letter]
        here, there = (side.player, letter), (side.player, target)
        shown_first, shown_second = self._shown.pop(here, None), self._shown.pop(there, None)
        if shown_first is not None:
            self._shown[there] = shown_first
        if shown_second is not None:
            self._shown[here] = shown_second
        held_first, held_second = self._disguise.pop(here, None), self._disguise.pop(there, None)
        if held_first is not None:
            self._disguise[there] = held_first
        if held_second is not None:
            self._disguise[here] = held_second
        for slot in (letter, target):
            index = side.slots[slot]
            if index is not None:
                side.roster[index].active_slot = slot

    # -- Illusion ---------------------------------------------------------------

    @staticmethod
    def _index_of(side: SideState, mon: PokemonState) -> int:
        return next(index for index, held in enumerate(side.roster) if held is mon)

    def _is_added(self, side: SideState, key: tuple[str, str]) -> bool:
        disguise = self._disguise.get(key)
        return disguise is not None and (side.player, disguise.index) in self._pairs

    def _phantom(self, side: SideState, mon: PokemonState) -> PokemonState:
        """A fresh entry for a Pokemon that is already shown in another position."""
        if mon.own:
            extra = copy.deepcopy(mon)
            extra.moves_revealed = []
        else:
            extra = PokemonState(own=False, name=mon.name, base_species=mon.base_species)
        extra.phantom = True
        side.roster.append(extra)
        return extra

    def _drop_phantom(self, side: SideState, index: int) -> None:
        """Remove an added entry and renumber everything that names a later one."""
        side.roster.pop(index)

        def shift(held: int) -> int:
            return held - 1 if held > index else held

        for letter, held in side.slots.items():
            if held is not None:
                side.slots[letter] = shift(held)
        for (player, _), disguise in self._disguise.items():
            if player == side.player:
                disguise.index = shift(disguise.index)
        renumbered = {}
        for (player, added), pair in self._pairs.items():
            if player == side.player:
                pair.owner, pair.phantom = shift(pair.owner), shift(pair.phantom)
            renumbered[(player, pair.phantom if player == side.player else added)] = pair
        self._pairs = renumbered

    def _end_pair(self, side: SideState, pair: _Pair) -> None:
        """The pair is resolved: forget it, its positions' disguises and the added entry."""
        del self._pairs[(side.player, pair.phantom)]
        for key in [
            key
            for key, d in self._disguise.items()
            if key[0] == side.player and d.index in (pair.owner, pair.phantom)
        ]:
            del self._disguise[key]
        self._drop_phantom(side, pair.phantom)

    def _replace(self, line: _Line) -> None:
        """`|replace|POKEMON|DETAILS`: the disguise at that position ended; POKEMON is the real one.

        Everything the channel showed at the position since the switch-in belongs to the real
        Pokemon (its own state does not change, only who it is shown as), and the Pokemon it was
        shown as is put back as it was before that switch-in.
        """
        _arity(line, 2, 3)
        side, letter, name = self._ref(line.args[0], line)
        shown_index = None if letter is None else side.slots.get(letter)
        if letter is None or shown_index is None:
            raise MalformedLineError("replace names a position nobody holds", line.raw)
        details = self._details(line.args[1], line)
        key = (side.player, letter)
        if key in self._disguise:
            real = self._locate_real(side, key, name, details, line)
            self._settle(side, letter, real, line)
        else:
            # An own request settled it already (`_settle_own`): the position holds the real one.
            real = side.roster[shown_index]
            if not (real.own and real.name == name):
                raise IllusionError("a replace with no disguise on record", line.raw)
        if real.base_species != details.base_species:
            raise IllusionError("the Pokemon shown in a replace is not the roster's", line.raw)
        real.name, real.brought = name, True
        real.species, real.level = details.species, details.level
        real.gender, real.shiny = details.gender or None, details.shiny
        real.ability = self.vocab.ability(ILLUSION)
        self._shown[key] = name
        hp = _optional(line, 2)
        if hp is not None:
            self._set_hp(real, hp, line)

    def _locate_real(
        self, side: SideState, key: tuple[str, str], name: str, details: _Details, line: _Line
    ) -> PokemonState:
        """The roster entry a replace names: the one called `name`, else the one never shown yet."""
        shown = self._disguise[key].index
        skip = {shown}
        for pair in self._pairs.values():
            if shown in (pair.owner, pair.phantom):
                skip |= {pair.owner, pair.phantom}
        candidates = [
            mon for index, mon in enumerate(side.roster) if index not in skip and not mon.phantom
        ]
        found = [mon for mon in candidates if mon.name == name] or [
            mon
            for mon in candidates
            if not mon.own
            and mon.name is None
            and mon.base_species == details.base_species
            and mon.species in (None, details.species)
        ]
        if len(found) > 1:
            raise IllusionError("a replace that fits more than one Pokemon", line.raw)
        if not found:
            raise IllusionError("a replace names a Pokemon outside the roster", line.raw)
        return found[0]

    def _settle(self, side: SideState, letter: str, real: PokemonState, line: _Line) -> None:
        """The Pokemon at `letter` is the roster entry `real`: carry over what was seen there
        since its switch-in and put the Pokemon it was shown as back."""
        disguise = self._disguise.pop((side.player, letter))
        shown = side.roster[disguise.index]
        if real.active_slot not in (None, letter):
            raise IllusionError("the replaced Pokemon is already in play", line.raw)
        if disguise.before is None:
            raise IllusionError("the disguise was not on the roster", line.raw)
        added = self._pairs.get((side.player, disguise.index))
        owned = next(
            (
                p
                for (pl, _), p in self._pairs.items()
                if pl == side.player and p.owner == disguise.index
            ),
            None,
        )
        self._carry(shown, disguise.before, real, letter)
        if added is not None:
            # The added entry was the Illusion user; the entry shown first is real as it stands.
            self._end_pair(side, added)
        elif owned is not None:
            self._settle_owned(side, disguise, owned)
        else:
            self._restore(side, disguise)
        side.slots[letter] = self._index_of(side, real)

    def _settle_owned(self, side: SideState, disguise: _Disguise, pair: _Pair) -> None:
        """The entry shown first was the Illusion user (its state is carried to it already);
        the added entry is the Pokemon it was shown as, and takes its place again."""
        twin = side.roster[pair.phantom]
        restored = self._restore(side, disguise)
        self._carry(twin, pair.phantom_before, restored, twin.active_slot)
        restored.name, restored.species, restored.level = twin.name, twin.species, twin.level
        restored.gender, restored.shiny = twin.gender, twin.shiny
        if twin.active_slot is not None:
            side.slots[twin.active_slot] = disguise.index
        self._end_pair(side, pair)

    def _restore(self, side: SideState, disguise: _Disguise) -> PokemonState:
        """Put the Pokemon the disguise was shown as back as it was before the switch-in."""
        assert disguise.before is not None
        restored = side.roster[disguise.index] = disguise.before
        if restored.active_slot is not None:
            # It was in play and left when the Illusion user took its place.
            self._switch_out(restored)
        return restored

    def _carry(
        self, shown: PokemonState, before: PokemonState, real: PokemonState, letter: str | None
    ) -> None:
        """Move what `shown` gained since `before` (the entry at the switch-in) to `real`."""
        real.hp_num, real.hp_den = shown.hp_num, shown.hp_den
        real.status, real.fainted = shown.status, shown.fainted
        if (shown.item, shown.item_known, shown.item_lost) != (
            before.item,
            before.item_known,
            before.item_lost,
        ):
            real.item, real.item_known, real.item_lost = (
                shown.item,
                shown.item_known,
                shown.item_lost,
            )
        if shown.ability != before.ability:
            real.ability = shown.ability
        if shown.mega != before.mega:
            real.mega, real.mega_stone = shown.mega, shown.mega_stone
        real.boosts, real.volatiles = dict(shown.boosts), set(shown.volatiles)
        real.derived = dict(shown.derived)
        real.types = None if shown.types is None else list(shown.types)
        real.temp_species, real.ability_suppressed = shown.temp_species, shown.ability_suppressed
        real.moves_revealed = real.moves_revealed + [
            move
            for move in shown.moves_revealed
            if move not in real.moves_revealed and move not in before.moves_revealed
        ]
        real.recent = copy.deepcopy(shown.recent)
        real.active_slot, real.brought = letter, True

    def _settle_own(self, entries: Sequence[Mapping[str, Any]], line: _Line) -> None:
        """An own request names the Pokemon in each position, so an Illusion user shown as another
        Pokemon is found here, before its `|replace|` (the own channel shows the disguise)."""
        side = self.state.own
        illusion = self.vocab.ability(ILLUSION)
        keyed = [((side.player, chr(ord("a") + i)), entry) for i, entry in enumerate(entries)]
        # An added entry last: its twin, if it was the Illusion user, settles both.
        keyed.sort(key=lambda item: self._is_added(side, item[0]))
        for key, entry in keyed:
            disguise = self._disguise.get(key)
            if disguise is None:
                continue
            real = self._own_by_ident(entry, line)
            added = self._pairs.get((side.player, disguise.index))
            shown_here = self._index_of(side, real) == (added.owner if added else disguise.index)
            is_illusion = (
                self.vocab.ability(entry.get("ability") or entry["baseAbility"]) == illusion
            )
            if not entry.get("active") or shown_here or not is_illusion:
                # Nothing to find: it is the Pokemon shown (an added entry's twin settles it).
                del self._disguise[key]
                continue
            self._settle(side, key[1], real, line)
        self._settle_left_pairs(entries, line)

    def _settle_left_pairs(self, entries: Sequence[Mapping[str, Any]], line: _Line) -> None:
        """A pair whose first-shown entry left play: if the request finds the real Pokemon in the
        added entry's position, the first one was the Illusion user, and the own team (its
        abilities are known) names it when exactly one Pokemon could be."""
        side = self.state.own
        illusion = self.vocab.ability(ILLUSION)
        for pair in [p for (player, _), p in self._pairs.items() if player == side.player]:
            twin = side.roster[pair.phantom]
            owner = side.roster[pair.owner]
            if (
                twin.active_slot is None
                or owner.active_slot is not None
                or pair.owner_before is None
            ):
                continue
            position = ord(twin.active_slot) - ord("a")
            if self._own_by_ident(entries[position], line) is not owner:
                continue
            users = [
                m for m in side.roster if m.ability == illusion and m is not owner and not m.phantom
            ]
            if len(users) == 1:
                self._carry(owner, pair.owner_before, users[0], None)
                self._settle_owned(side, _Disguise(pair.owner, pair.owner_before), pair)

    def _activate(self, line: _Line) -> None:
        _arity(line, 1, 4)
        if len(line.args) > 1 and to_id(line.args[1]) == "moveskillswap":
            # Both abilities changed; what each now holds is not stated per Pokemon.
            self._mon(line.args[0], line).ability = None
            if "of" in line.tags:
                self._mon(line.tags["of"], line).ability = None

    def _mega(self, line: _Line) -> None:
        # `|-mega|POKEMON|SPECIES|STONE`: the species is the one shown before the forme change.
        _arity(line, 3, 3)
        mon = self._mon(line.args[0], line)
        self.vocab.species(line.args[1])
        mon.mega, mon.mega_stone = True, self.vocab.item(line.args[2])
        mon.can_mega_evo = False

    def _transform(self, line: _Line) -> None:
        # `|-transform|POKEMON|TARGET`: the second field is the copied Pokemon's id.
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        target = self._mon(line.args[1], line)
        mon.temp_species = target.temp_species or target.species
        mon.volatiles.add("transform")
        mon.boosts = dict(target.boosts)  # `transformInto` copies the target's stat stages
        # The engine replaces the whole move list with the target's (and puts it back on
        # switching out), so the moves shown so far are not the ones it has now; what it
        # uses from here on is. The target's own list is not copied: the log does not say
        # which of its moves a copy keeps visible.
        if mon.moves_base is None:
            mon.moves_base = mon.moves_revealed
        mon.moves_revealed = []

    # -- actions and hit points ----------------------------------------------

    def _move(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        move = self.vocab.move(line.args[1])
        target = _optional(line, 2)
        shown: str | None = None
        if target and target != "null":  # "null": a move with no target ([notarget])
            side, letter, _ = self._ref(target, line)
            if letter is not None:
                shown = f"{'own' if side is self.state.own else 'opponent'}:{letter}"
        self._require_stall_outcome(line)
        self._actor = mon  # a called move (`[from]`) is still its user's
        if "from" not in line.tags and move not in mon.moves_revealed:
            mon.moves_revealed.append(move)
        if "from" in line.tags and line.tags["from"] != "lockedmove":
            return  # not the Pokemon's own action: a called, reflected or copied move
        if self.vocab.stall_moves is None:
            raise VocabularyError("no stall-moves table for this engine and format")
        recent = mon.recent
        recent.last_move, recent.last_target, recent.last_move_turn = move, shown, self.state.turn
        self._acted(mon)
        if move in self.vocab.stall_moves:
            self._stalling = mon

    def _faint(self, line: _Line) -> None:
        _arity(line, 1, 1)
        side, letter, _ = self._ref(line.args[0], line)
        mon = self._mon(line.args[0], line)
        mon.fainted, mon.hp_num = True, 0
        if letter is not None and side.slots.get(letter) == self._index_of(side, mon):
            side.slots[letter] = None
            self._disguise.pop((side.player, letter), None)
        self._switch_out(mon)
        mon.status = None

    def _hp_line(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        before, units = mon.hp_num, mon.hp_den
        self._set_hp(mon, line.args[1], line)
        self._reveal_from(mon, line)
        if line.kind == "-damage" and before is not None and mon.hp_den == units:
            lost = before - (mon.hp_num or 0)
            if lost > 0:
                mon.recent.hp_taken += lost
                if self._actor is not None and "from" not in line.tags:
                    self._actor.recent.hp_dealt += lost

    def _sethp(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        self._set_hp(mon, line.args[1], line, status=False)

    def _status(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        mon.status = self.vocab.status(line.args[1])
        self._reveal_from(mon, line)

    def _curestatus(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        self.vocab.status(line.args[1])
        mon.status = None

    def _cureteam(self, line: _Line) -> None:
        _arity(line, 1, 1)
        side, _, _ = self._ref(line.args[0], line)
        for mon in side.roster:
            mon.status = None

    # -- boosts --------------------------------------------------------------

    def _stat(self, text: str, line: _Line) -> str:
        if text not in STAT_KEYS:
            raise UnknownFieldError(f"unknown stat {text!r}", line.raw)
        return text

    def _boost(self, line: _Line) -> None:
        _arity(line, 3, 3)
        mon = self._mon(line.args[0], line)
        stat = self._stat(line.args[1], line)
        amount = _integer(line.args[2], line)
        if line.kind == "-boost":
            value = mon.boosts[stat] + amount
        elif line.kind == "-unboost":
            value = mon.boosts[stat] - amount
        else:
            value = amount
        if abs(value) > MAX_BOOST:
            raise MalformedLineError(
                f"boost {value} is outside -{MAX_BOOST}..{MAX_BOOST}", line.raw
            )
        mon.boosts[stat] = value
        self._reveal_from(mon, line)

    def _swapboost(self, line: _Line) -> None:
        _arity(line, 2, 3)
        source, target = self._mon(line.args[0], line), self._mon(line.args[1], line)
        named = _optional(line, 2)
        stats = list(STAT_KEYS) if named is None else [stat.strip() for stat in named.split(",")]
        for stat in stats:
            self._stat(stat, line)
            source.boosts[stat], target.boosts[stat] = target.boosts[stat], source.boosts[stat]

    def _invertboost(self, line: _Line) -> None:
        _arity(line, 1, 1)
        mon = self._mon(line.args[0], line)
        mon.boosts = {stat: -value for stat, value in mon.boosts.items()}

    def _clearboost(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self._mon(line.args[0], line).boosts = zero_boosts()

    def _clearallboost(self, line: _Line) -> None:
        _arity(line, 0, 0)
        for side in (self.state.own, self.state.opponent):
            for mon in side.roster:
                mon.boosts = zero_boosts()

    def _clearpositiveboost(self, line: _Line) -> None:
        _arity(line, 3, 3)
        mon = self._mon(line.args[0], line)
        self._mon(line.args[1], line)
        mon.boosts = {stat: min(value, 0) for stat, value in mon.boosts.items()}

    def _clearnegativeboost(self, line: _Line) -> None:
        _arity(line, 1, 1)
        mon = self._mon(line.args[0], line)
        mon.boosts = {stat: max(value, 0) for stat, value in mon.boosts.items()}

    def _copyboost(self, line: _Line) -> None:
        _arity(line, 2, 2)
        # The pinned engine (Psych Up, Costar) sets the first Pokemon's boosts from the second's,
        # the reverse of what sim/SIM-PROTOCOL.md says.
        copier, copied = self._mon(line.args[0], line), self._mon(line.args[1], line)
        copier.boosts = dict(copied.boosts)

    # -- field ---------------------------------------------------------------

    def _weather(self, line: _Line) -> None:
        _arity(line, 1, 1)
        found = None if line.args[0] == "none" else self.vocab.weather(line.args[0])
        if found != self.state.weather:  # `[upkeep]` repeats the weather that is already up
            self.state.weather_started = None if found is None else self.state.turn
        self.state.weather = found

    def _field_effect(self, line: _Line) -> tuple[str, str]:
        _arity(line, 1, 1)
        kind, found = self._resolve_effect(line.args[0], line)
        if kind not in ("move", "ability") and found not in self.vocab.terrains:
            raise UnknownFieldError(f"{line.args[0]!r} is not a field condition", line.raw)
        return kind, found

    def _fieldstart(self, line: _Line) -> None:
        _, found = self._field_effect(line)
        if found in self.vocab.terrains:
            if self.state.terrain != found:
                self.state.terrain_started = self.state.turn
            self.state.terrain = found
        else:
            self.state.field_conditions.add(found)
            self.state.field_started.setdefault(found, self.state.turn)

    def _fieldend(self, line: _Line) -> None:
        _, found = self._field_effect(line)
        if found in self.vocab.terrains:
            if self.state.terrain == found:
                self.state.terrain, self.state.terrain_started = None, None
        else:
            self.state.field_conditions.discard(found)
            self.state.field_started.pop(found, None)

    def _side_condition(self, line: _Line) -> tuple[SideState, str]:
        _arity(line, 2, 2)
        match = _SIDE.match(line.args[0])
        if match is None:
            raise MalformedLineError(f"{line.args[0]!r} is not a side", line.raw)
        kind, found = self._resolve_effect(line.args[1], line)
        if kind == "item":
            raise UnknownFieldError(f"{line.args[1]!r} is not a side condition", line.raw)
        return self._side(match.group(1), line), found

    def _sidestart(self, line: _Line) -> None:
        side, found = self._side_condition(line)
        side.condition_started.setdefault(found, self.state.turn)
        side.conditions[found] = side.conditions.get(found, 0) + 1

    def _sideend(self, line: _Line) -> None:
        side, found = self._side_condition(line)
        side.conditions.pop(found, None)
        side.condition_started.pop(found, None)

    def _swapsideconditions(self, line: _Line) -> None:
        _arity(line, 0, 0)
        own, opponent = self.state.own, self.state.opponent
        own.conditions, opponent.conditions = opponent.conditions, own.conditions
        own.condition_started, opponent.condition_started = (
            opponent.condition_started,
            own.condition_started,
        )

    # -- volatile statuses, items, abilities ---------------------------------

    def _volatile_start(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        kind, found = self._volatile(line.args[1], line)
        if kind == "ability":
            mon.ability = found
        mon.volatiles.add(found)
        if kind != "ability":
            self._track_derived(mon, found, start=True)
        if found == "typechange":
            mon.types = self._type_change(line)

    def _track_derived(self, mon: PokemonState, found: str, *, start: bool) -> None:
        """Keep the latest derived suffix per condition id (`stockpile1`, then `stockpile2`; the
        engine ends it as `-end|Stockpile`, and Perish Song counts down `perish3`..`perish0`
        with no `-end`). A version 1 table resolves nothing, so tracks nothing."""
        if self.vocab.conditions is None:
            return
        base = self.vocab.resolve_volatile(found)
        if start and base != found and found.startswith(base):
            mon.derived[base] = found[len(base) :]
        elif not start:
            mon.derived.pop(base, None)

    def _type_change(self, line: _Line) -> list[str]:
        """The Pokemon's types after `-start|POKEMON|typechange|TYPE[/TYPE]`: type ids of the
        vocabulary, or the engine's own typeless marker `???` (Burn Up). A copy
        (`[from] move: Reflect Type`) carries no type argument: the types changed to ones the
        channel does not say, an empty list. Any other type is refused."""
        given = line.args[2:]
        if not given:
            return []
        return [
            TYPELESS if part == TYPELESS else self.vocab.type_(part) for part in given[0].split("/")
        ]

    def _volatile_end(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        kind, found = self._volatile(line.args[1], line)
        mon.volatiles.discard(found)
        if kind != "ability":
            self._track_derived(mon, found, start=False)
        if found == "typechange":  # Mimicry: `-end|POKEMON|typechange|[silent]`
            mon.types = None

    def _item(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        mon.item, mon.item_known, mon.item_lost = self.vocab.item(line.args[1]), True, False
        # Frisk (#1171): `|-item|TARGET|ITEM|[from] ability: Frisk|[of] FRISKER`. The item is the
        # target's; the ability is the Frisk holder's, who is named by `[of]`.
        holder = line.tags.get("of")
        if holder is not None and to_id(line.tags.get("from", "")) == "abilityfrisk":
            self._mon(holder, line).ability = self.vocab.ability("Frisk")

    def _enditem(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        # An empty name is the engine's Fling: the thrown item was already gone when the move's
        # `onUpdate` ran (a Grassy Seed eaten on the hit that started the terrain), so
        # `getItem()` names nothing and the line says so. Any other empty name is refused.
        if not (line.args[1] == "" and line.tags.get("from") == FLING_SOURCE):
            self.vocab.item(line.args[1])
        mon.item, mon.item_known, mon.item_lost = None, True, True

    def _ability(self, line: _Line) -> None:
        _arity(line, 2, 3)
        extra = _optional(line, 2)
        # A third field is `boost` (Intimidate) or the ability being replaced (Trace).
        if extra not in (None, "boost"):
            self.vocab.ability(extra or "")
        mon = self._mon(line.args[0], line)
        mon.ability, mon.ability_suppressed = self.vocab.ability(line.args[1]), False
        source, holder = line.tags.get("from"), line.tags.get("of")
        if source is not None and holder is not None and to_id(source) in _ABILITY_COPIES:
            self._mon(holder, line).ability = mon.ability

    def _endability(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self._mon(line.args[0], line).ability_suppressed = True


class ChannelReducer(_Reducer):
    """The reducer fed one line at a time.

    ``state`` is what `reduce_channel` returns for the lines fed so far. It is
    live and changes with the next ``feed``: read it (or encode it) first.
    """


def reduce_channel(
    player: str,
    lines: Sequence[str],
    team: Sequence[Mapping[str, Any]],
    vocab: Vocabulary,
) -> BattleState:
    """Reduce one player's channel lines and own team to a `BattleState`.

    `lines` are the player's own protocol lines (ADR-008 §5), normally up to and
    including the request being answered; `team` is that player's own
    `PrivateTeamSpec.team`. Raises a `ReducerError` on anything unknown.
    """
    reducer = ChannelReducer(player, team, vocab)
    for raw in lines:
        reducer.feed(raw)
    return reducer.state
