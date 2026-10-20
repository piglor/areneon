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
* Deterministic. The same lines and team always give the same state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    MalformedLineError,
    OpponentSecretError,
    ReducerError,
    UnknownFieldError,
    UnknownLineError,
    VocabularyError,
)
from .state import STAT_KEYS, BattleState, PokemonState, SideState, zero_boosts
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

# Lines whose text may hold `[` or `|`-free prose: no tag parsing.
_FREE_TEXT = frozenset({"uhtml", "uhtmlchange", "-message", "-hint", "rule", "tier"})

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
}

_TAG = re.compile(r"^\[([a-z]+)\]\s*(.*)$")
_REF = re.compile(r"^(p[1-4])([a-d])?: (.+)$")
_SIDE = re.compile(r"^(p[1-4]): ")
_LEVEL = re.compile(r"^L(\d{1,3})$")
_HP = re.compile(r"^(\d+)(?:/(\d+)([gyr])?)?$")
MAX_VOLATILE_LENGTH = 64  # an opaque volatile id is short; a longer one is not an id
_VOLATILE = re.compile(r"^(?:(move|ability|item): )?([A-Za-z][A-Za-z0-9 '.-]*)$")
# Colour class the champions mod appends to a shared HP at exactly 50% or 20%.
_HP_COLOURS = {50: "gy", 20: "yr"}
# `[from]` effects after which the `[of]` Pokemon holds the new ability too.
_ABILITY_COPIES = frozenset({"abilitytrace", "moveroleplay"})
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
_SET_KEYS = ("species", "name", "item", "ability", "moves", "level", "gender")


@dataclass
class _Line:
    raw: str
    kind: str
    args: list[str]
    tags: dict[str, str] = field(default_factory=dict)


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
        if name in tags and (tags[name] != value or value):
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
        for pokemon_set in team:
            self.state.own.roster.append(self._own_mon(pokemon_set))
        names = [mon.name for mon in self.state.own.roster]
        if len(set(names)) != len(names):
            raise MalformedLineError(
                "own team has two Pokemon the protocol cannot tell apart by name"
            )
        self._handlers: dict[str, Callable[[_Line], None]] = {
            "player": self._player,
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
            if index is not None and side.roster[index].name == name:
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
        """A volatile's id. The engine-data export has no volatile table (engine-data v2
        would), so its shape is checked and the id kept as an opaque public string; only
        `ability:` names are looked up, because they also reveal the holder's ability."""
        match = _VOLATILE.match(text)
        if match is None or len(text) > MAX_VOLATILE_LENGTH:
            raise MalformedLineError(f"{text!r} is not a volatile id", line.raw)
        kind, name = match.groups()
        found = to_id(name)
        if kind == "ability":
            return kind, self.vocab.ability(name)
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
        self.state.turn = _integer(line.args[0], line)

    def _win(self, line: _Line) -> None:
        _arity(line, 1, 1)
        for label, side in (("own", self.state.own), ("opponent", self.state.opponent)):
            if side.username is not None and side.username == line.args[0]:
                self.state.winner = label
                return
        raise MalformedLineError(f"{line.args[0]!r} is neither player", line.raw)

    def _tie(self, line: _Line) -> None:
        _arity(line, 0, 0)
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
        listed = [self._apply_request_mon(entry, line) for entry in side_data.get("pokemon", [])]
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

    def _apply_request_mon(self, entry: object, line: _Line) -> PokemonState:
        self._keys(entry, _POKEMON_KEYS, line, "request pokemon")
        assert isinstance(entry, dict)
        match = _SIDE.match(str(entry["ident"]))
        name = str(entry["ident"])[match.end() :] if match else ""
        mon = next((m for m in self.state.own.roster if m.name == name), None)
        if mon is None:
            raise MalformedLineError(
                f"request lists {entry['ident']!r}, who is not in the own team", line.raw
            )
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

    def _bind(self, side: SideState, name: str, details: _Details, line: _Line) -> PokemonState:
        for mon in side.roster:
            if mon.name == name:
                return mon
        if side.roster and side.roster[0].own:
            raise MalformedLineError(f"{name!r} is not in the own team", line.raw)
        for mon in side.roster:
            if (
                mon.name is None
                and mon.base_species == details.base_species
                and mon.species in (None, details.species)
            ):
                mon.name = name
                return mon
        if side.teamsize is not None and len(side.roster) >= side.teamsize:
            raise MalformedLineError(f"{name!r} is more Pokemon than the team size", line.raw)
        mon = PokemonState(own=False, name=name, base_species=details.base_species)
        side.roster.append(mon)
        return mon

    def _switch(self, line: _Line) -> None:
        _arity(line, 3, 3)
        side, letter, name = self._ref(line.args[0], line)
        if letter is None or letter not in side.slots:
            raise MalformedLineError("a switch needs an active position", line.raw)
        details = self._details(line.args[1], line)
        mon = self._bind(side, name, details, line)
        previous = side.slots[letter]
        if previous is not None:
            self._switch_out(side.roster[previous])
        index = side.roster.index(mon)
        side.slots[letter] = index
        self._switch_out(mon)
        mon.active_slot, mon.brought = letter, True
        mon.species, mon.level = details.species, details.level
        mon.gender, mon.shiny = details.gender or None, details.shiny
        self._set_hp(mon, line.args[2], line)

    def _switch_out(self, mon: PokemonState) -> None:
        mon.active_slot = None
        mon.boosts = zero_boosts()
        mon.volatiles = set()
        mon.temp_species = None
        mon.ability_suppressed = False

    def _detailschange(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        details = self._details(line.args[1], line)
        mon.species, mon.level = details.species, details.level
        mon.gender, mon.shiny = details.gender or None, details.shiny
        hp = _optional(line, 2)
        if hp is not None:
            self._set_hp(mon, hp, line)

    def _formechange(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        mon.temp_species = self.vocab.species(line.args[1])

    def _swap(self, line: _Line) -> None:
        _arity(line, 2, 2)
        side, letter, name = self._ref(line.args[0], line)
        position = _integer(line.args[1], line)
        target = chr(ord("a") + position)
        if letter is None or letter not in side.slots or target not in side.slots:
            raise MalformedLineError("swap names an unknown position", line.raw)
        if side.slots[letter] is None or side.roster[side.slots[letter] or 0].name != name:
            raise MalformedLineError("swap of a Pokemon that is not in that position", line.raw)
        side.slots[letter], side.slots[target] = side.slots[target], side.slots[letter]
        for slot in (letter, target):
            index = side.slots[slot]
            if index is not None:
                side.roster[index].active_slot = slot

    def _replace(self, line: _Line) -> None:
        raise UnknownLineError("|replace| (Illusion ending) is not supported", line.raw)

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

    # -- actions and hit points ----------------------------------------------

    def _move(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        move = self.vocab.move(line.args[1])
        target = _optional(line, 2)
        if target and target != "null":  # "null": a move with no target ([notarget])
            self._ref(target, line)
        if "from" not in line.tags and move not in mon.moves_revealed:
            mon.moves_revealed.append(move)

    def _faint(self, line: _Line) -> None:
        _arity(line, 1, 1)
        side, letter, _ = self._ref(line.args[0], line)
        mon = self._mon(line.args[0], line)
        mon.fainted, mon.hp_num = True, 0
        if letter is not None and side.slots.get(letter) == side.roster.index(mon):
            side.slots[letter] = None
        self._switch_out(mon)
        mon.status = None

    def _hp_line(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        self._set_hp(mon, line.args[1], line)
        self._reveal_from(mon, line)

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
        source, target = self._mon(line.args[0], line), self._mon(line.args[1], line)
        target.boosts = dict(source.boosts)

    # -- field ---------------------------------------------------------------

    def _weather(self, line: _Line) -> None:
        _arity(line, 1, 1)
        self.state.weather = None if line.args[0] == "none" else self.vocab.weather(line.args[0])

    def _field_effect(self, line: _Line) -> tuple[str, str]:
        _arity(line, 1, 1)
        kind, found = self._resolve_effect(line.args[0], line)
        if kind not in ("move", "ability") and found not in self.vocab.terrains:
            raise UnknownFieldError(f"{line.args[0]!r} is not a field condition", line.raw)
        return kind, found

    def _fieldstart(self, line: _Line) -> None:
        _, found = self._field_effect(line)
        if found in self.vocab.terrains:
            self.state.terrain = found
        else:
            self.state.field_conditions.add(found)

    def _fieldend(self, line: _Line) -> None:
        _, found = self._field_effect(line)
        if found in self.vocab.terrains:
            if self.state.terrain == found:
                self.state.terrain = None
        else:
            self.state.field_conditions.discard(found)

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
        side.conditions[found] = side.conditions.get(found, 0) + 1

    def _sideend(self, line: _Line) -> None:
        side, found = self._side_condition(line)
        side.conditions.pop(found, None)

    def _swapsideconditions(self, line: _Line) -> None:
        _arity(line, 0, 0)
        own, opponent = self.state.own, self.state.opponent
        own.conditions, opponent.conditions = opponent.conditions, own.conditions

    # -- volatile statuses, items, abilities ---------------------------------

    def _volatile_start(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        kind, found = self._volatile(line.args[1], line)
        if kind == "ability":
            mon.ability = found
        mon.volatiles.add(found)

    def _volatile_end(self, line: _Line) -> None:
        _arity(line, 2, 3)
        mon = self._mon(line.args[0], line)
        _, found = self._volatile(line.args[1], line)
        mon.volatiles.discard(found)

    def _item(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
        mon.item, mon.item_known, mon.item_lost = self.vocab.item(line.args[1]), True, False

    def _enditem(self, line: _Line) -> None:
        _arity(line, 2, 2)
        mon = self._mon(line.args[0], line)
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
    reducer = _Reducer(player, team, vocab)
    for raw in lines:
        reducer.feed(raw)
    return reducer.state
