"""The reduced battle state of one player's channel (ADR-010 §3).

Perspective names, never `p1`/`p2`: `own` is the player the channel belongs to
and `opponent` the other side. The opponent's HP is the percentage the engine
shows (denominator 100); anything else it holds was revealed by a line. Own
exact values (stats, all four moves, PP) come from that player's own requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 2: `recent` per Pokemon, `condition_turns` per side, `elapsed` on the field (#1013).
STATE_SCHEMA_VERSION = 2
STAT_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def zero_boosts() -> dict[str, int]:
    return {stat: 0 for stat in STAT_KEYS}


@dataclass
class Recent:
    """What happened lately, read from the player's own channel (ADR-010 erratum A4 item 7, #1013).

    A turn is the span from one `|turn|` line to the next. `active_turns` is the number of `|turn|`
    lines since the switch-in (the same count as the world facts' `active_turns`). HP amounts are
    the sum of the Pokemon's `-damage` decreases in the units its side shows: exact hit points for
    an own Pokemon, points of 100 for an opponent's (damage it dealt is in the units of the Pokemon
    that took it). Everything resets on switching out or fainting."""

    active_turns: int = 0
    last_move: str | None = None
    last_target: str | None = (
        None  # `own:a` / `opponent:b` as the position was shown; None: no target
    )
    last_move_turn: int | None = None
    # The place among the turn's actions (moves and `|cant|` lines, from 1) of its latest action.
    last_action_turn: int | None = None
    last_action_order: int | None = None
    hp_taken: int = 0  # this turn so far
    hp_dealt: int = 0
    hp_taken_last_turn: int = 0  # the turn before the current one
    hp_dealt_last_turn: int = 0
    protect_streak: int = 0
    stall_turn: int | None = None  # the turn of the latest successful stalling move

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_turns": self.active_turns,
            "last_move": self.last_move,
            "last_target": self.last_target,
            "last_move_turn": self.last_move_turn,
            "last_action_turn": self.last_action_turn,
            "last_action_order": self.last_action_order,
            "hp_taken_last_turn": self.hp_taken_last_turn,
            "hp_dealt_last_turn": self.hp_dealt_last_turn,
            "hp_taken_this_turn": self.hp_taken,
            "hp_dealt_this_turn": self.hp_dealt,
            "protect_streak": self.protect_streak,
        }


@dataclass
class PokemonState:
    own: bool
    name: str | None = None
    base_species: str = ""
    species: str | None = None
    temp_species: str | None = None
    level: int | None = None
    gender: str | None = None
    shiny: bool = False
    hp_num: int | None = None
    hp_den: int | None = None
    status: str | None = None
    fainted: bool = False
    brought: bool | None = None
    active_slot: str | None = None
    boosts: dict[str, int] = field(default_factory=zero_boosts)
    volatiles: set[str] = field(default_factory=set)
    # The derived suffix of the volatiles the engine prints under a numbered or stat-suffixed name
    # (`stockpile2` -> {"stockpile": "2"}, `protosynthesisatk` -> {"protosynthesis": "atk"}), the
    # latest one per condition id; ended by the engine's `-end` under the bare name (which the raw
    # `volatiles` ids do not follow). Version 2 tables only; bookkeeping, not part of `to_dict`.
    derived: dict[str, str] = field(default_factory=dict)
    # The types a `typechange` gave the Pokemon (type ids, `???` for none; empty when the channel
    # does not say which), None while its species' own. Ends on switching out.
    types: list[str] | None = None
    moves_revealed: list[str] = field(default_factory=list)
    # The move list the Pokemon returns to on switching out: set when Transform replaces the
    # current one (not part of `to_dict`: it is bookkeeping, not an observation).
    moves_base: list[str] | None = None
    item: str | None = None
    item_known: bool = False
    item_lost: bool = False
    ability: str | None = None
    ability_suppressed: bool = False
    mega: bool = False
    mega_stone: str | None = None
    # Own side only: exact values from the player's own team and requests.
    stats: dict[str, int] | None = None
    moves_own: list[str] | None = None
    pp: dict[str, int] = field(default_factory=dict)
    can_mega_evo: bool = False
    # An entry the channel forced the reducer to add (Illusion): the same Pokemon shown twice
    # (one is an Illusion user), or a Pokemon that fainted shown as another. See `reducer.py`.
    phantom: bool = False
    recent: Recent = field(default_factory=Recent)

    @property
    def hp_fraction(self) -> float | None:
        if self.hp_num is None or not self.hp_den:
            return None
        return self.hp_num / self.hp_den

    def to_dict(self) -> dict[str, Any]:
        fraction = self.hp_fraction
        document = {
            "name": self.name,
            "base_species": self.base_species,
            "species": self.species,
            "temp_species": self.temp_species,
            "level": self.level,
            "gender": self.gender,
            "shiny": self.shiny,
            "hp_num": self.hp_num,
            "hp_den": self.hp_den,
            "hp_fraction": None if fraction is None else round(fraction, 6),
            "status": self.status,
            "fainted": self.fainted,
            "brought": self.brought,
            "active_slot": self.active_slot,
            "boosts": dict(sorted(self.boosts.items())),
            "volatiles": sorted(self.volatiles),
            "moves_revealed": list(self.moves_revealed),
            "item": self.item,
            "item_known": self.item_known,
            "item_lost": self.item_lost,
            "ability": self.ability,
            "ability_suppressed": self.ability_suppressed,
            "mega": self.mega,
            "mega_stone": self.mega_stone,
            "stats": None if self.stats is None else dict(sorted(self.stats.items())),
            "moves_own": None if self.moves_own is None else list(self.moves_own),
            "pp": dict(sorted(self.pp.items())),
            "can_mega_evo": self.can_mega_evo,
            "recent": self.recent.to_dict(),
        }
        if self.types is not None:  # only a typechange says so: every other state keeps its keys
            document["types"] = list(self.types)
        if self.phantom:  # only an added entry says so: every other state keeps its keys
            document["phantom"] = True
        return document


@dataclass
class SideState:
    player: str
    username: str | None = None
    teamsize: int | None = None
    roster: list[PokemonState] = field(default_factory=list)
    slots: dict[str, int | None] = field(default_factory=dict)
    conditions: dict[str, int] = field(default_factory=dict)
    # The turn each condition began (the first layer's), for `condition_turns`.
    condition_started: dict[str, int] = field(default_factory=dict)

    def to_dict(self, turn: int | None = None) -> dict[str, Any]:
        document = {
            "player": self.player,
            "username": self.username,
            "teamsize": self.teamsize,
            "roster": [mon.to_dict() for mon in self.roster],
            "slots": dict(sorted(self.slots.items())),
            "conditions": dict(sorted(self.conditions.items())),
        }
        if turn is not None:  # turns elapsed since each condition began (0: this turn)
            document["condition_turns"] = {
                name: turn - began for name, began in sorted(self.condition_started.items())
            }
        return document


@dataclass
class BattleState:
    player: str
    format_name: str | None = None
    gametype: str | None = None
    turn: int = 0
    started: bool = False
    request_kind: str | None = None
    weather: str | None = None
    terrain: str | None = None
    field_conditions: set[str] = field(default_factory=set)
    # The turn the weather, the terrain and each field condition began, for `elapsed`.
    weather_started: int | None = None
    terrain_started: int | None = None
    field_started: dict[str, int] = field(default_factory=dict)
    winner: str | None = None
    own: SideState = field(default_factory=lambda: SideState(player=""))
    opponent: SideState = field(default_factory=lambda: SideState(player=""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "player": self.player,
            "format_name": self.format_name,
            "gametype": self.gametype,
            "turn": self.turn,
            "started": self.started,
            "request_kind": self.request_kind,
            "weather": self.weather,
            "terrain": self.terrain,
            "field_conditions": sorted(self.field_conditions),
            "winner": self.winner,
            "elapsed": {
                "weather": None
                if self.weather_started is None
                else self.turn - self.weather_started,
                "terrain": None
                if self.terrain_started is None
                else self.turn - self.terrain_started,
                "field_conditions": {
                    name: self.turn - began for name, began in sorted(self.field_started.items())
                },
            },
            "own": self.own.to_dict(self.turn),
            "opponent": self.opponent.to_dict(self.turn),
        }
