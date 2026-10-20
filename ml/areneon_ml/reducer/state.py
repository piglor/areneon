"""The reduced battle state of one player's channel (ADR-010 §3).

Perspective names, never `p1`/`p2`: `own` is the player the channel belongs to
and `opponent` the other side. The opponent's HP is the percentage the engine
shows (denominator 100); anything else it holds was revealed by a line. Own
exact values (stats, all four moves, PP) come from that player's own requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATE_SCHEMA_VERSION = 1
STAT_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def zero_boosts() -> dict[str, int]:
    return {stat: 0 for stat in STAT_KEYS}


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
    moves_revealed: list[str] = field(default_factory=list)
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

    @property
    def hp_fraction(self) -> float | None:
        if self.hp_num is None or not self.hp_den:
            return None
        return self.hp_num / self.hp_den

    def to_dict(self) -> dict[str, Any]:
        fraction = self.hp_fraction
        return {
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
        }


@dataclass
class SideState:
    player: str
    username: str | None = None
    teamsize: int | None = None
    roster: list[PokemonState] = field(default_factory=list)
    slots: dict[str, int | None] = field(default_factory=dict)
    conditions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "username": self.username,
            "teamsize": self.teamsize,
            "roster": [mon.to_dict() for mon in self.roster],
            "slots": dict(sorted(self.slots.items())),
            "conditions": dict(sorted(self.conditions.items())),
        }


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
            "own": self.own.to_dict(),
            "opponent": self.opponent.to_dict(),
        }
