"""The battle-state spec as the packer and the projection read it (ADR-030 G4, #1464).

``SpecIndex`` reads a ``BattleStateSpec`` record through the contracts reader (bound to the
engine-data version 2 table) and gives each field its full shape (the group's dimensions, then the
field's own), the ids of its vocabulary and the index codec: 0 none, 1 unknown, 2 + the id's
position in the sorted union of its lists (the record's rule). No count or id is written here:
counts are the record's (which the TS reader holds to the engine), ids the table's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from areneon_contracts.battle_state_spec import (
    battle_state_spec_hash,
    read_battle_state_spec,
    vocabulary_ids,
)

# The fields the projection reads that record v1 does not hold: the four gaps #1464 found (the own
# Pokemon that were not brought, the own base species over a Mega Evolution is the preview
# species, raw numbered volatile ids, a fifth revealed move) and what the rebuild needed besides.
REQUIRED = (
    "active.boosts",
    "seat_active.own_moves_used",
    "seat_active.protocol_derived",
    "seat_active.protocol_volatiles",
    "seat_active.volatile_numbered",
    "seat_active.volatile_oldest_started",
    "seat_active.volatile_stat_raw",
    "seat_active.volatile_started_count",
    "seat_roster.ability_suppressed",
    "seat_roster.hp_den_shown",
    "seat_roster.own_can_mega_evo",
    "seat_roster.own_moves",
    "seat_roster.own_pp",
    "seat_roster.own_stats",
    "seat_roster.revealed_extra",
    "seat_roster.revealed_extra_at",
)
# The id a decoded "unknown" index stands for: no vocabulary holds it.
UNKNOWN_ID = "?unknown"
SEATS = ("p1", "p2")


class ProjectionError(ValueError):
    """A record, view or input the projection cannot work with (fixed words)."""


class SpecIndex:
    """A read record with its shapes and vocabularies."""

    def __init__(self, spec: Mapping[str, Any], table: Mapping[str, Any]) -> None:
        read_battle_state_spec(spec, table)
        self.spec = spec
        self.spec_hash = battle_state_spec_hash(spec)
        self.fields: dict[str, Mapping[str, Any]] = {f["id"]: f for f in spec["fields"]}
        missing = [name for name in REQUIRED if name not in self.fields]
        if missing:
            raise ProjectionError(
                "the record lacks fields the projection reads: " + ", ".join(missing)
            )
        self.groups: dict[str, Mapping[str, Any]] = {g["id"]: g for g in spec["groups"]}
        self._sizes: dict[str, int] = {}
        self._dim_ids: dict[str, list[str]] = {}
        for dim in spec["dims"]:
            if dim["kind"] == "count":
                self._sizes[dim["symbol"]] = int(dim["size"])
            else:
                ids = vocabulary_ids(table, dim["lists"])
                self._dim_ids[dim["symbol"]] = ids
                self._sizes[dim["symbol"]] = len(ids)
        self._vocab: dict[str, tuple[list[str], dict[str, int]]] = {}
        for field in spec["fields"]:
            if field["value"]["kind"] == "vocabulary":
                ids = vocabulary_ids(table, field["value"]["lists"])
                self._vocab[field["id"]] = (ids, {name: 2 + i for i, name in enumerate(ids)})

    def field(self, field_id: str) -> Mapping[str, Any]:
        found = self.fields.get(field_id)
        if found is None:
            raise ProjectionError(f"{field_id} is not a field of the record")
        return found

    def size(self, symbol: str) -> int:
        return self._sizes[symbol]

    def ids(self, symbol: str) -> list[str]:
        """The ids a vocabulary dimension indexes, in order."""
        return self._dim_ids[symbol]

    def shape(self, field_id: str) -> tuple[int, ...]:
        field = self.field(field_id)
        symbols = [*self.groups[field["group"]]["shape"], *field["extra"]]
        return tuple(self._sizes[s] for s in symbols)

    def dim_or_vocab(self, field_id: str) -> list[str]:
        """The ids a vocabulary-valued field indexes."""
        return self._vocab[field_id][0]

    def encode_id(self, field_id: str, name: str | None) -> int:
        if name is None:
            return 0
        return self._vocab[field_id][1].get(name, 1)

    def decode_id(self, field_id: str, index: int) -> str | None:
        if index == 0:
            return None
        if index == 1:
            return UNKNOWN_ID
        return self._vocab[field_id][0][index - 2]

    def zeros(self, field_id: str) -> Any:
        return _filled(self.shape(field_id), 0)


def _filled(shape: tuple[int, ...], value: int) -> Any:
    if not shape:
        return value
    return [_filled(shape[1:], value) for _ in range(shape[0])]
