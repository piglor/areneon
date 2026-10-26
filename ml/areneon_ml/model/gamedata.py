"""The game vocabularies and numeric attribute tables of the encoder (ADR-010 §3).

Everything here is read from the hashed engine-data export (``EngineData``,
exported from the pinned engine's data by a TypeScript script): which species,
moves, items, abilities, statuses, weathers and terrains exist, and their
numeric attributes. No list of game content and no type chart, damage rule or
legality rule is written in model code. Each vocabulary reserves index 0 for
"none" (the slot is empty) and index 1 for "unknown" (it exists, the
perspective cannot see it); the engine's ids follow from index 2, in the
export's sorted order. Attribute columns are scaled by the largest absolute
value in the table, so the scale also comes from the data.
"""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex, validate

from areneon_ml.reducer import Vocabulary

NONE = 0
UNKNOWN = 1
RESERVED = 2


class GameDataError(ValueError):
    """The engine data is unusable, or an id is not in it."""


@dataclass(frozen=True)
class IdVocab:
    """An ordered id vocabulary: index 0 none, 1 unknown, then the ids."""

    name: str
    ids: tuple[str, ...]
    index: Mapping[str, int]

    @classmethod
    def of(cls, name: str, ids: Sequence[str]) -> IdVocab:
        return cls(name, tuple(ids), {item: i + RESERVED for i, item in enumerate(ids)})

    def __len__(self) -> int:
        return len(self.ids) + RESERVED

    def lookup(self, item: str) -> int:
        found = self.index.get(item)
        if found is None:
            raise GameDataError(f"{item!r} is not a {self.name} of the engine data")
        return found

    def lookup_or_unknown(self, item: str) -> int:
        return self.index.get(item, UNKNOWN)


def _scaled(rows: list[list[float]], width: int) -> torch.Tensor:
    """A [RESERVED + rows, width] float32 table: the two reserved rows zero, columns / max |x|."""
    table = [[0.0] * width for _ in range(RESERVED)] + rows
    for column in range(width):
        scale = max((abs(row[column]) for row in table), default=0.0)
        if scale > 1.0:
            for row in table:
                row[column] /= scale
    return torch.tensor(table, dtype=torch.float32).reshape(len(table), width)


def _one_hot(position: int, size: int) -> list[float]:
    return [1.0 if i == position else 0.0 for i in range(size)]


def _tensor_digest(*tensors: torch.Tensor) -> str:
    """SHA-256 over the shapes and little-endian float32 bytes of CPU tensors, in order."""
    digest = hashlib.sha256()
    for tensor in tensors:
        flat = tensor.detach().to(torch.float32).contiguous()
        digest.update(repr(tuple(flat.shape)).encode())
        digest.update(ctypes.string_at(flat.data_ptr(), flat.numel() * 4))
    return digest.hexdigest()


@dataclass(frozen=True)
class GameData:
    """Vocabularies and attribute tables of one engine-data export."""

    engine_data_hash: str
    showdown_commit: str
    format_id: str
    vocabulary: Vocabulary
    types: IdVocab
    species: IdVocab
    moves: IdVocab
    items: IdVocab
    abilities: IdVocab
    statuses: IdVocab
    weathers: IdVocab
    terrains: IdVocab
    species_attributes: torch.Tensor
    move_attributes: torch.Tensor
    item_attributes: torch.Tensor
    move_base_pp: tuple[int, ...]
    # The engine's type chart as exported: ``type_chart[defender][attacker]`` is the code
    # ``types[defender].damage_taken[attacker]`` (positions in ``types`` order, without the two
    # reserved indices). The codes are categories; their meaning is never read here.
    type_chart: tuple[tuple[int, ...], ...] = ()
    chart_codes: int = 0  # the number of distinct codes the chart can hold: largest code + 1
    species_types: tuple[tuple[int, ...], ...] = ()  # per species index: its type positions
    move_type: tuple[int, ...] = ()  # per move index: its type position, -1 for the reserved rows
    move_status: tuple[bool, ...] = ()  # per move index: the export's category is Status

    @property
    def attribute_digest(self) -> str:
        """Hash of the attribute tables as built: a change in how they are built changes it."""
        return _tensor_digest(self.species_attributes, self.move_attributes, self.item_attributes)

    @classmethod
    def from_table(cls, table: Mapping[str, Any]) -> GameData:
        """Validate the export against its contract, then index it."""
        validate("engine-data", table)
        types = IdVocab.of("type", [t["id"] for t in table["types"]])
        targets: list[str] = table["move_targets"]
        flags: list[str] = table["move_flags"]
        categories = sorted({m["category"] for m in table["moves"]})
        species_rows = [
            [*(1.0 if t in s["types"] else 0.0 for t in types.ids)]
            + [float(s["base_stats"][k]) for k in ("hp", "atk", "def", "spa", "spd", "spe")]
            + [float(s["weight_hg"])]
            for s in table["species"]
        ]
        move_rows = [
            [*(1.0 if t == m["type"] else 0.0 for t in types.ids)]
            + _one_hot(categories.index(m["category"]), len(categories))
            + [
                float(m["base_power"]),
                float(m["accuracy"]),
                1.0 if m["always_hits"] else 0.0,
                float(m["priority"]),
                float(m["pp"]),
                float(m["crit_ratio"]),
            ]
            + _one_hot(targets.index(m["target"]), len(targets))
            + [1.0 if f in m["flags"] else 0.0 for f in flags]
            for m in table["moves"]
        ]
        item_rows = [
            [1.0 if i["is_berry"] else 0.0, float(i["fling_power"])] for i in table["items"]
        ]
        return cls(
            engine_data_hash=sha256_hex(jcs(table)),
            showdown_commit=table["showdown_commit"],
            format_id=table["format_id"],
            vocabulary=Vocabulary.from_engine_data(table),
            types=types,
            species=IdVocab.of("species", [s["id"] for s in table["species"]]),
            moves=IdVocab.of("move", [m["id"] for m in table["moves"]]),
            items=IdVocab.of("item", [i["id"] for i in table["items"]]),
            abilities=IdVocab.of("ability", [a["id"] for a in table["abilities"]]),
            statuses=IdVocab.of("status", [s["id"] for s in table["statuses"]]),
            weathers=IdVocab.of("weather", [w["id"] for w in table["weathers"]]),
            terrains=IdVocab.of("terrain", [t["id"] for t in table["terrains"]]),
            species_attributes=_scaled(species_rows, len(types.ids) + 7),
            move_attributes=_scaled(
                move_rows, len(types.ids) + len(categories) + 6 + len(targets) + len(flags)
            ),
            item_attributes=_scaled(item_rows, 2),
            move_base_pp=(0, 0, *(int(m["pp"]) for m in table["moves"])),
            type_chart=tuple(tuple(int(c) for c in t["damage_taken"]) for t in table["types"]),
            chart_codes=1 + max((c for t in table["types"] for c in t["damage_taken"]), default=-1),
            species_types=((),) * RESERVED
            + tuple(
                tuple(types.lookup(t) - RESERVED for t in s["types"]) for s in table["species"]
            ),
            move_type=(-1,) * RESERVED
            + tuple(types.lookup(m["type"]) - RESERVED for m in table["moves"]),
            # "Status" is the export's own category name (the schema enum), not a rule value.
            move_status=(False,) * RESERVED
            + tuple(m["category"] == "Status" for m in table["moves"]),
        )

    def matchup_cells(self, move: int, species: int, shown: int) -> list[int]:
        """The chart cells of ``move`` against the first ``shown`` types of ``species``.

        Each cell is ``RESERVED`` + the engine's code, as a category; a missing second type is
        ``NONE``. A move with no type in the export (a reserved row) gives ``NONE`` throughout.
        """
        attacker = self.move_type[move]
        types = self.species_types[species]
        if len(types) > shown:
            raise GameDataError(
                f"{self.species.ids[species - RESERVED]!r} shows more than {shown} types"
            )
        cells = [RESERVED + self.type_chart[t][attacker] for t in types] if attacker >= 0 else []
        return cells + [NONE] * (shown - len(cells))
