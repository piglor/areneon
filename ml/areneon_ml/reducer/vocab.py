"""The pinned engine's game vocabularies, as the reducer indexes them (ADR-010 §3).

Built from the hashed engine-data export (`engine-data.schema.json`): the
reducer asks it whether a species, move, item, ability, status, weather or
terrain exists and never carries such a list of its own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import VocabularyError

_NON_ID = re.compile(r"[^a-z0-9]")


def to_id(text: str) -> str:
    """The engine's id of a name: lowercase, letters and digits only."""
    return _NON_ID.sub("", text.lower())


def _ids(data: Mapping[str, Any], key: str) -> frozenset[str]:
    return frozenset(str(row["id"]) for row in data[key])


@dataclass(frozen=True)
class Vocabulary:
    """Id sets of one engine-data export; lookups raise `VocabularyError`."""

    showdown_commit: str
    format_id: str
    species_base: Mapping[str, str]
    moves: frozenset[str]
    items: frozenset[str]
    abilities: frozenset[str]
    statuses: frozenset[str]
    weathers: frozenset[str]
    terrains: frozenset[str]

    @classmethod
    def from_engine_data(cls, data: Mapping[str, Any]) -> Vocabulary:
        if data.get("schema_version") != 1:
            raise VocabularyError(
                f"unsupported engine-data schema_version {data.get('schema_version')!r}"
            )
        return cls(
            showdown_commit=str(data["showdown_commit"]),
            format_id=str(data["format_id"]),
            species_base={str(r["id"]): str(r["base_species"]) for r in data["species"]},
            moves=_ids(data, "moves"),
            items=_ids(data, "items"),
            abilities=_ids(data, "abilities"),
            statuses=_ids(data, "statuses"),
            weathers=_ids(data, "weathers"),
            terrains=_ids(data, "terrains"),
        )

    def _check(self, kind: str, name: str, known: Any) -> str:
        found = to_id(name)
        if found not in known:
            raise VocabularyError(f"unknown {kind} {name!r}")
        return found

    def species(self, name: str) -> str:
        return self._check("species", name, self.species_base)

    def base_species(self, species_id: str) -> str:
        return self.species_base[species_id]

    def move(self, name: str) -> str:
        return self._check("move", name, self.moves)

    def item(self, name: str) -> str:
        return self._check("item", name, self.items)

    def ability(self, name: str) -> str:
        return self._check("ability", name, self.abilities)

    def status(self, name: str) -> str:
        return self._check("status", name, self.statuses)

    def weather(self, name: str) -> str:
        return self._check("weather", name, self.weathers)
