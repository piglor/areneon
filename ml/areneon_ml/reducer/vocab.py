"""The pinned engine's game vocabularies, as the reducer indexes them (ADR-010 §3).

Built from the hashed engine-data export (`engine-data.schema.json`): the
reducer asks it whether a species, move, item, ability, status, weather or
terrain exists and never carries such a list of its own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from .errors import VocabularyError

DATA_DIR = Path(__file__).resolve().parent / "data"
_COPY_KEYS = frozenset(
    {
        "format_id",
        "not_copied",
        "schema_version",
        "shed_tail_copies",
        "showdown_commit",
        "source_path",
        "source_sha256",
        "volatile_copy_hash",
    }
)

_NON_ID = re.compile(r"[^a-z0-9]")


def to_id(text: str) -> str:
    """The engine's id of a name: lowercase, letters and digits only."""
    return _NON_ID.sub("", text.lower())


def _ids(data: Mapping[str, Any], key: str) -> frozenset[str]:
    return frozenset(str(row["id"]) for row in data[key])


def read_volatile_copy(commit: str, format_id: str) -> dict[str, Any] | None:
    """The committed copy rule for this engine and format, or None when there is none for it.
    A file that exists but is malformed, or whose hash does not match, is refused."""
    path = DATA_DIR / f"volatile-copy.{format_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _COPY_KEYS or data["schema_version"] != 1:
        raise VocabularyError(f"{path.name} is not a volatile-copy table")
    body = {k: v for k, v in data.items() if k != "volatile_copy_hash"}
    if sha256_hex(jcs(body)) != data["volatile_copy_hash"]:
        raise VocabularyError(f"{path.name} does not match its hash")
    if data["format_id"] != format_id:
        raise VocabularyError(f"{path.name} is for another format")
    if data["showdown_commit"] != commit:
        return None  # built for another pin: the switch that needs it fails closed
    return data


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
    # The volatile-copy rule of the pinned engine (`data/volatile-copy.<format>.json`, exported by
    # `export-volatile-copy.ts`): a Baton Pass copies every volatile except `not_copied`, a Shed
    # Tail only `shed_tail_copies`. None when no file matches this export (a switch that needs it
    # then raises `VocabularyError`).
    not_copied: frozenset[str] | None = None
    shed_tail_copies: frozenset[str] | None = None

    @classmethod
    def from_engine_data(cls, data: Mapping[str, Any]) -> Vocabulary:
        if data.get("schema_version") != 1:
            raise VocabularyError(
                f"unsupported engine-data schema_version {data.get('schema_version')!r}"
            )
        copy = read_volatile_copy(str(data["showdown_commit"]), str(data["format_id"]))
        return cls(
            not_copied=None if copy is None else frozenset(copy["not_copied"]),
            shed_tail_copies=None if copy is None else frozenset(copy["shed_tail_copies"]),
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

    def passed_volatiles(self, cause: str, held: frozenset[str] | set[str]) -> set[str]:
        """The volatiles of `held` that a `batonpass` or `shedtail` switch hands over."""
        if self.not_copied is None or self.shed_tail_copies is None:
            raise VocabularyError("no volatile-copy table for this engine and format")
        if cause == "shedtail":
            return {v for v in held if v in self.shed_tail_copies}
        return {v for v in held if v not in self.not_copied}

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
