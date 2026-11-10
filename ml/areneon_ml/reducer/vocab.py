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

# The engine-data schema_version that adds the condition list (version 1 has none).
WITH_CONDITIONS = 2
DATA_DIR = Path(__file__).resolve().parent / "data"
# Some conditions print their volatile under a derived name: Stockpile as `stockpile1`..`3`,
# Protosynthesis and Quark Drive as `protosynthesisatk` (the boosted stat's id appended). The
# engine's condition id is the name without that suffix.
_DERIVED_SUFFIX = re.compile(r"(?:\d+|atk|def|spa|spd|spe)$")
# `-start` / `-end` words the engine prints that are no condition id and no name of a move,
# ability or item: the two type messages and Supreme Overlord's count (`fallen1`..`fallen5`, the
# ability's own message) and Perish Song's countdown (`perish3`..`perish0`). They are protocol
# text read from the engine's source, not game rules.
PROTOCOL_VOLATILES = frozenset({"typechange", "typeadd", "fallen", "perish"})
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


_STALL_KEYS = frozenset(
    {
        "format_id",
        "schema_version",
        "showdown_commit",
        "source_path",
        "source_sha256",
        "stall_moves",
        "stall_moves_hash",
    }
)


def read_stall_moves(commit: str, format_id: str) -> frozenset[str] | None:
    """The committed stalling moves (Protect family, Endure, Quick/Wide Guard) for this engine and
    format (`data/stall-moves.<format>.json`, exported by `export-stall-moves.ts`), or None when
    there is none for this pin. A malformed file or a hash mismatch is refused."""
    path = DATA_DIR / f"stall-moves.{format_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _STALL_KEYS or data["schema_version"] != 1:
        raise VocabularyError(f"{path.name} is not a stall-moves table")
    body = {k: v for k, v in data.items() if k != "stall_moves_hash"}
    if sha256_hex(jcs(body)) != data["stall_moves_hash"]:
        raise VocabularyError(f"{path.name} does not match its hash")
    if data["format_id"] != format_id:
        raise VocabularyError(f"{path.name} is for another format")
    if data["showdown_commit"] != commit:
        return None  # built for another pin: the Protect streak then fails closed
    return frozenset(data["stall_moves"])


_HISTORY_KEYS = frozenset(
    {
        "action_sources",
        "direct_end_items",
        "eat_tags",
        "eaten_berries",
        "external_action_sources",
        "format_id",
        "history_facts_data_hash",
        "item_use_sources",
        "last_item_restorers",
        "last_item_takers",
        "null_cant_reasons",
        "protect_breaker_moves",
        "schema_version",
        "showdown_commit",
        "slot_conditions",
        "stall_counter",
    }
)


def read_history_facts(commit: str, format_id: str) -> dict[str, Any] | None:
    """The committed history-facts data for this engine and format
    (`data/history-facts.<format>.json`, exported by `export-history-facts-data.ts`: the stall
    counter constants, the move-line action sources, the item and slot-condition rules), or None
    when there is none for this pin (the history facts are then left out of the state). A
    malformed file or a hash mismatch is refused."""
    path = DATA_DIR / f"history-facts.{format_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _HISTORY_KEYS or data["schema_version"] != 1:
        raise VocabularyError(f"{path.name} is not a history-facts table")
    body = {k: v for k, v in data.items() if k != "history_facts_data_hash"}
    if sha256_hex(jcs(body)) != data["history_facts_data_hash"]:
        raise VocabularyError(f"{path.name} does not match its hash")
    if data["format_id"] != format_id:
        raise VocabularyError(f"{path.name} is for another format")
    if data["showdown_commit"] != commit:
        return None
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
    types: frozenset[str]
    # The engine's condition ids (engine-data version 2 only; None for a version 1 table, whose
    # volatile ids are then kept as opaque public strings).
    conditions: frozenset[str] | None
    # The volatile-copy rule of the pinned engine (`data/volatile-copy.<format>.json`, exported by
    # `export-volatile-copy.ts`): a Baton Pass copies every volatile except `not_copied`, a Shed
    # Tail only `shed_tail_copies`. None when no file matches this export (a switch that needs it
    # then raises `VocabularyError`).
    not_copied: frozenset[str] | None = None
    shed_tail_copies: frozenset[str] | None = None
    # The ids of the moves that add the engine's `stall` volatile
    # (`data/stall-moves.<format>.json`), which the Protect streak counts. None when no file
    # matches this export: a move line then raises `VocabularyError`, never a guess.
    stall_moves: frozenset[str] | None = None
    # The ids of the moves whose engine target is the user's own side (`allySide`, from the
    # export's move table): Quick Guard and Wide Guard print nothing when the condition is
    # already up (#1281).
    ally_side_moves: frozenset[str] = frozenset()
    # What the history facts (`history.py`) need beyond the tables above: the committed
    # history-facts data file merged with the stall moves, the side-target stall moves and the
    # volatile copy table, as the TS module's `HistoryFactsData`. None when any of them is missing
    # for this pin: the state then carries no history facts.
    history_data: Mapping[str, Any] | None = None

    @classmethod
    def from_engine_data(cls, data: Mapping[str, Any]) -> Vocabulary:
        if data.get("schema_version") not in (1, 2):
            raise VocabularyError(
                f"unsupported engine-data schema_version {data.get('schema_version')!r}"
            )
        copy = read_volatile_copy(str(data["showdown_commit"]), str(data["format_id"]))
        stall = read_stall_moves(str(data["showdown_commit"]), str(data["format_id"]))
        history = read_history_facts(str(data["showdown_commit"]), str(data["format_id"]))
        ally_side = frozenset(str(r["id"]) for r in data["moves"] if r.get("target") == "allySide")
        merged: dict[str, Any] | None = None
        if history is not None and stall is not None and copy is not None:
            merged = {
                **{k: v for k, v in history.items() if k not in ("history_facts_data_hash",)},
                "stall_moves": sorted(stall),
                "ally_side_moves": sorted(ally_side & stall),
                "volatile_not_copied": list(copy["not_copied"]),
                "shed_tail_copies": list(copy["shed_tail_copies"]),
            }
        return cls(
            history_data=merged,
            ally_side_moves=frozenset(
                str(r["id"]) for r in data["moves"] if r.get("target") == "allySide"
            ),
            stall_moves=read_stall_moves(str(data["showdown_commit"]), str(data["format_id"])),
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
            types=_ids(data, "types"),
            conditions=_ids(data, "conditions")
            if data["schema_version"] == WITH_CONDITIONS
            else None,
        )

    def passed_volatiles(self, cause: str, held: frozenset[str] | set[str]) -> set[str]:
        """The volatiles of `held` that a `batonpass` or `shedtail` switch hands over."""
        if self.not_copied is None or self.shed_tail_copies is None:
            raise VocabularyError("no volatile-copy table for this engine and format")
        if cause == "shedtail":
            return {v for v in held if v in self.shed_tail_copies}
        return {
            v
            for v in held
            if v not in self.not_copied and _DERIVED_SUFFIX.sub("", v) not in self.not_copied
        }

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

    def type_(self, name: str) -> str:
        return self._check("type", name, self.types)

    def volatile(self, found: str, kind: str | None = None) -> None:
        """Check a volatile id against the engine's condition ids (version 2 tables only).

        The engine prints a volatile under its condition id, under a derived name (Stockpile as
        `stockpile1`; the condition id without the number or stat suffix) or under the name of the
        move, ability or item that started it (`Fire Spin`, `Illusion`), or is one of the
        `PROTOCOL_VOLATILES`: any of these resolves. A name given with its kind (`move: Taunt`)
        resolves only against that kind's vocabulary or the conditions."""
        if self.conditions is not None:
            self.resolve_volatile(found, kind)

    def resolve_volatile(self, found: str, kind: str | None = None) -> str:
        """The vocabulary id a volatile resolves to (see `volatile`): the id itself, or the id
        without its derived number or stat suffix. Version 2 tables only."""
        if self.conditions is None:
            raise VocabularyError(f"cannot resolve volatile {found!r} without a condition list")
        stripped = _DERIVED_SUFFIX.sub("", found)
        sources = {"move": self.moves, "ability": self.abilities, "item": self.items}
        searched = [sources[kind]] if kind in sources else list(sources.values())
        if found in self.conditions or found in PROTOCOL_VOLATILES:
            return found
        if stripped in self.conditions or stripped in PROTOCOL_VOLATILES:
            return stripped
        if any(found in vocabulary for vocabulary in searched):
            return found
        raise VocabularyError(f"unknown volatile {found!r}")

    def status(self, name: str) -> str:
        return self._check("status", name, self.statuses)

    def weather(self, name: str) -> str:
        return self._check("weather", name, self.weathers)
