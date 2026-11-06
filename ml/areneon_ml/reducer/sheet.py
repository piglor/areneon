"""The engine-sent team sheet of an open-format game (ADR-027 section 5, #1150).

An open-sheet format makes the engine print one `|showteam|<p1|p2>|<packed>` line per side
between the preview lines and the `|request|` of Team Preview. The packed team is read here
with a parser written from the upstream `sim/TEAMS.md` (the packed format) and this
repository's committed open-regime channels; no client code was used.

One Pokemon is twelve `|`-separated fields, Pokemon are `]`-separated:

    NAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|EXTRAS

The engine's sheet carries no nickname (it prints the species name as NAME and leaves SPECIES
blank), no Stat Points (EVS), no IVs and no extras (happiness, ball, hidden power type,
Gigantamax, Dynamax level, Tera type); any of those present is refused, as is any other
shape. Species, item, ability and move names must resolve in the engine-data vocabulary. The
nature has no vocabulary in the engine-data table, so it is checked for shape only.

The reducer accepts a sheet in ANY channel: it does not know the format's regime. Whether a channel
may hold a `|showteam|` line at all (never in a closed-sheet format, exactly one per side in an open
one) is decided by the TypeScript gate (`playerChannel`, `readEpisode`, the training loader); the
sheet is read here but never encoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import MalformedLineError, UnknownFieldError
from .vocab import Vocabulary, to_id

FIELDS = 12
MAX_LEVEL = 100
MAX_POKEMON = 24  # a sanity bound on the list, not a team size rule (the preview count is checked)
_NATURE = re.compile(r"^[A-Za-z]{1,16}$")
_LEVEL = re.compile(r"^\d{1,3}$")


@dataclass(frozen=True)
class SheetMon:
    """One Pokemon of a sheet, as built: ids, in the sheet's own order."""

    species: str
    item: str | None
    ability: str
    moves: tuple[str, ...]
    nature: str | None
    gender: str | None
    level: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "item": self.item,
            "ability": self.ability,
            "moves": list(self.moves),
            "nature": self.nature,
            "gender": self.gender,
            "level": self.level,
        }


def parse_sheet(packed: str, vocab: Vocabulary, raw: str) -> list[SheetMon]:
    """The Pokemon of a packed sheet, or a typed refusal."""
    entries = packed.split("]")
    if not 1 <= len(entries) <= MAX_POKEMON:
        raise MalformedLineError("a sheet lists between one and 24 Pokemon", raw)
    return [_pokemon(entry.split("|"), vocab, raw) for entry in entries]


def _pokemon(fields: list[str], vocab: Vocabulary, raw: str) -> SheetMon:
    if len(fields) != FIELDS:
        raise MalformedLineError(f"a sheet Pokemon has {FIELDS} fields, not {len(fields)}", raw)
    name, nickname, item, ability, moves, nature, points, gender, ivs, shiny, level, extras = fields
    if nickname != "":
        raise UnknownFieldError("a sheet shows no nickname", raw)
    if points != "" or ivs != "":
        raise UnknownFieldError("a sheet shows no Stat Points and no IVs", raw)
    if shiny != "" or extras != "":
        raise UnknownFieldError("a sheet carries no shiny flag and no extra fields", raw)
    if name == "" or ability == "" or moves == "":
        raise MalformedLineError("a sheet Pokemon lacks its species, ability or moves", raw)
    if gender not in ("", "M", "F"):
        raise UnknownFieldError(f"unknown sheet gender {gender!r}", raw)
    if nature != "" and _NATURE.match(nature) is None:
        raise MalformedLineError("a sheet nature is a name", raw)
    if level != "" and (_LEVEL.match(level) is None or not 1 <= int(level) <= MAX_LEVEL):
        raise MalformedLineError("a sheet level is a number from 1 to 100", raw)
    names = moves.split(",")
    if "" in names:
        raise MalformedLineError("a sheet move list has an empty move", raw)
    return SheetMon(
        species=vocab.species(name),
        item=vocab.item(item) if item else None,
        ability=vocab.ability(ability),
        moves=tuple(vocab.move(move) for move in names),
        nature=to_id(nature) or None,
        gender=gender or None,
        level=int(level) if level else MAX_LEVEL,
    )
