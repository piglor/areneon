"""The ``opponent_sheet`` block as encoder inputs (ADR-027 slice 4, ADR-025 erratum E4).

A query under an ``opponent-sheet/1`` checkpoint carries, beside its observation, the closed-regime
block the Node side builds from the opponent belief (``PolicyQuery.opponent_sheet``). The block has
one slot per previewed opponent Pokémon, in preview order, which is the order of the reducer's
opponent roster (a phantom entry, an Illusion twin, is no preview slot). This module turns one slot
into the extra columns of that opponent's token:

* categories: the source tag of each field (brought, item, ability, nature, moves), the block's
  refusal code (the same on every opponent token), and the entry ids (items, abilities and moves are
  ids of the engine-data table; a nature has no vocabulary in the table, so it is a SHA-256 bucket);
* numbers: every ppm divided by 10^6, ``other_ppm`` as its own number, ``slots_known`` over 4.

Own tokens and padding entries are NONE and 0. An id the engine data lacks is an ``EncodingError``
(nothing is guessed). The ``k`` values and the bucket count are constants hashed into the encoder
version; a block with more entries than ``k`` is refused (the builder cuts to ``k``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from areneon_ml.model.gamedata import NONE, RESERVED, GameData, GameDataError

PPM = 1_000_000
K_ITEM, K_ABILITY, K_NATURE, K_MOVES = 4, 3, 3, 8
MOVE_SLOTS = 4  # slots_known counts the four move slots
NATURE_BUCKETS = 4096
FIELDS = ("brought", "item", "ability", "nature", "moves")

# Closed words of the contract (opponent-sheet.schema.json), in its order; a test holds them equal.
SOURCES = ("sheet", "revealed", "predicted", "unknown")
REFUSALS = (
    "reader_flagged",
    "team_size",
    "species_not_in_prior",
    "forme_not_in_prior",
    "brought_exceeds_picked",
    "disguise_assignments_exceeded",
    "evidence_has_no_support",
    "legality_redraws_exhausted",
    "empty_distribution",
    "disguise_unsettled",
)
SOURCE_CATEGORIES = RESERVED + len(SOURCES)
REFUSAL_CATEGORIES = RESERVED + len(REFUSALS)
NATURE_CATEGORIES = RESERVED + NATURE_BUCKETS

#: Categorical columns of an opponent token: (column, vocabulary kind).
CAT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sheet_refusal", "sheet_refusal"),
    *((f"sheet_{f}_source", "sheet_source") for f in FIELDS),
    *((f"sheet_item_{j}", "item") for j in range(K_ITEM)),
    *((f"sheet_ability_{j}", "ability") for j in range(K_ABILITY)),
    *((f"sheet_nature_{j}", "sheet_nature") for j in range(K_NATURE)),
    *((f"sheet_move_{j}", "move") for j in range(K_MOVES)),
)  # fmt: skip
#: Numeric columns of an opponent token.
NUM_COLUMNS: tuple[str, ...] = (
    "sheet_brought",
    *(f"sheet_item_{j}_ppm" for j in range(K_ITEM)),
    "sheet_item_other",
    *(f"sheet_ability_{j}_ppm" for j in range(K_ABILITY)),
    "sheet_ability_other",
    *(f"sheet_nature_{j}_ppm" for j in range(K_NATURE)),
    "sheet_nature_other",
    *(f"sheet_move_{j}_ppm" for j in range(K_MOVES)),
    "sheet_moves_known",
)

CONSTANTS: dict[str, Any] = {
    "k": [K_ITEM, K_ABILITY, K_NATURE, K_MOVES],
    "sources": list(SOURCES),
    "refusals": list(REFUSALS),
    "ppm": PPM,
    "move_slots": MOVE_SLOTS,
    "nature": f"sha256 first 4 bytes big-endian modulo {NATURE_BUCKETS}, offset by RESERVED",
    "no_item": "an item entry with a null id is NONE",
}


class SheetError(ValueError):
    """The block cannot be read as columns."""


def nature_category(nature: str) -> int:
    digest = hashlib.sha256(nature.encode("utf-8")).digest()
    return RESERVED + int.from_bytes(digest[:4], "big") % NATURE_BUCKETS


def source_category(source: str) -> int:
    return RESERVED + SOURCES.index(source)


def none_columns() -> tuple[list[int], list[float]]:
    """The columns of an own token: nothing of the sheet."""
    return [NONE] * len(CAT_COLUMNS), [0.0] * len(NUM_COLUMNS)


def _padded(ids: Sequence[int], width: int) -> list[int]:
    return [*ids, *([NONE] * (width - len(ids)))]


def _ppms(values: Sequence[int], width: int) -> list[float]:
    return [*(v / PPM for v in values), *([0.0] * (width - len(values)))]


def _lookup(vocabulary: Any, name: str, ident: str) -> int:
    try:
        index: int = vocabulary.lookup(ident)
    except GameDataError as error:
        raise SheetError(f"the opponent_sheet names a {name} the engine data lacks") from error
    return index


def _ability(game: GameData, ident: str) -> int:
    return _lookup(game.abilities, "ability", ident)


def _item(game: GameData, ident: str | None) -> int:
    return NONE if ident is None else _lookup(game.items, "item", ident)


def slot_columns(
    game: GameData, block: Mapping[str, Any], slot: Mapping[str, Any]
) -> tuple[list[int], list[float]]:
    """The columns of one opponent token from its slot of the block."""
    refusal = block["source"].get("refusal")
    item, ability, nature, moves = slot["item"], slot["ability"], slot["nature"], slot["moves"]
    for name, field, k in (
        ("item", item, K_ITEM),
        ("ability", ability, K_ABILITY),
        ("nature", nature, K_NATURE),
        ("moves", moves, K_MOVES),
    ):
        if len(field["entries"]) > k:
            raise SheetError(f"the opponent_sheet {name} field has more entries than k")
    cats = [
        NONE if refusal is None else RESERVED + REFUSALS.index(refusal),
        source_category(slot["brought_source"]),
        source_category(item["source"]),
        source_category(ability["source"]),
        source_category(nature["source"]),
        source_category(moves["source"]),
        *_padded([_item(game, e["id"]) for e in item["entries"]], K_ITEM),
        *_padded([_ability(game, e["id"]) for e in ability["entries"]], K_ABILITY),
        *_padded([nature_category(e["id"]) for e in nature["entries"]], K_NATURE),
        *_padded([_lookup(game.moves, "move", e["id"]) for e in moves["entries"]], K_MOVES),
    ]  # fmt: skip
    nums = [
        slot["brought_ppm"] / PPM,
        *_ppms([e["ppm"] for e in item["entries"]], K_ITEM),
        item["other_ppm"] / PPM,
        *_ppms([e["ppm"] for e in ability["entries"]], K_ABILITY),
        ability["other_ppm"] / PPM,
        *_ppms([e["ppm"] for e in nature["entries"]], K_NATURE),
        nature["other_ppm"] / PPM,
        *_ppms([e["inclusion_ppm"] for e in moves["entries"]], K_MOVES),
        moves["slots_known"] / MOVE_SLOTS,
    ]
    return cats, nums
