"""The ``opponent_sheet`` block as encoder inputs (ADR-027 section 2, ADR-025 erratum E4).

ADR-027 section 2, "Encoding": "Each field becomes the probability-weighted sum of its entries'
embeddings plus `other_ppm` times an "other" embedding, plus a source embedding and the top
probability as a scalar; moves are the inclusion-weighted sum; the result is concatenated to that
opponent Pokémon's existing row (no extra tokens)."

This module is the encoder's half. A query under an ``opponent-sheet/1`` checkpoint carries the
closed-regime block (``PolicyQuery.opponent_sheet``); its slots are the opponent's preview entries
in order, which is the order of the reducer's opponent roster (a phantom entry is no preview slot).
One slot becomes these columns of that opponent's token (own tokens and padding are NONE and 0):

* categories embedded and concatenated as any token column: the block's refusal code (the same on
  every opponent token) and the source tag of each of the five fields (brought, item, ability,
  nature, moves);
* per field the entry ids (``sheet_<field>_<j>``, in the item, ability and move vocabularies; a null
  item id, "holds no item", is NONE, the vocabulary's own none row; a nature has no vocabulary in
  the engine-data table, so it is a SHA-256 bucket) and the weights ``ppm / 10^6`` of the entries
  and of ``other_ppm``. The NETWORK reads these two sets of columns apart from the others: it
  embeds each id, weights it and sums (``PolicyNetwork._sheet_sums``), adds ``other`` times a
  per-field embedding, and concatenates the result to the token's categories; the weights are not
  inputs of the numeric layer;
* scalars: ``brought_ppm`` and the top probability of each field (the largest entry's ppm).

An unknown field has no entries and no ``other`` mass (the reader rule), so its sum is zero and its
source tag says why. An id the engine data lacks is a ``SheetError`` (nothing is guessed). The ``k``
values and the bucket count are constants hashed into the encoder version; a block with more
entries than ``k`` is refused (the builder cuts to ``k``).
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

FIELDS = ("brought", "item", "ability", "nature", "moves")
#: The fields that have entries: (field, vocabulary kind of the ids, k, has an ``other`` mass).
ENTRY_FIELDS: tuple[tuple[str, str, int, bool], ...] = (
    ("item", "item", K_ITEM, True),
    ("ability", "ability", K_ABILITY, True),
    ("nature", "sheet_nature", K_NATURE, True),
    ("moves", "move", K_MOVES, False),
)


def _entry_name(field: str, j: int) -> str:
    return f"sheet_{'move' if field == 'moves' else field}_{j}"


#: Categorical columns of an opponent token: (column, vocabulary kind). The entry ids come last.
CAT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sheet_refusal", "sheet_refusal"),
    *((f"sheet_{f}_source", "sheet_source") for f in FIELDS),
    *((_entry_name(f, j), kind) for f, kind, k, _ in ENTRY_FIELDS for j in range(k)),
)
SCALAR_COLUMNS: tuple[str, ...] = (
    "sheet_brought",
    *(f"sheet_{f}_top" for f, _, _, _ in ENTRY_FIELDS),
)
#: Numeric columns of an opponent token: the scalars, then the entry and ``other`` weights.
NUM_COLUMNS: tuple[str, ...] = (
    *SCALAR_COLUMNS,
    *(
        name
        for f, _, k, other in ENTRY_FIELDS
        for name in (
            *(f"{_entry_name(f, j)}_ppm" for j in range(k)),
            *((f"sheet_{f}_other",) if other else ()),
        )
    ),
)  # fmt: skip


def entry_columns(
    layout_cat: Sequence[tuple[str, str]], layout_num: Sequence[str]
) -> list[tuple[str, str, list[int], list[int], int | None]]:
    """Per entry field in a layout: (field, kind, id columns, weight columns, other column)."""
    cat = [name for name, _ in layout_cat]
    out: list[tuple[str, str, list[int], list[int], int | None]] = []
    for field, kind, k, other in ENTRY_FIELDS:
        if _entry_name(field, 0) not in cat:
            continue
        ids = [cat.index(_entry_name(field, j)) for j in range(k)]
        weights = [layout_num.index(f"{_entry_name(field, j)}_ppm") for j in range(k)]
        out.append(
            (field, kind, ids, weights, layout_num.index(f"sheet_{field}_other") if other else None)
        )
    return out


def plain_columns(
    layout_cat: Sequence[tuple[str, str]], layout_num: Sequence[str]
) -> tuple[list[int], list[int]]:
    """The categorical columns embedded as usual and the numeric columns that feed the numeric
    layer (all but the sheet's entry weights), as indices into the layout."""
    ids = {name for f, _, k, _ in ENTRY_FIELDS for name in (_entry_name(f, j) for j in range(k))}
    weights = {
        name
        for name in layout_num
        if name.startswith("sheet_") and (name.endswith("_ppm") or name.endswith("_other"))
    }
    return (
        [i for i, (name, _) in enumerate(layout_cat) if name not in ids],
        [i for i, name in enumerate(layout_num) if name not in weights],
    )


CONSTANTS: dict[str, Any] = {
    "encoding": "per field: sum of ppm/10^6 times the embedding of each entry id, plus other/10^6 "
    "times a learned per-field embedding (not for moves), concatenated to the token's categories; "
    "scalars: brought and the top probability of each field",
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


def _weights(values: Sequence[int], width: int) -> list[float]:
    return [*(v / PPM for v in values), *([0.0] * (width - len(values)))]


def _lookup(vocabulary: Any, name: str, ident: str) -> int:
    try:
        index: int = vocabulary.lookup(ident)
    except GameDataError as error:
        raise SheetError(f"the opponent_sheet names a {name} the engine data lacks") from error
    return index


def _item(game: GameData, ident: str | None) -> int:
    return NONE if ident is None else _lookup(game.items, "item", ident)


def _ability(game: GameData, ident: str) -> int:
    return _lookup(game.abilities, "ability", ident)


def _move(game: GameData, ident: str) -> int:
    return _lookup(game.moves, "move", ident)


def _source_of(slot: Mapping[str, Any], field: str) -> str:
    source: str = slot["brought_source"] if field == "brought" else slot[field]["source"]
    return source


def slot_columns(
    game: GameData, block: Mapping[str, Any], slot: Mapping[str, Any]
) -> tuple[list[int], list[float]]:
    """The columns of one opponent token from its slot of the block."""
    refusal = block["source"].get("refusal")
    fields = {f: slot[f] for f, _, _, _ in ENTRY_FIELDS}
    ids = {
        "item": lambda e: _item(game, e["id"]),
        "ability": lambda e: _ability(game, e["id"]),
        "nature": lambda e: nature_category(e["id"]),
        "moves": lambda e: _move(game, e["id"]),
    }
    cats = [
        NONE if refusal is None else RESERVED + REFUSALS.index(refusal),
        *(source_category(_source_of(slot, f)) for f in FIELDS),
    ]  # fmt: skip
    nums = [slot["brought_ppm"] / PPM]
    weights: list[float] = []
    for field, _, k, other in ENTRY_FIELDS:
        entries = fields[field]["entries"]
        if len(entries) > k:
            raise SheetError(f"the opponent_sheet {field} field has more entries than k")
        values = [e["inclusion_ppm" if field == "moves" else "ppm"] for e in entries]
        cats.extend(_padded([ids[field](e) for e in entries], k))
        nums.append(max(values, default=0) / PPM)
        weights.extend(_weights(values, k))
        if other:
            weights.append(fields[field]["other_ppm"] / PPM)
    return cats, [*nums, *weights]
