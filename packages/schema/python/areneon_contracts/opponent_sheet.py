"""The reader rules of the ``opponent_sheet`` block that the contract profile cannot say.

ADR-027 sections 2 and 5, ADR-025 erratum E4. The shapes are ``opponent-sheet.schema.json``
(copied into ``PolicyQuery`` and ``TrainingExample``). This is the Python twin of
``packages/showdown-bridge/src/opponent-sheet.ts``: the same rules, and the shared corpus holds
both to them. ``sheet_problem`` returns the first problem as fixed text that names a slot and a
field and never a value, or None.

The rules (closed regime; the open regime's ``engine-sheet`` kind is checked for the shape
ADR-027 gives it):

* an entry list is sorted by ppm descending then id ascending (a null item id last) with unique
  ids;
* ``other_ppm`` is ``max(0, 1000000 - sum of the entries)`` and the sum is at most 1000000 plus
  one per entry (each entry is rounded alone); a field of source ``unknown`` has no entries and
  ``other_ppm`` 0 (moves: ``slots_known`` 0);
* a field of source ``revealed`` holds its value alone at 1000000 ppm (moves: ``slots_known`` 4
  and four entries at 1000000); a move list holds at least ``slots_known`` entries at 1000000;
* ``brought_source`` revealed means 1000000 ppm and unknown means 0;
* the source kind fits the regime: ``belief`` has no field of source ``sheet``;
  ``revealed-only`` has no predicted field; ``engine-sheet`` and ``oracle-diagnostic`` hold
  every item, ability, nature and move field at source ``sheet``; only a ``belief`` block has a
  refusal, and a refused block is unknown everywhere with ``brought_ppm`` 0.
"""

from __future__ import annotations

import itertools
from typing import Any

FULL = 1_000_000
MOVE_SLOTS = 4
_FIELDS = ("item", "ability", "nature", "moves")


def _order_key(entry: tuple[str | None, int]) -> tuple[int, int, str]:
    ident, value = entry
    return (-value, 1 if ident is None else 0, ident or "")


def _list_problem(where: str, source: str, entries: list[tuple[str | None, int]]) -> str | None:
    if source == "unknown":
        return None if not entries else f"{where}: a field of source unknown has no entries"
    if not entries:
        return f"{where}: a field with a source other than unknown has entries"
    keys = [_order_key(e) for e in entries]
    if any(a >= b for a, b in itertools.pairwise(keys)):
        return f"{where}: the entries must be sorted by ppm descending then id, and unique"
    return None


def _id_field_problem(where: str, field: dict[str, Any]) -> str | None:
    entries = [(e["id"], e["ppm"]) for e in field["entries"]]
    problem = _list_problem(where, field["source"], entries)
    if problem is not None:
        return problem
    if field["source"] == "unknown":
        return (
            None
            if field["other_ppm"] == 0
            else f"{where}: a field of source unknown has other_ppm 0"
        )
    total = sum(v for _, v in entries)
    if total > FULL + len(entries):
        return f"{where}: the entries sum to more than 1000000 plus one per entry"
    if field["other_ppm"] != max(0, FULL - total):
        return f"{where}: other_ppm must be max(0, 1000000 - the sum of the entries)"
    if field["source"] == "revealed" and not (len(entries) == 1 and entries[0][1] == FULL):
        return f"{where}: a revealed field holds its one value at 1000000 ppm"
    return None


def _moves_problem(where: str, field: dict[str, Any]) -> str | None:
    entries = [(e["id"], e["inclusion_ppm"]) for e in field["entries"]]
    problem = _list_problem(where, field["source"], entries)
    if problem is not None:
        return problem
    if field["source"] == "unknown":
        return (
            None
            if field["slots_known"] == 0
            else f"{where}: a field of source unknown has slots_known 0"
        )
    if sum(1 for _, v in entries if v == FULL) < field["slots_known"]:
        return f"{where}: slots_known counts revealed moves, which are held at 1000000 ppm"
    full_four = field["slots_known"] == MOVE_SLOTS and all(
        v == FULL for _, v in entries[:MOVE_SLOTS]
    )
    if field["source"] == "revealed" and not full_four:
        return f"{where}: a revealed move field has all four moves known"
    return None


def _fit_problem(at: str, kind: str, refusal: str | None, slot: dict[str, Any]) -> str | None:
    """Whether the slot's sources and brought_ppm fit the block's kind and refusal."""
    fields = [slot[name]["source"] for name in _FIELDS]
    bought = slot["brought_source"]
    ppm = slot["brought_ppm"]
    wrong = {
        "brought_ppm does not fit brought_source": (bought == "unknown" and ppm != 0)
        or (bought == "revealed" and ppm != FULL),
        "a refused block is unknown everywhere": refusal is not None
        and not (ppm == 0 and bought == "unknown" and all(f == "unknown" for f in fields)),
        "a belief block has no field of source sheet": kind == "belief" and "sheet" in fields,
        "a revealed-only block has revealed or unknown fields only": kind == "revealed-only"
        and ("sheet" in fields or "predicted" in fields or bought != "unknown"),
        "a sheet block holds every item, ability, nature and move field at source sheet": kind
        in ("engine-sheet", "oracle-diagnostic")
        and not all(f == "sheet" for f in fields),
        "an engine-sheet block takes brought_ppm from the belief or leaves it unknown": kind
        == "engine-sheet"
        and bought == "revealed",
    }
    return next((f"{at}: {text}" for text, bad in wrong.items() if bad), None)


def sheet_problem(block: dict[str, Any]) -> str | None:
    """The first problem of a block (already valid against the schema), or None."""
    kind = block["source"]["kind"]
    refusal = block["source"].get("refusal")
    if refusal is not None and kind != "belief":
        return "only a belief block can carry a refusal"
    for i, slot in enumerate(block["slots"]):
        at = f"slot {i}"
        problem = (
            _fit_problem(at, kind, refusal, slot)
            or _id_field_problem(f"{at} item", slot["item"])
            or _id_field_problem(f"{at} ability", slot["ability"])
            or _id_field_problem(f"{at} nature", slot["nature"])
            or _moves_problem(f"{at} moves", slot["moves"])
        )
        if problem is not None:
            return problem
    return None
