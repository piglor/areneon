"""The ``search_labels`` block of a labelled generation's ``generation.json`` (ADR-026 E2, #1290).

The block holds ``search_config_hash``, ``search_identity_hash``, ``selected``, ``labelled``,
``refused`` (a list of ``{reason, count}``) and ``recomputed``: two hashes and counts only.

A generation in which nothing was labelled writes no label files, so these counts are what tells
it from a broken labeller. ``selected`` decisions were hashed in, ``labelled`` got a label and the
rest are ``refused`` by reason (sorted, unique), so ``labelled + sum(refused) == selected``.
``recomputed`` Episodes were labelled again by the process that wrote the stream. This module reads
the block and decides nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_HASH = re.compile(r"[0-9a-f]{64}")
_KEYS = frozenset(
    {
        "search_config_hash",
        "search_identity_hash",
        "selected",
        "labelled",
        "refused",
        "recomputed",
    }
)


class LabelCountsError(ValueError):
    """A ``search_labels`` block breaks the rules of its contract."""


@dataclass(frozen=True)
class LabelCounts:
    search_config_hash: str
    search_identity_hash: str
    selected: int
    labelled: int
    refused: tuple[tuple[str, int], ...]  # sorted by reason, unique
    recomputed: int


def _count(value: object, what: str) -> int:
    if type(value) is not int or value < 0:
        raise LabelCountsError(f"{what} is not a non-negative integer")
    return value


def read_label_counts(raw: object, what: str = "search_labels") -> LabelCounts:
    """The closed block, its counts checked against each other."""
    if not isinstance(raw, dict):
        raise LabelCountsError(f"{what} is not an object")
    if frozenset(raw) != _KEYS:
        raise LabelCountsError(f"{what} keys are not exactly {sorted(_KEYS)}")
    hashes = []
    for key in ("search_config_hash", "search_identity_hash"):
        value = raw[key]
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise LabelCountsError(f"{what} {key} is not a SHA-256")
        hashes.append(value)
    refused_raw = raw["refused"]
    if not isinstance(refused_raw, list):
        raise LabelCountsError(f"{what} refused is not a list")
    refused: list[tuple[str, int]] = []
    for item in refused_raw:
        if not isinstance(item, dict) or frozenset(item) != {"reason", "count"}:
            raise LabelCountsError(f"{what} refused entry is not {{reason, count}}")
        reason = item["reason"]
        if not isinstance(reason, str) or not reason:
            raise LabelCountsError(f"{what} refused reason is not a string")
        refused.append((reason, _count(item["count"], f"{what} refused count")))
    reasons = [reason for reason, _ in refused]
    if reasons != sorted(set(reasons)):
        raise LabelCountsError(f"{what} refused is not sorted by reason and unique")
    selected = _count(raw["selected"], f"{what} selected")
    labelled = _count(raw["labelled"], f"{what} labelled")
    if labelled + sum(count for _, count in refused) != selected:
        raise LabelCountsError(f"{what}: labelled plus refused is not selected")
    return LabelCounts(
        hashes[0],
        hashes[1],
        selected,
        labelled,
        tuple(refused),
        _count(raw["recomputed"], f"{what} recomputed"),
    )


def label_counts_dict(counts: LabelCounts) -> dict[str, Any]:
    return {
        "search_config_hash": counts.search_config_hash,
        "search_identity_hash": counts.search_identity_hash,
        "selected": counts.selected,
        "labelled": counts.labelled,
        "refused": [{"reason": r, "count": c} for r, c in counts.refused],
        "recomputed": counts.recomputed,
    }
