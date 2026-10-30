"""Reader for the usage weight table (ADR-020 section 4).

The schema is closed and integer-only; this reader adds what the contract profile cannot say, as
the TS reader does: sorted and unique ids, every id resolving in the team vocabulary the table was
built for (abilities and moves against the species' own roster entry), the exact sums (every list is
1,000,000 parts per million, entries plus truncated plus unmapped plus its semantic key), the
top-K limits, the Stat Point limit and the integer roots of ``w_species`` and the teammate weights.
``usage_table_hash`` is SHA-256 of the JCS bytes. No Smogon data lives here.
"""

from __future__ import annotations

import itertools
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
PPM_TOTAL = 10**6
PPM_DIGITS = 6


class UsageWeightTableError(ValueError):
    """A usage weight table the reader refuses."""


def integer_root(value: int, root: int) -> int:
    """The largest ``x`` with ``x ** root <= value`` (exact integer arithmetic only)."""
    if value < 0 or root < 1:
        raise UsageWeightTableError("integer root of a negative value or a root below 1")
    if root == 1 or value <= 1:
        return value
    low, high = 1, 1 << (value.bit_length() // root + 1)
    while low < high:
        mid = (low + high + 1) // 2
        if mid**root <= value:
            low = mid
        else:
            high = mid - 1
    return low


def rooted_weight(ppm: int, root: int) -> int:
    """``floor((ppm * 10^(6 * (root - 1)))^(1/root))``: the weight a ppm becomes under ``root``."""
    return integer_root(ppm * 10 ** (PPM_DIGITS * (root - 1)), root)


def _ordered(ids: list[Any], what: str) -> None:
    for before, after in itertools.pairwise(ids):
        if not before < after:
            raise UsageWeightTableError(f"{what}: {after} is repeated or out of order")


def _check(name: str, value: object) -> None:
    try:
        validate(name, value)
    except ContractError as exc:
        raise UsageWeightTableError(str(exc)) from exc


def _spread_key(spread: dict[str, Any]) -> tuple[Any, ...]:
    return (spread["nature"], *(spread["stat_points"][stat] for stat in STATS))


def _sum_to_total(entries: list[dict[str, Any]], masses: list[int], where: str) -> None:
    if sum(e["ppm"] for e in entries) + sum(masses) != PPM_TOTAL:
        raise UsageWeightTableError(f"{where}: the masses do not sum to 1,000,000 ppm")


def _check_species(  # noqa: PLR0912 - one rule per list kind, kept in one place
    species: dict[str, Any],
    table: dict[str, Any],
    vocabulary: dict[str, Any],
    roster: dict[str, dict[str, Any]],
) -> None:
    sid = species["id"]
    where = f"species {sid}"
    entry = roster[sid]
    limits = table["derivation"]
    if species["w_species"] != rooted_weight(species["usage_ppm"], limits["species_root"]):
        raise UsageWeightTableError(f"{where}: w_species is not the integer root of usage_ppm")

    abilities = species["abilities"]
    _ordered([e["id"] for e in abilities["entries"]], f"{where} abilities")
    for e in abilities["entries"]:
        if e["id"] not in entry["abilities"]:
            raise UsageWeightTableError(f"{where}: ability {e['id']} is not one the roster lists")
    _sum_to_total(abilities["entries"], [abilities["unmapped_ppm"]], f"{where} abilities")

    items = species["items"]
    _ordered([e["id"] for e in items["entries"]], f"{where} items")
    if len(items["entries"]) > limits["k_items"]:
        raise UsageWeightTableError(f"{where} items: more entries than k_items")
    for e in items["entries"]:
        if e["id"] not in vocabulary["items"]:
            raise UsageWeightTableError(f"{where}: item {e['id']} is not in the vocabulary")
    _sum_to_total(
        items["entries"],
        [items["truncated_ppm"], items["unmapped_ppm"], items["no_item_ppm"]],
        f"{where} items",
    )

    moves = species["moves"]
    _ordered([e["id"] for e in moves["entries"]], f"{where} moves")
    if len(moves["entries"]) > limits["k_moves"]:
        raise UsageWeightTableError(f"{where} moves: more entries than k_moves")
    for e in moves["entries"]:
        if e["id"] not in entry["moves"]:
            raise UsageWeightTableError(f"{where}: move {e['id']} is not in the roster move pool")
    _sum_to_total(
        moves["entries"],
        [moves["truncated_ppm"], moves["unmapped_ppm"], moves["empty_ppm"]],
        f"{where} moves",
    )

    spreads = species["spreads"]
    if len(spreads["entries"]) > limits["k_spreads"]:
        raise UsageWeightTableError(f"{where} spreads: more entries than k_spreads")
    natures = {n["id"] for n in vocabulary["natures"]}
    for k, spread in enumerate(spreads["entries"]):
        if spread["nature"] not in natures:
            raise UsageWeightTableError(
                f"{where}: nature {spread['nature']} is not in the vocabulary"
            )
        if sum(spread["stat_points"].values()) > vocabulary["rules"]["ev_limit"]:
            raise UsageWeightTableError(
                f"{where}: spread {k} holds more Stat Points than the format's limit"
            )
        if k > 0 and _spread_key(spreads["entries"][k - 1]) >= _spread_key(spread):
            raise UsageWeightTableError(f"{where} spreads: spread {k} is repeated or out of order")
    _sum_to_total(
        spreads["entries"], [spreads["truncated_ppm"], spreads["unmapped_ppm"]], f"{where} spreads"
    )

    mates = species["teammates"]
    _ordered([e["id"] for e in mates["entries"]], f"{where} teammates")
    if len(mates["entries"]) > limits["k_teammates"]:
        raise UsageWeightTableError(f"{where} teammates: more entries than k_teammates")
    for e in mates["entries"]:
        if e["id"] not in roster:
            raise UsageWeightTableError(f"{where}: teammate {e['id']} is not in the roster")
        if e["weight"] != rooted_weight(e["ppm"], limits["teammate_root"]):
            raise UsageWeightTableError(f"{where}: a teammate weight is not the root of its ppm")
    if mates["entries"] or mates["truncated_ppm"] or mates["unmapped_ppm"]:
        _sum_to_total(
            mates["entries"], [mates["truncated_ppm"], mates["unmapped_ppm"]], f"{where} teammates"
        )


def read_usage_weight_table(value: object, vocabulary: dict[str, Any]) -> dict[str, Any]:
    """The schema, then the reader rules, against ``vocabulary`` (a TeamVocabulary document).

    The table must name the vocabulary by hash, and so its format and engine-data table.
    """
    _check("usage-weight-table", value)
    _check("team-vocabulary", vocabulary)
    table: dict[str, Any] = value  # type: ignore[assignment]
    if table["vocabulary_hash"] != sha256_hex(jcs(vocabulary)):
        raise UsageWeightTableError("vocabulary_hash is not the hash of the given vocabulary")
    if table["format_id"] != vocabulary["format_id"]:
        raise UsageWeightTableError("format_id differs from the vocabulary's")
    if table["engine_data_hash"] != vocabulary["engine_data_hash"]:
        raise UsageWeightTableError("engine_data_hash differs from the vocabulary's")
    _ordered([s["source_id"] for s in table["sources"]], "sources")

    roster = {entry["id"]: entry for entry in vocabulary["roster"]}
    _ordered([s["id"] for s in table["species"]], "species")
    usage = sum(s["usage_ppm"] for s in table["species"]) + table["unmapped_species_ppm"]
    if usage != PPM_TOTAL:
        raise UsageWeightTableError("species: the usage shares do not sum to 1,000,000 ppm")
    for species in table["species"]:
        if species["id"] not in roster:
            raise UsageWeightTableError(f"species {species['id']} is not in the vocabulary roster")
        _check_species(species, table, vocabulary, roster)
    return table


def usage_table_hash(table: dict[str, Any]) -> str:
    """SHA-256 of the table's JCS bytes: ``usageTableHash``."""
    return sha256_hex(jcs(table))
