"""The usage weight table reader (ADR-020 section 4; #976).

The Python reader refuses exactly what the TS one does: every reader-invalid corpus sample, the
schema-invalid ones, and an ``evs`` key anywhere. The valid samples hash to the digests the TS side
recorded, and the integer root is exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.usage_weight_table import (
    UsageWeightTableError,
    integer_root,
    read_usage_weight_table,
    rooted_weight,
    usage_table_hash,
)

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts"
HASHES: dict[str, str] = json.loads((SAMPLES / "hashes.json").read_text(encoding="utf-8"))


def load(path: str) -> Any:
    return json.loads((SAMPLES / path).read_text(encoding="utf-8"))


VOCABULARY = load("v1/team-vocabulary/valid/golden.json")
READER_INVALID = {
    "unsorted-species": "species: .* repeated or out of order",
    "unsorted-sources": "sources: .* repeated or out of order",
    "species-sum": "usage shares do not sum",
    "species-root": "w_species is not the integer root",
    "unknown-species": "is not in the vocabulary roster",
    "duplicate-item": "items: .* repeated or out of order",
    "unsorted-moves": "moves: .* repeated or out of order",
    "unsorted-spreads": "spread 1 is repeated or out of order",
    "unsorted-teammates": "teammates: .* repeated or out of order",
    "unknown-item": "item aaaaaaaaa is not in the vocabulary",
    "unknown-move": "move aaaaaaaaa is not in the roster move pool",
    "unknown-ability": "ability aaaaaaaaa is not one the roster lists",
    "unknown-nature": "nature aaaaaaaaa is not in the vocabulary",
    "unknown-teammate": "teammate aaaaaaaaa is not in the roster",
    "abilities-sum": "abilities: the masses do not sum",
    "items-sum": "items: the masses do not sum",
    "moves-sum": "moves: the masses do not sum",
    "spreads-sum": "spreads: the masses do not sum",
    "teammates-sum": "teammates: the masses do not sum",
    "teammate-root": "teammate weight is not the root",
    "spread-over-limit": "more Stat Points than the format's limit",
    "too-many-items": "more entries than k_items",
    "too-many-moves": "more entries than k_moves",
    "too-many-spreads": "more entries than k_spreads",
    "too-many-teammates": "more entries than k_teammates",
    "stale-vocabulary-hash": "vocabulary_hash is not the hash",
    "other-format": "format_id differs",
    "other-engine-data": "engine_data_hash differs",
}


def names(kind: str) -> list[str]:
    prefix = f"v1/usage-weight-table/{kind}/"
    return sorted(p[len(prefix) :].removesuffix(".json") for p in HASHES if p.startswith(prefix))


def test_the_valid_samples_read_and_hash_as_the_ts_side_does() -> None:
    assert names("valid") == ["no-teammates", "root-1", "v1"]
    for name in names("valid"):
        path = f"v1/usage-weight-table/valid/{name}.json"
        table = read_usage_weight_table(load(path), VOCABULARY)
        assert usage_table_hash(table) == HASHES[path]


@pytest.mark.parametrize("name", sorted(READER_INVALID))
def test_a_reader_invalid_sample_is_refused(name: str) -> None:
    value = load(f"v1/usage-weight-table/reader-invalid/{name}.json")
    with pytest.raises(UsageWeightTableError, match=READER_INVALID[name]):
        read_usage_weight_table(value, VOCABULARY)


def test_every_reader_invalid_sample_has_its_expected_message() -> None:
    assert names("reader-invalid") == sorted(READER_INVALID)


def test_every_schema_invalid_sample_is_refused() -> None:
    assert len(names("invalid")) >= 18
    for name in names("invalid"):
        with pytest.raises(UsageWeightTableError):
            read_usage_weight_table(load(f"v1/usage-weight-table/invalid/{name}.json"), VOCABULARY)


def test_an_evs_key_is_refused_because_the_schema_is_closed() -> None:
    table = load("v1/usage-weight-table/valid/v1.json")
    table["species"][0]["spreads"]["entries"][0]["evs"] = {}
    with pytest.raises(UsageWeightTableError):
        read_usage_weight_table(table, VOCABULARY)


@pytest.mark.parametrize(
    ("value", "root", "expected"),
    [
        (0, 2, 0),
        (1, 3, 1),
        (15, 2, 3),
        (16, 2, 4),
        (10**12, 2, 10**6),
        (10**12 - 1, 2, 10**6 - 1),
        (10**18 * 7, 3, 1_912_931),
        (123456789, 1, 123456789),
    ],
)
def test_the_integer_root_is_exact(value: int, root: int, expected: int) -> None:
    got = integer_root(value, root)
    assert got == expected
    assert got**root <= value < (got + 1) ** root


def test_rooted_weight_scales_so_a_full_share_stays_a_million() -> None:
    assert rooted_weight(10**6, 1) == 10**6
    assert rooted_weight(10**6, 2) == 10**6
    assert rooted_weight(1, 2) == 1000
    assert rooted_weight(600_000, 2) == 774_596
    assert rooted_weight(10**6, 3) == 10**6


def test_the_root_refuses_a_negative_value_and_a_zero_root() -> None:
    for args in ((-1, 2), (4, 0)):
        with pytest.raises(UsageWeightTableError):
            integer_root(*args)
