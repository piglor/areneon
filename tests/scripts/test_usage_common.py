"""The shared pieces of the offline usage derivations: decimals, units, truncation, largest remainder."""

# ruff: noqa: E501  (toy files and expected records are kept whole so they read as written)

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from scripts.usage_common import (
    DerivationError,
    NameTable,
    fraction,
    largest_remainder,
    parse_source,
    to_units,
    top_k,
)


def test_parse_source_reads_floats_as_exact_decimals() -> None:
    parsed = parse_source('{"a": 0.1, "b": 3, "c": 131622.82570831427}')
    assert parsed["a"] == Decimal("0.1")
    assert isinstance(parsed["b"], int)
    assert fraction(parsed["c"]) == Fraction(13162282570831427, 10**11)


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_parse_source_refuses_duplicates_and_non_numbers(text: str) -> None:
    with pytest.raises(DerivationError):
        parse_source(text)


@pytest.mark.parametrize("value", [True, "1", -1, Decimal("-0.5"), None])
def test_fraction_refuses_non_numbers_and_negatives(value: object) -> None:
    with pytest.raises(DerivationError):
        fraction(value)


def test_to_units_rounds_half_to_even() -> None:
    assert to_units(Fraction(131622, 1)) == 131_622_000_000
    assert to_units(Fraction(5, 2 * 10**6)) == 2  # 2.5 units -> even
    assert to_units(Fraction(7, 2 * 10**6)) == 4  # 3.5 units -> even
    assert to_units(Fraction(1, 3 * 10**6)) == 0


def test_name_table_finds_an_exact_id_an_exact_name_or_the_engine_id_of_a_name() -> None:
    table = NameTable(
        [
            {"id": "farfetchd", "name": "Farfetch\N{RIGHT SINGLE QUOTATION MARK}d"},
            {"id": "rillaboom", "name": "Rillaboom"},
        ]
    )
    assert table.resolve("rillaboom") == "rillaboom"
    assert table.resolve("Rillaboom") == "rillaboom"
    assert table.resolve("Farfetch\N{RIGHT SINGLE QUOTATION MARK}d") == "farfetchd"
    assert table.resolve("Farfetch'd") == "farfetchd"  # the engine's toID of the source's name
    assert table.resolve("FARFETCH-D") == "farfetchd"
    assert table.resolve("Farfetch'dd") is None  # an id the table lacks is never guessed
    assert table.resolve("Rilla boom") == "rillaboom"
    assert table.resolve("''") is None
    assert table.resolve("rillabo") is None


def test_top_k_orders_by_weight_then_id() -> None:
    weights = {"b": Fraction(2), "a": Fraction(2), "c": Fraction(5), "d": Fraction(1)}
    assert top_k(weights, 3) == ["c", "a", "b"]
    assert top_k(weights, 10) == ["c", "a", "b", "d"]


def test_largest_remainder_sums_exactly_and_breaks_ties_by_id() -> None:
    assert largest_remainder(10, {"b": Fraction(1), "a": Fraction(1), "c": Fraction(1)}) == {
        "a": 4,
        "b": 3,
        "c": 3,
    }
    # 2/3 and 1/3 of 10: quotas 6.67 and 3.33 -> the larger remainder takes the spare unit.
    assert largest_remainder(10, {"x": Fraction(2), "y": Fraction(1)}) == {"x": 7, "y": 3}
    assert largest_remainder(0, {"x": Fraction(0)}) == {"x": 0}


def test_largest_remainder_respects_caps() -> None:
    out = largest_remainder(10, {"a": Fraction(1), "b": Fraction(1)}, caps={"a": 3})
    assert out == {"a": 3, "b": 7}
    # A quota above its cap passes the surplus on.
    assert largest_remainder(4, {"a": Fraction(9), "b": Fraction(1)}, caps={"a": 1}) == {
        "a": 1,
        "b": 3,
    }


def test_largest_remainder_refuses_what_cannot_be_scaled() -> None:
    with pytest.raises(DerivationError):
        largest_remainder(5, {"a": Fraction(0)})
    with pytest.raises(DerivationError):
        largest_remainder(5, {"a": Fraction(1)}, caps={"a": 2})
