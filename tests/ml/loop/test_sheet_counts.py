"""Opponent sheet block counts and E4's 1% validity rule (ADR-025 erratum E4 point 5)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from areneon_ml.loop.sheet_counts import (
    MAX_REFUSED_PPM,
    SheetCounts,
    SheetCountsError,
    read_sheet_counts,
    total,
)
from areneon_ml.report.model import GenerationFacts
from areneon_ml.report.render import _sheet_section


def counts(requests: int, refused: int, *refusals: tuple[str, int]) -> SheetCounts:
    return SheetCounts(requests, refused, tuple(refusals))


def test_the_share_and_the_one_percent_rule() -> None:
    assert counts(0, 0).refused_share_ppm == 0
    assert counts(100, 1).refused_share_ppm == MAX_REFUSED_PPM
    assert counts(100, 1).valid  # exactly 1% is not above it
    assert not counts(100, 2).valid
    assert counts(3, 1).refused_share_ppm == 333333  # rounded down


def test_counts_are_summed_with_the_refusals_by_code() -> None:
    a = counts(10, 2, ("a", 1), ("b", 1))
    b = counts(5, 1, ("b", 1))
    assert total([a, b]) == counts(15, 3, ("a", 1), ("b", 2))
    assert a.as_dict() == {
        "requests": 10,
        "refused": 2,
        "refusals": [{"code": "a", "count": 1}, {"code": "b", "count": 1}],
    }
    assert read_sheet_counts(a.as_dict()) == a


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "exactly"),
        ({"requests": 1, "refused": 0}, "exactly"),
        ({"requests": 1.5, "refused": 0, "refusals": []}, "requests"),
        ({"requests": 1, "refused": 0, "refusals": {}}, "not a list"),
        ({"requests": 1, "refused": 0, "refusals": [3]}, "code and count"),
        ({"requests": 1, "refused": 1, "refusals": [{"code": "", "count": 1}]}, "not text"),
        ({"requests": 1, "refused": 2, "refusals": [{"code": "a", "count": 2}]}, "more than"),
        ({"requests": 3, "refused": 2, "refusals": [{"code": "a", "count": 1}]}, "add up"),
        (
            {
                "requests": 3,
                "refused": 2,
                "refusals": [{"code": "b", "count": 1}, {"code": "a", "count": 1}],
            },
            "sorted",
        ),
    ],
)
def test_a_sheet_object_must_follow_its_rules(raw: Any, message: str) -> None:
    with pytest.raises(SheetCountsError, match=message):
        read_sheet_counts(raw)


def generation(number: int, sheet: SheetCounts | None) -> GenerationFacts:
    return GenerationFacts(number, "0" * 64, None, 1, 1, 0, 2, 3, None, {}, None, None, sheet=sheet)


def test_the_dashboard_shows_the_blocks_and_flags_an_invalid_run() -> None:
    model: Any = SimpleNamespace(
        generations=[generation(1, counts(100, 0)), generation(2, counts(100, 5, ("x", 5)))]
    )
    html = _sheet_section(model)
    assert "Opponent sheet blocks" in html
    assert "valid" in html
    assert "INVALID: refused share above 1%" in html
    assert "x 5" in html
    none: Any = SimpleNamespace(generations=[generation(1, None)])
    assert _sheet_section(none) == ""
