"""The KO counts of the damage fixture, recounted outside the TypeScript probe (#1166).

The `ko` entries of fixtures/mechanics/<format>/damage/cases.json are expectations the probe is
held to, so they must not come from the code under test. This recounts them from each case's own
roll vectors and current HP with plain arithmetic: the rolls at or above the HP, and the ordered
pairs of two rolls of the no-crit vector whose sum reaches it (before any HP clamp).
"""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "mechanics"
    / "gen9championsvgc2026regmc"
    / "damage"
    / "cases.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def test_the_fixture_has_cases() -> None:
    assert len(CASES) == 12


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_ko_counts_are_recounted_from_the_vectors(case: dict[str, Any]) -> None:
    expected = case["expected"]
    hp = expected["hp_current"]
    no_crit, crit = expected["no_crit"], expected["crit"]
    assert len(no_crit) == len(crit) == 16
    assert expected["ko"] == {
        "kind": "counts",
        "ohko_no_crit": sum(1 for roll in no_crit if roll >= hp),
        "ohko_crit": sum(1 for roll in crit if roll >= hp),
        "thko_no_crit": sum(1 for a in no_crit for b in no_crit if a + b >= hp),
    }
