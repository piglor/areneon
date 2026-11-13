"""The Threatened flag of a Battle decision (ADR-010 erratum A11.2, #1366): the Python twin of the
TS ``classifySlot`` held equal to it over the rows the TS side wrote from the committed corpora,
and hand-built decisions for every branch of the reading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from areneon_ml.learner.config import THREAT_ROW_FEATURE_SETS
from areneon_ml.learner.situation import (
    SITUATIONS,
    classify_slot,
    has_threat_rows,
    is_threatened,
    last_request,
    living_active_positions,
    threatened_flag,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "fixtures/situation/rows.json"


def request_line(body: dict[str, Any]) -> str:
    return "|request|" + json.dumps(body)


# -- parity with the TS module ------------------------------------------------------------


def test_the_twin_equals_classify_slot_over_the_corpora_rows() -> None:
    entries = json.loads(FIXTURE.read_text())["entries"]
    seen: set[str] = set()
    flags = 0
    for entry in entries:
        log = ["|start|", request_line(entry["request"])]
        positions = living_active_positions(entry["request"])
        ours = [
            {"position": p, "situation": classify_slot(entry["rows"], p)} for p in positions or []
        ]
        assert ours == entry["slots"], entry["id"]
        assert is_threatened(log, entry["rows"]) is entry["threatened"], entry["id"]
        seen.update(str(s["situation"]) for s in ours)
        flags += entry["threatened"]
    assert seen == set(SITUATIONS)
    assert 0 < flags < len(entries)
    # the sample holds the null-KO fallback, a priority bracket and unsupported rows
    rows = [r for e in entries for r in e["rows"]["threats"]]
    assert any(r["kind"] == "computed" and r["ohko_rolls"] is None for r in rows)
    assert any(r["kind"] == "unsupported" for r in rows)
    priorities = [p for e in entries for p in e["rows"]["foe_priorities"]]
    assert any(p["kind"] == "computed" and p["priority"] > 0 for p in priorities)


def test_the_feature_set_list_is_the_ts_readers() -> None:
    source = (ROOT / "packages/showdown-bridge/src/training-run-config.ts").read_text()
    found = re.search(r"THREAT_ROW_FEATURE_SETS: readonly string\[\] = \[([^\]]*)\]", source)
    assert found is not None
    assert tuple(re.findall(r'"([^"]+)"', found.group(1))) == THREAT_ROW_FEATURE_SETS


# -- the reading, hand-built ----------------------------------------------------------------


def threat(own: int, foe: int, move: str, **over: Any) -> dict[str, Any]:
    row = {
        "own": own,
        "mega": False,
        "foe": foe,
        "move": move,
        "kind": "computed",
        "mean_ppm": 500_000,
        "max_ppm": 600_000,
        "min_ppm": 400_000,
        "ohko_rolls": 3,
    }
    row.update(over)
    return row


def speed(own: int, foe: int, relation: str, **over: Any) -> dict[str, Any]:
    return {"own": own, "mega": False, "foe": foe, "kind": "computed", "relation": relation} | over


def prio(foe: int, move: str, priority: int, **over: Any) -> dict[str, Any]:
    return {"foe": foe, "move": move, "kind": "computed", "priority": priority} | over


def rows(threats: list[Any], speeds: list[Any], priorities: list[Any]) -> dict[str, Any]:
    return {"threats": threats, "speeds": speeds, "foe_priorities": priorities}


def test_a_reaching_foe_that_acts_first_threatens() -> None:
    # the foe is faster at priority 0
    assert (
        classify_slot(rows([threat(0, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", 0)]), 0)
        == "threatened"
    )
    # or has priority above 0 whatever the speed
    assert (
        classify_slot(rows([threat(0, 1, "m")], [speed(0, 1, "own_first")], [prio(1, "m", 1)]), 0)
        == "threatened"
    )
    # a negative priority acts after the own default-priority action, whatever the speed
    assert (
        classify_slot(rows([threat(0, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", -1)]), 0)
        == "exposed_slower"
    )


def test_a_slower_or_tied_foe_is_exposed_slower_and_no_reach_is_safe() -> None:
    for relation in ("own_first", "tie"):
        found = classify_slot(
            rows([threat(0, 1, "m")], [speed(0, 1, relation)], [prio(1, "m", 0)]), 0
        )
        assert found == "exposed_slower"
    assert classify_slot(rows([threat(0, 1, "m", ohko_rolls=0)], [], []), 0) == "safe"
    assert classify_slot(rows([], [], []), 0) == "safe"
    # a threat on another own position does not count
    other = rows([threat(1, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", 0)])
    assert classify_slot(other, 0) == "safe"


def test_null_ko_counts_fall_back_to_the_maximum_share() -> None:
    first = [speed(0, 1, "foe_first")]
    reaching = threat(0, 1, "m", ohko_rolls=None, max_ppm=1_000_000)
    assert classify_slot(rows([reaching], first, [prio(1, "m", 0)]), 0) == "threatened"
    short = threat(0, 1, "m", ohko_rolls=None, max_ppm=999_999)
    assert classify_slot(rows([short], first, [prio(1, "m", 0)]), 0) == "safe"


def test_unsupported_missing_and_mega_rows_never_threaten_nor_guess() -> None:
    first = [speed(0, 1, "foe_first")]
    assert (
        classify_slot(rows([threat(0, 1, "m", kind="unsupported")], first, [prio(1, "m", 0)]), 0)
        == "safe"
    )
    assert (
        classify_slot(rows([threat(0, 1, "m", mega=True)], first, [prio(1, "m", 0)]), 0) == "safe"
    )
    # a missing or unsupported priority row counts as priority 0, so the speed row decides
    assert classify_slot(rows([threat(0, 1, "m")], first, []), 0) == "threatened"
    unsupported = [prio(1, "m", 5, kind="unsupported")]
    assert classify_slot(rows([threat(0, 1, "m")], first, unsupported), 0) == "threatened"
    assert (
        classify_slot(rows([threat(0, 1, "m")], [speed(0, 1, "own_first")], unsupported), 0)
        == "exposed_slower"
    )
    # a missing, unsupported or Mega speed row is "not first"
    for speeds in (
        [],
        [speed(0, 1, "foe_first", kind="unsupported")],
        [speed(0, 1, "foe_first", mega=True)],
    ):
        assert (
            classify_slot(rows([threat(0, 1, "m")], speeds, [prio(1, "m", 0)]), 0)
            == "exposed_slower"
        )
    # the row of another foe or move is not the row asked about
    other = [prio(2, "m", 1), prio(1, "other", 1)]
    assert classify_slot(rows([threat(0, 1, "m")], [], other), 0) == "exposed_slower"
    assert classify_slot(rows([threat(0, 1, "m")], [speed(0, 2, "foe_first")], []), 0) == (
        "exposed_slower"
    )


def move_request(conditions: list[str], active: list[bool]) -> dict[str, Any]:
    return {
        "active": [{} for _ in range(sum(active))],
        "side": {
            "pokemon": [
                {"condition": c, "active": a} for c, a in zip(conditions, active, strict=True)
            ]
        },
    }


def test_only_living_active_slots_of_a_move_request_count() -> None:
    body = move_request(["50/100", "0 fnt", "80/100", "100/100"], [True, True, False, False])
    assert living_active_positions(body) == [0]
    threatened = rows([threat(1, 1, "m")], [speed(1, 1, "foe_first")], [prio(1, "m", 0)])
    # position 1 would be Threatened but is fainted; position 0 has no threat
    assert is_threatened([request_line(body)], threatened) is False
    benched = rows([threat(2, 1, "m")], [speed(2, 1, "foe_first")], [prio(1, "m", 0)])
    assert is_threatened([request_line(body)], benched) is False  # not active
    on_slot = rows([threat(0, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", 0)])
    assert is_threatened([request_line(body)], on_slot) is True


def test_a_request_that_is_not_a_move_request_has_no_threatened_slot() -> None:
    hot = rows([threat(0, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", 0)])
    side = {"pokemon": [{"condition": "100/100", "active": True}]}
    forced = {"forceSwitch": [True], "side": side}
    assert living_active_positions(forced) is None
    assert is_threatened([request_line(forced)], hot) is False
    preview = {"teamPreview": True, "active": [{}], "side": side}
    assert living_active_positions(preview) is None
    assert is_threatened([request_line(preview)], hot) is False
    assert is_threatened(["|turn|1"], hot) is False  # no request line at all
    assert last_request(["|request|[1]"]) is None  # not an object
    assert last_request(["|request|{}", "|turn|2"]) == {}
    # the LAST request line is the one read
    both = [request_line(move_request(["1/100"], [True])), request_line(forced)]
    assert is_threatened(both, hot) is False


def test_the_rows_are_present_exactly_when_all_three_arrays_are() -> None:
    assert has_threat_rows(rows([], [], []))
    assert not has_threat_rows(None)
    assert not has_threat_rows({"rows": [], "threats": []})
    assert not has_threat_rows({"threats": [], "speeds": []})


def test_the_example_flag_is_none_without_the_rows() -> None:
    hot = rows([threat(0, 1, "m")], [speed(0, 1, "foe_first")], [prio(1, "m", 0)])
    log = [request_line(move_request(["100/100"], [True]))]
    assert threatened_flag(log, hot) is True
    assert threatened_flag(log, rows([], [], [])) is False
    assert threatened_flag(log, None) is None
    assert threatened_flag(log, {"rows": []}) is None
