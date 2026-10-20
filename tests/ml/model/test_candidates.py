"""The candidate parser (ADR-010 §3): round trip over the whole committed choice corpus."""

from __future__ import annotations

import json

import pytest
from areneon_ml.model.candidates import EVENTS, SlotAction, UnsupportedCandidate, parse_choice

from .support import FORMAT_DIR, ROOT, sample

MANY = 1000
CORPUS = sorted((ROOT / "fixtures" / "showdown").glob("**/choices/*.json"))


def test_the_corpus_is_there() -> None:
    assert len(CORPUS) >= 1
    assert FORMAT_DIR / "choices" / "preview.json" in CORPUS


def test_every_candidate_of_the_corpus_round_trips_to_the_identical_string() -> None:
    seen: set[str] = set()
    count = 0
    for path in CORPUS:
        for candidate in json.loads(path.read_text())["candidates"]:
            choice = candidate["choice"]
            assert parse_choice(choice).format() == choice, f"{path.name}: {choice}"
            seen.add(choice)
            count += 1
    for name in ("policy-query/valid/battle.json", "policy-query/valid/preview.json"):
        for candidate in sample(name)["candidates"]:
            assert parse_choice(candidate["choice"]).format() == candidate["choice"]
    assert count > MANY


def test_the_corpus_has_every_part_the_head_embeds() -> None:
    kinds: set[str] = set()
    events: set[str | None] = set()
    targets: set[int | None] = set()
    for path in CORPUS:
        for candidate in json.loads(path.read_text())["candidates"]:
            parsed = parse_choice(candidate["choice"])
            kinds.add(parsed.kind)
            for slot in parsed.slots:
                kinds.add(slot.kind)
                events.add(slot.event)
                targets.add(slot.target)
    assert kinds == {"team", "joint", "pass", "move", "switch"}
    assert {None, "mega"} <= events
    assert {None, 1, 2, -1, -2} <= targets


def test_parts_are_read_from_the_canonical_string() -> None:
    parsed = parse_choice("move 3 -2 megax, switch 4")
    assert parsed.kind == "joint"
    assert parsed.slots == (SlotAction("move", 3, -2, "megax"), SlotAction("switch", 4))
    assert parsed.parts == ("move 3 -2 megax", "switch 4")
    assert parse_choice("pass, move 4").slots[0] == SlotAction("pass")
    team = parse_choice("team 3126")
    assert (team.kind, team.team, team.parts) == ("team", (3, 1, 2, 6), ("3126",))
    assert set(EVENTS) == {"mega", "megax", "megay", "terastallize"}


@pytest.mark.parametrize(
    "choice",
    [
        "",
        "default",
        "team",
        "team 0",
        "team 1123",
        "team 12a",
        "move 0",
        "move 1 mega2",
        "move 1 zmove",
        "move 1 2 3",
        "move 1 0",
        "switch 0",
        "switch 1",
        "pass",
        "move 1, move 2, move 3",
        "move 1,move 2",
        "move 1 ,move 2",
        "shift, pass",
        "Move 1, move 2",
        " move 1, move 2",
        "move 1, move 2\n",
    ],
)
def test_an_unknown_shape_is_refused(choice: str) -> None:
    with pytest.raises(UnsupportedCandidate):
        parse_choice(choice)
