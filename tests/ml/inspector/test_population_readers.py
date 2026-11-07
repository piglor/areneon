"""The population readers (ADR-021 section 8): closed key sets pinned to a tiny loop's files."""

from __future__ import annotations

from typing import Any

import pytest
from areneon_ml.inspector import population_readers as pr
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout

from tests.ml.inspector.population_support import POP_FIXTURE

GEN = POP_FIXTURE / "generations" / "g000001"


def load(name: str) -> Any:
    return layout.read_json(GEN / name)


def test_the_key_sets_are_the_keys_the_loop_wrote() -> None:
    """A change to the simulator side's writer (a key added, renamed or dropped) fails here."""
    plan = load("plan.json")
    assert {frozenset(entry) for entry in plan} == {pr.PLAN_KEYS[0]}
    seat_keys = {frozenset(seat) for entry in plan for seat in entry["seats"].values()}
    assert seat_keys == {
        pr.MEMBER_SEAT_KEYS[0],
        pr.SNAPSHOT_SEAT_KEYS[0],
        pr.BASELINE_SEAT_KEYS[0],
    }
    assert {entry["kind"] for entry in plan} == set(pr.KINDS)
    generation = load("generation.json")
    assert frozenset(generation) == pr.GENERATION_KEYS[0]
    assert {frozenset(game) for game in generation["games"]} == {pr.GAME_KEYS[0]}
    assert {frozenset(m) for m in generation["members"]} == {pr.MEMBER_KEYS[0]}
    assert frozenset(generation["manifest"]) == pr.VERSION_MANIFEST_KEYS[0]


def test_the_fixture_files_are_read() -> None:
    plan = pr.read_population_plan(load("plan.json"))
    generation = pr.read_population_generation(load("generation.json"), 1)
    assert [p.game for p in plan] == list(range(len(plan))) == [g.game for g in generation.games]
    assert {p.kind for p in plan} == set(pr.KINDS)
    assert [m.member for m in generation.members] == ["m0", "m1"]
    past = [p for p in plan if p.kind == "past"]
    assert past
    assert {s.kind for p in past for s in p.seats.values()} == {"member", "snapshot"}
    assert pr.snapshot_parts("m1@g0") == ("m1", 0)


def test_an_unknown_or_missing_key_fails_loudly() -> None:
    plan = load("plan.json")
    plan[0]["new_field"] = 1
    with pytest.raises(InspectError, match=r"keys this inspector does not know: new_field"):
        pr.read_population_plan(plan)
    plan = load("plan.json")
    del plan[0]["pair"]
    with pytest.raises(InspectError, match=r"lacks pair"):
        pr.read_population_plan(plan)
    plan = load("plan.json")
    plan[0]["seats"]["p1"]["extra"] = 1
    with pytest.raises(InspectError, match=r"extra"):
        pr.read_population_plan(plan)
    generation = load("generation.json")
    generation["new_field"] = 1
    with pytest.raises(InspectError, match=r"new_field"):
        pr.read_population_generation(generation, 1)
    generation = load("generation.json")
    generation["games"][0]["extra"] = 1
    with pytest.raises(InspectError, match=r"extra"):
        pr.read_population_generation(generation, 1)
    generation = load("generation.json")
    generation["members"][0]["extra"] = 1
    with pytest.raises(InspectError, match=r"extra"):
        pr.read_population_generation(generation, 1)


def test_values_it_cannot_read_are_refused() -> None:
    def plan_with(change: Any) -> list[Any]:
        plan: list[Any] = load("plan.json")
        change(plan)
        return plan

    cases = [
        (lambda p: p[0].update(game=5), "is numbered 5"),
        (lambda p: p[0].update(half=1), "is not half 1 of pair 0"),
        (lambda p: p[0].update(kind="league"), "kind is not one of"),
        (lambda p: p[0].update(member="x0"), "no learning member"),
        (lambda p: p[0].update(seed="seed"), "not a sodium seed"),
        (lambda p: p[0]["seats"].update(p1={"member": "z"}), "no learning member"),
        (lambda p: p[0]["seats"].update(p1={"snapshot": "z", "policy_id": "p"}), "no past-pool"),
        (lambda p: p[0]["teams"].update(p1=-1), "not an integer"),
        (lambda p: p[1].update(seed=p[0]["seed"]), None),
    ]
    for change, words in cases:
        if words is None:
            continue
        with pytest.raises(InspectError, match=words):
            pr.read_population_plan(plan_with(change))
    with pytest.raises(InspectError, match=r"not a list"):
        pr.read_population_plan({})
    generation = load("generation.json")
    for change, words in (
        (lambda g: g.update(schema_version=1), "not schema 2"),
        (lambda g: g.update(generation=9), "not schema 2"),
        (lambda g: g["games"][0].update(outcome="x"), "outcome is not played or forfeit"),
        (lambda g: g["games"][0].update(winner="p3"), "winner is not"),
        (lambda g: g["games"][0].update(kind="x"), "kind is not one of"),
        (lambda g: g["games"][0].update(forfeit={}), "has a forfeit record but was played"),
        (
            lambda g: g["members"][0].update(episode_hashes=["b" * 64, "a" * 64]),
            "sorted and unique",
        ),
        (lambda g: g["members"][0].update(member="q"), "no learning member"),
        (lambda g: g.update(forfeit_causes=[]), "forfeit_causes is not an object"),
    ):
        broken = load("generation.json")
        change(broken)
        with pytest.raises(InspectError, match=words):
            pr.read_population_generation(broken, 1)
    assert generation["forfeits"] == 0


def test_a_forfeit_record_is_read_whole() -> None:
    generation = load("generation.json")
    row = generation["games"][0]
    row.update(
        outcome="forfeit",
        winner="p2",
        forfeit={"loser": "p1", "cause": "crash", "request_seq": 3, "detail": "x"},
    )
    generation["forfeits"] = 1
    generation["forfeit_causes"] = {"crash": 1}
    read = pr.read_population_generation(generation, 1)
    assert read.games[0].forfeit is not None
    assert read.games[0].forfeit.cause == "crash"
    assert read.forfeit_causes == {"crash": 1}
    row["forfeit"]["loser"] = "p3"
    with pytest.raises(InspectError, match=r"loser is not p1 or p2"):
        pr.read_population_generation(generation, 1)
    row["forfeit"] = {"loser": "p1", "cause": "crash", "request_seq": 3}
    with pytest.raises(InspectError, match=r"lacks detail"):
        pr.read_population_generation(generation, 1)


def test_the_manifest_member_facts_and_snapshot_ids() -> None:
    manifest = layout.read_json(GEN / layout.MANIFEST)
    assert pr.manifest_member(manifest, "m0")["checkpoints"]
    with pytest.raises(InspectError, match=r"no facts for m9"):
        pr.manifest_member(manifest, "m9")
    with pytest.raises(InspectError, match=r"<member>@g<generation>"):
        pr.snapshot_parts("m1")
