"""The strict readers (ADR-017 section 7): closed key sets pinned to a real loop's files."""

from __future__ import annotations

from typing import Any

import pytest
from areneon_ml.inspector import readers
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout

from tests.ml.inspector.support import FIXTURE

GEN = FIXTURE / "generations" / "g000001"


def load(name: str) -> Any:
    return layout.read_json(GEN / name)


def test_the_key_sets_are_the_keys_the_loop_wrote() -> None:
    """A change to the loop's writer (a key added, renamed or dropped) fails here, not in a page."""
    plan = load("plan.json")
    assert {frozenset(entry) for entry in plan} == {readers.PLAN_KEYS[0]}
    assert {frozenset(entry["teams"]) for entry in plan} == {frozenset(readers.SEATS)}
    generation = load("generation.json")
    assert frozenset(generation) == readers.GENERATION_KEYS[0]
    assert {frozenset(game) for game in generation["games"]} == {readers.GAME_KEYS[0]}
    assert frozenset(generation["manifest"]) == readers.VERSION_MANIFEST_KEYS[0]
    assert frozenset(load("team-set.json")) == readers.TEAM_SET_KEYS[0]
    baselines = [e["opponent"] for e in plan if e["opponent"] != "self"]
    assert baselines
    assert {frozenset(b) for b in baselines} == {readers.BASELINE_KEYS[0]}


def test_the_fixture_files_are_read() -> None:
    plan = readers.read_plan(load("plan.json"))
    generation = readers.read_generation(load("generation.json"), 1)
    teams = readers.read_team_set(load("team-set.json"))
    assert [p.game for p in plan] == list(range(len(plan))) == [g.game for g in generation.games]
    assert {p.opponent for p in plan} == {"self", "random-legal@v1", "actor-rule@v1"}
    assert sorted(generation.team_hashes) == list(generation.team_hashes)
    assert set(generation.team_hashes) <= set(teams)


def test_an_unknown_or_missing_key_fails_loudly() -> None:
    plan = load("plan.json")
    plan[0]["new_field"] = 1
    with pytest.raises(InspectError, match="keys this inspector does not know: new_field"):
        readers.read_plan(plan)
    plan = load("plan.json")
    del plan[0]["baseline_seed"]
    with pytest.raises(InspectError, match="lacks baseline_seed"):
        readers.read_plan(plan)
    generation = load("generation.json")
    generation["new_field"] = 1
    with pytest.raises(InspectError, match="new_field"):
        readers.read_generation(generation, 1)
    generation = load("generation.json")
    generation["games"][0]["extra"] = 1
    with pytest.raises(InspectError, match="extra"):
        readers.read_generation(generation, 1)
    generation = load("generation.json")
    generation["manifest"]["extra"] = "x"
    with pytest.raises(InspectError, match="extra"):
        readers.read_generation(generation, 1)
    team_set = load("team-set.json")
    team_set["extra"] = 1
    with pytest.raises(InspectError, match="extra"):
        readers.read_team_set(team_set)
    with pytest.raises(InspectError, match="not an object"):
        readers.read_team_set([])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p[0].update(opponent={"actor": "x"}), "lacks policy_id"),
        (lambda p: p[0].update(policy_seat="p3"), "policy_seat"),
        (lambda p: p[0].update(policy_seat="p1"), "only a self-play game"),
        (lambda p: p[0].update(game=5), "is numbered 5"),
        (lambda p: p[0].update(seed="nope"), "not a sodium seed"),
        (lambda p: p[0]["teams"].update(p1=-1), "teams p1"),
        (lambda p: p[0].update(baseline_seed="7"), "baseline_seed"),
        (lambda p: p[1].update(seed=p[0]["seed"]), "repeats a seed"),
        (lambda p: p.append("x"), "not an object"),
    ],
)
def test_the_plan_refuses(change: Any, message: str) -> None:
    plan = load("plan.json")
    change(plan)
    with pytest.raises(InspectError, match=message):
        readers.read_plan(plan)
    with pytest.raises(InspectError, match="not a list"):
        readers.read_plan({})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda g: g.update(schema_version=2), "schema 1"),
        (lambda g: g.update(generation=2), "schema 1"),
        (lambda g: g["games"][0].update(outcome="lost"), "outcome"),
        (lambda g: g["games"][0].update(winner="p3"), "winner"),
        (lambda g: g["games"][0].update(game=3), "is numbered 3"),
        (lambda g: g["games"][0].update(turns=-1), "turns"),
        (lambda g: g["games"][0].update(seed=3), "seed"),
        (
            lambda g: g["games"][0].update(forfeit={"loser": "p1", "cause": "x", "request_seq": 1}),
            "but was played",
        ),
        (lambda g: g.update(episode_hashes=["x"]), "SHA-256"),
        (lambda g: g.update(forfeit_causes=[]), "forfeit_causes"),
        (lambda g: g.update(examples=True), "examples"),
        (lambda g: g.update(training_set_hash="x"), "training_set_hash"),
    ],
)
def test_the_generation_refuses(change: Any, message: str) -> None:
    generation = load("generation.json")
    change(generation)
    with pytest.raises(InspectError, match=message):
        readers.read_generation(generation, 1)


def test_a_forfeit_row_is_read() -> None:
    generation = load("generation.json")
    generation["games"][0].update(
        outcome="forfeit",
        winner="p2",
        forfeit={"loser": "p1", "cause": "latency", "request_seq": 3, "detail": "slow"},
    )
    generation["forfeits"] = 1
    generation["forfeit_causes"] = {"latency": 1}
    read = readers.read_generation(generation, 1)
    forfeit = read.games[0].forfeit
    assert forfeit is not None
    assert (forfeit.loser, forfeit.cause, forfeit.request_seq, forfeit.detail) == (
        "p1",
        "latency",
        3,
        "slow",
    )
    assert read.forfeit_causes == {"latency": 1}
    generation["games"][0]["forfeit"] = {"loser": "p1", "cause": "latency", "request_seq": 3}
    assert readers.read_generation(generation, 1).games[0].forfeit.detail is None  # type: ignore[union-attr]
    generation["games"][0]["forfeit"]["loser"] = "p3"
    with pytest.raises(InspectError, match="loser"):
        readers.read_generation(generation, 1)


def test_the_team_set_must_be_sorted_and_unique() -> None:
    teams = load("team-set.json")
    teams["teams"] = list(reversed(teams["teams"]))
    with pytest.raises(InspectError, match="sorted and unique"):
        readers.read_team_set(teams)
    teams["schema_version"] = 2
    with pytest.raises(InspectError, match="schema 1"):
        readers.read_team_set(teams)
