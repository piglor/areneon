"""Training-game replays: reading the two Episodes of a game, and refusing what is not one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import replay
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout

from tests.ml.inspector.support import FIXTURE, copy_run, edit, edit_episode


def episode_files(run: Path, generation: int) -> list[Path]:
    return sorted((layout.generation_dir(run, generation) / "episodes").glob("*.json"))


def test_a_game_reads_as_its_two_own_views() -> None:
    _, games = replay.read_games(FIXTURE, 1, [0])
    (game,) = games
    assert (game.generation, game.row.game, game.run_key) == (1, 0, "inspector-fixture")
    assert [v.seat for v in game.views] == ["p1", "p2"]
    own = {}
    for path in episode_files(FIXTURE, 1):
        episode = json.loads(path.read_text())
        if episode["provenance"]["collection"]["seed"] == game.row.seed:
            own[episode["player"]] = episode
    for view in game.views:
        episode = own[view.seat]
        assert list(view.lines) == episode["channel"]
        assert list(view.team) == episode["team"]["team"]
        assert [d.choice for d in view.decisions] == [d["choice"] for d in episode["decisions"]]
        assert [d.request_seq for d in view.decisions] == [
            d["identity"]["request_seq"] for d in episode["decisions"]
        ]
        assert view.turns == game.row.turns
        assert view.played_by == game.pair_policy_id
    # an own channel never holds the other side's request: every request is for its own seat
    for view in game.views:
        requests = [
            json.loads(x[len("|request|") :]) for x in view.lines if x.startswith("|request|")
        ]
        assert {r["side"]["id"] for r in requests} == {view.seat}


def test_every_played_game_when_none_is_named_and_a_baseline_seat_is_named() -> None:
    _, games = replay.read_games(FIXTURE, 2, None)
    assert [g.row.game for g in games] == [0, 1, 2, 3, 4]
    baseline = games[3]
    assert baseline.plan.policy_seat == "p2"  # the pair sits in p2: the baseline is p1
    assert baseline.views[0].played_by == baseline.plan.opponent != baseline.pair_policy_id
    assert baseline.views[1].played_by == baseline.pair_policy_id
    other = games[4]
    assert other.views[1].played_by == other.plan.opponent
    assert games[0].views[0].played_by == games[0].views[1].played_by == games[0].pair_policy_id


def forfeit_game(run: Path, generation: int, game: int) -> None:
    gen = layout.generation_dir(run, generation)
    seed = layout.read_json(gen / "plan.json")[game]["seed"]
    gone = []
    for path in episode_files(run, generation):
        if json.loads(path.read_text())["provenance"]["collection"]["seed"] == seed:
            gone.append(path.stem)
            path.unlink()

    def change(listing: Any) -> None:
        listing["games"][game].update(
            outcome="forfeit",
            winner=None,
            turns=0,
            forfeit={"loser": "p1", "cause": "latency", "request_seq": 2},
        )
        listing["forfeits"] += 1
        listing["forfeit_causes"] = {"latency": 1}
        listing["episode_hashes"] = sorted(set(listing["episode_hashes"]) - set(gone))

    def record(value: Any) -> None:
        value.update(
            played=value["played"] - 1,
            forfeits=value["forfeits"] + 1,
            episodes=value["episodes"] - 2,
        )

    edit(run, generation, "generation.json", change)
    edit(run, generation, "league-run.json", record)


def test_a_forfeited_game_has_no_replay(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    forfeit_game(run, 1, 1)
    with pytest.raises(InspectError, match="forfeited: it has no replay"):
        replay.read_games(run, 1, [1])
    _, games = replay.read_games(run, 1, None)  # the whole generation skips it
    assert [g.row.game for g in games] == [0, 2, 3, 4]


def test_what_is_not_a_replayable_game_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    with pytest.raises(InspectError, match="plays no games"):
        replay.read_games(run, 0, None)
    with pytest.raises(InspectError, match="has no game 5"):
        replay.read_games(run, 1, [5])
    raw = json.loads((run / layout.CONFIG).read_text())
    raw["lineage_id"] = "private-main"
    layout.write_json(run / layout.CONFIG, raw)
    with pytest.raises(InspectError, match="not a public one"):
        replay.read_games(run, 1, [0])


def test_a_changed_generation_fails_its_manifest(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    plan = layout.generation_dir(run, 1) / "plan.json"
    plan.write_bytes(plan.read_bytes().replace(b"sodium", b"sodiun", 1))
    with pytest.raises(InspectError, match="g000001"):
        replay.read_games(run, 1, [0])


def test_the_files_must_tell_one_story(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    edit(run, 1, "league-run.json", lambda v: v.update(generation=2))
    with pytest.raises(InspectError, match="g000001"):
        replay.read_games(run, 1, [0])
    run = copy_run(tmp_path / "b")

    def other_seed(value: Any) -> None:
        value["games"][0]["seed"] = "sodium," + "0" * 32

    edit(run, 1, "generation.json", other_seed)
    with pytest.raises(InspectError, match="differ"):
        replay.read_games(run, 1, [0])
    run = copy_run(tmp_path / "c")

    def drop_game(value: Any) -> None:
        value["games"].pop()

    edit(run, 1, "generation.json", drop_game)
    with pytest.raises(InspectError, match=r"generation\.json"):
        replay.read_games(run, 1, [0])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.update(data_class="private"), "not a synthetic"),
        (lambda e: e["provenance"]["collection"].update(run_id="other"), "not a league Episode"),
        (lambda e: e.pop("decisions"), "not a valid Episode"),
        (lambda e: e["channel"].append("not a protocol line"), "not a valid Episode|not protocol"),
        (lambda e: e["channel"].insert(3, "|split|p1"), "|split|"),
        (
            lambda e: e["decisions"][0]["identity"].update(request_seq=99),
            "request the channel lacks",
        ),
        (
            lambda e: e["decisions"][1]["identity"].update(
                request_seq=e["decisions"][0]["identity"]["request_seq"]
            ),
            "two decisions answer one request",
        ),
        (lambda e: e["result"].update(turns=77), "result differs"),
    ],
)
def test_an_episode_that_is_not_a_clean_own_channel_is_refused(
    tmp_path: Path, change: Any, message: str
) -> None:
    run = copy_run(tmp_path)
    edit_episode(run, 1, 0, change)
    with pytest.raises(InspectError, match=message):
        replay.read_games(run, 1, None)


def test_an_episode_per_seat_is_needed(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    gen = layout.generation_dir(run, 1)
    seed = layout.read_json(gen / "plan.json")[0]["seed"]
    mine = [p for p in episode_files(run, 1) if seed in p.read_text()]
    assert len(mine) == 2
    mine[0].unlink()

    def drop(listing: Any) -> None:
        listing["episode_hashes"].remove(mine[0].stem)

    edit(run, 1, "generation.json", drop)
    with pytest.raises(InspectError, match="one Episode per seat"):
        replay.read_games(run, 1, [0])


def test_an_episode_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    game = replay.read_games(FIXTURE, 1, [0])[1][0]
    bad = tmp_path / "x.json"
    for content in (None, b"{not json"):
        if content is not None:
            bad.write_bytes(content)
        with pytest.raises(InspectError, match="not a valid Episode"):
            replay.read_seat(bad, "inspector-fixture", game.plan, game.pair_policy_id, game.row)


def test_the_ids_of_a_page_are_link_ids() -> None:
    assert replay.GAME_NAME.format(game=7) == "game-000007"
    assert replay.to_id("Farfetch\u2019d, L50") == "farfetchdl50"
