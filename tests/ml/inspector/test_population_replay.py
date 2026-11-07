"""Population-game replays: each member seat's own Episode, side by side (ADR-021, ADR-017 4a)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import command, replay, replay_command, replay_page
from areneon_ml.inspector.pool import SCRIPT, load_pool
from areneon_ml.inspector.population_replay import read_population_games
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_command import Selection
from areneon_ml.loop import __main__ as loop_main
from areneon_ml.loop import layout

from tests.ml.inspector.population_support import (
    POP_FIXTURE,
    copy_pop_run,
    edit,
    gen_dir,
    pop_pool,
    rename_episode,
)
from tests.ml.inspector.support import FIXTURE, ROOT
from tests.ml.inspector.test_command import loader
from tests.ml.inspector.test_pages import audit
from tests.ml.inspector.test_replay_pages import CANARIES, LOG, NICK, POLICY, RUN, problems, table


def games(generation: int, which: list[int] | None, member: str | None = None) -> list[Any]:
    return read_population_games(POP_FIXTURE, generation, which, pop_pool(), member)[1]


def build(g: replay.GameReplay) -> str:
    return replay_page.replay_page(g, replay_page.dex_for(table(), g))


def test_a_game_reads_as_its_two_own_views_with_who_held_each_seat() -> None:
    (ladder,) = games(2, [0])
    assert (ladder.generation, ladder.row.game) == (2, 0)
    assert ladder.population is not None
    assert ladder.population.holders == {"p1": ("member", "m0"), "p2": ("member", "m1")}
    assert [v.seat for v in ladder.views] == ["p1", "p2"]
    assert [v.played_by for v in ladder.views] == ["m0", "m1"]
    for view in ladder.views:
        requests = [json.loads(x[9:]) for x in view.lines if x.startswith("|request|")]
        assert {r["side"]["id"] for r in requests} == {view.seat}


def test_the_seats_of_the_other_kinds_are_labelled_by_their_holder() -> None:
    every = games(2, None)
    assert len(every) == 24
    kinds = {g.population.kind: g for g in every if g.population is not None}
    assert set(kinds) == {"ladder", "self", "past", "anchor"}
    past = kinds["past"].population
    assert past is not None
    assert sorted(kind for kind, _ in past.holders.values()) == ["member", "snapshot"]
    anchor = kinds["anchor"].population
    assert anchor is not None
    assert sorted(kind for kind, _ in anchor.holders.values()) == ["baseline", "member"]
    held = next(iter(kinds["self"].population.holders.values()))
    assert held == ("member", kinds["self"].population.member)


def test_a_self_play_pair_swaps_its_teams_between_the_halves() -> None:
    selfs = [g for g in games(2, None) if g.population and g.population.kind == "self"]
    by_pair: dict[int, list[Any]] = {}
    for g in selfs:
        assert g.population is not None
        by_pair.setdefault(g.population.pair, []).append(g)
    for pair in by_pair.values():
        first, second = sorted(pair, key=lambda g: g.population.half)
        assert first.views[0].team == second.views[1].team
        assert first.views[1].team == second.views[0].team
        assert first.views[0].team != first.views[1].team


def test_the_member_filter_keeps_the_games_that_member_holds_a_seat_in() -> None:
    mine = games(2, None, "m1")
    assert mine
    assert all(
        g.population is not None and "m1" in {label for _, label in g.population.holders.values()}
        for g in mine
    )
    assert len(mine) < 24
    assert {g.row.game for g in games(2, None, "m0")} | {g.row.game for g in mine} == set(range(24))
    with pytest.raises(InspectError, match=r"no member m9"):
        games(2, None, "m9")


def test_a_named_game_out_of_range_and_generation_zero_are_refused() -> None:
    with pytest.raises(InspectError, match=r"has no game 99"):
        games(1, [99])
    with pytest.raises(InspectError, match=r"generation 0"):
        games(0, None)


def test_a_page_labels_each_panel_and_notes_the_pair(tmp_path: Path) -> None:
    (past,) = [g for g in games(2, [6]) if g.population and g.population.kind == "past"]
    raw = build(past)
    assert problems(raw) == []
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    data = json.loads(block)
    labels = [v["label"] for v in data["views"]]
    assert labels == [
        "p1's own channel, played by learning member m0",
        "p2's own channel, played by frozen past-pool snapshot m1@g1",
    ]
    assert any("past game scheduled for member m0" in n for n in data["notes"])


def test_the_anchor_page_notes_the_scripted_baseline() -> None:
    (anchor,) = [g for g in games(2, [8]) if g.population and g.population.kind == "anchor"]
    (block,) = [
        b for attrs, b in audit(build(anchor)).scripts if attrs.get("type") == "application/json"
    ]
    notes = json.loads(block)["notes"]
    assert any("scripted baseline (actor-rule@v1)" in n for n in notes)
    assert any("scripted baseline actor-rule@v1" in v["label"] for v in json.loads(block)["views"])


def test_hostile_strings_stay_out_of_the_markup() -> None:
    g = games(2, [6])[0]
    views = []
    for view in g.views:
        lines = list(view.lines)
        lines.insert(len(lines) // 2, f"|-message|{LOG}")
        lines.insert(len(lines) // 2, f"|{NICK}|{LOG}")
        team = tuple({**s, "name": NICK, "item": LOG} for s in view.team)
        decisions = tuple(dataclasses.replace(d, policy=POLICY) for d in view.decisions)
        views.append(dataclasses.replace(view, lines=tuple(lines), team=team, decisions=decisions))
    info = dataclasses.replace(
        g.population,
        member=NICK,
        holders={"p1": ("member", NICK), "p2": ("baseline", POLICY)},
    )
    raw = build(dataclasses.replace(g, run_key=RUN, views=tuple(views), population=info))
    assert problems(raw) == []
    assert not [c for c in CANARIES if c in raw]


def test_a_page_is_a_pure_function_of_the_files_and_builds_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build(games(1, [2])[0])

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert build(games(1, [2])[0]) == first


def test_the_command_writes_pages_for_a_member_and_an_index(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    done = replay_command.replay_games(
        run,
        None,
        Selection(2, [0, 1, 12], "m1"),
        repo=ROOT,
        pool_loader=loader(pop_pool()),
    )
    out = tmp_path / "run.inspect"
    assert done["pages"] == 3
    assert Path(done["out"]) == out
    assert sorted(p.name for p in (out / "replays" / "g000002").iterdir()) == [
        "game-000000.html",
        "game-000001.html",
        "game-000012.html",
    ]
    # the inspector links the pages that exist
    inspected = command.inspect_run(
        run, out, repo=ROOT, node="node", pool_loader=loader(pop_pool())
    )
    assert inspected["pages"] == 7
    games_page = (out / "games-g000002.html").read_text()
    assert games_page.count('href="replays/g000002/') == 3
    assert (out / "replays" / "g000002" / "game-000000.html").is_file()  # kept across a rebuild


def test_member_applies_to_a_population_run_only(tmp_path: Path) -> None:
    with pytest.raises(InspectError, match=r"--member applies to a population run only"):
        replay_command.replay_games(FIXTURE, tmp_path / "o", Selection(1, [0], "m0"), repo=ROOT)


def test_the_inspector_builds_a_population_run_and_a_single_lineage_run_unchanged(
    tmp_path: Path,
) -> None:
    run = copy_pop_run(tmp_path)
    done = command.inspect_run(run, None, repo=ROOT, node="node", pool_loader=loader(pop_pool()))
    out = Path(done["out"])
    assert "population run" in (out / "index.html").read_text()
    assert not (run / "run.inspect").exists()


def test_a_replay_refuses_what_the_model_refuses(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    edit(run, 2, "generation.json", lambda g: g["games"][0].update(seed="sodium," + "0" * 32))
    with pytest.raises(InspectError, match=r"plan.json and generation.json differ"):
        read_population_games(run, 2, [0], pop_pool())
    run = copy_pop_run(tmp_path / "b")
    rename_episode(run, 2, "m0", 0, lambda e: e["channel"].insert(3, "|split|p1"))
    with pytest.raises(InspectError, match=r"split|valid Episode|no Episode"):
        read_population_games(run, 2, None, pop_pool())
    run = copy_pop_run(tmp_path / "c")
    for f in (gen_dir(run, 2) / "members" / "m0" / "episodes").glob("*.json"):
        f.write_text("not json")
    with pytest.raises(InspectError):
        read_population_games(run, 2, None, pop_pool())


def test_a_forfeited_game_has_no_replay(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    gen = gen_dir(run, 2)
    seed = layout.read_json(gen / "plan.json")[0]["seed"]
    gone: dict[str, set[str]] = {}
    for folder in (gen / "members").glob("*/episodes"):
        for path in sorted(folder.glob("*.json")):
            collection = json.loads(path.read_text())["provenance"]["collection"]
            if collection["seed"] == seed:
                gone.setdefault(folder.parent.name, set()).add(path.stem)
    # the pair's seed names both of its games; take the Episodes of game 0 only (its half 0)
    first = {
        m: {
            h
            for h in hashes
            if json.loads((gen / "members" / m / "episodes" / f"{h}.json").read_text())["player"]
            == ("p1" if m == "m0" else "p2")
        }
        for m, hashes in gone.items()
    }
    # game 0 is m0 on p1 against m1 on p2; half 1 swaps the policies, so p1 of m0 is only in game 0
    for member, hashes in first.items():
        for h in hashes:
            (gen / "members" / member / "episodes" / f"{h}.json").unlink()

    def forfeit(generation: dict[str, Any]) -> None:
        generation["games"][0].update(
            outcome="forfeit",
            winner="p2",
            forfeit={"loser": "p1", "cause": "crash", "request_seq": 1, "detail": "x"},
        )
        generation["forfeits"] = 1
        generation["forfeit_causes"] = {"crash": 1}
        for entry in generation["members"]:
            entry["episode_hashes"] = [
                h for h in entry["episode_hashes"] if h not in first.get(entry["member"], set())
            ]

    edit(run, 2, "generation.json", forfeit)
    with pytest.raises(InspectError, match=r"was forfeited: it has no replay"):
        read_population_games(run, 2, [0], pop_pool())


def test_the_cli_reads_a_population_run_for_inspect_and_replay_and_refuses_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = copy_pop_run(tmp_path)
    cases = [
        ["inspect", "--run-dir", str(run), "--resimulate", "1:0"],
        [
            "replay",
            "--run-dir",
            str(run),
            "--progress-dir",
            str(tmp_path / "p"),
            "--generation",
            "1",
        ],
    ]
    for argv in cases:
        assert loop_main.main(argv) != 0
        assert "population run is not supported by the inspector yet" in capsys.readouterr().err
    assert layout.read_json(run / "run-config.json")["population"]


@pytest.mark.skipif(
    not (ROOT / "vendor" / "pokemon-showdown" / "dist" / "sim").is_dir(),
    reason="needs the built engine checkout",
)
def test_the_real_pool_script_prints_the_committed_population_listing() -> None:
    """The Node side, run for real on the population config, gives the committed pool."""
    assert load_pool(ROOT, "node", POP_FIXTURE / layout.CONFIG) == pop_pool()
    done = subprocess.run(
        ["node", str(ROOT / SCRIPT), "--config", str(POP_FIXTURE / layout.CONFIG)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == (POP_FIXTURE / "pool.json").read_text()


def test_a_named_game_the_member_holds_no_seat_in_is_an_error() -> None:
    # game 0 is a ladder game between m0 and m1; game 4 is m0 against itself
    assert len(games(2, [0], "m1")) == 1
    with pytest.raises(InspectError, match=r"game 4: m1 holds no seat in it"):
        games(2, [4], "m1")


def test_a_replay_verifies_what_the_pages_verify(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    gen = gen_dir(run, 1)
    source = sorted((gen / "members" / "m0" / "episodes").glob("*.json"))[0]
    shutil.move(source, gen / "members" / "m1" / "episodes" / source.name)

    def move(generation: dict[str, Any]) -> None:
        a, b = generation["members"]
        a["episode_hashes"].remove(source.stem)
        b["episode_hashes"] = sorted([*b["episode_hashes"], source.stem])

    edit(run, 1, "generation.json", move)
    with pytest.raises(InspectError, match=r"not in the directory of"):
        read_population_games(run, 1, [0], pop_pool())
    run = copy_pop_run(tmp_path / "b")
    rename_episode(
        run,
        1,
        "m0",
        0,
        lambda e: e["provenance"]["collection"].update(opponent_policy_id="actor-rule@v1x"),
    )
    with pytest.raises(InspectError, match=r"names another opponent|no Episode"):
        read_population_games(run, 1, [0], pop_pool())
    run = copy_pop_run(tmp_path / "c")
    edit(run, 1, "generation.json", lambda g: g["members"].__delitem__(-1))
    with pytest.raises(InspectError, match=r"other members than the config"):
        read_population_games(run, 1, [0], pop_pool())
