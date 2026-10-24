"""Engine re-simulation of a training game (ADR-017 4a, #926): accepted only when it matches."""

# ruff: noqa: E501  (page and command assertions are kept whole so they read as written)

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import command, replay, replay_page, resimulate, resimulate_command
from areneon_ml.inspector import resimulate_page as rpage
from areneon_ml.inspector.model import read_run_dir_config
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import __main__ as loop_main
from areneon_ml.loop import layout

from tests.ml.inspector.support import FIXTURE, ROOT, copy_run, edit_episode
from tests.ml.inspector.test_pages import LS, PS, audit
from tests.ml.inspector.test_replay import episode_files, forfeit_game
from tests.ml.inspector.test_replay_command import tree
from tests.ml.inspector.test_replay_pages import (
    ATTRS,
    CANARIES,
    ENGINE_DATA,
    LOG,
    NICK,
    POLICY,
    RUN,
    TAGS,
    table,
)

SIM = ROOT / "vendor" / "pokemon-showdown" / "dist" / "sim"
HAVE_ENGINE = SIM.is_dir() and shutil.which("node") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="the pinned engine is not built")


def game(run: Path = FIXTURE, index: int = 0) -> replay.GameReplay:
    return replay.read_games(run, 1, [index])[1][0]


def fake_result(g: replay.GameReplay, **change: Any) -> dict[str, Any]:
    """What the engine would print for a game that reproduces the Episodes (no engine needed)."""
    log = ["|player|p1|p1|", "|player|p2|p2|", "|teampreview", "|start"]
    log += [f"|turn|{t}" for t in range(1, g.row.turns + 1)]
    log += ["|tie"] if g.row.winner is None else [f"|win|{g.row.winner}"]
    decisions = [
        {
            "player": v.seat,
            "request_seq": d.request_seq,
            "choice": d.choice,
            **({"refused": d.refused} if d.refused else {}),
            "candidates": [{"choice": d.choice, "pending": False}],
        }
        for v in g.views
        for d in v.decisions
    ]
    result = {
        "winner": g.row.winner,
        "turns": g.row.turns,
        "omniscient_log": log,
        "channels": {v.seat: list(v.lines) for v in g.views},
        "team_hashes": {"p1": "0" * 64, "p2": "1" * 64},
        "decisions": decisions,
    }
    return {**result, **change}


def runner_of(result: dict[str, Any] | str) -> resimulate.Runner:
    text = result if isinstance(result, str) else json.dumps(result)
    return lambda request: text


def accepted(run: Path = FIXTURE, index: int = 0) -> resimulate.Resimulated:
    g = game(run, index)
    return resimulate.accept(g, json.dumps(fake_result(g)))


# ---- the request ---------------------------------------------------------------------------------


def test_the_request_is_the_seed_both_teams_and_every_recorded_choice() -> None:
    g = game()
    request = resimulate.request_for(g, "fmt", 7)
    assert set(request) == {"format_id", "seed", "max_turns", "teams", "choices"}
    assert request["seed"] == g.row.seed
    assert request["max_turns"] == 7
    assert request["teams"] == {v.seat: list(v.team) for v in g.views}
    assert len(request["choices"]) == sum(len(v.decisions) for v in g.views)
    assert {c["player"] for c in request["choices"]} == {"p1", "p2"}
    assert set(request["choices"][0]) == {"player", "request_seq", "choice"}


def test_a_refused_choice_is_in_the_request_as_it_was_submitted() -> None:
    g = game()
    view = g.views[0]
    refused = dataclasses.replace(view.decisions[0], refused="[Unavailable choice] x")
    marked = dataclasses.replace(
        g,
        views=(dataclasses.replace(view, decisions=(refused, *view.decisions[1:])), g.views[1]),
    )
    request = resimulate.request_for(marked, "f", 5)
    assert request["choices"][0] == {
        "player": "p1",
        "request_seq": refused.request_seq,
        "choice": refused.choice,
    }


# ---- acceptance ----------------------------------------------------------------------------------


def test_a_result_that_reproduces_both_channels_is_accepted() -> None:
    g = game()
    done = resimulate.accept(g, json.dumps(fake_result(g)))
    assert done.game == g
    assert done.omniscient_log[-1].startswith(("|win|", "|tie"))
    assert [d["choice"] for d in done.decisions["p1"]] == [d.choice for d in g.views[0].decisions]


def test_a_run_is_resimulated_through_the_runner_and_the_run_is_read_only() -> None:
    g = game()
    seen = []

    def run(request: dict[str, Any]) -> str:
        seen.append(request)
        return json.dumps(fake_result(g))

    done, format_id = resimulate.resimulate(FIXTURE, 1, 0, runner=run)
    assert format_id == "gen9championsvgc2026regmc"
    assert done.game == g
    assert seen[0]["max_turns"] == read_run_dir_config(FIXTURE).generation.max_turns


@pytest.mark.parametrize("seat", ["p1", "p2"])
def test_a_channel_that_differs_is_refused_with_its_position_and_no_line(seat: str) -> None:
    g = game()
    result = fake_result(g)
    channel = list(result["channels"][seat])
    channel[3] = "|SECRET-LINE-TEXT"
    result["channels"][seat] = channel
    with pytest.raises(InspectError, match=rf"{seat}: .*first difference at line 3") as caught:
        resimulate.accept(g, json.dumps(result))
    assert "SECRET" not in str(caught.value)


def test_a_channel_that_is_longer_or_shorter_is_refused() -> None:
    g = game()
    for change in (lambda c: [*c, "|extra"], lambda c: c[:-1]):
        result = fake_result(g)
        result["channels"]["p2"] = change(result["channels"]["p2"])
        with pytest.raises(InspectError, match="differs from the recorded Episode"):
            resimulate.accept(g, json.dumps(result))


def test_another_winner_or_turn_count_is_refused() -> None:
    g = game()
    other = "p1" if g.row.winner != "p1" else "p2"
    for change in ({"winner": other}, {"turns": g.row.turns + 1}):
        with pytest.raises(InspectError, match="winner or turn count differs"):
            resimulate.accept(g, json.dumps(fake_result(g, **change)))


def test_a_log_that_ends_in_another_result_is_refused() -> None:
    g = game()
    ending = "|win|p2" if g.row.winner != "p2" else "|tie"
    log = [*fake_result(g)["omniscient_log"][:-1], ending]
    with pytest.raises(InspectError, match="ends in another result"):
        resimulate.accept(g, json.dumps(fake_result(g, omniscient_log=log)))


def test_choices_the_engine_took_differently_are_refused() -> None:
    g = game()
    for key, value in (("choice", "move 9 9"), ("refused", "[Unavailable choice] x")):
        result = fake_result(g)
        result["decisions"][1][key] = value
        with pytest.raises(InspectError, match="differ from the recorded decisions"):
            resimulate.accept(g, json.dumps(result))
    result = fake_result(g)
    result["decisions"].pop()
    with pytest.raises(InspectError, match="differ from the recorded decisions"):
        resimulate.accept(g, json.dumps(result))


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[]",
        '{"winner":null}',
        "null",
    ],
)
def test_output_that_is_not_the_result_is_refused(bad: str) -> None:
    with pytest.raises(InspectError):
        resimulate.accept(game(), bad)


def test_every_field_of_the_result_is_type_checked() -> None:
    g = game()
    changes: list[dict[str, Any]] = [
        {"extra": 1},
        {"turns": "5"},
        {"turns": True},
        {"winner": "p3"},
        {"decisions": {}},
        {"decisions": [{"player": "p1"}]},
        {"omniscient_log": "text"},
        {"omniscient_log": [1]},
        {"channels": {"p1": []}},
        {"channels": {"p1": [], "p2": [], "p3": []}},
        {"channels": {"p1": "x", "p2": []}},
    ]
    for change in changes:
        result = {**fake_result(g), **change}
        with pytest.raises(InspectError):
            resimulate.accept(g, json.dumps(result))
    result = fake_result(g)
    del result["team_hashes"]
    with pytest.raises(InspectError):
        resimulate.accept(g, json.dumps(result))


# ---- tampered files and refused games --------------------------------------------------------------


def test_a_tampered_episode_channel_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    g = game(run)
    genuine = fake_result(g)  # the engine's channels, as the untouched Episodes had them
    path = episode_files(run, 1)[0]
    seed = g.row.seed

    def change(episode: dict[str, Any]) -> None:
        episode["channel"][2] = "|tampered"

    for index, file in enumerate(episode_files(run, 1)):
        if json.loads(file.read_text())["provenance"]["collection"]["seed"] == seed:
            edit_episode(run, 1, index, change)
            break
    assert path
    with pytest.raises(InspectError, match="differs from the recorded Episode"):
        resimulate.resimulate(run, 1, 0, runner=runner_of(genuine))


def test_a_tampered_choice_list_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    genuine = fake_result(game(run))
    seed = game(run).row.seed

    def change(episode: dict[str, Any]) -> None:
        episode["decisions"][-1]["choice"] = "move 4 4"

    for index, file in enumerate(episode_files(run, 1)):
        if json.loads(file.read_text())["provenance"]["collection"]["seed"] == seed:
            edit_episode(run, 1, index, change)
            break
    with pytest.raises(InspectError, match="differ from the recorded decisions"):
        resimulate.resimulate(run, 1, 0, runner=runner_of(genuine))


def test_a_forfeited_game_and_a_missing_game_are_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    forfeit_game(run, 1, 1)

    def never(request: dict[str, Any]) -> str:
        raise AssertionError("the engine must not run")

    with pytest.raises(InspectError, match="forfeited: it has no replay"):
        resimulate.resimulate(run, 1, 1, runner=never)
    with pytest.raises(InspectError, match="no game 9"):
        resimulate.resimulate(run, 1, 9, runner=never)
    with pytest.raises(InspectError, match="plays no games"):
        resimulate.resimulate(run, 0, 0, runner=never)


# ---- the node runner -----------------------------------------------------------------------------


def test_the_node_runner_reports_a_missing_node_a_failing_script_and_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(InspectError, match=r"cannot run .*FileNotFoundError"):
        resimulate.node_runner(ROOT, str(tmp_path / "no-node"))({})
    if shutil.which("node") is not None:
        with pytest.raises(InspectError, match=r"failed \(exit 1\)"):
            resimulate.node_runner(tmp_path, "node")({})

    def slow(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired("node", 1)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(InspectError, match="TimeoutExpired"):
        resimulate.node_runner(ROOT, "node")({})


@needs_engine
def test_the_engine_reproduces_every_fixture_game_from_the_episodes() -> None:
    runner = resimulate.node_runner(ROOT, "node")
    for index in range(5):
        done, _ = resimulate.resimulate(FIXTURE, 1, index, runner=runner)
        # the engine's own log holds what a player never saw: a request-free, split-free full log
        assert not any(x.startswith("|request|") for x in done.omniscient_log)
        assert not any(x.startswith("|split|") for x in done.omniscient_log)


@needs_engine
def test_the_engine_refuses_a_choice_list_that_is_not_the_battles() -> None:
    g = game()
    request = resimulate.request_for(g, "gen9championsvgc2026regmc", 4)
    run = resimulate.node_runner(ROOT, "node")
    assert json.loads(run(request))["channels"]["p1"] == list(g.views[0].lines)
    # a turn cap that ends the battle early: the recorded choices are not the ones it took
    early = {**request, "max_turns": 1}
    with pytest.raises(InspectError, match="failed"):
        run(early)
    # a choice list that leaves a request unanswered aborts the battle
    short = json.loads(json.dumps(request))
    short["choices"].pop()
    with pytest.raises(InspectError, match="failed"):
        run(short)
    # an input that is not the request is refused before anything runs
    with pytest.raises(InspectError, match="failed"):
        run({"format_id": "x"})


# ---- the command: where it writes ----------------------------------------------------------------


def test_the_command_writes_one_page_and_leaves_the_run_exactly_as_it_was(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    before = tree(run)
    done = resimulate_command.resimulate_game(
        run, None, (1, 0), repo=ROOT, runner=runner_of(fake_result(game(run)))
    )
    out = tmp_path / "run.inspect"
    assert Path(done["out"]) == out
    assert done["page"] == "resimulations/g000001/game-000000.html"
    assert sorted(tree(out)) == [done["page"]]
    assert done["bytes"] == len(tree(out)[done["page"]])
    assert tree(run) == before  # every file of the run, manifests included, byte for byte
    layout.verify_generation(layout.generation_dir(run, 1), 1)  # the manifest hashes still match


def test_a_refused_game_writes_nothing(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    g = game(run)
    before = tree(run)
    out = tmp_path / "pages"
    result = fake_result(g)
    result["channels"]["p1"] = result["channels"]["p1"][:-1]
    with pytest.raises(InspectError):
        resimulate_command.resimulate_game(run, out, (1, 0), repo=ROOT, runner=runner_of(result))
    assert not out.exists()
    assert tree(run) == before


def test_the_output_may_not_be_inside_the_run(tmp_path: Path) -> None:
    run = copy_run(tmp_path)

    def never(request: dict[str, Any]) -> str:
        raise AssertionError("the engine must not run")

    for out in (run, run / "pages"):
        with pytest.raises(InspectError, match="inside the run directory"):
            resimulate_command.resimulate_game(run, out, (1, 0), repo=ROOT, runner=never)


def test_the_target_is_a_generation_and_a_game() -> None:
    assert resimulate_command.parse_target("3:12") == (3, 12)
    for bad in ("3", "3:", ":4", "a:b", "3:4:5", "-1:2", "3,4"):
        with pytest.raises(InspectError, match="GENERATION:GAME"):
            resimulate_command.parse_target(bad)


def test_a_later_inspect_keeps_the_resimulation_pages(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "run.inspect"
    resimulate_command.resimulate_game(
        run, out, (1, 0), repo=ROOT, runner=runner_of(fake_result(game(run)))
    )
    kept = tree(out)
    command.check_target(out)  # an output that holds only inspector pages and kept directories
    command.write_pages(out, {"index.html": "<p>x</p>"})
    assert {k: v for k, v in tree(out).items() if k.startswith("resimulations/")} == kept
    assert (out / "index.html").is_file()


def test_the_cli_writes_the_page_or_refuses(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = copy_run(tmp_path)
    good = fake_result(game(run))
    monkeypatch.setattr(loop_main, "node_runner", lambda repo, node: runner_of(good))
    args = ["inspect", "--run-dir", str(run), "--out", str(tmp_path / "pages")]
    assert loop_main.main([*args, "--resimulate", "1:0"]) == 0
    done = json.loads(capsys.readouterr().out)
    assert done["page"] == "resimulations/g000001/game-000000.html"
    assert (tmp_path / "pages" / done["page"]).is_file()
    assert loop_main.main([*args, "--resimulate", "1:x"]) == loop_main.EXIT_REFUSED
    assert "GENERATION:GAME" in capsys.readouterr().err
    bad = {**good, "turns": good["turns"] + 1}
    monkeypatch.setattr(loop_main, "node_runner", lambda repo, node: runner_of(bad))
    assert loop_main.main([*args, "--resimulate", "1:0"]) == loop_main.EXIT_REFUSED
    assert "turn count differs" in capsys.readouterr().err


@needs_engine
def test_the_cli_re_runs_a_fixture_game_on_the_engine(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "pages"
    code = loop_main.main(
        ["inspect", "--run-dir", str(FIXTURE), "--out", str(out), "--resimulate", "1:2"]
    )
    assert code == 0
    page_file = out / json.loads(capsys.readouterr().out)["page"]
    assert rpage.LABEL in page_file.read_text(encoding="utf-8").lower()


# ---- the page ------------------------------------------------------------------------------------


def build(done: resimulate.Resimulated | None = None, **kwargs: Any) -> str:
    done = done or accepted()
    return rpage.resimulation_page(done, rpage.dex_for(table(), done), **kwargs)


def problems(raw: str) -> list[str]:
    found = []
    a = audit(raw)
    tags = TAGS | {"strong"}
    if not set(a.tags) <= tags:
        found.append(f"elements the template does not have: {sorted(set(a.tags) - tags)}")
    if not {name for _, name, _ in a.attrs} <= ATTRS:
        found.append("an attribute the template does not have")
    if any(name.startswith("on") or name == "style" for _, name, _ in a.attrs):
        found.append("an event handler or style attribute")
    if any(
        name == "href" and re.fullmatch(r"(\.\./)*[a-z0-9_/.#-]+", value or "") is None
        for _, name, value in a.attrs
    ):
        found.append("a href that is not a local page")
    if raw.count("</script") != a.tags.count("script") or a.tags.count("script") != 2:
        found.append("a script end or start outside the template's own two")
    if raw.count("<!--") != 0:
        found.append("a comment opener in the page")
    ours = [s for s in a.scripts if s[0].get("type") != "application/json"]
    if len(ours) != 1 or ours[0][1] != replay_page.assets("omniscient")[0]:
        found.append("the executable script is not exactly ours")
    data = [s for s in a.scripts if s[0].get("type") == "application/json"]
    if len(data) != 1 or data[0][0].get("id") != "replay-data":
        found.append("not exactly one data block")
    for _, body in data:
        if any(c in body for c in "<>&" + LS + PS):
            found.append("an unescaped character in a data block")
    found += [f"a canary appears verbatim in the markup: {c[:12]!r}" for c in CANARIES if c in raw]
    return found


def hostile() -> resimulate.Resimulated:
    done = accepted()
    log = list(done.omniscient_log)
    log.insert(4, f"|-message|{LOG}")
    log.insert(4, f"|{NICK}|{LOG}")
    views = tuple(
        dataclasses.replace(
            v,
            team=tuple({**s, "name": NICK, "item": LOG} for s in v.team),
            decisions=tuple(dataclasses.replace(d, policy=POLICY) for d in v.decisions),
        )
        for v in done.game.views
    )
    decisions = {
        seat: tuple(
            {**d, "choice": NICK, "refused": LOG, "candidates": [{"choice": LOG, "pending": True}]}
            for d in found
        )
        for seat, found in done.decisions.items()
    }
    game_ = dataclasses.replace(done.game, run_key=RUN, views=views)
    return dataclasses.replace(done, game=game_, omniscient_log=tuple(log), decisions=decisions)


def data_of(raw: str) -> dict[str, Any]:
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    return json.loads(block)  # type: ignore[no-any-return]


def test_the_page_is_labelled_omniscient_and_checked_and_has_the_template_only() -> None:
    raw = build()
    assert problems(raw) == []
    assert replay_page.assets("omniscient")[0].startswith('var AR_MODE = "omniscient";\n')
    text = "".join(audit(raw).text)
    assert "Engine re-simulation, checked against both recorded channels." in text
    assert "both sides' hidden information" in text
    assert "Do not share or commit this page" in text
    assert re.fullmatch(r"resimulations/g000001/game-000000\.html", rpage.page_path(1, 0))
    assert audit(raw).tags.count("style") == 1


def test_the_data_block_holds_the_engines_log_both_teams_and_no_verdict() -> None:
    done = accepted()
    data = data_of(build(done))
    assert set(data) == {"title", "views", "dex", "links", "notes"}
    (view,) = data["views"]
    assert view["lines"] == list(done.omniscient_log)
    assert set(view["teams"]) == {"p1", "p2"}
    assert [s["seat"] for s in view["seats"]] == ["p1", "p2"]
    assert rpage.LABEL in view["label"]
    assert view["decision_points"]
    for point in view["decision_points"]:
        assert set(point) == {
            "seat",
            "kind",
            "frame",
            "request_seq",
            "chosen",
            "refused",
            "policy",
            "candidates",
        }
    assert "verdict" not in json.dumps(data).lower()
    assert 0 < len(data["dex"]["species"]) < 60


def test_the_nav_links_back_only_when_the_inspector_index_exists() -> None:
    assert re.findall(r'href="([^"]+)"', build()) == []
    assert re.findall(r'href="([^"]+)"', build(inspector=True)) == ["../../index.html"]


def test_hostile_strings_stay_out_of_the_markup() -> None:
    raw = build(hostile())
    assert problems(raw) == []
    data = data_of(raw)
    (view,) = data["views"]
    assert f"|-message|{LOG}" in view["lines"]
    assert view["teams"]["p1"][0]["name"] == NICK
    assert any(p["chosen"] == NICK and p["refused"] == LOG for p in view["decision_points"])
    assert RUN in "".join(audit(raw).text)  # the run key reaches the page only as escaped HTML text


def test_the_checker_catches_what_it_checks_for() -> None:
    raw = build()
    assert any("canary" in p for p in problems(raw.replace("<h1>", f"<h1>{NICK}", 1)))
    assert any("handler" in p for p in problems(raw.replace("<h1>", '<h1 onclick="x">', 1)))
    assert any(
        "not exactly ours" in p
        for p in problems(raw.replace("</script>\n</body>", "alert(1)</script>\n</body>"))
    )
    assert any("comment" in p for p in problems(raw.replace("<h1>", "<h1><!-- x -->", 1)))


def test_the_page_builds_with_sockets_blocked_and_is_a_pure_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    replay_page.assets.cache_clear()
    assert build() == expected
    assert not re.search(r"https?:|//[a-z]", expected.replace("<!doctype", ""))
    assert build(accepted(FIXTURE, 1)) != expected


def test_a_tie_is_worded_as_a_tie() -> None:
    done = accepted()
    tied = dataclasses.replace(
        done,
        game=dataclasses.replace(done.game, row=dataclasses.replace(done.game.row, winner=None)),
    )
    assert rpage._result(tied).startswith("tie after")
    assert rpage._result(done).startswith(("p1 won", "p2 won", "tie"))


def test_the_engine_data_table_is_the_committed_export() -> None:
    assert ENGINE_DATA.is_file()
