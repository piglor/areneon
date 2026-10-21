"""``python -m areneon_ml.loop replay``: where it writes, what it keeps, how it refuses."""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import command, replay_command
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_command import Selection
from areneon_ml.loop import __main__ as loop_main
from areneon_ml.loop import layout

from tests.ml.inspector.support import ROOT, copy_run, fixture_pool
from tests.ml.inspector.test_command import loader


def tree(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_it_writes_pages_and_an_index_under_replays_and_nothing_else(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    before = tree(run)
    done = replay_command.replay_games(run, None, Selection(1, [0, 3]), repo=ROOT)
    out = tmp_path / "run.inspect"
    assert Path(done["out"]) == out
    assert done["pages"] == 2
    assert sorted(tree(out)) == [
        "replays/g000001/game-000000.html",
        "replays/g000001/game-000003.html",
        "replays/index.html",
    ]
    assert done["bytes"] == sum(len(v) for v in tree(out).values())
    assert tree(run) == before  # the run directory is untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run", "run.inspect"]  # no partial file


def test_a_second_call_adds_pages_and_keeps_the_earlier_ones(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)
    first = tree(out)
    done = replay_command.replay_games(run, out, Selection(2, None), repo=ROOT)
    assert done["pages"] == 5
    assert done["replays"] == 6
    after = tree(out)
    assert after["replays/g000001/game-000000.html"] == first["replays/g000001/game-000000.html"]
    index = after["replays/index.html"].decode()
    assert "g000001/game-000000.html" in index
    assert "g000002/game-000004.html" in index
    again = replay_command.replay_games(run, out, Selection(2, None), repo=ROOT)  # same bytes
    assert tree(out) == after
    assert again["pages"] == 5


def test_the_output_may_not_be_inside_the_run(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    for out in (run, run / "pages"):
        with pytest.raises(InspectError, match="inside the run directory"):
            replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)
    assert not (run / "pages").exists()


def test_a_refusal_writes_nothing(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    with pytest.raises(InspectError, match="has no game"):
        replay_command.replay_games(run, out, Selection(1, [0, 9]), repo=ROOT)
    assert not out.exists()
    with pytest.raises(InspectError, match="plays no games"):
        replay_command.replay_games(run, out, Selection(0, None), repo=ROOT)
    assert not out.exists()


def test_it_refuses_an_output_path_that_is_not_a_regular_file(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    target = out / "replays" / "g000001" / "game-000000.html"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(InspectError, match="not a regular file"):
        replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)
    target.unlink()
    target.mkdir()
    with pytest.raises(InspectError, match="not a regular file"):
        replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)


def test_a_failed_write_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)
    assert [p for p in out.rglob("*") if p.is_file()] == []


def test_it_builds_with_sockets_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    replay_command.replay_games(run, tmp_path / "open", Selection(1, [0]), repo=ROOT)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    replay_command.replay_games(run, tmp_path / "blocked", Selection(1, [0]), repo=ROOT)
    assert tree(tmp_path / "blocked") == tree(tmp_path / "open")


def test_a_game_list_is_numbers_and_ranges() -> None:
    assert replay_command.parse_games(None) is None
    assert replay_command.parse_games("3") == [3]
    assert replay_command.parse_games("5, 2-4,3") == [2, 3, 4, 5]
    for bad in ("", "a", "4-2", "1,,2", "-1", "1-"):
        with pytest.raises(InspectError, match="--game"):
            replay_command.parse_games(bad)


def test_the_engine_data_may_be_named(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    export = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc" / "engine-data.json"
    done = replay_command.replay_games(
        run, tmp_path / "pages", Selection(1, [0]), repo=tmp_path, engine_data=export
    )
    assert done["pages"] == 1
    with pytest.raises(InspectError, match="engine-data"):  # no default beside a repo without it
        replay_command.replay_games(run, tmp_path / "other", Selection(1, [0]), repo=tmp_path)


def test_an_inspector_build_keeps_the_replays_and_links_to_them(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    replay_command.replay_games(run, out, Selection(1, [1, 3]), repo=ROOT)
    kept = tree(out)
    command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    after = tree(out)
    for name, data in kept.items():
        assert after[name] == data  # the replay pages are the same bytes
    assert "games-g000001.html" in after
    games = after["games-g000001.html"].decode()
    assert games.count(">replay</a>") == 2
    assert 'href="replays/g000001/game-000001.html"' in games
    assert 'href="replays/g000001/game-000003.html"' in games
    assert "not built" in games
    other = after["games-g000002.html"].decode()
    assert ">replay</a>" not in other
    # a link resolves, and a replay page links back to the games table row
    for link in re.findall(r'href="(replays/[^"]+)"', games):
        assert (out / link).is_file()
    page = (out / "replays/g000001/game-000001.html").read_text()
    assert 'href="../../games-g000001.html#g1"' in page
    assert (out / "games-g000001.html").is_file()
    # an inspector build with no replays shows none, and a rebuild is the same bytes
    plain = tmp_path / "plain"
    command.inspect_run(run, plain, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert ">replay</a>" not in (plain / "games-g000001.html").read_text()
    assert "none (forfeit)" not in (plain / "games-g000001.html").read_text()


def test_an_inspector_build_that_fails_keeps_the_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    replay_command.replay_games(run, out, Selection(1, [0]), repo=ROOT)
    command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    kept = {k: v for k, v in tree(out).items() if k.startswith("replays/")}
    real = os.replace
    failed = {"yes": False}

    def flaky(src: Any, dst: Any) -> None:
        if (
            str(dst) == str(out) and not failed["yes"]
        ):  # the swap of the finished set over the target
            failed["yes"] = True
            raise OSError("swap failed")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="swap failed"):
        command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    monkeypatch.undo()
    assert {k: v for k, v in tree(out).items() if k.startswith("replays/")} == kept
    assert (out / "index.html").is_file()  # the old set is still whole
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pages", "run"]

    def flaky_target(src: Any, dst: Any) -> None:
        if str(dst).endswith(".old") and not failed["yes"]:  # the old set cannot be set aside
            failed["yes"] = True
            raise OSError("second swap failed")
        real(src, dst)

    failed["yes"] = False
    monkeypatch.setattr(os, "replace", flaky_target)
    with pytest.raises(OSError, match="second swap failed"):
        command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    monkeypatch.undo()
    assert {k: v for k, v in tree(out).items() if k.startswith("replays/")} == kept
    assert (out / "index.html").is_file()


def test_the_existing_replays_are_found_by_name_only(tmp_path: Path) -> None:
    base = tmp_path / "replays"
    (base / "g000002").mkdir(parents=True)
    (base / "g000002" / "game-000004.html").write_text("x")
    (base / "g000002" / "game-4.html").write_text("x")  # not our name
    (base / "g000002" / "notes.txt").write_text("x")
    (base / "other").mkdir()
    (base / "other" / "game-000001.html").write_text("x")
    (base / "g000003").symlink_to(tmp_path)
    (base / "g000002" / "game-000005.html").symlink_to(tmp_path / "nowhere")
    assert command.existing_replays(tmp_path) == {(2, 4)}
    assert command.existing_replays(tmp_path / "none") == set()
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "replays").symlink_to(tmp_path)
    assert command.existing_replays(tmp_path / "x") == set()


def test_the_cli_writes_pages_and_reports_one_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run = copy_run(tmp_path)
    code = loop_main.main(
        [
            "replay", "--run-dir", str(run), "--generation", "1", "--game", "0-1",
            "--out", str(tmp_path / "pages"),
        ]
    )  # fmt: skip
    assert code == 0
    done = json.loads(capsys.readouterr().out)
    assert done["pages"] == 2
    assert (tmp_path / "pages" / "replays" / "index.html").is_file()
    assert (
        loop_main.main(["replay", "--run-dir", str(run), "--generation", "0"])
        == loop_main.EXIT_REFUSED
    )
    assert "plays no games" in capsys.readouterr().err
    assert (
        loop_main.main(["replay", "--run-dir", str(run), "--generation", "1", "--game", "x"])
        == loop_main.EXIT_REFUSED
    )
    assert not (tmp_path / "run.inspect").exists()
    assert layout.read_json(run / layout.CONFIG)  # still there
