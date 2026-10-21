"""Progress-game replays: reads, checks, pages and the command (ADR-017 sections 3, 4b, 4c)."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import re
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import page
from areneon_ml.inspector import progress_replay as pr
from areneon_ml.inspector import progress_replay_command as command
from areneon_ml.inspector import progress_replay_page as page_module
from areneon_ml.inspector.command import PROGRESS_REPLAYS, check_target, write_pages
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_page import assets, read_engine_data
from areneon_ml.loop import __main__ as loop_main
from areneon_ml.loop import layout
from areneon_ml.progress.records import log_hash, read_game_record, record_hash
from areneon_ml.progress.run import ProgressRequest, progress_generation
from areneon_ml.progress.seeds import progress_game_seed

from tests.ml.inspector.test_pages import LS, PS, audit
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
    sample,
)

ROOT = Path(__file__).resolve().parents[3]
ENGINE_DATA = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc" / "engine-data.json"
NICK = "</script><img src=x onerror=alert(1)>"
LOG = "a </script> b <!-- c " + LS + " d " + PS + " e \"q\" 's' & <b>"
POLICY = "policy</script><svg onload=alert(3)>"
RUN = "run\"'</script><svg onload=alert(2)>" + LS + PS
CHOICE = "move 1 </script><b>"
CANARIES = (NICK, LOG, POLICY, RUN, CHOICE)
TAGS = {
    "html", "head", "meta", "title", "style", "body", "header", "nav", "a", "main", "h1", "p",
    "strong", "div", "noscript", "footer", "script",
}  # fmt: skip
ATTRS = {"charset", "class", "content", "href", "http-equiv", "id", "lang", "name", "type"}
GAMES = 3


def make_consistent(out: Path, request: ProgressRequest) -> None:
    """Make the stand-in's progress.json agree with the corpus game: every game a tie in 4 turns."""
    record = layout.read_json(out / "progress.json")
    cap = record["plan"]["max_turns"]
    for row in record["games"]:
        row["outcome"] = {"kind": "played", "winner": "tie", "turns": 4}
    for match in record["matches"]:
        match.update(
            wins=0,
            ties=match["games"],
            losses=0,
            turn_cap_ties=match["games"] if cap < 4 else 0,
            turns_total=4 * match["games"],
        )
    layout.write_json(out / "progress.json", record)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(run dir, progress root) of a committed run with generation 1 evaluated."""
    base = tmp_path_factory.mktemp("progress")
    run = base / "run"
    config = make_run(run, generations=2)
    root = base / "run.progress"
    player = FakeProgressPlayer(config, committed_manifest(run, 0), make_consistent)
    progress_generation(run, 1, root, player, games=GAMES, record_games=2)
    return run, root


@pytest.fixture
def root(built: tuple[Path, Path], tmp_path: Path) -> Path:
    target = tmp_path / "run.progress"
    shutil.copytree(built[1], target)
    return target


def gdir(root: Path, generation: int = 1) -> Path:
    return root / layout.generation_name(generation)


def reseal(root: Path, generation: int = 1) -> None:
    path = gdir(root, generation)
    manifest = layout.read_json(path / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(path)
    layout.write_json(path / layout.MANIFEST, manifest)


def edit_game(
    root: Path, name: str, change: Callable[[dict[str, Any]], None], *, rehash: bool = True
) -> None:
    """Change one stored game and keep progress.json's hash list and the manifest in step."""
    path = gdir(root)
    found = layout.read_json(path / "games" / name)
    before = record_hash(found)
    change(found)
    if rehash:
        found["omniscient_log_sha256"] = log_hash(found["omniscient_log"])
    layout.write_json(path / "games" / name, found)
    progress = layout.read_json(path / "progress.json")
    for match in progress["matches"]:
        match["game_record_hashes"] = sorted(
            record_hash(found) if h == before else h for h in match["game_record_hashes"]
        )
    layout.write_json(path / "progress.json", progress)
    reseal(root)


def games_of(root: Path, **kwargs: Any) -> list[pr.ProgressGame]:
    return pr.read_progress_games(root, 1, **kwargs)


def table() -> dict[str, Any]:
    return read_engine_data(ENGINE_DATA, "gen9championsvgc2026regmc")


def build(game: pr.ProgressGame, *, inspector: bool = False) -> str:
    return page_module.replay_page(game, page_module.dex_for(table(), game), inspector=inspector)


# ---- reading --------------------------------------------------------------------------------


def test_it_reads_every_stored_game_and_places_the_decisions(root: Path) -> None:
    found = games_of(root)
    assert (
        len(found) == 6 * 2
    )  # six matches here (two baselines and a previous pair, two modes), two stored games each
    first = found[0]
    assert (first.run_id, first.generation) == (dataset_config().run_key, 1)
    assert first.mode in ("argmax", "sample")
    lines = first.record["omniscient_log"]
    by_seat = {s: [p for p in first.points if p["seat"] == s] for s in ("p1", "p2")}
    turn_at = {
        int(line.split("|")[2]): i for i, line in enumerate(lines) if line.startswith("|turn|")
    }
    assert by_seat["p1"][0]["kind"] == "preview"
    assert (
        by_seat["p1"][0]["frame"]
        == lines.index(next(x for x in lines if x.startswith("|teampreview"))) + 1
    )
    assert by_seat["p1"][1]["kind"] == "move"
    assert by_seat["p1"][1]["frame"] == turn_at[1] + 1
    forced = [p for p in by_seat["p2"] if p["kind"] == "forceSwitch"]
    assert forced, "the corpus game has a forced switch of p2"
    frames = [p["frame"] for p in first.points]
    assert frames == sorted(frames)
    # the frame is the state BEFORE the replacement switch line is applied
    assert lines[forced[0]["frame"]].startswith("|switch|p2")


def test_it_selects_by_match_and_game_and_refuses_what_is_not_there(root: Path) -> None:
    only = games_of(root, match="actor-rule-sample", games=[1])
    assert [(g.match_id, g.game) for g in only] == [("actor-rule-sample", 1)]
    assert only[0].candidate_seat == "p2"
    with pytest.raises(InspectError, match="has no match nope"):
        games_of(root, match="nope")
    with pytest.raises(InspectError, match="no stored game matches"):
        games_of(root, games=[7])
    with pytest.raises(InspectError, match="no stored game matches"):
        games_of(root, match="actor-rule-sample", games=[0, 2])  # game 2 is not recorded


def test_it_refuses_a_progress_directory_that_differs_from_its_manifest(root: Path) -> None:
    path = gdir(root) / "games" / "actor-rule-sample-0.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(InspectError, match="differ from its manifest"):
        games_of(root)
    with pytest.raises(InspectError, match="progress"):
        pr.read_progress_games(root, 5)


def test_a_non_synthetic_record_is_refused_by_the_schema(root: Path) -> None:
    edit_game(root, "actor-rule-sample-0.json", lambda g: g.update(data_class="public"))
    with pytest.raises(InspectError, match=r"games/actor-rule-sample-0\.json"):
        games_of(root)


def test_a_private_lineage_run_is_refused(
    root: Path, built: tuple[Path, Path], tmp_path: Path
) -> None:
    run = tmp_path / "run"
    shutil.copytree(built[0], run)
    config = layout.read_json(run / layout.CONFIG)
    config["lineage_id"] = "private-main"
    layout.write_json(run / layout.CONFIG, config)
    with pytest.raises(InspectError, match="not a public one"):
        pr.read_progress_games(root, 1, run_dir=run)


def test_a_run_dir_ties_the_record_to_its_run(root: Path, built: tuple[Path, Path]) -> None:
    assert len(pr.read_progress_games(root, 1, run_dir=built[0])) == 12


def test_a_log_hash_that_does_not_recompute_is_refused(root: Path) -> None:
    def change(g: dict[str, Any]) -> None:
        g["omniscient_log"][5] = "|rule|Tampered"

    edit_game(root, "actor-rule-sample-0.json", change, rehash=False)
    with pytest.raises(InspectError, match="omniscient_log_sha256"):
        games_of(root)


def test_the_log_must_end_as_the_recorded_outcome_says(root: Path) -> None:
    edit_game(
        root, "actor-rule-sample-0.json", lambda g: g["omniscient_log"].__setitem__(-1, "|win|p1")
    )
    with pytest.raises(InspectError, match="ends unlike the recorded outcome"):
        games_of(root)


def test_a_game_must_be_the_one_it_is_named_for_and_listed(root: Path) -> None:
    path = gdir(root) / "games"
    shutil.copy(path / "actor-rule-sample-0.json", path / "actor-rule-sample-5.json")
    reseal(root)
    with pytest.raises(InspectError, match="not the game it is named for"):
        games_of(root)
    (path / "actor-rule-sample-5.json").unlink()
    shutil.copy(path / "actor-rule-sample-0.json", path / "unknown-match-0.json")
    reseal(root)
    with pytest.raises(InspectError, match="not a game of a match"):
        games_of(root)


def test_a_game_the_progress_record_does_not_list_is_refused(root: Path) -> None:
    path = gdir(root)
    found = layout.read_json(path / "games" / "actor-rule-sample-0.json")
    found["policies"]["p2"] = "actor-rule@v1-edited"
    found["omniscient_log_sha256"] = log_hash(found["omniscient_log"])
    layout.write_json(path / "games" / "actor-rule-sample-0.json", found)
    reseal(root)
    with pytest.raises(InspectError, match=r"not listed by progress\.json"):
        games_of(root)


def test_the_outcome_must_agree_with_the_game_row(root: Path) -> None:
    path = gdir(root)
    progress = layout.read_json(path / "progress.json")
    row = next(
        r for r in progress["games"] if r["match_id"] == "actor-rule-sample" and r["game"] == 0
    )
    row["outcome"] = {
        "kind": "forfeit",
        "loser": "candidate",
        "cause": "crash",
        "request_seq": 2,
        "turns": 4,
    }
    match = next(m for m in progress["matches"] if m["match_id"] == "actor-rule-sample")
    match.update(
        ties=match["ties"] - 1,
        losses=1,
        forfeits=[{"side": "candidate", "cause": "crash", "count": 1}],
    )
    layout.write_json(path / "progress.json", progress)
    reseal(root)
    with pytest.raises(InspectError, match="outcome differs"):
        games_of(root)


def test_another_runs_game_is_refused(root: Path) -> None:
    edit_game(root, "actor-rule-sample-0.json", lambda g: g.update(run_id="other"))
    with pytest.raises(InspectError, match="seed"):
        games_of(root)


def test_a_progress_json_that_cannot_be_read_or_breaks_a_rule_is_refused(root: Path) -> None:
    path = gdir(root) / "progress.json"
    good = path.read_bytes()
    path.write_text(" " + good.decode())
    reseal(root)
    with pytest.raises(InspectError, match=r"progress\.json cannot be read"):
        games_of(root)
    record = json.loads(good)
    record["matches"][0]["wins"] += 1
    layout.write_json(path, record)
    reseal(root)
    with pytest.raises(InspectError, match=r"^progress\.json: "):
        games_of(root)
    record = json.loads(good)
    record["generation"] = 2
    layout.write_json(path, record)
    reseal(root)
    with pytest.raises(InspectError, match="another generation"):
        games_of(root)


def test_a_game_that_has_no_row_in_progress_json_is_refused(root: Path) -> None:
    seed = progress_game_seed(dataset_config().run_key, 7)
    edit_game(root, "actor-rule-sample-0.json", lambda g: g.update(game=7, seed=seed))
    with pytest.raises(InspectError, match="no game row"):
        games_of(root)


def test_outcome_by_seat() -> None:
    row: dict[str, Any] = {
        "candidate_seat": "p2",
        "outcome": {"kind": "played", "winner": "candidate", "turns": 3},
    }
    assert pr.seat_outcome(row)["winner"] == "p2"
    row["outcome"]["winner"] = "opponent"
    assert pr.seat_outcome(row)["winner"] == "p1"
    row["outcome"] = {
        "kind": "forfeit",
        "loser": "opponent",
        "cause": "crash",
        "request_seq": 1,
        "turns": 2,
    }
    assert pr.seat_outcome(row)["loser"] == "p1"


# ---- placing decisions ------------------------------------------------------------------------


def cand(*choices: str) -> list[dict[str, Any]]:
    return [{"choice": c, "pending": False} for c in choices]


def small(decisions: list[dict[str, Any]], lines: list[str] | None = None) -> dict[str, Any]:
    return {
        "omniscient_log": lines
        or [
            "|player|p1|a|", "|player|p2|b|", "|teampreview|4", "|start",
            "|switch|p1a: A|A, L50|10/10", "|turn|1", "|move|p1a: A|X|p2a: B", "|faint|p1a: A",
            "|upkeep", "|switch|p1a: C|C, L50|10/10", "|turn|2", "|move|p1a: C|X|p2a: B",
        ],
        "decisions": decisions,
        "policies": {"p1": "one", "p2": "two"},
    }  # fmt: skip


def d(player: str, seq: int, candidates: list[str], chosen: str, **more: str) -> dict[str, Any]:
    return {
        "player": player,
        "request_seq": seq,
        "candidates": cand(*candidates),
        "chosen": chosen,
        **more,
    }


def test_a_forced_switch_is_placed_at_the_replacement_after_a_faint() -> None:
    record = small(
        [
            d("p1", 1, ["team 1234"], "team 1234"),
            d("p1", 2, ["move 1 1", "switch 3, move 1"], "move 1 1"),
            d("p1", 4, ["switch 3, pass", "switch 4, pass"], "switch 3, pass"),
            d("p1", 5, ["move 1 1"], "move 1 1"),
        ]
    )
    points = pr.decision_points(record)
    assert [(p["kind"], p["frame"]) for p in points] == [
        ("preview", 3), ("move", 6), ("forceSwitch", 9), ("move", 11)
    ]  # fmt: skip
    assert record["omniscient_log"][9].startswith("|switch|p1")


def test_a_forced_switch_without_a_faint_is_placed_at_the_end_of_its_turn() -> None:
    record = small(
        [d("p1", 1, ["move 1"], "move 1"), d("p1", 2, ["switch 2, pass"], "switch 2, pass")],
        ["|player|p1|a|", "|turn|1", "|move|p1a: A|U-turn", "|turn|2", "|move|p1a: A|X"],
    )
    assert [(p["kind"], p["frame"]) for p in pr.decision_points(record)] == [
        ("move", 2),
        ("forceSwitch", 3),
    ]


def test_a_refused_choice_and_its_retry_share_a_position() -> None:
    record = small(
        [
            d("p1", 2, ["move 1"], "move 1", refused="[Unavailable choice] x"),
            d("p1", 3, ["move 1", "move 2"], "move 2"),
        ]
    )
    points = pr.decision_points(record)
    assert [p["frame"] for p in points] == [6, 6]
    assert [p["refused"] for p in points] == ["[Unavailable choice] x", None]
    assert (
        pr.decision_points(small([d("p1", 2, ["move 1"], "move 1", refused="r")]))[0]["frame"] == 6
    )


def test_a_team_preview_decision_without_a_teampreview_line_is_frame_zero() -> None:
    record = small([d("p1", 1, ["team 12"], "team 12")], ["|player|p1|a|", "|start"])
    assert pr.decision_points(record)[0]["frame"] == 0


def test_decisions_that_do_not_fit_the_log_are_refused() -> None:
    too_many = small([d("p1", n, ["move 1"], "move 1") for n in (1, 2, 3)])
    with pytest.raises(InspectError, match="turn 3, which the log lacks"):
        pr.decision_points(too_many)
    backwards = small([d("p1", 3, ["move 1"], "move 1"), d("p1", 2, ["move 1"], "move 1")])
    with pytest.raises(InspectError, match="rising request numbers"):
        pr.decision_points(backwards)


def test_log_result_reads_the_winner_through_the_player_names() -> None:
    lines = ["|player|p1|Ann|", "|player|p2|Bo|", "|turn|1", "|turn|2", "|win|Bo"]
    assert pr.log_result(lines) == ("p2", 2)
    assert pr.log_result(["|turn|3", "|tie"]) == ("tie", 3)
    assert pr.log_result(["|turn|1"]) == (None, 1)


# ---- pages -----------------------------------------------------------------------------------


def problems(raw: str) -> list[str]:
    found = []
    a = audit(raw)
    if not set(a.tags) <= TAGS:
        found.append(f"elements the template does not have: {sorted(set(a.tags) - TAGS)}")
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
    if len(ours) != 1 or ours[0][1] != assets()[0]:
        found.append("the executable script is not exactly ours")
    data = [s for s in a.scripts if s[0].get("type") == "application/json"]
    if len(data) != 1 or data[0][0].get("id") != "replay-data":
        found.append("not exactly one data block")
    for _, body in data:
        if any(c in body for c in "<>&" + LS + PS):
            found.append("an unescaped character in a data block")
    found += [f"a canary appears verbatim in the markup: {c[:12]!r}" for c in CANARIES if c in raw]
    return found


def hostile(game: pr.ProgressGame) -> pr.ProgressGame:
    record = copy.deepcopy(game.record)
    record["omniscient_log"].insert(8, f"|-message|{LOG}")
    record["omniscient_log"].insert(9, f"|{NICK}|{LOG}")
    for seat in ("p1", "p2"):
        for s in record["teams"][seat]["team"]:
            s["name"], s["item"] = NICK, LOG
        record["policies"][seat] = POLICY
    for decision in record["decisions"]:
        decision["candidates"][0]["choice"] = CHOICE
    points = tuple({**p, "policy": POLICY, "chosen": CHOICE, "refused": LOG} for p in game.points)
    return dataclasses.replace(game, run_id=RUN, record=record, points=points, match_id="m")


def test_the_page_is_the_template_the_script_and_one_data_block(root: Path) -> None:
    game = games_of(root)[0]
    raw = build(game)
    assert problems(raw) == []
    script, style = assets()
    assert script in raw
    assert style in raw
    assert page_module.WARNING in raw
    assert "omniscient" in raw.lower()
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    data = json.loads(block)
    assert set(data) == {"title", "views", "dex", "links", "notes"}
    (view,) = data["views"]
    assert view["omniscient"] is True
    assert view["lines"] == game.record["omniscient_log"]
    assert set(view["teams"]) == {"p1", "p2"}
    assert [s["seat"] for s in view["seats"]] == ["p1", "p2"]
    assert len(view["decision_points"]) == len(game.record["decisions"])
    assert "candidate" in view["seats"][0]["label"]
    assert "opponent" in view["seats"][1]["label"]
    assert view["result"] == "tie after 4 turns"
    assert page_module.WARNING in data["notes"]
    assert 0 < len(data["dex"]["species"]) < 60


def test_the_csp_is_first_in_head_names_our_hashes_and_nothing_remote_is_in_the_page(
    root: Path,
) -> None:
    raw = build(games_of(root)[0])
    assert not re.search(r"https?:|//[a-z]", raw.replace("<!doctype", ""))
    head = raw[raw.index("<head>") + len("<head>") :].lstrip()
    assert head.startswith('<meta http-equiv="Content-Security-Policy"')
    content = re.search(r'content="([^"]+)"', raw)
    assert content is not None
    csp = dict(part.strip().split(" ", 1) for part in content[1].replace("&#x27;", "'").split(";"))
    assert csp["default-src"] == "'none'"
    assert csp["img-src"] == "'none'"
    assert csp["base-uri"] == "'none'"
    assert csp["form-action"] == "'none'"
    (ours,) = [s for attrs, s in audit(raw).scripts if attrs.get("type") != "application/json"]
    style = re.search(r"<style>(.*?)</style>", raw, re.S)
    assert style is not None
    for directive, source in (("script-src", ours), ("style-src", style[1])):
        digest = base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        assert csp[directive] == f"'sha256-{digest}'"
    assert "unsafe-inline" not in raw


def test_hostile_strings_stay_out_of_the_markup(root: Path) -> None:
    game = hostile(games_of(root)[0])
    raw = build(game)
    assert problems(raw) == []
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    data = json.loads(block)
    view = data["views"][0]
    assert f"|-message|{LOG}" in view["lines"]
    assert view["teams"]["p1"][0]["name"] == NICK
    assert view["decision_points"][0]["policy"] == POLICY
    assert view["decision_points"][0]["chosen"] == CHOICE
    assert view["decision_points"][0]["refused"] == LOG
    assert POLICY in view["seats"][0]["label"]
    assert RUN in "".join(audit(raw).text)  # the run id reaches the page only as escaped HTML text


def test_the_checker_catches_what_it_checks_for(root: Path) -> None:
    raw = build(games_of(root)[0])
    assert problems(raw) == []
    assert any("canary" in p for p in problems(raw.replace("<h1>", f"<h1>{NICK}", 1)))
    assert any("handler" in p for p in problems(raw.replace("<h1>", '<h1 onclick="x">', 1)))
    assert any(
        "not a local page" in p
        for p in problems(
            raw.replace('<a href="../index.html"', '<a href="https://example.test/"', 1)
        )
    )


def test_the_nav_links_back_to_the_inspector_only_when_it_exists(root: Path) -> None:
    game = games_of(root)[0]
    assert 'href="../../index.html"' not in build(game)
    assert 'href="../../index.html"' in build(game, inspector=True)
    assert problems(build(game, inspector=True)) == []


def test_page_names_are_ids_only() -> None:
    assert (
        page_module.page_path(1, "actor-rule-sample", 3)
        == "progress-replays/g000001/actor-rule-sample-3.html"
    )
    with pytest.raises(ValueError, match="not a link id"):
        page_module.page_name("../x", 1)


def test_the_forfeit_and_previous_wording(root: Path) -> None:
    game = games_of(root, match="previous-sample", games=[0])[0]
    assert (
        "previous generation's frozen pair"
        in page_module.payload(game, {})["views"][0]["seats"][1]["label"]
    )
    record = copy.deepcopy(game.record)
    record["outcome"] = {
        "kind": "forfeit",
        "loser": "p2",
        "cause": "crash",
        "request_seq": 4,
        "turns": 3,
    }
    text = page_module.payload(dataclasses.replace(game, record=record), {})["views"][0]["result"]
    assert text == "p2 forfeited (crash) at its request 4 after 3 turns"
    record["outcome"] = {"kind": "played", "winner": "p1", "turns": 3}
    assert (
        page_module.payload(dataclasses.replace(game, record=record), {})["views"][0]["result"]
        == "p1 won after 3 turns"
    )


# ---- the command -----------------------------------------------------------------------------


def tree(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_it_writes_pages_and_an_index_under_progress_replays_and_nothing_else(
    root: Path, tmp_path: Path
) -> None:
    before = tree(root)
    done = command.replay_progress(root, 1, None, repo=ROOT, match="actor-rule-sample")
    out = tmp_path / "run.inspect"
    assert Path(done["out"]) == out
    assert done["pages"] == 2
    assert sorted(tree(out)) == [
        "progress-replays/g000001/actor-rule-sample-0.html",
        "progress-replays/g000001/actor-rule-sample-1.html",
        "progress-replays/index.html",
    ]
    assert tree(root) == before  # the progress root is untouched
    index = (out / "progress-replays" / "index.html").read_text()
    assert "g000001/actor-rule-sample-0.html" in index
    assert page.esc(page_module.WARNING) in index
    again = command.replay_progress(root, 1, None, repo=ROOT, match="actor-rule-sample")
    assert again["pages"] == 2
    assert tree(out) == {k: v for k, v in tree(out).items()}


def test_a_second_call_adds_pages_and_keeps_the_earlier_ones(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pages"
    command.replay_progress(root, 1, out, repo=ROOT, match="actor-rule-sample", games=[0])
    first = tree(out)
    done = command.replay_progress(root, 1, out, repo=ROOT, match="random-legal-argmax")
    assert done["replays"] == 3
    after = tree(out)
    name = "progress-replays/g000001/actor-rule-sample-0.html"
    assert after[name] == first[name]
    assert command.existing_pages(out) == [
        (1, "actor-rule-sample", 0), (1, "random-legal-argmax", 0), (1, "random-legal-argmax", 1)
    ]  # fmt: skip


def test_it_links_back_to_the_inspector_when_its_index_exists(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pages"
    out.mkdir()
    (out / "index.html").write_text("<!doctype html>")
    command.replay_progress(root, 1, out, repo=ROOT, match="actor-rule-sample", games=[0])
    raw = (out / "progress-replays" / "g000001" / "actor-rule-sample-0.html").read_text()
    assert 'href="../../index.html"' in raw
    assert 'href="../index.html"' in (out / "progress-replays" / "index.html").read_text()


def test_the_output_may_not_be_inside_the_progress_directory_or_the_run(
    root: Path, built: tuple[Path, Path]
) -> None:
    for out in (root, root / "pages"):
        with pytest.raises(InspectError, match="inside the progress directory"):
            command.replay_progress(root, 1, out, repo=ROOT)
    run = built[0]
    with pytest.raises(InspectError, match="inside the run directory"):
        command.replay_progress(root, 1, run / "pages", repo=ROOT, run_dir=run)
    assert not (root / "pages").exists()


def test_a_refusal_writes_nothing(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pages"
    with pytest.raises(InspectError, match="no stored game"):
        command.replay_progress(root, 1, out, repo=ROOT, games=[9])
    assert not out.exists()


def test_existing_pages_skips_what_is_not_a_page(tmp_path: Path) -> None:
    assert command.existing_pages(tmp_path) == []
    base = tmp_path / PROGRESS_REPLAYS
    (base / "g000002").mkdir(parents=True)
    (base / "g000002" / "a-1.html").write_text("x")
    (base / "g000002" / "b-1.html").symlink_to(tmp_path)
    (base / "g000002" / "notes.txt").write_text("x")
    (base / "other").mkdir()
    (base / "other" / "c-1.html").write_text("x")
    (base / "g000003").symlink_to(tmp_path)
    (base / "stray.html").write_text("x")
    assert command.existing_pages(tmp_path) == [(2, "a", 1)]
    shutil.rmtree(base)
    base.symlink_to(tmp_path)
    assert command.existing_pages(tmp_path) == []


def test_the_default_output_is_beside_the_progress_directory(tmp_path: Path) -> None:
    assert command.default_out(tmp_path / "x.progress") == tmp_path / "x.inspect"
    assert command.default_out(tmp_path / "p") == tmp_path / "p.inspect"
    assert command.default_out(tmp_path / ".progress") == tmp_path / ".progress.inspect"


def test_it_builds_with_sockets_blocked(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    command.replay_progress(root, 1, tmp_path / "blocked", repo=ROOT, games=[0])


def test_the_inspector_keeps_the_progress_pages_it_did_not_build(
    root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pages"
    command.replay_progress(root, 1, out, repo=ROOT, games=[0])
    before = tree(out / PROGRESS_REPLAYS)
    check_target(out)
    write_pages(out, {"index.html": "<!doctype html>"})
    assert tree(out / PROGRESS_REPLAYS) == before
    assert (out / "index.html").is_file()
    (out / "stray.txt").write_text("x")
    with pytest.raises(InspectError, match="holds something other"):
        check_target(out)


def test_a_failed_inspector_write_puts_the_progress_pages_back(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pages"
    command.replay_progress(root, 1, out, repo=ROOT, games=[0])
    before = tree(out)
    real = layout.sync_dir

    def boom(path: Path) -> None:
        if path == out.parent:
            raise OSError("disk full")
        real(path)

    monkeypatch.setattr("areneon_ml.inspector.command.layout.sync_dir", boom)
    with pytest.raises(OSError, match="disk full"):
        write_pages(out, {"index.html": "<!doctype html>"})
    assert tree(out).get("progress-replays/index.html") == before["progress-replays/index.html"]


# ---- the loop's command line ----------------------------------------------------------------


def test_the_command_line_replays_progress_games(
    root: Path, built: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli"
    code = loop_main.main(
        ["replay", "--progress-dir", str(root), "--generation", "1", "--match", "actor-rule-sample",
         "--game", "0", "--out", str(out), "--run-dir", str(built[0])]
    )  # fmt: skip
    assert code == 0
    assert json.loads(capsys.readouterr().out)["pages"] == 1
    assert (out / "progress-replays" / "g000001" / "actor-rule-sample-0.html").is_file()
    assert (
        loop_main.main(
            [
                "replay",
                "--progress-dir",
                str(root),
                "--generation",
                "1",
                "--game",
                "9",
                "--out",
                str(out),
            ]
        )
        == loop_main.EXIT_REFUSED
    )


def test_the_command_line_needs_a_run_or_a_progress_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        ["replay", "--generation", "1"],
        ["replay", "--run-dir", str(tmp_path), "--generation", "1", "--match", "x"],
    ):
        with pytest.raises(SystemExit):
            loop_main.main(argv)
    assert "needs --run-dir" in capsys.readouterr().err


def test_a_game_record_is_not_an_episode_or_a_training_input() -> None:
    record = sample("progress-game-record/valid/played")
    assert record["data_class"] == "synthetic"
    assert "episode" not in record


def corpus_game(name: str) -> pr.ProgressGame:
    record = read_game_record(sample(f"progress-game-record/valid/{name}"))
    return pr.ProgressGame(
        record["run_id"], record["generation"], record["match_id"], record["game"], "sample",
        "baseline", "p1", record, pr.decision_points(record),
    )  # fmt: skip


@pytest.mark.parametrize("name", ["played", "forfeit"])
def test_a_page_builds_from_each_committed_corpus_sample(name: str) -> None:
    game = corpus_game(name)
    raw = build(game)
    assert problems(raw) == []
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    view = json.loads(block)["views"][0]
    assert len(view["decision_points"]) == len(game.record["decisions"])
    assert view["lines"] == game.record["omniscient_log"]
