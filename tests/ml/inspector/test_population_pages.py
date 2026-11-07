"""The population pages: offline, escaped, content-secured, deterministic (ADR-017 4c, ADR-021)."""

from __future__ import annotations

import dataclasses
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import population_render as pages
from areneon_ml.inspector.pool import read_pool
from areneon_ml.inspector.population_model import (
    PopulationModel,
    build_population_model,
    read_population_config,
)
from areneon_ml.inspector.population_readers import PopSeat
from areneon_ml.inspector.readers import Forfeit

from tests.ml.inspector.population_support import (
    POP_FIXTURE,
    copy_pop_run,
    pop_model,
    pop_pool,
    quarantine,
)
from tests.ml.inspector.test_pages import (
    CANARIES,
    HOSTILE_ATTR,
    HOSTILE_LABEL,
    HOSTILE_LOG,
    HOSTILE_NAME,
    HOSTILE_RUN,
    audit,
    problems,
)


def hostile() -> PopulationModel:
    """The fixture run with hostile strings in every field the population pages show."""
    raw = json.loads((POP_FIXTURE / "pool.json").read_text())
    for team in raw["teams"][:5]:
        team["labels"] = [{"label_id": HOSTILE_LABEL, "value": HOSTILE_LOG}]
        team["split"] = HOSTILE_LABEL
        for item in team["sets"]:
            item["name"] = HOSTILE_NAME
            item["moves"] = [HOSTILE_NAME, HOSTILE_LOG]
    model = dataclasses.replace(
        pop_model(), pool=read_pool(raw), run_key=HOSTILE_RUN, lineage_id=HOSTILE_LABEL
    )
    gens = []
    for gen in model.generations:
        games = []
        for index, original in enumerate(gen.games):
            game = original
            plan = game.plan
            if plan.seats["p2"].kind == "baseline":
                seats = {**plan.seats, "p2": PopSeat("baseline", HOSTILE_NAME, HOSTILE_NAME)}
                game = dataclasses.replace(game, plan=dataclasses.replace(plan, seats=seats))
            if index == 0:
                row = dataclasses.replace(
                    game.row,
                    outcome="forfeit",
                    winner=None,
                    forfeit=Forfeit("p1", HOSTILE_LOG, 3, HOSTILE_ATTR),
                )
                game = dataclasses.replace(game, row=row)
            games.append(game)
        gens.append(
            dataclasses.replace(
                gen,
                games=tuple(games),
                forfeit_causes={HOSTILE_LOG: 1},
                selection={"failed": HOSTILE_NAME, "kind": "none"},
                members=tuple(
                    dataclasses.replace(m, member=HOSTILE_LABEL, copied_from=HOSTILE_NAME)
                    for m in gen.members
                ),
                snapshots=tuple(
                    dataclasses.replace(s, policy_id=HOSTILE_NAME) for s in gen.snapshots
                ),
            )
        )
    population = {**model.population, "shares": {HOSTILE_NAME: 1}}
    return dataclasses.replace(model, generations=tuple(gens), population=population)


def test_the_fixture_run_builds_clean_pages() -> None:
    built = pages.build_population_pages(pop_model())
    assert set(built) >= {
        "index.html",
        "distributions.html",
        "games-g000001.html",
        "games-g000002.html",
        "drawn-g000001.html",
        "drawn-g000002.html",
        "teams-0001.html",
    }
    for name, raw in built.items():
        assert problems(raw) == [], name


def test_the_games_table_names_the_member_on_each_seat_the_kind_and_the_snapshot() -> None:
    raw = pages.build_population_pages(pop_model())["games-g000002.html"]
    a = audit(raw)
    text = "".join(a.text)
    for head in ("p1 played by", "p2 played by", "Kind", "Scheduled for", "Past-pool snapshot"):
        assert head in text
    rows = re.findall(r'<tr id="g(\d+)">(.*?)</tr>', raw)
    assert len(rows) == 24
    by_game = {int(n): re.sub(r"<[^>]+>", "|", body) for n, body in rows}
    assert "member m0" in by_game[0]
    assert "member m1" in by_game[0]
    assert "ladder" in by_game[0]
    past = [body for body in by_game.values() if "snapshot m" in body]
    assert len(past) == 4
    assert all(re.search(r"m\d@g\d \(member m\d, generation \d\)", body) for body in past)
    assert any("baseline actor-rule@v1" in body for body in by_game.values())
    assert 'id="filter"' in raw
    assert "not built" in raw


def test_the_replay_cell_links_only_pages_that_exist() -> None:
    model = pop_model()
    plain = pages.build_population_pages(model)["games-g000001.html"]
    linked = pages.build_population_pages(model, {(1, 0), (2, 3)})["games-g000001.html"]
    assert 'href="replays/g000001/game-000000.html"' in linked
    assert "game-000003" not in linked  # generation 2's game is not on generation 1's page
    assert "replay</a>" not in plain
    for raw in (plain, linked):
        assert problems(raw) == []


def test_the_index_shows_members_the_copy_and_the_past_pool() -> None:
    raw = pages.build_population_pages(pop_model())["index.html"]
    text = " ".join("".join(audit(raw).text).split())
    assert "Training data of population run" in text
    assert "copy of m0" in text
    assert "copy: m0 into m1; 4 pairs" in text
    assert "applied at the start of generation 2" in text
    assert "none: lower_bound" in text
    assert "m0@g0, m1@g0, m0@g1, m1@g1" in text
    assert "Generation 0 (the seeded pairs, no games)" in text
    assert "ladder self past anchor" in text


def test_every_page_has_the_csp_first_in_head_and_nothing_remote() -> None:
    for name, raw in pages.build_population_pages(pop_model()).items():
        assert not re.search(r"https?:|//[a-z]", raw.replace("<!doctype", "")), name
        head = raw[raw.index("<head>") + len("<head>") :].lstrip()
        assert head.startswith('<meta http-equiv="Content-Security-Policy"'), name
        assert "unsafe-inline" not in raw
        assert "unsafe-eval" not in raw


def test_hostile_strings_stay_text() -> None:
    built = pages.build_population_pages(hostile(), {(1, 1)})
    seen: set[str] = set()
    for name, raw in built.items():
        assert problems(raw) == [], name
        a = audit(raw)
        text = "".join(a.text)
        seen |= {c for c in CANARIES if c in text}
        for _, body in (s for s in a.scripts if s[0].get("type") == "application/json"):
            seen |= {c for c in CANARIES if c in json.loads(body)["run"]}
    assert {HOSTILE_NAME, HOSTILE_LOG, HOSTILE_RUN} <= seen
    assert HOSTILE_LABEL in "".join(audit(built["index.html"]).text)
    assert HOSTILE_ATTR not in built["games-g000001.html"]  # only inside an escaped attribute


def test_the_checker_catches_what_it_checks_for() -> None:
    """Negative control: a population page that splices a canary in unescaped fails the checks."""
    raw = pages.build_population_pages(pop_model())["games-g000001.html"]
    assert problems(raw) == []
    spliced = raw.replace("<h1>", f"<h1>{HOSTILE_NAME}", 1)
    assert any("canary" in p for p in problems(spliced))
    remote = raw.replace('<a href="distributions.html">', '<a href="https://example.test/">', 1)
    assert any("not a local page" in p for p in problems(remote))


def test_the_pages_are_deterministic_and_build_with_sockets_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = pages.build_population_pages(pop_model())

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert pages.build_population_pages(pop_model()) == first


def test_teams_results_are_listed_by_kind_and_holder() -> None:
    raw = pages.build_population_pages(pop_model())["teams-0001.html"]
    text = "".join(audit(raw).text)
    assert "Opponent kind" in text
    assert "past" in text
    assert "baseline actor-rule@v1" in text


def test_a_quarantined_member_is_shown_as_discarded_and_held(tmp_path: Path) -> None:
    run = copy_pop_run(tmp_path)
    quarantine(run, 2, "m0", held=True, cause="non_finite_loss")
    model = build_population_model(run, pop_pool(), read_population_config(run))
    raw = pages.build_population_pages(model)["index.html"]
    assert problems(raw) == []
    text = " ".join("".join(audit(raw).text).split())
    assert "quarantined (non_finite_loss), update discarded; pair held" in text
    assert "quarantined, update discarded: m0 (non_finite_loss)" in text
    assert "each slot is replaced at the start of generation 3" in text
    plain = " ".join(
        "".join(audit(pages.build_population_pages(pop_model())["index.html"]).text).split()
    )
    assert "quarantined" not in plain
