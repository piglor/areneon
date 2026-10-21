"""The model of a run: every number recomputed from the files, and every mismatch refused."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.inspector import model as model_module
from areneon_ml.inspector.model import build_model, read_run_dir_config
from areneon_ml.inspector.pool import Pool
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout

from tests.ml.inspector.support import (
    FIXTURE,
    copy_run,
    edit,
    edit_episode,
    fixture_model,
    fixture_pool,
    reseal,
)


def build(run: Path, pool: Pool | None = None) -> Any:
    return build_model(run, pool or fixture_pool(), read_run_dir_config(run))


def test_the_numbers_equal_a_recomputation_from_the_files() -> None:
    model = fixture_model()
    pool = fixture_pool()
    index = {t.hash: i for i, t in enumerate(pool.teams)}
    draws: Counter[str] = Counter()
    results: dict[str, Counter[str]] = {}
    for number in (1, 2):
        base = FIXTURE / "generations" / f"g{number:06d}"
        plan = json.loads((base / "plan.json").read_text())
        games = json.loads((base / "generation.json").read_text())["games"]
        drawn = set()
        for entry, row in zip(plan, games, strict=True):
            for seat in ("p1", "p2"):
                team = pool.teams[entry["teams"][seat]].hash
                draws[team] += 1
                drawn.add(team)
                tally = results.setdefault(team, Counter())
                tally["games"] += 1
                tally[
                    "tie" if row["winner"] is None else ("win" if row["winner"] == seat else "loss")
                ] += 1
        generation = model.generations[number - 1]
        assert generation.drawn == tuple(sorted(drawn))
        assert len(generation.games) == len(plan)
    assert model.cumulative_drawn == draws
    for team, use in model.use.items():
        assert use.draws == draws[team]
        assert sum(t["games"] for t in use.results.values()) == draws[team]
        for result in ("win", "tie", "loss"):
            assert (
                sum(t[result] for t in use.results.values()) == results.get(team, Counter())[result]
            )
        assert len(use.games) == use.draws
    assert sum(1 for i in index if draws[i]) == len(draws)
    assert model.pool is pool or model.pool.teams == pool.teams
    assert set(model.manifest_hashes) == {0, 1, 2}


def test_the_games_know_their_seats_and_results() -> None:
    games = fixture_model().generations[0].games
    baseline = next(g for g in games if g.plan.opponent == "random-legal@v1")
    assert baseline.kind == "random-legal@v1"
    assert {baseline.played_by("p1"), baseline.played_by("p2")} == {"pair", "baseline"}
    assert all(g.played_by("p1") == "pair" for g in games if g.kind == "self-play")
    assert {g.result("p1") for g in games} <= {"win", "loss", "tie", "forfeit"}


def test_a_plan_index_outside_the_pool_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    size = len(fixture_pool().teams)

    def change(plan: Any) -> None:
        plan[0]["teams"]["p1"] = (
            size  # the seeded plan hashes to plan_hash, so repair the record too
        )

    edit(run, 1, "plan.json", change)
    record = layout.read_json(layout.generation_dir(run, 1) / "league-run.json")
    record["plan_hash"] = sha256_hex(
        jcs(layout.read_json(layout.generation_dir(run, 1) / "plan.json"))
    )
    edit(run, 1, "league-run.json", lambda _: record)
    with pytest.raises(InspectError, match="outside the pool"):
        build(run)


def test_a_plan_that_is_not_the_records_plan_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    edit(
        run,
        1,
        "plan.json",
        lambda plan: plan[0]["teams"].update(p1=(plan[0]["teams"]["p1"] + 1) % 5),
    )
    with pytest.raises(InspectError, match="plan_hash"):
        build(run)


def test_an_episode_whose_team_is_not_its_seats_team_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)

    def change(episode: dict[str, Any]) -> None:
        episode["team"]["team"][0]["item"] = "Nothing Like It"

    edit_episode(run, 1, 0, change)
    with pytest.raises(InspectError, match="own team is not the team its plan seat names"):
        build(run)


def test_an_episode_whose_result_differs_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    edit_episode(
        run,
        1,
        0,
        lambda e: e["result"].update(winner="p1" if e["result"]["winner"] is None else None),
    )
    with pytest.raises(InspectError, match=r"result differs from generation\.json"):
        build(run)
    run = copy_run(tmp_path / "again")
    edit_episode(run, 2, 3, lambda e: e["result"].update(turns=99))
    with pytest.raises(InspectError, match="result differs"):
        build(run)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.update(data_class="private"), "not a synthetic"),
        (lambda e: e.update(rights="restricted"), "not a synthetic"),
        (
            lambda e: e["provenance"]["collection"].update(run_id="other"),
            "not a league Episode of this run",
        ),
        (lambda e: e["provenance"]["collection"].update(kind="evaluation"), "not a league Episode"),
        (
            lambda e: e["provenance"]["collection"].update(opponent_team_data_class="private"),
            "not a league Episode",
        ),
        (lambda e: e.pop("result"), "not an Episode this inspector reads"),
        (lambda e: e.update(player="p3"), "not a league Episode"),
    ],
)
def test_an_episode_that_is_not_public_synthetic_league_data_is_refused(
    tmp_path: Path, change: Any, message: str
) -> None:
    run = copy_run(tmp_path)
    edit_episode(run, 1, 0, change)
    with pytest.raises(InspectError, match=message):
        build(run)


def test_episode_files_must_be_the_listed_ones_and_hash_to_their_name(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    gen = layout.generation_dir(run, 1)
    first = sorted((gen / "episodes").glob("*.json"))[0]
    first.write_bytes(first.read_bytes() + b" ")
    reseal(run, 1)
    with pytest.raises(InspectError, match="does not hash to its name"):
        build(run)
    run = copy_run(tmp_path / "second")
    gen = layout.generation_dir(run, 1)
    sorted((gen / "episodes").glob("*.json"))[0].unlink()
    reseal(run, 1)
    with pytest.raises(InspectError, match="not the episode_hashes"):
        build(run)


def test_a_changed_file_fails_its_manifest(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    plan = layout.generation_dir(run, 2) / "plan.json"
    plan.write_bytes(plan.read_bytes().replace(b"sodium", b"sodiun", 1))
    with pytest.raises(InspectError, match="differ from its manifest"):
        build(run)


def test_a_team_outside_the_pool_or_not_drawn_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    edit(run, 2, "team-set.json", lambda t: t.update(teams=sorted([*t["teams"], "f" * 64])))
    with pytest.raises(InspectError, match="not in the pool"):
        build(run)
    run = copy_run(tmp_path / "b")
    pool = fixture_pool()
    other = next(
        t.hash
        for t in pool.teams
        if t.hash not in layout.read_json(layout.generation_dir(run, 2) / "team-set.json")["teams"]
    )
    edit(run, 2, "team-set.json", lambda t: t.update(teams=sorted([*t["teams"], other])))
    with pytest.raises(InspectError, match="not the union of the teams drawn so far"):
        build(run)
    run = copy_run(tmp_path / "c")
    edit(run, 1, "generation.json", lambda g: g.update(team_hashes=g["team_hashes"][:-1]))
    with pytest.raises(InspectError, match="team_hashes are not the teams its plan draws"):
        build(run)


@pytest.mark.parametrize(
    ("file", "change", "message"),
    [
        ("generation.json", lambda g: g.update(forfeits=1), "forfeit and played counts"),
        (
            "generation.json",
            lambda g: g["games"][0].update(seed="sodium," + "0" * 32),
            "seed differs",
        ),
        (
            "generation.json",
            lambda g: g["games"][4].update(opponent="policy:other"),
            "opponent differs",
        ),
        (
            "generation.json",
            lambda g: g.update(games=g["games"][:-1]),
            "different numbers of games",
        ),
        (
            "generation.json",
            lambda g: g.update(forfeit_causes={"latency": 1}),
            "forfeit_causes are not",
        ),
        ("league-run.json", lambda r: r.update(generation=2), "does not carry the seed"),
        (
            "league-run.json",
            lambda r: r.update(episodes=r["episodes"] - 2, played=r["played"] - 1, forfeits=1),
            "forfeit and played counts",
        ),
    ],
)
def test_the_files_must_agree(tmp_path: Path, file: str, change: Any, message: str) -> None:
    run = copy_run(tmp_path)
    edit(run, 1, file, change)
    with pytest.raises(InspectError, match=message):
        build(run)


def test_a_forfeited_game_with_an_episode_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)

    def forfeit(generation: Any) -> None:
        generation["games"][0].update(
            outcome="forfeit", forfeit={"loser": "p1", "cause": "latency", "request_seq": 2}
        )
        generation["forfeits"] = 1
        generation["forfeit_causes"] = {"latency": 1}

    def record(value: Any) -> None:
        value.update(played=value["played"] - 1, forfeits=1, episodes=value["episodes"] - 2)

    edit(run, 1, "generation.json", forfeit)
    edit(run, 1, "league-run.json", record)
    with pytest.raises(InspectError, match="a forfeited game has an Episode"):
        build(run)


def test_a_missing_episode_and_a_stray_one_are_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    gen = layout.generation_dir(run, 1)
    listing = layout.read_json(gen / "generation.json")
    # Drop one seat's Episode from files and list; the game then has no Episode.
    victim = sorted((gen / "episodes").glob("*.json"))[0]
    victim.unlink()
    edit(
        run,
        1,
        "generation.json",
        lambda g: g.update(
            episode_hashes=[h for h in listing["episode_hashes"] if h != victim.stem]
        ),
    )
    with pytest.raises(InspectError, match="no Episode"):
        build(run)


def test_the_pool_must_match_the_config(tmp_path: Path) -> None:
    pool = fixture_pool()
    config = read_run_dir_config(FIXTURE)
    with pytest.raises(InspectError, match="source does not match"):
        build_model(
            FIXTURE,
            Pool("dataset", "a" * 64, pool.records_before, pool.excluded_abilities, pool.teams),
            config,
        )
    with pytest.raises(InspectError, match="excluded abilities"):
        build_model(FIXTURE, Pool("interim", None, pool.records_before, (), pool.teams), config)
    run = copy_run(tmp_path)
    raw = json.loads((run / layout.CONFIG).read_text())
    raw["team_dataset"] = {"path": "x", "dataset_hash": "b" * 64}
    layout.write_json(run / layout.CONFIG, raw)
    with pytest.raises(InspectError, match="dataset is not the one"):
        build(run)


def test_a_non_public_lineage_is_refused(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    raw = json.loads((run / layout.CONFIG).read_text())
    for lineage in ("private-main", "owner-private", "PRIVATE"):
        raw["lineage_id"] = lineage
        layout.write_json(run / layout.CONFIG, raw)
        with pytest.raises(InspectError, match="not a public one"):
            build(run)


def test_the_run_directory_must_be_whole(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    with pytest.raises(InspectError, match="no generations"):
        model_module.generation_numbers(tmp_path)
    (layout.generations_dir(run) / "stray").mkdir()
    with pytest.raises(InspectError, match="not a generation of this run"):
        build(run)
    (layout.generations_dir(run) / "stray").rmdir()
    (layout.generations_dir(run) / ".g000003.partial").mkdir()
    assert model_module.generation_numbers(run) == [0, 1, 2]  # an unfinished generation is not read
    shutil.rmtree(layout.generation_dir(run, 1))
    with pytest.raises(InspectError, match="no gap"):
        build(run)
    with pytest.raises(InspectError, match=r"run-config\.json"):
        read_run_dir_config(tmp_path)


def test_generation_zero_plays_no_games(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    edit(run, 0, "team-set.json", lambda t: t.update(teams=["a" * 64]))
    with pytest.raises(InspectError, match="no games"):
        build(run)


def test_the_files_are_canonical_json(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    plan = layout.generation_dir(run, 1) / "plan.json"
    plan.write_text(json.dumps(json.loads(plan.read_text()), indent=1))
    reseal(run, 1)
    with pytest.raises(InspectError, match="JCS"):
        build(run)
