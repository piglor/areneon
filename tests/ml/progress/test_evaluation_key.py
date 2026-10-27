"""The evaluation-key option of the progress evaluation (#1177)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import __main__ as module
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import EXIT_REFUSED, main
from areneon_ml.progress import evaluation_key as option
from areneon_ml.progress import run as run_module
from areneon_ml.progress import seeds as seeds_module
from areneon_ml.progress.looks import Look, LookError, append_look, looks_path, read_looks
from areneon_ml.progress.records import (
    ProgressRecordError,
    read_game_record,
    read_progress_record,
)
from areneon_ml.progress.run import ProgressError, verify_progress_dir
from areneon_ml.progress.seeds import progress_game_seed

from tests.ml.model.support import FORMAT_DIR
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
    sample,
)

KEY = "paired-eval-1"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def fake(run: Path) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))


def play(run: Path, root: Path, player: Any, **kwargs: Any) -> dict[str, Any]:
    found: dict[str, Any] = run_module.progress_generation(
        run, 1, root, player, 2, 2, (), 4, **kwargs
    )
    return found


def test_the_key_replaces_the_run_key_in_the_plan_and_reaches_the_worker(
    run: Path, tmp_path: Path
) -> None:
    player = fake(run)
    play(run, tmp_path / "p", player, evaluation_key=KEY)
    (request,) = player.requests
    assert request.evaluation_key == KEY
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert record["plan"]["evaluation_key"] == KEY
    assert record["run_id"] == dataset_config().run_key  # the run's own id is unchanged
    games = sorted((tmp_path / "p" / "g000001" / "games").glob("*.json"))
    assert games
    for path in games:
        stored = read_game_record(layout.read_json(path))
        assert stored["evaluation_key"] == KEY
        assert stored["seed"] == progress_game_seed(KEY, stored["pair"])
    assert option.node_args(KEY) == ("--evaluation-key", KEY)


def test_two_runs_on_one_key_store_the_same_seed_for_every_pair(run: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    make_run(other, generations=2, config=dataset_config("other-run", (21, 22)))
    seeds = []
    for index, (directory, config) in enumerate(
        ((run, dataset_config()), (other, dataset_config("other-run", (21, 22))))
    ):
        player = FakeProgressPlayer(config, committed_manifest(directory, 0))
        play(directory, tmp_path / f"p{index}", player, evaluation_key=KEY)
        files = sorted((tmp_path / f"p{index}" / "g000001" / "games").glob("*.json"))
        seeds.append([layout.read_json(f)["seed"] for f in files])
    assert seeds[0] == seeds[1]
    # Negative control: without the key the two runs' seeds differ.
    plain = []
    for index, (directory, config) in enumerate(
        ((run, dataset_config()), (other, dataset_config("other-run", (21, 22))))
    ):
        player = FakeProgressPlayer(config, committed_manifest(directory, 0))
        play(directory, tmp_path / f"q{index}", player)
        files = sorted((tmp_path / f"q{index}" / "g000001" / "games").glob("*.json"))
        plain.append([layout.read_json(f)["seed"] for f in files])
    assert plain[0] != plain[1]


def test_without_the_option_the_record_keeps_its_bytes(run: Path, tmp_path: Path) -> None:
    play(run, tmp_path / "p", fake(run))
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert "evaluation_key" not in record["plan"]
    for path in (tmp_path / "p" / "g000001" / "games").glob("*.json"):
        assert "evaluation_key" not in layout.read_json(path)


def test_the_runs_own_key_and_a_malformed_key_are_refused_before_anything_starts(
    run: Path, tmp_path: Path
) -> None:
    player = fake(run)
    with pytest.raises(ProgressError, match="omit the option"):
        play(run, tmp_path / "p", player, evaluation_key=dataset_config().run_key)
    with pytest.raises(ProgressError, match="id token"):
        play(run, tmp_path / "p", player, evaluation_key="not a token")
    assert player.requests == []
    assert not (tmp_path / "p").exists()


def test_the_committed_plan_seed_check_uses_the_key(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, str, int]] = []
    monkeypatch.setattr(
        run_module, "check_not_training_seeds", lambda d, k, n: seen.append((d, k, n))
    )
    play(run, tmp_path / "p", fake(run), evaluation_key=KEY)
    play(run, tmp_path / "q", fake(run))
    assert [key for _, key, _ in seen] == [KEY, dataset_config().run_key]


def test_a_key_whose_seed_equals_a_committed_plan_seed_is_refused(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        seeds_module, "committed_plan_seeds", lambda _: {progress_game_seed(KEY, 0)}
    )
    player = fake(run)
    with pytest.raises(ProgressError, match="equals a seed of a committed plan"):
        play(run, tmp_path / "p", player, evaluation_key=KEY)
    assert player.requests == []
    # The run key's own seeds are not the ones in that set, so without the key the plan is allowed.
    play(run, tmp_path / "q", fake(run))


def test_a_committed_directory_is_kept_only_for_the_same_key(run: Path, tmp_path: Path) -> None:
    keyed, plain = tmp_path / "keyed", tmp_path / "plain"
    play(run, keyed, fake(run), evaluation_key=KEY)
    play(run, plain, fake(run))
    assert play(run, keyed, fake(run), evaluation_key=KEY)["skipped"] is True
    with pytest.raises(ProgressError, match="keyed by an evaluation key"):
        play(run, keyed, fake(run))
    with pytest.raises(ProgressError, match="not keyed by the evaluation key asked for"):
        play(run, keyed, fake(run), evaluation_key="another-key")
    with pytest.raises(ProgressError, match="not keyed by the evaluation key asked for"):
        play(run, plain, fake(run), evaluation_key=KEY)
    # A lenient read (report, replay) accepts either when none is asked for, and ties one asked for.
    config = dataset_config()
    assert verify_progress_dir(keyed / "g000001", config, run, 1)["plan"]["evaluation_key"] == KEY
    assert verify_progress_dir(plain / "g000001", config, run, 1)
    with pytest.raises(ProgressError, match="not keyed by the evaluation key asked for"):
        verify_progress_dir(keyed / "g000001", config, run, 1, evaluation_key="x")


def test_a_stored_game_that_does_not_name_the_records_key_is_refused(
    run: Path, tmp_path: Path
) -> None:
    keyed = tmp_path / "keyed"
    play(run, keyed, fake(run), evaluation_key=KEY)
    path = next((keyed / "g000001" / "games").glob("*.json"))
    stored = layout.read_json(path)
    del stored["evaluation_key"]
    stored["seed"] = progress_game_seed(dataset_config().run_key, stored["pair"])
    layout.write_json(path, stored)
    manifest = layout.read_json(keyed / "g000001" / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(keyed / "g000001")
    layout.write_json(keyed / "g000001" / layout.MANIFEST, manifest)
    with pytest.raises(ProgressError, match="its evaluation key is not the record's"):
        verify_progress_dir(keyed / "g000001", dataset_config(), run, 1)


def test_a_worker_whose_games_carry_another_key_commits_nothing(run: Path, tmp_path: Path) -> None:
    def tamper(out: Path, _request: Any) -> None:
        for path in (out / "games").glob("*.json"):
            stored = layout.read_json(path)
            stored["evaluation_key"] = "other-key"
            stored["seed"] = progress_game_seed("other-key", stored["pair"])
            layout.write_json(path, stored)

    player = FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), tamper)
    with pytest.raises(ProgressError, match="is not the game it is named for"):
        play(run, tmp_path / "p", player, evaluation_key=KEY)


def test_the_readers_recompute_the_seed_from_the_key_and_refuse_its_misuse() -> None:
    game = sample("progress-game-record/valid/plan-label")
    assert read_game_record(game)["seed"] == progress_game_seed(KEY.replace(KEY, "paired-key"), 0)
    wrong = {**game, "seed": progress_game_seed(game["run_id"], 0)}
    with pytest.raises(ProgressRecordError, match="progress-seed/1"):
        read_game_record(wrong)
    with pytest.raises(ProgressRecordError, match="both-seats game"):
        read_game_record({**game, "evaluation_key": game["run_id"]})
    no_pair = {k: v for k, v in game.items() if k != "pair"}
    with pytest.raises(ProgressRecordError, match="both-seats game"):
        read_game_record(no_pair)
    record = sample("progress-evaluation-record-2/valid/plan-label")
    assert read_progress_record(record)["plan"]["evaluation_key"] == "paired-key"
    bad = copy.deepcopy(record)
    bad["plan"]["evaluation_key"] = bad["run_id"]
    with pytest.raises(ProgressRecordError, match="must differ from run_id"):
        read_progress_record(bad)
    v1 = sample("progress-evaluation-record/valid/baseline-match")
    v1["plan"]["evaluation_key"] = "paired-key"
    with pytest.raises(ProgressRecordError):
        read_progress_record(v1)


def test_the_option_takes_an_id_token_that_is_not_the_run_key() -> None:
    assert option.check(None, "run") is None
    assert option.check("k-1", "run") == "k-1"
    with pytest.raises(option.EvaluationKeyError, match="id token"):
        option.check("-bad", "run")
    with pytest.raises(option.EvaluationKeyError, match="run's own key"):
        option.check("run", "run")
    parser = argparse.ArgumentParser()
    option.add_argument(parser)
    assert parser.parse_args(["--evaluation-key", "k"]).evaluation_key == "k"
    assert option.node_args(None) == ()


def test_the_look_log_names_the_key_and_refuses_a_malformed_one(tmp_path: Path) -> None:
    log = tmp_path / "x.looks.jsonl"
    append_look(log, "progress", "run", [Look(1, "actor-rule-argmax", 2, KEY)])
    append_look(log, "progress", "run", [Look(1, "actor-rule-argmax", 2)])
    lines = log.read_text().splitlines()
    assert json.loads(lines[0])["matches"][0]["evaluation_key"] == KEY
    assert "evaluation_key" not in json.loads(lines[1])["matches"][0]
    assert read_looks(log).count == 2
    with pytest.raises(LookError, match="match entry is malformed"):
        append_look(tmp_path / "y.looks.jsonl", "progress", "run", [Look(1, "m", 2, "bad key")])


def test_the_command_passes_the_key_on_and_logs_it(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))
    monkeypatch.setattr(module, "NodeProgressPlayer", lambda **_: built)
    data = FORMAT_DIR / "engine-data.json"
    common = ["progress", "--run-dir", str(run), "--engine-data", str(data)]
    common += ["--generation", "1", "--games", "2", "--record-games", "0"]
    root = tmp_path / "p"
    assert main([*common, "--evaluation-key", KEY, "--progress-dir", str(root)]) == 0
    capsys.readouterr()
    assert (
        json.loads((root / "g000001" / "progress.json").read_text())["plan"]["evaluation_key"]
        == KEY
    )
    (entry,) = [json.loads(x) for x in looks_path(root, run).read_text().splitlines()]
    assert {m["evaluation_key"] for m in entry["matches"]} == {KEY}
    # The run's own key and a malformed one are refused, and nothing is logged.
    for bad in (dataset_config().run_key, "not a token"):
        code = main([*common, "--evaluation-key", bad, "--progress-dir", str(tmp_path / "q")])
        assert code == EXIT_REFUSED
    assert not looks_path(tmp_path / "q", run).exists()
