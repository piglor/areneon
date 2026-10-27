"""The eval-dataset option of the progress evaluation (ADR-020 E-17.1, slice 4)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.progress import run as run_module
from areneon_ml.progress.eval_dataset import (
    EvalDataset,
    EvalDatasetError,
    add_arguments,
    from_values,
    validation_teams,
)
from areneon_ml.progress.looks import Look, append_look, looks_path, read_looks
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import ProgressError, max_pairs, verify_progress_dir

from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

OTHER = "7" * 64


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


@pytest.fixture
def other(tmp_path: Path) -> EvalDataset:
    """A second dataset directory: Python reads only its manifest (6 validation teams)."""
    directory = tmp_path / "other-dataset"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"splits": [{"split": "dev", "teams": 8}, {"split": "validation", "teams": 6}]})
    )
    return EvalDataset(directory, OTHER)


def fake(run: Path) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))


def play(run: Path, root: Path, player: Any, games: int = 2, **kwargs: Any) -> dict[str, Any]:
    found: dict[str, Any] = run_module.progress_generation(
        run, 1, root, player, games, 1, (), 4, **kwargs
    )
    return found


def test_the_record_names_the_evaluation_dataset_and_the_worker_is_asked_for_it(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    player = fake(run)
    play(run, tmp_path / "p", player, eval_dataset=other)
    (request,) = player.requests
    assert request.eval_dataset == other
    assert other.node_args() == (
        "--eval-dataset-dir",
        str(other.path),
        "--eval-dataset-hash",
        OTHER,
    )
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert record["teams"]["evaluation_dataset_hash"] == OTHER
    assert record["teams"]["dataset_hash"] == "5" * 64  # the run's own dataset stays named


def test_without_the_option_the_record_keeps_its_bytes(run: Path, tmp_path: Path) -> None:
    play(run, tmp_path / "p", fake(run))
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert "evaluation_dataset_hash" not in record["teams"]


def test_the_runs_own_training_dataset_is_refused_before_anything_starts(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    player = fake(run)
    own = EvalDataset(other.path, "5" * 64)
    with pytest.raises(ProgressError, match="run's own training dataset"):
        play(run, tmp_path / "p", player, eval_dataset=own)
    assert player.requests == []
    assert not (tmp_path / "p").exists()


def test_more_games_than_the_evaluation_datasets_pairs_is_refused(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    player = fake(run)
    with pytest.raises(ProgressError, match="at most 3"):
        play(run, tmp_path / "p", player, games=8, eval_dataset=other)  # 4 pairs, 3 possible
    assert player.requests == []
    assert max_pairs(dataset_config(), other) == 3
    play(run, tmp_path / "p", player, games=6, eval_dataset=other)  # 3 pairs: allowed


def test_a_record_that_does_not_name_the_asked_dataset_commits_nothing(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    def tamper(out: Path, _request: Any) -> None:
        record = layout.read_json(out / "progress.json")
        record["teams"]["evaluation_dataset_hash"] = "8" * 64
        layout.write_json(out / "progress.json", record)

    player = FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), tamper)
    with pytest.raises(ProgressError, match="not the evaluation dataset asked for"):
        play(run, tmp_path / "p", player, eval_dataset=other)
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == [".lock"]


def test_a_committed_directory_is_kept_only_for_the_same_dataset(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    root = tmp_path / "p"
    play(run, root, fake(run), eval_dataset=other)
    again = fake(run)
    assert play(run, root, again, eval_dataset=other)["skipped"] is True
    assert again.requests == []
    with pytest.raises(ProgressError, match="another dataset's validation split"):
        play(run, root, again)  # the option left out: a progress directory of its own
    with pytest.raises(ProgressError, match="not the evaluation dataset asked for"):
        play(run, root, again, eval_dataset=EvalDataset(other.path, "9" * 64))
    config = dataset_config()
    # reading a root asks nothing of the option, but a dataset that is asked for must match
    found = verify_progress_dir(root / "g000001", config, run, 1)
    assert found["teams"]["evaluation_dataset_hash"]
    with pytest.raises(ProgressError, match="not the evaluation dataset asked for"):
        verify_progress_dir(
            root / "g000001", config, run, 1, None, EvalDataset(other.path, "9" * 64)
        )


def test_a_default_directory_is_not_kept_for_an_evaluation_dataset(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    root = tmp_path / "p"
    play(run, root, fake(run))
    with pytest.raises(ProgressError, match="not the evaluation dataset asked for"):
        play(run, root, fake(run), eval_dataset=other)


def test_the_record_reader_refuses_the_runs_own_hash_as_the_evaluation_dataset(
    run: Path, tmp_path: Path, other: EvalDataset
) -> None:
    play(run, tmp_path / "p", fake(run), eval_dataset=other)
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    record["teams"]["evaluation_dataset_hash"] = record["teams"]["dataset_hash"]
    with pytest.raises(ProgressRecordError, match=r"must differ from teams\.dataset_hash"):
        read_progress_record(record, dataset_config())


def test_every_look_is_logged_by_the_command(run: Path, tmp_path: Path, other: EvalDataset) -> None:
    # The command logs through `append_look` after the evaluation (see test_looks); here the root's
    # sibling log is unchanged by the evaluation itself, and a look is appended by the command.
    root = tmp_path / "p"
    play(run, root, fake(run), eval_dataset=other)
    log = looks_path(root, run)
    append_look(log, "progress", "k", [Look(1, "actor-rule-argmax", 2)])
    assert read_looks(log).count == 1


def test_the_option_flags_go_together_and_the_hash_is_hex(tmp_path: Path) -> None:
    assert from_values(None, None) is None
    with pytest.raises(EvalDatasetError, match="go together"):
        from_values(tmp_path, None)
    with pytest.raises(EvalDatasetError, match="go together"):
        from_values(None, OTHER)
    with pytest.raises(EvalDatasetError, match="64 lower-case hex"):
        from_values(tmp_path, "ABC")
    assert from_values(tmp_path, OTHER) == EvalDataset(tmp_path.resolve(), OTHER)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--eval-dataset-dir", str(tmp_path), "--eval-dataset-hash", OTHER])
    assert from_values(args.eval_dataset_dir, args.eval_dataset_hash) is not None


def test_an_unreadable_manifest_refuses(tmp_path: Path) -> None:
    with pytest.raises(EvalDatasetError, match="cannot read the validation split"):
        validation_teams(tmp_path / "nowhere")
    shutil.copytree(Path(__file__).parent, tmp_path / "x", dirs_exist_ok=True)
    with pytest.raises(ProgressError, match="cannot read the validation split"):
        max_pairs(dataset_config(), EvalDataset(tmp_path / "nowhere", OTHER))
