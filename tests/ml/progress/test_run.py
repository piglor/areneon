"""The progress command's run: reads, refusals, the atomic commit, the progress root (ADR-017)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import sha256_hex
from areneon_ml.loop import layout
from areneon_ml.loop.play import PlayError
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import (
    ProgressError,
    ProgressRequest,
    committed_generations,
    progress_generation,
    progress_lock,
    progress_root,
    summary_of,
    verify_progress_dir,
)
from areneon_ml.progress.seeds import progress_game_seed

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import make_loop, reseal
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A committed run of two generations; the tests only read it (each copies what it edits)."""
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def copied(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(run, target)
    return target


def player(run: Path, change: Any = None) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), change)


def test_the_progress_root_is_the_sibling_and_never_inside_the_run(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    assert progress_root(run, None) == (tmp_path / "r.progress").resolve()
    assert progress_root(run, tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()
    with pytest.raises(ProgressError, match="inside the run"):
        progress_root(run, run / "progress")
    with pytest.raises(ProgressError, match="inside the run"):
        progress_root(run, run)
    with pytest.raises(ProgressError, match="contain the run"):
        progress_root(run, tmp_path)


def test_a_generation_is_evaluated_and_committed_whole(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    fake = player(run)
    done = progress_generation(run, 1, root, fake, games=2, record_games=1)
    assert done["skipped"] is False
    assert [m["match_id"] for m in done["matches"]] == [
        "actor-rule-argmax",
        "actor-rule-sample",
        "random-legal-argmax",
        "random-legal-sample",
    ]
    target = root / "g000001"
    assert sorted(p.name for p in target.iterdir()) == ["games", "manifest.json", "progress.json"]
    assert sorted(p.name for p in (target / "games").iterdir()) == [
        f"{m}-0.json" for m in done_ids(done)
    ]
    assert sorted(p.name for p in root.iterdir()) == [".lock", "g000001"]  # no partial left
    manifest = layout.read_json(target / "manifest.json")
    assert manifest["files"] == layout.hash_tree(target)
    (request,) = fake.requests
    assert request.out_dir.name == ".g000001.partial"
    assert request.games == 2
    assert request.generation == 1
    assert request.policy_id_sample != request.policy_id_argmax
    assert request.manifest_sha256 == sha256_hex(
        (layout.generation_dir(run, 1) / layout.MANIFEST).read_bytes()
    )
    # The run directory was only read.
    assert not any(p.name.startswith("progress") for p in run.iterdir())


def done_ids(done: dict[str, Any]) -> list[str]:
    return [m["match_id"] for m in done["matches"]]


def test_a_committed_directory_is_verified_and_never_rewritten(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    before = (root / "g000001" / "manifest.json").read_bytes()
    again = player(run)
    done = progress_generation(run, 1, root, again, games=2, record_games=1)
    assert done["skipped"] is True
    assert again.requests == []
    assert (root / "g000001" / "manifest.json").read_bytes() == before
    # A changed committed file is caught when it is verified, not skipped.
    progress = root / "g000001" / "progress.json"
    progress.write_bytes(progress.read_bytes().replace(b'"turns":5', b'"turns":6', 1))
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        progress_generation(run, 1, root, again, games=2, record_games=1)


def test_a_refusal_commits_nothing_and_removes_the_partial_directory(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"

    class Failing(FakeProgressPlayer):
        def progress(self, request: ProgressRequest) -> dict[str, Any]:
            super().progress(request)
            raise PlayError("incident")

    with pytest.raises(PlayError):
        progress_generation(
            run, 1, root, Failing(dataset_config(), committed_manifest(run, 0)), 2, 1
        )
    assert sorted(p.name for p in root.iterdir()) == [".lock"]


def test_a_stale_partial_directory_is_deleted_at_start(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    (root / ".g000009.partial").mkdir(parents=True)
    (root / ".g000009.partial" / "x").write_text("half")
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    assert sorted(p.name for p in root.iterdir()) == [".lock", "g000001"]


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        (
            "another generation's manifest hash",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d.update(generation_manifest_sha256="0" * 64)
            ),
            "is not the record asked for",
        ),
        (
            "another frozen pair",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d["checkpoints"].update(preview="0" * 64)
            ),
            "frozen pair's",
        ),
        (
            "another game count",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d["plan"].update(recorded_games_per_match=0)
            ),
            "recorded_games_per_match long",
        ),
        (
            "another run",
            lambda out, r: rewrite(out / "progress.json", lambda d: d.update(run_id="other-run")),
            "another run",
        ),
        (
            "an extra game file",
            lambda out, r: (out / "games" / "stray.json").write_text("{}"),
            "other files than the record lists",
        ),
        (
            "a missing game file",
            lambda out, r: (out / "games" / "actor-rule-argmax-0.json").unlink(),
            "No such file",
        ),
        (
            "a game of another match",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d.update(match_id="random-legal-sample"),
            ),
            "is not the game it is named for",
        ),
        (
            "a game whose bytes differ from the listed hash",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d["decisions"][0].update(refused="x"),
            ),
            "differ from game_record_hashes",
        ),
        (
            "a game with a bad log hash",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d["omniscient_log"].append("|x"),
            ),
            "omniscient_log_sha256",
        ),
    ],
)
def test_the_files_the_simulator_side_wrote_are_checked_before_anything_commits(
    run: Path, tmp_path: Path, name: str, change: Any, message: str
) -> None:
    root = tmp_path / "p"
    with pytest.raises((ProgressError, ProgressRecordError, OSError), match=message):
        progress_generation(run, 1, root, player(run, change), games=2, record_games=1)
    assert sorted(p.name for p in root.iterdir()) == [".lock"], name


def rewrite(path: Path, change: Any) -> None:
    value = json.loads(path.read_text())
    change(value)
    layout.write_json(path, value)


def test_a_run_without_a_team_dataset_has_no_validation_split(tmp_path: Path) -> None:
    run_dir = tmp_path / "plain"
    make_loop(run_dir, tiny_config()).run(1)
    with pytest.raises(ProgressError, match="no team dataset"):
        progress_generation(run_dir, 1, tmp_path / "p", player(run_dir), 2, 1)


def test_a_generation_that_is_not_committed_is_refused(run: Path, tmp_path: Path) -> None:
    for generation in (3, -1):
        with pytest.raises(ProgressError, match="is not committed"):
            progress_generation(run, generation, tmp_path / "p", player(run), 2, 1)


def test_sizes_are_checked_first(run: Path, tmp_path: Path) -> None:
    for games, record in ((0, 0), (2, 3), (2, -1)):
        with pytest.raises(ProgressError, match="positive"):
            progress_generation(run, 1, tmp_path / "p", player(run), games, record)


def test_a_run_that_is_not_synthetic_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "records" / "battle.json"
    rewrite(path, lambda d: d.update(data_class="private"))
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="synthetic"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_records_that_name_another_checkpoint_than_the_manifest_are_refused(
    run: Path, tmp_path: Path
) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "records" / "preview.json"
    rewrite(path, lambda d: d.update(checkpoint_id="0" * 64))
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="names another checkpoint"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_generation_under_another_run_config_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(run_config_hash="0" * 64))
    with pytest.raises(ProgressError, match="another run config"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_manifest_without_a_frozen_pair_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(checkpoints={"battle": "a" * 64}))
    with pytest.raises(ProgressError, match="names no frozen pair"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_progress_seed_equal_to_a_training_seed_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    config = dataset_config()
    plan_file = layout.generation_dir(copy, 1) / "plan.json"
    plan = layout.read_json(plan_file)
    plan[0]["seed"] = progress_game_seed(config.run_key, 1)
    layout.write_json(plan_file, plan)
    reseal(copy, 1)
    with pytest.raises(ProgressError, match="game 1 equals a seed"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_the_team_set_must_be_the_one_the_manifest_names(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "team-set.json"
    rewrite(path, lambda d: d["teams"].append("f" * 64))
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(files=layout.hash_tree(path.parent)))
    with pytest.raises(ProgressError, match=r"team-set\.json differs"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_two_commands_never_share_a_root(tmp_path: Path) -> None:
    root = tmp_path / "p"
    with progress_lock(root), pytest.raises(ProgressError, match="holds"), progress_lock(root):
        pass
    with progress_lock(root):  # released when the holder leaves
        assert (root / ".lock").exists()


def test_a_committed_directory_of_another_generation_or_run_is_refused(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    # g000001's files presented as g000002: its manifest and record are not generation 2's.
    shutil.copytree(root / "g000001", root / "g000002")
    with pytest.raises(layout.RunDirError, match="not generation 2"):
        progress_generation(run, 2, root, player(run), games=2, record_games=1)
    # Its record names another committed generation.
    other = tmp_path / "q"
    progress_generation(run, 2, other, player(run), games=2, record_games=1)
    target = other / "g000002"
    rewrite(target / "progress.json", lambda d: d.update(generation=1))
    rewrite(target / layout.MANIFEST, lambda d: d.update(files=layout.hash_tree(target)))
    with pytest.raises(ProgressError, match="not of the committed generation"):
        verify_progress_dir(target, dataset_config(), run, 2)
    rewrite(target / layout.MANIFEST, lambda d: d.update(run_id="other"))
    with pytest.raises((ProgressError, ProgressRecordError)):
        verify_progress_dir(target, dataset_config(), run, 2)


def test_every_committed_generation_is_listed_and_summarised(run: Path, tmp_path: Path) -> None:
    assert committed_generations(run) == [0, 1, 2]
    assert committed_generations(tmp_path) == []
    root = tmp_path / "p"
    done = progress_generation(run, 0, root, player(run), games=3, record_games=0)
    first = done["matches"][0]
    assert first["interval"] == "conditional on the fixed validation plan"
    assert first["wilson_low_ppm"] <= first["score_ppm"] <= first["wilson_high_ppm"]
    record = read_progress_record(layout.read_json(root / "g000000" / "progress.json"))
    assert summary_of(record) == done["matches"]
    assert not list((root / "g000000").rglob("*-0.json"))  # --record-games 0 stores no game
