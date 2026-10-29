"""A run that declares ``generation.play`` is evaluated in its training mode too, without a
``--modes`` flag (ADR-023 section 1, Evaluation, #1046)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.modes import ExtraMode

from tests.ml.learner.support import edited_config
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

TAU = 500_000


def tempered(config: RunConfig) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw.update(config.raw)
        raw["generation"]["play"] = {"temperature_ppm": TAU}

    return edited_config(edit)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, RunConfig]:
    base = tmp_path_factory.mktemp("tempered-progress")
    config = tempered(dataset_config())
    make_run(base / "a", generations=2, config=config)
    return base / "a", config


def evaluate(
    run_dir: Path, config: RunConfig, root: Path, **more: Any
) -> tuple[dict[str, Any], FakeProgressPlayer]:
    fake = FakeProgressPlayer(config, committed_manifest(run_dir, 0))
    done: dict[str, Any] = run_module.progress_generation(
        run_dir, 1, root, fake, 4, 1, (), 4, **more
    )
    return done, fake


def match_ids(root: Path) -> list[str]:
    record = layout.read_json(root / "g000001" / "progress.json")
    return [m["match_id"] for m in record["matches"]]


def tree(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_the_training_mode_is_added_to_argmax_and_temperature_one(
    run: tuple[Path, RunConfig], tmp_path: Path
) -> None:
    run_dir, config = run
    _, fake = evaluate(run_dir, config, tmp_path / "p")
    (request,) = fake.requests
    assert [e.mode for e in request.extra_modes] == [ExtraMode(TAU, None)]
    assert request.extra_modes[0].policy_id == policy_id(
        request.preview,
        request.battle,
        "sample",
        temperature_ppm=TAU,
        features=features_hash(config),
    )
    assert match_ids(tmp_path / "p") == [
        f"{actor}-{mode}"
        for actor in ("actor-rule", "random-legal")
        for mode in ("argmax", "sample", "sample-t500000")
    ]


def test_naming_the_mode_again_changes_nothing(run: tuple[Path, RunConfig], tmp_path: Path) -> None:
    run_dir, config = run
    evaluate(run_dir, config, tmp_path / "x")
    evaluate(run_dir, config, tmp_path / "y", modes=(ExtraMode(TAU, None),))
    assert tree(tmp_path / "x") == tree(tmp_path / "y")
    # another extra mode is played beside it, in the order given then the training mode
    evaluate(run_dir, config, tmp_path / "z", modes=(ExtraMode(None, 30_000),))
    assert "actor-rule-sample-th30000" in match_ids(tmp_path / "z")
    assert "actor-rule-sample-t500000" in match_ids(tmp_path / "z")


def test_a_run_without_the_field_plays_only_what_was_asked(tmp_path: Path) -> None:
    config = dataset_config()
    make_run(tmp_path / "plain", generations=2, config=config)
    evaluate(tmp_path / "plain", config, tmp_path / "p")
    assert match_ids(tmp_path / "p") == [
        f"{actor}-{mode}"
        for actor in ("actor-rule", "random-legal")
        for mode in ("argmax", "sample")
    ]
