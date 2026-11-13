"""A tiny loop with ``ppo.shaping`` (ADR-010 erratum A11.1, #1365): the Battle report carries the
shaping keys, an independent rerun is byte-identical and a kill anywhere resumes to the same
hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests

GENERATIONS = 3


def shaping_config() -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["shaping"] = {"coef_micro": 1_000_000}

    return edited_config(edit)


def loop(run_dir: Path, on_stage: Any = None) -> Any:
    config = shaping_config()
    return make_loop(run_dir, config, FakePlayer(config=config), on_stage)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("shaping")
    loop(run_dir).run(GENERATIONS)
    return run_dir


def test_the_battle_report_carries_the_shaping_keys_and_the_preview_does_not(
    finished: Path,
) -> None:
    for generation in range(1, GENERATIONS + 1):
        report = json.loads(
            (layout.generation_dir(finished, generation) / "train-report.json").read_bytes()
        )
        assert report["battle"]["shaping_coef_micro"] == 1_000_000
        assert report["battle"]["shaping_mean_abs_micro"] >= 0
        assert "shaping_coef_micro" not in report["preview"]


def test_an_independent_rerun_gives_identical_generations(finished: Path, tmp_path: Path) -> None:
    loop(tmp_path).run(GENERATIONS)
    assert manifests(tmp_path) == manifests(finished)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_run_killed_anywhere_resumes_with_identical_hashes(
    finished: Path, tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        loop(tmp_path, killer(generation, stage)).run(GENERATIONS)
    loop(tmp_path).run(GENERATIONS)
    assert manifests(tmp_path) == manifests(finished)
    for g in range(GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(tmp_path, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()
