"""A tiny loop with ``model.options.aux_move_head`` and ``ppo.aux_moves`` (ADR-010 erratum A12.2,
#1384): the Battle report carries the move-reveal keys, an independent rerun is byte-identical and
a kill anywhere resumes to the same hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests

GENERATIONS = 3


def aux_config() -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["options"] = {"aux_move_head": True}
        raw["ppo"]["aux_moves"] = {"coef_micro": 50_000}

    return edited_config(edit)


def loop(run_dir: Path, on_stage: Any = None) -> Any:
    config = aux_config()
    return make_loop(run_dir, config, FakePlayer(config=config), on_stage)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("aux-moves")
    loop(run_dir).run(GENERATIONS)
    return run_dir


def test_the_battle_report_carries_the_move_reveal_keys_and_the_preview_does_not(
    finished: Path,
) -> None:
    for generation in range(1, GENERATIONS + 1):
        report = json.loads(
            (layout.generation_dir(finished, generation) / "train-report.json").read_bytes()
        )
        battle = report["battle"]
        assert battle["aux_moves_steps"] > 0
        assert battle["aux_moves_loss"] > 0
        assert 0 <= battle["aux_moves_top1"] <= 1
        assert battle["aux_moves_masked"] >= 0
        assert not any(k.startswith("aux_moves") for k in report["preview"])


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
