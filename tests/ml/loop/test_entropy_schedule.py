"""A tiny loop with a linear ``ppo.entropy_schedule`` (ADR-023 section 3, #1048): the report and
sidecar carry the weight, an independent rerun is byte-identical and a kill anywhere resumes to
the same hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests

GENERATIONS = 3


def loop(run_dir: Path, on_stage: Any = None) -> Any:
    config = entropy_config()
    return make_loop(run_dir, config, FakePlayer(config=config), on_stage)


def entropy_config() -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["entropy_schedule"] = {"kind": "linear", "final_micro": 0, "generations": 2}

    return edited_config(edit)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("entropy")
    loop(run_dir).run(GENERATIONS)
    return run_dir


def test_report_and_sidecar_carry_the_weight_of_each_update(finished: Path) -> None:
    e0 = entropy_config().raw["ppo"]["entropy_coef_micro"]
    rule = [e0, e0 - e0 // 2, 0]
    for generation in range(1, GENERATIONS + 1):
        folder = layout.generation_dir(finished, generation)
        report = json.loads((folder / "train-report.json").read_bytes())
        sidecar = json.loads((folder / f"learner-{generation}.json").read_bytes())
        assert sidecar["entropy_coef_micro"] == rule[generation - 1]
        assert all(r["entropy_coef_micro"] == rule[generation - 1] for r in report.values())
    zero = json.loads((layout.generation_dir(finished, 0) / "learner-0.json").read_bytes())
    assert "entropy_coef_micro" not in zero


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
