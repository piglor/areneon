"""The KL anchor through the loop (ADR-019 section 4, #968): a warm-started run (with and without
a value warm-up) whose PPO updates are anchored to the cloned pair, the train report and sidecar
keys it writes, and a kill anywhere resumes to the same hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import Killed, killer, manifests
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_config, warm_loop

RUN_GENERATIONS = 3
SCHEDULE = {"kind": "linear", "final_micro": 0, "generations": 2}


def anchor_config(ids: dict[str, str], warmup: int | None) -> RunConfig:
    base = warm_config(ids, f"tiny-anchor-{warmup}").raw

    def edit(raw: dict[str, Any]) -> None:
        raw.update(base)
        raw["ppo"]["anchor"] = {"kl_micro": 100_000, "schedule": SCHEDULE}
        if warmup is not None:
            raw["ppo"]["value_warmup"] = {"generations": warmup}

    return edited_config(edit)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


@pytest.fixture(scope="module", params=[None, 1], ids=["no-warmup", "warmup-1"])
def finished(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, clone_dir: Path
) -> tuple[Path, int | None]:
    run_dir = tmp_path_factory.mktemp("anchor")
    import_clone(clone_dir, run_dir)
    warm_loop(run_dir, anchor_config(clone_ids(clone_dir), request.param)).run(RUN_GENERATIONS)
    return run_dir, request.param


def test_the_train_report_and_sidecar_carry_the_anchor(finished: tuple[Path, int | None]) -> None:
    run_dir, warmup = finished
    skipped = warmup or 0
    # anchor update t = update - warmup; beta = 100000, 50000, 0 for t = 0, 1, 2
    rule = [100_000, 50_000, 0]
    for generation in range(1, RUN_GENERATIONS + 1):
        folder = layout.generation_dir(run_dir, generation)
        report = layout.read_json(folder / "train-report.json")
        update = generation - 1
        expected = 0 if update < skipped else rule[update - skipped]
        for role_report in report.values():
            assert role_report["anchor_beta_micro"] == expected
            assert role_report["anchor_kl"] >= 0
            if update < skipped:
                assert role_report["anchor_kl"] == 0
                assert role_report["value_warmup"] is True
        sidecar = json.loads((folder / f"learner-{generation}.json").read_bytes())
        assert sidecar["anchor_beta_micro"] == expected
    zero = json.loads((layout.generation_dir(run_dir, 0) / "learner-0.json").read_bytes())
    assert "anchor_beta_micro" not in zero


def test_an_independent_rerun_gives_identical_generations(
    finished: tuple[Path, int | None], tmp_path: Path, clone_dir: Path
) -> None:
    run_dir, warmup = finished
    again = tmp_path / "run"
    import_clone(clone_dir, again)
    warm_loop(again, anchor_config(clone_ids(clone_dir), warmup)).run(RUN_GENERATIONS)
    assert manifests(again) == manifests(run_dir)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_run_killed_anywhere_resumes_with_identical_hashes(
    finished: tuple[Path, int | None], tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    complete, warmup = finished
    config = anchor_config(clone_ids(clone_dir), warmup)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(RUN_GENERATIONS)
    warm_loop(run_dir, config).run(RUN_GENERATIONS)
    assert manifests(run_dir) == manifests(complete)
    for g in range(RUN_GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(complete, g) / name
            ).read_bytes()


def test_the_anchor_pair_must_be_in_the_store(tmp_path: Path, clone_dir: Path) -> None:
    config = anchor_config(clone_ids(clone_dir), None)
    with pytest.raises(LoopError, match="warm_start"):
        warm_loop(tmp_path / "never-imported", config).run(1)
