"""The value warm-up through the loop (ADR-019 section 4, #1111): a warm-started run whose first
learner updates train the value head only, and a kill anywhere in or after it resumes to the
same hashes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.warm_start import import_clone
from safetensors.torch import load as load_tensors

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import Killed, killer, manifests
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_config, warm_loop

WARM_GENERATIONS = 2
RUN_GENERATIONS = 3


def warmup_config(ids: dict[str, str]) -> RunConfig:
    base = warm_config(ids, "tiny-warmup").raw

    def edit(raw: dict[str, Any]) -> None:
        raw.update(base)
        raw["ppo"]["value_warmup"] = {"generations": WARM_GENERATIONS}

    return edited_config(edit)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory, clone_dir: Path) -> Path:
    run_dir = tmp_path_factory.mktemp("warmup")
    import_clone(clone_dir, run_dir)
    warm_loop(run_dir, warmup_config(clone_ids(clone_dir))).run(RUN_GENERATIONS)
    return run_dir


def weights(run_dir: Path, generation: int, role: str) -> dict[str, torch.Tensor]:
    path = layout.generation_dir(run_dir, generation) / f"learner-{generation}.safetensors"
    prefix = f"net.{role}."
    return {
        name[len(prefix) :]: tensor
        for name, tensor in load_tensors(path.read_bytes()).items()
        if name.startswith(prefix)
    }


def moved(run_dir: Path, generation: int, role: str) -> set[str]:
    before, after = weights(run_dir, generation - 1, role), weights(run_dir, generation, role)
    return {name for name in before if not torch.equal(before[name], after[name])}


@pytest.mark.parametrize("role", ["preview", "battle"])
def test_the_policy_is_byte_identical_through_the_warm_up_and_moves_after(
    finished: Path, role: str
) -> None:
    for generation in range(1, WARM_GENERATIONS + 1):
        changes = moved(finished, generation, role)
        assert changes
        assert all(name.startswith("value.") for name in changes)
    assert any(not n.startswith("value.") for n in moved(finished, RUN_GENERATIONS, role))


def test_the_train_report_marks_only_the_warm_up_updates(finished: Path) -> None:
    for generation in range(1, RUN_GENERATIONS + 1):
        report = layout.read_json(layout.generation_dir(finished, generation) / "train-report.json")
        assert all(
            ("value_warmup" in role_report) == (generation <= WARM_GENERATIONS)
            for role_report in report.values()
        )


@pytest.mark.parametrize(
    ("generation", "stage"), [(1, "trained"), (2, "played"), (2, "written"), (3, "played")]
)
def test_a_run_killed_in_or_after_the_warm_up_resumes_with_identical_hashes(
    finished: Path, tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    config = warmup_config(clone_ids(clone_dir))
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(RUN_GENERATIONS)
    warm_loop(run_dir, config).run(RUN_GENERATIONS)
    assert manifests(run_dir) == manifests(finished)
    for g in range(RUN_GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()
