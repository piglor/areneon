"""A tiny loop with ``model.options.shared_trunk`` (ADR-010 erratum A4 item 5, #1009): every
generation's two checkpoints hold equal trunk tensors, an independent rerun is byte-identical and
a kill anywhere resumes to the same learner-state hashes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_ml.checkpoint.files import checkpoint_bytes
from areneon_ml.loop import layout
from areneon_ml.model.network import ROLES, check_shared_trunk

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests

GENERATIONS = 3


def shared() -> Any:
    return edited_config(lambda raw: raw["model"].update(options={"shared_trunk": True}))


def loop(run_dir: Path, on_stage: Any = None) -> Any:
    config = shared()
    return make_loop(run_dir, config, FakePlayer(config=config), on_stage)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("shared-trunk")
    loop(run_dir).run(GENERATIONS)
    return run_dir


def test_the_learner_holds_one_trunk_after_every_generation(tmp_path: Path) -> None:
    running = loop(tmp_path)
    running.run(GENERATIONS)
    learner = running._learner
    assert learner is not None
    assert learner.shared_trunk
    check_shared_trunk(learner.networks)
    for role in ROLES:
        assert checkpoint_bytes(learner.networks[role], running.game)


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
