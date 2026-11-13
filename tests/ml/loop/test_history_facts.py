"""A tiny loop with ``model.encoder.options.history_facts`` (ADR-022 erratum E-22.5.4, slice H5):
the learner trains on tokens that carry the history facts, an independent rerun is byte-identical
and a kill anywhere resumes to the same learner-state hashes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests

GENERATIONS = 3


def with_history() -> Any:
    return edited_config(
        lambda raw: raw["model"]["encoder"].update(options={"history_facts": True})
    )


def loop(run_dir: Path, on_stage: Any = None) -> Any:
    config = with_history()
    return make_loop(run_dir, config, FakePlayer(config=config), on_stage)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("history-facts")
    loop(run_dir).run(GENERATIONS)
    return run_dir


def test_the_run_trains_with_the_option_and_names_it_in_its_checkpoints(finished: Path) -> None:
    running = loop(finished)
    assert running.config.model.encoder.options
    assert "history_facts" in running.config.model.as_dict()["encoder"]["options"]
    assert (layout.generation_dir(finished, GENERATIONS) / "manifest.json").exists()


def test_an_independent_rerun_gives_identical_generations(finished: Path, tmp_path: Path) -> None:
    loop(tmp_path).run(GENERATIONS)
    assert manifests(tmp_path) == manifests(finished)


@pytest.mark.parametrize(("generation", "stage"), [(1, "trained"), (2, "played")])
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
