"""``ppo.anchor.threatened`` through the loop (ADR-010 erratum A11.2, #1366): a warm-started run
whose anchor weighs Threatened decisions apart, the report and sidecar keys it writes, and a kill
anywhere resumes to the same hashes.

The loop's stand-in simulator side writes examples with no ``candidate_features``, so the Threatened
flag is a stand-in here (a deterministic function of the example's log) and the reader's
feature-set rule is lifted: the real flag is ``learner/situation.py`` (``tests/ml/learner/
test_situation.py``, held equal to the TS module), the real per-decision weighting is
``tests/ml/learner/test_anchor_threatened.py``. This test pins the plumbing: both betas in the
report and sidecar, the same bytes on a rerun and a resume."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner import config as run_config
from areneon_ml.learner import data
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import Killed, killer, manifests
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_config, warm_loop

RUN_GENERATIONS = 3
HOT_SCHEDULE = {"kind": "linear", "final_micro": 0, "generations": 2}


@pytest.fixture(scope="module", autouse=True)
def stand_in_flag() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_config, "_check_threatened_anchor", lambda raw: None)
        patch.setattr(data, "threatened_flag", lambda log, features: len(log) % 2 == 0)
        yield


def threatened_config(ids: dict[str, str], warmup: int | None) -> RunConfig:
    base = warm_config(ids, f"tiny-threatened-{warmup}").raw

    def edit(raw: dict[str, Any]) -> None:
        raw.update(base)
        raw["ppo"]["anchor"] = {
            "kl_micro": 0,
            "schedule": {"kind": "constant"},
            "threatened": {"kl_micro": 200_000, "schedule": HOT_SCHEDULE},
        }
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
    run_dir = tmp_path_factory.mktemp("threatened")
    import_clone(clone_dir, run_dir)
    warm_loop(run_dir, threatened_config(clone_ids(clone_dir), request.param)).run(RUN_GENERATIONS)
    return run_dir, request.param


def test_the_report_and_sidecar_carry_both_betas(finished: tuple[Path, int | None]) -> None:
    run_dir, warmup = finished
    skipped = warmup or 0
    # threatened beta at anchor update t = update - warmup: 200000, 100000, 0
    rule = [200_000, 100_000, 0]
    for generation in range(1, RUN_GENERATIONS + 1):
        folder = layout.generation_dir(run_dir, generation)
        report = layout.read_json(folder / "train-report.json")
        update = generation - 1
        expected = 0 if update < skipped else rule[update - skipped]
        for role, role_report in report.items():
            assert role_report["anchor_beta_micro"] == 0  # no anchor outside Threatened
            if role == "preview":
                assert "anchor_kl_threatened" not in role_report
                continue
            assert role_report["anchor_threatened_beta_micro"] == expected
            assert role_report["anchor_threatened_steps"] > 0
            assert role_report["anchor_kl_threatened"] >= 0
            if update < skipped:
                assert role_report["anchor_kl_threatened"] == 0
        sidecar = json.loads((folder / f"learner-{generation}.json").read_bytes())
        assert sidecar["anchor_threatened_beta_micro"] == expected
    zero = json.loads((layout.generation_dir(run_dir, 0) / "learner-0.json").read_bytes())
    assert "anchor_threatened_beta_micro" not in zero


def test_an_independent_rerun_gives_identical_generations(
    finished: tuple[Path, int | None], tmp_path: Path, clone_dir: Path
) -> None:
    run_dir, warmup = finished
    again = tmp_path / "run"
    import_clone(clone_dir, again)
    warm_loop(again, threatened_config(clone_ids(clone_dir), warmup)).run(RUN_GENERATIONS)
    assert manifests(again) == manifests(run_dir)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_run_killed_anywhere_resumes_with_identical_hashes(
    finished: tuple[Path, int | None], tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    complete, warmup = finished
    config = threatened_config(clone_ids(clone_dir), warmup)
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
