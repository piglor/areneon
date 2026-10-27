"""A clone made with model options warm-starts only a run with the same options (#1168)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.checkpoint.files import load_checkpoint
from areneon_ml.learner.clone import clone
from areneon_ml.learner.clone_config import read_clone_config
from areneon_ml.learner.clone_run import write_clone
from areneon_ml.loop import layout
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.warm_start import import_clone
from areneon_ml.model.encoder import encoder_version

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.learner.test_clone import VERSION, config_for, teacher_stream
from tests.ml.loop.support import FakePlayer, make_loop
from tests.ml.model.support import game

OPTION = {"candidate_matchup": True}
REFUSED = "another model section|other model options"


def _clone_with(tmp_path: Path, name: str, options: dict[str, Any] | None) -> dict[str, Any]:
    text, manifest = teacher_stream()
    raw = json.loads(json.dumps(config_for(manifest, epochs=1).raw))
    raw["seeds"]["run_key"] = f"clone-{name}"
    if options is not None:
        raw["model"]["encoder"]["options"] = options
    config = read_clone_config(raw)
    cloned = clone(config, game(), text, manifest)
    report: dict[str, Any] = write_clone(tmp_path / name, config, game(), cloned, VERSION)
    return report


def _run_config(options: dict[str, Any] | None, ids: dict[str, str]) -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"preview": ids["preview"], "battle": ids["battle"]}
        raw["seeds"]["run_key"] = "options-run"
        if options is not None:
            raw["model"]["encoder"]["options"] = options

    return edited_config(edit)


def _start(
    tmp_path: Path,
    name: str,
    clone_options: dict[str, Any] | None,
    run_options: dict[str, Any] | None,
) -> Path:
    report = _clone_with(tmp_path, name, clone_options)
    run_dir = tmp_path / f"run-{name}"
    ids = import_clone(tmp_path / name, run_dir)
    assert ids == report["checkpoints"]
    config = _run_config(run_options, ids)
    make_loop(run_dir, config, FakePlayer(config=config)).run(1)
    return run_dir


def test_a_clone_with_the_option_warm_starts_a_run_with_it(tmp_path: Path) -> None:
    run_dir = _start(tmp_path, "with", OPTION, OPTION)
    zero = layout.verify_generation(layout.generation_dir(run_dir, 0), 0)
    config = read_clone_config(
        json.loads((tmp_path / "with" / "clone-run-config.json").read_text())
    )
    assert config.model.encoder.options == (("candidate_matchup", True),)
    for role, found in zero["checkpoints"].items():
        data = (run_dir / "checkpoints" / f"{found}.safetensors").read_bytes()
        net = load_checkpoint(data, game(), found, role, configured=config.model)
        assert net.config.layout_changing
    assert encoder_version(game(), config.model.encoder) != encoder_version(
        game(), tiny_config().model.encoder
    )


def test_a_clone_without_the_option_is_refused_by_a_run_with_it(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match=REFUSED):
        _start(tmp_path, "plain", None, OPTION)


def test_a_clone_with_the_option_is_refused_by_a_run_without_it(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match=REFUSED):
        _start(tmp_path, "opt", OPTION, None)
