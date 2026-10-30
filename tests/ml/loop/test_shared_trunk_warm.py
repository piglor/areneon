"""A Battle-only warm start with ``shared_trunk``, ``ppo.value_warmup`` and ``ppo.anchor`` (#1009):
both roles hold the cloned Battle trunk, the trunk is frozen through the warm-up while Preview's
heads train, the anchor reaches the trunk afterwards, and the run is deterministic and resumable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import load_checkpoint
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.driver import export_provenance
from areneon_ml.loop.warm_start import import_clone, record_path
from areneon_ml.model.network import check_shared_trunk

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.loop.support import Killed, killer, manifests
from tests.ml.loop.test_warm_start import TEACHER, clone_ids, clone_output, warm_loop
from tests.ml.model.support import game

GENERATIONS = 3
WARMUP = 2
BATTLE = ("battle",)
MODEL = {**tiny_config().model.as_dict(), "options": {"shared_trunk": True}}


def config_for(ids: dict[str, str]) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["options"] = {"shared_trunk": True}
        raw["warm_start"] = {"battle": ids["battle"]}
        raw["seeds"]["run_key"] = "shared-warm"
        raw["ppo"]["value_warmup"] = {"generations": WARMUP}
        raw["ppo"]["anchor"] = {"kl_micro": 100_000, "schedule": {"kind": "constant"}}

    return edited_config(edit)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"), model=MODEL)


@pytest.fixture(scope="module")
def config(clone_dir: Path) -> RunConfig:
    return config_for(clone_ids(clone_dir))


def started(root: Path, clone_dir: Path, config: RunConfig, hook: Any = None) -> Path:
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    warm_loop(run_dir, config, hook).run(GENERATIONS)
    return run_dir


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, config: RunConfig) -> Path:
    return started(tmp_path_factory.mktemp("shared-warm"), clone_dir, config)


def network(run_dir: Path, generation: int, role: str) -> Any:
    found = layout.verify_generation(layout.generation_dir(run_dir, generation), generation)
    checkpoint = found["checkpoints"][role]
    data = (run_dir / "checkpoints" / f"{checkpoint}.safetensors").read_bytes()
    return load_checkpoint(data, game(), checkpoint, role)


def state(run_dir: Path, generation: int, role: str) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in network(run_dir, generation, role).state_dict().items()}


def test_the_trunk_is_the_clones_and_equal_in_both_files_every_generation(finished: Path) -> None:
    for g in range(GENERATIONS + 1):
        pair = {role: network(finished, g, role) for role in ("preview", "battle")}
        check_shared_trunk(pair)
    zero_battle = state(finished, 0, "battle")
    zero_preview = state(finished, 0, "preview")
    for key in network(finished, 0, "battle").trunk_keys():
        assert torch.equal(zero_battle[key], zero_preview[key])
    records = layout.generation_dir(finished, 0) / "records"
    assert (records / "battle.json").read_bytes() == record_path(finished, "battle").read_bytes()
    preview = read_record(layout.read_json(records / "preview.json"))
    assert preview.training_set_hashes == ()
    assert preview.parent is None
    assert not record_path(finished, "preview").exists()


def test_the_trunk_is_frozen_through_the_warm_up_and_preview_heads_move(finished: Path) -> None:
    trunk = network(finished, 0, "battle").trunk_keys()
    first = {role: state(finished, 0, role) for role in ("preview", "battle")}
    for g in range(1, WARMUP + 1):
        for role in ("preview", "battle"):
            now = state(finished, g, role)
            assert all(torch.equal(first[role][k], now[k]) for k in trunk)
    heads = [k for k in first["preview"] if k not in set(trunk) and k.startswith("score.")]
    one = state(finished, 1, "preview")
    assert any(not torch.equal(first["preview"][k], one[k]) for k in heads)
    value = [k for k in first["battle"] if k.startswith("value.")]
    battle_one = state(finished, 1, "battle")
    assert any(not torch.equal(first["battle"][k], battle_one[k]) for k in value)
    # the first update after the warm-up moves the trunk (the anchor and PPO reach it)
    after = state(finished, WARMUP + 1, "battle")
    assert any(not torch.equal(first["battle"][k], after[k]) for k in trunk)


def test_the_reports_mark_the_warm_role_only(finished: Path) -> None:
    first = layout.read_json(layout.generation_dir(finished, 1) / "train-report.json")
    assert first["battle"]["value_warmup"] is True
    assert "value_warmup" not in first["preview"]
    last = layout.read_json(layout.generation_dir(finished, WARMUP + 1) / "train-report.json")
    assert last["battle"]["anchor_beta_micro"] == 100_000
    assert "anchor_beta_micro" not in last["preview"]


def test_an_independent_rerun_is_identical(
    finished: Path, tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    again = started(tmp_path, clone_dir, config)
    assert manifests(again) == manifests(finished)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_killed_run_resumes_with_identical_hashes(
    finished: Path, tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    config = config_for(clone_ids(clone_dir))
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(GENERATIONS)
    warm_loop(run_dir, config).run(GENERATIONS)
    assert manifests(run_dir) == manifests(finished)
    for g in range(GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()


def test_provenance_exports_and_classifies(finished: Path, tmp_path: Path) -> None:
    out = tmp_path / "export"
    export_provenance(finished, out)
    records = [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()]
    graph = ProvenanceGraph(records, json.loads((out / "sources.json").read_text()))
    last = layout.latest_complete(finished)
    assert last is not None
    final = last.manifest["checkpoints"]
    assert TEACHER in graph.ancestry(final["battle"]).sources
    assert TEACHER in graph.ancestry(final["preview"]).sources
