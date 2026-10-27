"""A Battle-only warm start (#1165, ADR-019 section 4): ``warm_start`` naming the Battle checkpoint
alone, Team Preview starting from the seeded init exactly as a scratch run, and ``ppo.value_warmup``
and ``ppo.anchor`` acting on the Battle role only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.learner.trainer import Learner, LearnerError
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import main
from areneon_ml.loop.driver import LoopError, export_provenance, sources_registry
from areneon_ml.loop.warm_start import WarmStartError, import_clone, record_path
from areneon_ml.model.network import init_network

from tests.ml.learner.support import edited_config, networks, tiny_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests
from tests.ml.loop.test_warm_start import TEACHER, clone_ids, clone_output, warm_loop
from tests.ml.model.support import game

GENERATIONS = 3
BATTLE = ("battle",)


def battle_config(
    ids: dict[str, str], *, warmup: int | None = None, anchor: bool = False, key: str = "bo"
) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"battle": ids["battle"]}
        raw["seeds"]["run_key"] = key
        if warmup is not None:
            raw["ppo"]["value_warmup"] = {"generations": warmup}
        if anchor:
            raw["ppo"]["anchor"] = {"kl_micro": 100_000, "schedule": {"kind": "constant"}}

    return edited_config(edit)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


@pytest.fixture(scope="module")
def ids(clone_dir: Path) -> dict[str, str]:
    return clone_ids(clone_dir)


def started(root: Path, clone_dir: Path, config: RunConfig, generations: int) -> Path:
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    warm_loop(run_dir, config).run(generations)
    return run_dir


@pytest.fixture(scope="module")
def full(
    tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, ids: dict[str, str]
) -> tuple[Path, RunConfig]:
    config = battle_config(ids, warmup=1, anchor=True)
    return started(tmp_path_factory.mktemp("full"), clone_dir, config, GENERATIONS), config


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, ids: dict[str, str]) -> Path:
    """The same Battle-only start with neither the warm-up nor the anchor."""
    return started(tmp_path_factory.mktemp("plain"), clone_dir, battle_config(ids), 1)


# -- the reader ----------------------------------------------------------------------------------


def test_the_readers_take_a_battle_only_warm_start_and_its_companions(ids: dict[str, str]) -> None:
    config = battle_config(ids, warmup=2, anchor=True)
    assert config.warm_start == {"battle": ids["battle"]}
    assert config.ppo.value_warmup == 2
    assert config.ppo.anchor is not None
    assert config.hash == sha256_hex(jcs(config.raw))
    assert "preview" not in config.raw["warm_start"]
    both = {"preview": "a" * 64, "battle": "b" * 64}
    kept = edited_config(lambda raw: raw.update(warm_start=both))
    assert kept.warm_start == both
    assert kept.raw["warm_start"] == both  # a both-role record is read and hashed as it was
    raw = json.loads(json.dumps(tiny_config().raw))
    raw["warm_start"] = {"preview": "a" * 64}
    with pytest.raises(Exception, match="battle"):
        read_run_config(raw)


# -- generation 0 --------------------------------------------------------------------------------


def test_generation_zero_is_the_seeded_preview_and_the_cloned_battle(
    full: tuple[Path, RunConfig], ids: dict[str, str], tmp_path: Path
) -> None:
    run_dir, _config = full
    scratch = tmp_path / "scratch"
    make_loop(scratch, tiny_config(), FakePlayer(config=tiny_config())).run(0)
    mine = layout.verify_generation(layout.generation_dir(run_dir, 0), 0)["checkpoints"]
    seeded = layout.verify_generation(layout.generation_dir(scratch, 0), 0)["checkpoints"]
    assert mine["preview"] == seeded["preview"]
    assert mine["battle"] == ids["battle"]
    zero = layout.generation_dir(run_dir, 0) / "records"
    battle = read_record(layout.read_json(zero / "battle.json"))
    assert (zero / "battle.json").read_bytes() == record_path(run_dir, "battle").read_bytes()
    assert battle.training_set_hashes == (TEACHER,)
    preview = read_record(layout.read_json(zero / "preview.json"))
    assert preview.training_set_hashes == ()
    assert preview.parent is None
    assert not record_path(run_dir, "preview").exists()


def test_generation_one_descends_from_the_seeded_init_and_the_clone(
    full: tuple[Path, RunConfig],
) -> None:
    run_dir, _ = full
    for role in ("preview", "battle"):
        parent = read_record(
            layout.read_json(layout.generation_dir(run_dir, 0) / "records" / f"{role}.json")
        )
        child = read_record(
            layout.read_json(layout.generation_dir(run_dir, 1) / "records" / f"{role}.json")
        )
        assert child.parent == parent.checkpoint_id


# -- value warm-up and anchor act on the Battle role only ---------------------------------------


def test_preview_trains_as_plain_ppo_and_battle_is_warmed_and_anchored(
    full: tuple[Path, RunConfig], plain: Path
) -> None:
    run_dir, _ = full
    first = layout.read_json(layout.generation_dir(run_dir, 1) / "train-report.json")
    assert {"value_warmup", "anchor_kl", "anchor_beta_micro"}.isdisjoint(first["preview"])
    assert first["battle"]["value_warmup"] is True
    assert first["battle"]["anchor_kl"] == 0
    assert first["battle"]["anchor_beta_micro"] == 0
    # after the warm-up the Battle update is anchored and moves the policy
    second = layout.read_json(layout.generation_dir(run_dir, 2) / "train-report.json")
    assert {"value_warmup", "anchor_kl", "anchor_beta_micro"}.isdisjoint(second["preview"])
    assert second["battle"]["anchor_beta_micro"] == 100_000
    assert "value_warmup" not in second["battle"]
    # Preview's first update is the update of a run with neither feature
    base = layout.read_json(layout.generation_dir(plain, 1) / "train-report.json")
    assert first["preview"] == base["preview"]
    assert (
        layout.verify_generation(layout.generation_dir(run_dir, 1), 1)["checkpoints"]["preview"]
        == layout.verify_generation(layout.generation_dir(plain, 1), 1)["checkpoints"]["preview"]
    )


# -- determinism and resume ----------------------------------------------------------------------


def test_an_independent_rerun_is_identical(
    full: tuple[Path, RunConfig], tmp_path: Path, clone_dir: Path
) -> None:
    run_dir, config = full
    again = started(tmp_path, clone_dir, config, GENERATIONS)
    assert manifests(again) == manifests(run_dir)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "written"), (0, "committed"), (1, "trained"), (2, "played")]
)
def test_a_killed_run_resumes_with_identical_hashes(
    full: tuple[Path, RunConfig], tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    complete, config = full
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(GENERATIONS)
    warm_loop(run_dir, config).run(GENERATIONS)
    assert manifests(run_dir) == manifests(complete)
    for g in range(GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(complete, g) / name
            ).read_bytes()


# -- provenance ----------------------------------------------------------------------------------


def test_the_battle_lineage_names_the_clone_and_the_preview_lineage_does_not(
    full: tuple[Path, RunConfig], ids: dict[str, str], tmp_path: Path
) -> None:
    run_dir, _ = full
    assert sources_registry(run_dir)[TEACHER] == []
    out = tmp_path / "export"
    export_provenance(run_dir, out)
    records = [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()]
    assert len(records) == 2 * (GENERATIONS + 1)
    graph = ProvenanceGraph(records, json.loads((out / "sources.json").read_text()))
    last = layout.latest_complete(run_dir)
    assert last is not None
    final = last.manifest["checkpoints"]
    assert TEACHER in graph.ancestry(final["battle"]).sources
    assert ids["battle"] in graph.ancestry(final["battle"]).checkpoints
    # Preview's own generation 0 is the seeded init: no teacher set behind it. Later Preview
    # generations train on games the cloned Battle network played, so they do descend from it.
    zero = layout.verify_generation(layout.generation_dir(run_dir, 0), 0)["checkpoints"]
    assert graph.ancestry(zero["preview"]).sources == frozenset()
    assert graph.ancestry(zero["preview"]).checkpoints == frozenset({zero["preview"]})
    assert ids["battle"] in graph.ancestry(final["preview"]).checkpoints


# -- import-clone --------------------------------------------------------------------------------


def test_import_with_the_battle_role_copies_that_checkpoint_and_record_only(
    tmp_path: Path, clone_dir: Path, ids: dict[str, str]
) -> None:
    assert import_clone(clone_dir, tmp_path / "run", BATTLE) == {"battle": ids["battle"]}
    assert (tmp_path / "run" / "checkpoints" / f"{ids['battle']}.safetensors").is_file()
    assert not (tmp_path / "run" / "checkpoints" / f"{ids['preview']}.safetensors").exists()
    assert record_path(tmp_path / "run", "battle").is_file()
    assert not record_path(tmp_path / "run", "preview").exists()


def test_import_with_the_battle_role_ignores_a_damaged_preview_but_needs_the_battle(
    tmp_path: Path, clone_dir: Path, ids: dict[str, str]
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(clone_dir, broken)
    (broken / "checkpoints" / f"{ids['preview']}.safetensors").unlink()
    assert import_clone(broken, tmp_path / "run", BATTLE) == {"battle": ids["battle"]}
    (broken / "records" / f"{ids['battle']}.json").unlink()
    with pytest.raises(WarmStartError, match="exactly one battle record"):
        import_clone(broken, tmp_path / "run2", BATTLE)


def test_the_command_takes_roles_battle(
    tmp_path: Path, clone_dir: Path, ids: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["import-clone", "--clone-dir", str(clone_dir), "--run-dir", str(tmp_path / "r")]
    assert main([*args, "--roles", "battle"]) == 0
    assert json.loads(capsys.readouterr().out) == {"checkpoints": {"battle": ids["battle"]}}


def test_a_battle_only_start_needs_the_battle_pair_in_the_store(
    tmp_path: Path, ids: dict[str, str]
) -> None:
    with pytest.raises(LoopError, match="import-clone"):
        warm_loop(tmp_path, battle_config(ids)).run(0)


# -- the learner ---------------------------------------------------------------------------------


def test_the_learner_wants_anchors_for_the_warm_started_roles_only(ids: dict[str, str]) -> None:
    config = battle_config(ids, anchor=True)
    built = networks(config)
    device = torch.device("cpu")
    anchor = init_network("battle", game(), config.model, 5)
    learner = Learner(config, built, device, {"battle": anchor})
    assert set(learner.anchors) == {"battle"}
    with pytest.raises(LearnerError, match="warm-started"):
        Learner(config, networks(config), device, {"preview": anchor, "battle": anchor})
    with pytest.raises(LearnerError, match="warm-started"):
        Learner(config, networks(config), device, {"preview": anchor})
