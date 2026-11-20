"""GPU policy inference for training self-play (ADR-030 step 4, #1458).

The device of ``generation.inference`` in the config, the run config hash and the policy id;
#1195's claims for such a run (plan, seeds, teams and ids on rerun; every committed generation
verifies; retraining from a committed stream reproduces the learner state, a CPU learner here);
the divergence record that names the device; the workers' device counts in run metrics; and the
driver's flags for the simulator side (the GPU worker's Python, the rollback flag, the stats
directory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, validate
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Frozen, Loop
from areneon_ml.loop.metrics import MetricsError, metrics_path, read_run_metrics
from areneon_ml.loop.play import NodePlayer, PlayRequest, read_inference_stats
from areneon_ml.loop.records import policy_id

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.loop.support import FakePlayer, make_loop

GENERATIONS = 2
GPU = {"kind": "batched", "max_batch": 2, "device": "cuda"}
CPU = {"kind": "batched", "max_batch": 2}
CORPUS = Path(__file__).resolve().parents[3] / "fixtures/contracts/v1"


def gpu_config(device: str | None = "cuda") -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["generation"]["inference"] = {"kind": "batched", "max_batch": 2}
        if device is not None:
            raw["generation"]["inference"]["device"] = device

    return edited_config(edit)


class NoisyPlayer(FakePlayer):
    """Plays like a GPU run: a replayed generation is other bytes (timing and device differ)."""

    def perturbation(self) -> int:
        return 11 * len(self.requests)


def test_policy_id_vectors_are_pinned_and_the_device_enters_only_for_the_gpu() -> None:
    # Computed outside the code with Python hashlib over the JCS bytes; the TS test pins the same.
    a, b = "a" * 64, "b" * 64
    assert (
        policy_id(a, b, inference={"kind": "batched", "max_batch": 4, "device": "cuda"})
        == "policy:32ffe17f08e8e31c890a6c461a811d6928f7168bf3fec8bb76f04baefca252fb"
    )
    assert (
        policy_id(a, b, "sample", temperature_ppm=500_000, inference=GPU)
        == "policy:a12c36fe31f5fcff65500cfb3696c3d842c23c94ce748c7567d5323af9fcd721"
    )
    # #1195's CPU batching keeps its pinned id.
    assert (
        policy_id(a, b, inference={"kind": "batched", "max_batch": 4})
        == "policy:c1cd82a47b9ef4532ae97297f0b913625474a39e410224d496ac9b2b007c1fb0"
    )
    assert policy_id(a, b, inference=GPU) != policy_id(a, b, inference=CPU)


@pytest.mark.parametrize("device", ["cpu", "rocm", "", None])
def test_policy_id_refuses_a_device_it_cannot_encode(device: object) -> None:
    with pytest.raises(ValueError, match="inference"):
        policy_id("a" * 64, "b" * 64, inference={**CPU, "device": device})


def test_the_config_reader_takes_the_device_and_the_hash_names_it() -> None:
    assert gpu_config().generation.inference_device == "cuda"
    assert gpu_config(None).generation.inference_device is None
    assert tiny_config().generation.inference_device is None
    assert gpu_config().hash != gpu_config(None).hash


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_the_config_reader_refuses_another_device(device: str) -> None:
    with pytest.raises((RunConfigError, ContractError)):
        gpu_config(device)


def test_frozen_policy_id_names_the_device() -> None:
    pair = {"preview": "a" * 64, "battle": "b" * 64}
    gpu = Frozen(0, pair, "c" * 64, None, (), None, None, 2, "cuda")
    assert gpu.policy_id == policy_id(pair["preview"], pair["battle"], inference=GPU)


def test_the_pair_plays_under_the_gpu_policy_id(tmp_path: Path) -> None:
    config = gpu_config()
    player = FakePlayer(config=config)
    make_loop(tmp_path, config, player).run(GENERATIONS)
    for request in player.requests:
        assert request.policy_id == policy_id(request.preview, request.battle, inference=GPU)
        assert request.policy_id != policy_id(request.preview, request.battle, inference=CPU)


def noisy_run(run_dir: Path) -> Loop:
    config = gpu_config()
    return make_loop(run_dir, config, NoisyPlayer(config=config))


def test_a_rerun_keeps_plan_seeds_teams_and_ids(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    noisy_run(first).run(GENERATIONS)
    config = gpu_config()
    make_loop(second, config, FakePlayer(config=config)).run(GENERATIONS)
    for g in range(1, GENERATIONS + 1):
        one, two = layout.generation_dir(first, g), layout.generation_dir(second, g)
        assert (one / "plan.json").read_bytes() == (two / "plan.json").read_bytes()
    one, two = layout.generation_dir(first, 1), layout.generation_dir(second, 1)
    assert (one / "league-run.json").read_bytes() == (two / "league-run.json").read_bytes()
    assert (one / "team-set.json").read_bytes() == (two / "team-set.json").read_bytes()


def test_every_committed_generation_verifies(tmp_path: Path) -> None:
    noisy_run(tmp_path).run(GENERATIONS)
    for g in range(GENERATIONS + 1):
        layout.verify_generation(layout.generation_dir(tmp_path, g), g)


def test_retraining_from_a_committed_stream_reproduces_the_learner_state(tmp_path: Path) -> None:
    loop = noisy_run(tmp_path)
    loop.run(GENERATIONS)
    before, after = (layout.generation_dir(tmp_path, g) for g in (1, 2))
    manifest = json.loads((after / layout.MANIFEST).read_text())
    learner = loop._new_learner()
    restore_state(
        learner,
        (before / "learner-1.safetensors").read_bytes(),
        (before / "learner-1.json").read_bytes(),
        json.loads((before / layout.MANIFEST).read_text())["learner_state_hash"],
    )
    pair = json.loads((after / "league-run.json").read_text())["pair_policy_id"]
    found = trajectories(
        after / "stream.jsonl",
        json.loads((after / "training-set.json").read_text()),
        loop._encoders,
        pair,
    )
    learner.train(found, None)
    assert pack_state(learner).hash == manifest["learner_state_hash"]


def divergence(**changes: Any) -> dict[str, Any]:
    return {
        "max_batch": 2,
        "device": "cuda",
        "decisions": 2000,
        "sampled": 256,
        "differing": 3,
        "max_prior_diff_ppm": 41,
        "target_met": True,
        **changes,
    }


def device_counts(**changes: Any) -> dict[str, Any]:
    return {
        "device": "cuda",
        "workers": 4,
        "device_batches": 900,
        "cpu_batches": 30,
        "allocation_fallbacks": 1,
        "forced": False,
        **changes,
    }


def metrics_with(config: Any, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((CORPUS / "run-metrics/valid/cpu.json").read_text())
    record["run_config_hash"] = config.hash
    record["generation"] = 1
    record["threads"] = config.threads
    record.update(extra)
    return record


def test_the_divergence_record_names_the_configs_device() -> None:
    config = gpu_config()
    found = read_run_metrics(metrics_with(config, inference_divergence=divergence()), config, 1)
    assert found["inference_divergence"]["device"] == "cuda"
    without = divergence()
    del without["device"]
    with pytest.raises(MetricsError, match="device"):
        read_run_metrics(metrics_with(config, inference_divergence=without), config, 1)
    cpu = gpu_config(None)
    with pytest.raises(MetricsError, match="device"):
        read_run_metrics(metrics_with(cpu, inference_divergence=divergence()), cpu, 1)


def test_the_device_counts_are_in_the_contract_and_tie_to_the_config() -> None:
    config = gpu_config()
    validate("run-metrics", metrics_with(config, inference_device=device_counts()))
    assert (
        read_run_metrics(metrics_with(config, inference_device=device_counts()), config, 1)[
            "inference_device"
        ]
        == device_counts()
    )
    cpu = gpu_config(None)
    with pytest.raises(MetricsError, match="device"):
        read_run_metrics(metrics_with(cpu, inference_device=device_counts()), cpu, 1)
    with pytest.raises(MetricsError, match="fallback"):
        read_run_metrics(
            metrics_with(config, inference_device=device_counts(allocation_fallbacks=5)), config, 1
        )
    with pytest.raises(MetricsError, match="forced"):
        read_run_metrics(
            metrics_with(config, inference_device=device_counts(forced=True)), config, 1
        )


def test_the_workers_stats_files_are_summed_and_anything_odd_is_ignored(tmp_path: Path) -> None:
    one = {
        "device": "cuda",
        "device_batches": 10,
        "cpu_batches": 0,
        "allocation_fallbacks": 0,
        "forced": False,
    }
    two = {**one, "device_batches": 3, "cpu_batches": 7, "allocation_fallbacks": 1}
    (tmp_path / "inference-1.json").write_text(json.dumps(one))
    (tmp_path / "inference-2.json").write_text(json.dumps(two))
    (tmp_path / "inference-3.json").write_text("{")
    (tmp_path / "other.json").write_text(json.dumps(one))
    assert read_inference_stats(tmp_path) == {
        "device": "cuda",
        "workers": 2,
        "device_batches": 13,
        "cpu_batches": 7,
        "allocation_fallbacks": 1,
        "forced": False,
    }
    assert read_inference_stats(tmp_path / "missing") == {}


class Recording(NodePlayer):
    """A NodePlayer whose script call is recorded; it leaves one worker's stats file."""

    def _run(self, script: str, *args: str, **_: Any) -> str:
        self.calls.append(list(args))  # type: ignore[attr-defined]
        if "--inference-stats-dir" in args:
            where = Path(args[args.index("--inference-stats-dir") + 1])
            stats = {
                "device": "cuda",
                "device_batches": 5,
                "cpu_batches": 0,
                "allocation_fallbacks": 0,
                "forced": self.inference_cpu,
            }
            if self.inference_cpu:
                stats["device_batches"], stats["cpu_batches"] = 0, 5
            (where / "inference-9.json").write_text(json.dumps(stats))
        return json.dumps({"generation": 1})


def recording(tmp_path: Path, **fields: Any) -> Recording:
    player = Recording(
        repo=tmp_path,
        node="node",
        worker_python="/cpu/python",
        run_dir=tmp_path,
        config_path=tmp_path / "run-config.json",
        engine_data=tmp_path / "engine-data.json",
        encoder_version="enc",
        memory_guard=False,
        **fields,
    )
    player.calls = []  # type: ignore[attr-defined]
    return player


def request(tmp_path: Path) -> PlayRequest:
    return PlayRequest(1, "policy:" + "0" * 64, "a" * 64, "b" * 64, tmp_path / "out")


def test_the_player_passes_the_gpu_flags_and_reads_the_counts(tmp_path: Path) -> None:
    player = recording(tmp_path, gpu_inference=True, inference_worker_python="/gpu/python")
    player.play(request(tmp_path))
    [args] = player.calls  # type: ignore[attr-defined]
    assert args[args.index("--inference-worker-python") + 1] == "/gpu/python"
    assert "--inference-cpu" not in args
    assert "--inference-stats-dir" in args
    stats_dir = Path(args[args.index("--inference-stats-dir") + 1])
    assert not stats_dir.exists()  # a scratch directory, removed after the call
    assert not str(stats_dir).startswith(str(tmp_path / "out"))  # never in a hashed directory
    assert player.last_inference_device == {
        "device": "cuda",
        "workers": 1,
        "device_batches": 5,
        "cpu_batches": 0,
        "allocation_fallbacks": 0,
        "forced": False,
    }


def test_the_rollback_flag_reaches_the_play_script(tmp_path: Path) -> None:
    player = recording(tmp_path, gpu_inference=True, inference_cpu=True)
    player.play(request(tmp_path))
    [args] = player.calls  # type: ignore[attr-defined]
    assert "--inference-cpu" in args
    assert "--inference-worker-python" not in args
    assert player.last_inference_device["forced"] is True


def test_a_run_without_the_gpu_passes_none_of_the_flags(tmp_path: Path) -> None:
    player = recording(tmp_path)
    player.play(request(tmp_path))
    [args] = player.calls  # type: ignore[attr-defined]
    assert not any(a.startswith("--inference") for a in args)
    assert player.last_inference_device == {}


def test_the_loop_records_the_device_counts_in_run_metrics(tmp_path: Path) -> None:
    config = gpu_config()
    player = FakePlayer(config=config)
    player.last_inference_device = device_counts()  # type: ignore[attr-defined]
    make_loop(tmp_path, config, player).run(1)
    found = json.loads(metrics_path(tmp_path, 1).read_text())
    assert found["inference_device"] == device_counts()


def test_the_run_command_refuses_the_gpu_flags_for_a_run_without_the_gpu(tmp_path: Path) -> None:
    from areneon_ml.loop.__main__ import gpu_flags_problem  # noqa: PLC0415

    assert gpu_flags_problem(gpu_config(), "/gpu/python", inference_cpu=False) is None
    assert gpu_flags_problem(gpu_config(), None, inference_cpu=True) is None
    assert "--inference-worker-python" in str(gpu_flags_problem(gpu_config(), None, False))
    assert "device" in str(gpu_flags_problem(gpu_config(None), "/gpu/python", False))
    assert "device" in str(gpu_flags_problem(tiny_config(), None, True))
    assert gpu_flags_problem(tiny_config(), None, False) is None


def test_the_corpus_sample_reads() -> None:
    raw = json.loads((CORPUS / "training-run-config/valid/inference-gpu.json").read_text())
    assert read_run_config(raw).generation.inference_device == "cuda"
