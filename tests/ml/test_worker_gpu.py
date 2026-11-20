"""GPU policy inference for training self-play (ADR-030 step 4, #1458): the worker.

It states the device in the handshake and refuses every other hello (an evaluation's batch 1
above all), fails closed to its CPU path on an allocation failure and counts it, starts on the
CPU path under the rollback flag, and leaves its counts where the driver reads them. No test
here needs a GPU: the device path is a device copy on the CPU, and the allocation failure is
simulated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import load_checkpoint
from areneon_ml.model.placement import InferencePlacement
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker import __main__ as worker_main
from areneon_ml.worker.__main__ import EXIT_DEVICE, main, write_inference_stats
from areneon_ml.worker.checkpoints import CheckpointStore
from areneon_ml.worker.protocol import check
from areneon_ml.worker.serve import EXIT_OK, EXIT_REFUSED, Worker
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import (
    ENGINE_DATA,
    ENGINE_DATA_HASH,
    MAX_LINE,
    battle_query,
    game,
    hello,
    line,
    make_run_dir,
    make_worker,
    preview_query,
    session,
)

GPU = {"kind": "batched", "max_batch": 2, "device": "cuda"}
CPU = {"kind": "batched", "max_batch": 2}


def hello_with(inference: dict[str, Any] | None) -> dict[str, Any]:
    frame = hello()
    if inference is not None:
        frame["message"]["inference"] = inference
    return frame


def at(root: Path, name: str) -> Path:
    found = root / name
    found.mkdir()
    return found


def gpu_worker(root: Path, placement: InferencePlacement | None = None) -> Worker:
    """A worker that states the GPU setting; its device path is a CPU copy (no GPU in a test)."""
    run_dir, preview, battle = make_run_dir(root)
    store = CheckpointStore(run_dir)
    networks = {
        role: load_checkpoint(store.load(found), game(), found, role=role)
        for role, found in (("preview", preview), ("battle", battle))
    }
    chosen = placement if placement is not None else InferencePlacement(torch.device("cpu"))
    return Worker(
        scorer=NetworkScorer(game(), networks, placement=chosen),
        engine_data_hash=ENGINE_DATA_HASH,
        checkpoints=[
            {"role": "preview", "checkpoint_id": preview},
            {"role": "battle", "checkpoint_id": battle},
        ],
        max_line_bytes=MAX_LINE,
        inference_max_batch=2,
        inference_device="cuda",
    )


def without_elapsed(reply: dict[str, Any]) -> dict[str, Any]:
    found = json.loads(json.dumps(reply))
    recommendation = found["reply"].get("recommendation")
    if recommendation is not None:
        recommendation["elapsed_ms"] = 0
    return found


def test_a_gpu_worker_states_the_device_in_ready(tmp_path: Path) -> None:
    code, [ready] = session(gpu_worker(tmp_path), line(hello_with(GPU)))
    assert code == EXIT_OK
    check("policy-handshake", ready)
    assert ready["message"]["inference"] == GPU


@pytest.mark.parametrize(
    "asked",
    [None, CPU, {"kind": "batched", "max_batch": 3, "device": "cuda"}],
    ids=["evaluation-batch-1", "cpu-batching", "other-max-batch"],
)
def test_a_gpu_worker_refuses_every_other_hello(
    tmp_path: Path, asked: dict[str, Any] | None
) -> None:
    code, frames = session(
        gpu_worker(tmp_path), line(hello_with(asked)), line(battle_query(query_id=1))
    )
    assert code == EXIT_REFUSED
    assert len(frames) == 1  # the ready names the difference; no query is answered


def test_a_cpu_batching_worker_refuses_a_gpu_hello(tmp_path: Path) -> None:
    code, _ = session(make_worker(tmp_path, inference_max_batch=2), line(hello_with(GPU)))
    assert code == EXIT_REFUSED


def test_the_device_needs_a_batching(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        Worker(
            scorer=make_worker(at(tmp_path, "w"))._scorer,
            engine_data_hash=ENGINE_DATA_HASH,
            checkpoints=[],
            max_line_bytes=MAX_LINE,
            inference_device="cuda",
        )
    with pytest.raises(ValueError, match="device"):
        Worker(
            scorer=make_worker(at(tmp_path, "x"))._scorer,
            engine_data_hash=ENGINE_DATA_HASH,
            checkpoints=[],
            max_line_bytes=MAX_LINE,
            inference_max_batch=2,
            inference_device="rocm",
        )


def test_the_device_path_answers_as_the_cpu_scorer_does_on_a_cpu_copy(tmp_path: Path) -> None:
    gpu = gpu_worker(at(tmp_path, "g"))
    cpu = make_worker(at(tmp_path, "c"), inference_max_batch=2)
    queries = [battle_query(query_id=1), preview_query(query_id=2)]
    assert [without_elapsed(r) for r in gpu.answer_batch(queries)] == [
        without_elapsed(r) for r in cpu.answer_batch(queries)
    ]
    stats = gpu._scorer.placement_stats()  # type: ignore[attr-defined]
    assert stats == {
        "device": "cuda",
        "device_batches": 2,  # one padded pass per role
        "cpu_batches": 0,
        "allocation_fallbacks": 0,
        "forced": False,
    }


def oom_once(network: torch.nn.Module) -> list[int]:
    """Make ``network``'s next forward pass raise an allocation failure, once; count the calls."""
    calls: list[int] = []
    original = network.forward

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(*args, **kwargs)

    network.forward = failing  # type: ignore[method-assign]
    return calls


def test_an_allocation_failure_fails_closed_to_the_cpu_path_for_the_rest_of_the_session(
    tmp_path: Path,
) -> None:
    placement = InferencePlacement(torch.device("cpu"))
    worker = gpu_worker(at(tmp_path, "g"), placement)
    reference = make_worker(at(tmp_path, "c"), inference_max_batch=2)
    calls = oom_once(placement.device_network("battle"))
    first = worker.answer_batch([battle_query(query_id=1)])
    assert calls == [1]  # the device copy raised; it is never asked again
    # The query was answered on the CPU path, exactly as the CPU worker answers it.
    assert [without_elapsed(r) for r in first] == [
        without_elapsed(r) for r in reference.answer_batch([battle_query(query_id=1)])
    ]
    worker.answer_batch([battle_query(query_id=2), preview_query(query_id=3)])
    assert calls == [1]
    assert not placement.on_device
    assert placement.stats() == {
        "device": "cuda",
        "device_batches": 0,
        "cpu_batches": 3,  # the retried battle pass, then one pass per role
        "allocation_fallbacks": 1,
        "forced": False,
    }


def test_an_allocation_failure_on_the_cpu_path_is_a_defect(tmp_path: Path) -> None:
    placement = InferencePlacement(torch.device("cpu"), forced_cpu=True)
    gpu_worker(tmp_path, placement)
    scorer_network = placement.cpu_network("battle")
    oom_once(scorer_network)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        placement.run("battle", lambda network, device: network(torch.zeros(1)))


def test_an_allocation_failure_while_placing_the_networks_starts_on_the_cpu_path(
    tmp_path: Path,
) -> None:
    def refuse(network: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    placement = InferencePlacement(torch.device("cpu"), copy_to=refuse)
    worker = gpu_worker(tmp_path, placement)
    worker.answer_batch([battle_query(query_id=1)])
    assert placement.stats() == {
        "device": "cuda",
        "device_batches": 0,
        "cpu_batches": 1,
        "allocation_fallbacks": 1,
        "forced": False,
    }


def test_the_rollback_flag_starts_on_the_cpu_path_and_keeps_the_stated_setting(
    tmp_path: Path,
) -> None:
    placement = InferencePlacement(torch.device("cpu"), forced_cpu=True)
    worker = gpu_worker(tmp_path, placement)
    code, frames = session(worker, line(hello_with(GPU)), line(battle_query(query_id=1)))
    assert code == EXIT_OK
    assert frames[0]["message"]["inference"] == GPU
    assert placement.stats() == {
        "device": "cuda",
        "device_batches": 0,
        "cpu_batches": 1,
        "allocation_fallbacks": 0,
        "forced": True,
    }


def test_value_and_single_queries_go_through_the_placement(tmp_path: Path) -> None:
    placement = InferencePlacement(torch.device("cpu"))
    scorer = gpu_worker(tmp_path, placement)._scorer
    view = encoder_input(battle_query())
    assert scorer.probabilities(view)  # type: ignore[attr-defined]
    assert placement.stats()["device_batches"] == 1


def test_the_stats_file_holds_the_counts(tmp_path: Path) -> None:
    placement = InferencePlacement(torch.device("cpu"), forced_cpu=True)
    path = write_inference_stats(tmp_path, placement.stats())
    assert path.parent == tmp_path
    assert path.name.startswith("inference-")
    assert json.loads(path.read_text()) == placement.stats()


def args(run_dir: Path, preview: str, battle: str, *extra: str) -> list[str]:
    return [
        "--run-dir",
        str(run_dir),
        "--preview",
        preview,
        "--battle",
        battle,
        "--engine-data",
        str(ENGINE_DATA),
        "--max-line-bytes",
        str(MAX_LINE),
        *extra,
    ]


def test_the_command_line_refuses_a_cuda_worker_where_torch_has_no_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(
        args(run_dir, preview, battle, "--inference-max-batch", "2", "--inference-device", "cuda")
    )
    assert code == EXIT_DEVICE
    assert "--inference-cpu" in capsys.readouterr().err  # it names the rollback flag


@pytest.mark.parametrize(
    "extra",
    [
        ("--inference-device", "cuda"),  # a device without a batching
        ("--inference-max-batch", "2", "--inference-force-cpu"),  # rollback without a device
        ("--inference-max-batch", "2", "--inference-stats-dir", "x"),  # stats without a device
        ("--inference-max-batch", "2", "--inference-device", "cpu"),  # not a device choice
    ],
)
def test_the_command_line_refuses_device_flags_out_of_place(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    with pytest.raises(SystemExit):
        worker_main.build(args(run_dir, preview, battle, *extra))


def test_the_rollback_flag_builds_a_worker_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker = worker_main.build(
        args(
            run_dir,
            preview,
            battle,
            "--inference-max-batch",
            "2",
            "--inference-device",
            "cuda",
            "--inference-force-cpu",
            "--inference-stats-dir",
            str(tmp_path),
        )
    )
    assert worker.inference_stats_dir == tmp_path
    code, [ready] = session(worker, line(hello_with(GPU)))
    assert code == EXIT_OK
    assert ready["message"]["inference"] == GPU
