"""GPU policy inference for training self-play (ADR-030 step 4, #1458): the worker.

It states the device in the handshake and refuses every other hello (an evaluation's batch 1
above all), fails closed to its CPU path on an allocation failure and counts it, starts on the
CPU path under the rollback flag, and leaves its counts where the driver reads them. No test
here needs a GPU: the device path is a device copy on the CPU, and the allocation failure is
simulated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import load_checkpoint
from areneon_ml.loop.play import read_inference_stats
from areneon_ml.model.network import PolicyNetwork
from areneon_ml.model.placement import InferencePlacement
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker import __main__ as worker_main
from areneon_ml.worker.__main__ import EXIT_DEVICE, DeviceStatsFile, main, write_inference_stats
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
    sample,
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
    scorer = NetworkScorer(game(), networks, placement=chosen)
    worker = Worker(
        scorer=scorer,
        engine_data_hash=ENGINE_DATA_HASH,
        checkpoints=[
            {"role": "preview", "checkpoint_id": preview},
            {"role": "battle", "checkpoint_id": battle},
        ],
        max_line_bytes=MAX_LINE,
        inference_max_batch=2,
        inference_device="cuda",
    )
    worker.placement_stats = scorer.placement_stats
    return worker


def without_elapsed(reply: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(json.dumps(reply))
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
    stats = gpu.placement_stats()
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

    network.forward = failing
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
    def refuse(network: PolicyNetwork, device: torch.device) -> PolicyNetwork:
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
    assert scorer.probabilities(view)
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


# --- #1502: the fail-closed covers every form torch raises when CUDA memory runs out -----------
# Recorded on the RTX 5070 Ti with torch 2.14.1+cu130: the allocator's own error (a RuntimeError
# subclass), and, when the context could not be created under a host address-space limit, a plain
# RuntimeError; the other forms are what CUDA and cuBLAS say when a kernel or handle cannot
# allocate.
PLAIN_ALLOCATION_FAILURES = [
    "Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling "
    "NumCudaDevices() that might have already set an error? Error 2: out of memory",
    "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported at some "
    "other API call, so the stacktrace below might be incorrect.",
    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    "CUDA out of memory. Tried to allocate 20.00 MiB",
]
LOUD_ERRORS: list[BaseException] = [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA error: device-side assert triggered"),
    RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x8 and 16x8)"),
    ValueError("out of memory"),  # not a RuntimeError, whatever it says
]


def failing_once(network: torch.nn.Module, error: BaseException) -> list[int]:
    """Make ``network``'s next forward pass raise ``error``, once; count the calls."""
    calls: list[int] = []
    original = network.forward

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise error
        return original(*args, **kwargs)

    network.forward = failing
    return calls


@pytest.mark.parametrize("message", PLAIN_ALLOCATION_FAILURES)
def test_a_plain_runtime_error_in_a_pass_fails_closed_to_the_cpu_path(
    tmp_path: Path, message: str
) -> None:
    placement = InferencePlacement(torch.device("cpu"))
    worker = gpu_worker(at(tmp_path, "g"), placement)
    reference = make_worker(at(tmp_path, "c"), inference_max_batch=2)
    calls = failing_once(placement.device_network("battle"), RuntimeError(message))
    first = worker.answer_batch([battle_query(query_id=1)])
    assert calls == [1]
    assert [without_elapsed(r) for r in first] == [
        without_elapsed(r) for r in reference.answer_batch([battle_query(query_id=1)])
    ]
    worker.answer_batch([preview_query(query_id=2)])
    assert not placement.on_device
    assert placement.stats() == {
        "device": "cuda",
        "device_batches": 0,
        "cpu_batches": 2,
        "allocation_fallbacks": 1,
        "forced": False,
    }


@pytest.mark.parametrize("message", PLAIN_ALLOCATION_FAILURES)
def test_a_plain_runtime_error_while_placing_the_networks_starts_on_the_cpu_path(
    tmp_path: Path, message: str
) -> None:
    def refuse(network: PolicyNetwork, device: torch.device) -> PolicyNetwork:
        raise RuntimeError(message)

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


@pytest.mark.parametrize("error", LOUD_ERRORS, ids=lambda e: str(e)[:30])
def test_every_other_error_in_a_pass_is_loud_and_leaves_the_device_path_alone(
    tmp_path: Path, error: BaseException
) -> None:
    placement = InferencePlacement(torch.device("cpu"))
    worker = gpu_worker(tmp_path, placement)
    failing_once(placement.device_network("battle"), error)
    with pytest.raises(type(error)):
        worker.answer_batch([battle_query(query_id=1)])
    assert placement.on_device
    assert placement.stats()["allocation_fallbacks"] == 0


@pytest.mark.parametrize("error", LOUD_ERRORS, ids=lambda e: str(e)[:30])
def test_every_other_error_while_placing_the_networks_is_loud(
    tmp_path: Path, error: BaseException
) -> None:
    def refuse(network: PolicyNetwork, device: torch.device) -> PolicyNetwork:
        raise error

    with pytest.raises(type(error)):
        gpu_worker(tmp_path, InferencePlacement(torch.device("cpu"), copy_to=refuse))


def test_a_plain_runtime_error_on_the_cpu_path_is_a_defect(tmp_path: Path) -> None:
    placement = InferencePlacement(torch.device("cpu"), forced_cpu=True)
    gpu_worker(tmp_path, placement)
    with pytest.raises(RuntimeError, match="out of memory"):
        placement.run("battle", lambda network, device: raise_runtime("CUDA error: out of memory"))


def raise_runtime(message: str) -> Any:
    raise RuntimeError(message)


def test_the_command_line_worker_survives_a_context_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole of ``build()``: the first CUDA operation (the context) raises a plain error."""
    run_dir, preview, battle = make_run_dir(tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "x")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")  # restored afterwards, whatever was there
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a, **k: None)
    original = torch.nn.Module.to

    def refusing(self: torch.nn.Module, *a: Any, **k: Any) -> Any:
        target = a[0] if a else k.get("device")
        if isinstance(target, (str, torch.device)) and torch.device(target).type == "cuda":
            raise RuntimeError(PLAIN_ALLOCATION_FAILURES[0])
        return original(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", refusing)
    worker = worker_main.build(
        args(run_dir, preview, battle, "--inference-max-batch", "2", "--inference-device", "cuda")
    )
    code, frames = session(worker, line(hello_with(GPU)), line(battle_query(query_id=1)))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    assert worker.placement_stats() == {
        "device": "cuda",
        "device_batches": 0,
        "cpu_batches": 1,
        "allocation_fallbacks": 1,
        "forced": False,
    }


# --- #1502: a real device-transfer check, for local runs on a GPU (skipped in CI) ---------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (local runs only)")
def test_the_cuda_device_answers_as_the_cpu_scorer_does(tmp_path: Path) -> None:
    """The 'device path' tests above use a device copy on the CPU, so they cannot catch a bug in
    the transfer of the networks or of a batch: this one runs tiny networks on the card."""
    cpu = make_worker(at(tmp_path, "c"), inference_max_batch=2)._scorer
    placement = InferencePlacement(torch.device("cuda"))
    gpu = gpu_worker(at(tmp_path, "g"), placement)._scorer
    assert isinstance(cpu, NetworkScorer)
    assert isinstance(gpu, NetworkScorer)
    assert placement.on_device
    for role in ("preview", "battle"):
        found = next(placement.device_network(role).parameters())
        assert found.device.type == "cuda"
    views = [
        encoder_input(sample("policy-query/valid/battle-evaluate.json")),
        encoder_input(sample("policy-query/valid/preview-evaluate.json")),
    ]
    for view in views:
        got, want = gpu.evaluate(view), cpu.evaluate(view)
        torch.testing.assert_close(
            torch.tensor(got[0]), torch.tensor(want[0]), atol=1e-4, rtol=1e-4
        )
        assert got[1] == pytest.approx(want[1], abs=1e-4)
    items = [(view, None) for view in views]
    for got_all, want_all in zip(gpu.score_batch(items), cpu.score_batch(items), strict=True):
        assert isinstance(got_all, tuple)
        assert isinstance(want_all, tuple)
        torch.testing.assert_close(
            torch.tensor(got_all[0]), torch.tensor(want_all[0]), atol=1e-4, rtol=1e-4
        )
    stats = placement.stats()
    assert stats["device_batches"] >= 4
    assert stats["cpu_batches"] == 0
    assert stats["allocation_fallbacks"] == 0


# --- #1502: the counts file a worker leaves ----------------------------------------------------


def counts(**changes: Any) -> dict[str, Any]:
    return {
        "device": "cuda",
        "device_batches": 3,
        "cpu_batches": 1,
        "allocation_fallbacks": 0,
        "forced": False,
        **changes,
    }


def test_two_counts_files_of_one_pid_are_two_files(tmp_path: Path) -> None:
    """A reused pid must not overwrite another worker's counts."""
    first = write_inference_stats(tmp_path, counts())
    second = write_inference_stats(tmp_path, counts(device_batches=9))
    assert first != second
    assert first.name.startswith(f"inference-{os.getpid()}-")
    assert sorted(p.name for p in tmp_path.glob("inference-*.json")) == sorted(
        [first.name, second.name]
    )
    assert read_inference_stats(tmp_path)["device_batches"] == 12
    assert not list(tmp_path.glob(".*partial"))


def test_a_stats_file_rewrites_its_own_file_and_two_of_them_do_not_share_one(
    tmp_path: Path,
) -> None:
    now = [0.0]
    live = counts()
    one = DeviceStatsFile(tmp_path, lambda: dict(live), clock=lambda: now[0])
    other = DeviceStatsFile(tmp_path, lambda: dict(live), clock=lambda: now[0])
    one.write()
    live["device_batches"] = 7
    one.write()
    other.write()
    files = sorted(tmp_path.glob("inference-*.json"))
    assert len(files) == 2
    assert read_inference_stats(tmp_path)["workers"] == 2


def test_the_heartbeat_writes_at_most_once_an_interval(tmp_path: Path) -> None:
    now = [100.0]
    live = counts(device_batches=1)
    stats = DeviceStatsFile(tmp_path, lambda: dict(live), clock=lambda: now[0], interval_s=1.0)

    def on_disk() -> int:
        [path] = tmp_path.glob("inference-*.json")
        return int(json.loads(path.read_text())["device_batches"])

    stats.beat()  # the first beat writes
    assert on_disk() == 1
    live["device_batches"] = 2
    now[0] += 0.5
    stats.beat()  # too soon
    assert on_disk() == 1
    now[0] += 0.5
    stats.beat()
    assert on_disk() == 2
    live["device_batches"] = 5
    stats.write()  # the end of the session writes whatever the clock says
    assert on_disk() == 5


def test_a_worker_without_counts_leaves_no_file(tmp_path: Path) -> None:
    stats = DeviceStatsFile(tmp_path, lambda: None)
    stats.write()
    stats.beat()
    assert list(tmp_path.iterdir()) == []


def test_a_failing_counts_write_is_reported_and_never_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = DeviceStatsFile(tmp_path / "missing", counts)
    broken.write()
    broken.write()
    broken.beat()
    assert capsys.readouterr().err.count("could not leave the device counts") == 1  # once


class StubWorker:
    """What ``main`` needs of a worker: where to leave counts, and a session to serve."""

    def __init__(self, stats_dir: Path | None, code: int = 0, error: Exception | None = None):
        self.inference_stats_dir = stats_dir
        self.profile_dir: Path | None = None
        self.code, self.error = code, error
        self.live = counts(device_batches=0)
        self.seen_at_start: list[str] = []
        self.heartbeat: Any = lambda: None

    def placement_stats(self) -> dict[str, Any] | None:
        return dict(self.live)

    def serve(self, stdin: Any, stdout: Any) -> int:
        if self.inference_stats_dir is not None and self.inference_stats_dir.is_dir():
            self.seen_at_start = sorted(p.name for p in self.inference_stats_dir.iterdir())
        self.live["device_batches"] = 4
        self.heartbeat()
        if self.error is not None:
            raise self.error
        return self.code


def test_main_leaves_the_counts_at_the_start_and_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker killed mid-session still leaves a file, so it is counted."""
    stub = StubWorker(tmp_path)
    monkeypatch.setattr(worker_main, "build", lambda argv: stub)
    assert main([]) == 0
    assert len(stub.seen_at_start) == 1  # there before the first query was served
    [path] = tmp_path.glob("inference-*.json")
    assert json.loads(path.read_text())["device_batches"] == 4  # and rewritten at the end


def test_a_failing_counts_write_does_not_change_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = StubWorker(tmp_path / "missing", code=3)
    monkeypatch.setattr(worker_main, "build", lambda argv: stub)
    assert main([]) == 3
    assert "device counts" in capsys.readouterr().err


def test_a_failing_counts_write_does_not_hide_the_error_of_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubWorker(tmp_path / "missing", error=RuntimeError("the session failed"))
    monkeypatch.setattr(worker_main, "build", lambda argv: stub)
    with pytest.raises(RuntimeError, match="the session failed"):
        main([])


def test_a_worker_without_a_counts_directory_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubWorker(None, code=2)
    monkeypatch.setattr(worker_main, "build", lambda argv: stub)
    assert main([]) == 2
    assert list(tmp_path.iterdir()) == []


def test_the_batched_session_beats_after_each_batch(tmp_path: Path) -> None:
    worker = gpu_worker(tmp_path)
    beats: list[int] = []
    worker.heartbeat = lambda: beats.append(1)
    session(
        worker,
        line(hello_with(GPU)),
        line(battle_query(query_id=1)),
        line(preview_query(query_id=2)),
    )
    assert len(beats) >= 1
