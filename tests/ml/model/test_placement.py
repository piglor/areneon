"""The device-placement helpers of GPU policy inference (ADR-030 step 4, #1458, hardening #1502).

``is_allocation_failure`` decides which errors fail closed to the CPU path (an out-of-memory error
of torch, in whatever form torch raises it, and nothing else), ``batch_to`` moves a collated batch
to a device, and ``configure_gpu_inference`` sets a GPU worker's process up. None needs a GPU: a
``meta`` device carries a batch's shapes without memory, the CUDA calls are stubbed, and the real
device-equality test below is skipped where torch has no CUDA.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.placement import InferencePlacement, batch_to, is_allocation_failure
from areneon_ml.model.runtime import DeviceError, configure_gpu_inference
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import battle_query, make_worker, preview_query

# What torch raised when CUDA memory ran out, recorded on the RTX 5070 Ti with torch 2.14.1+cu130
# (#1502): the allocator's own error (a RuntimeError subclass), and, when the context could not be
# created under a host address-space limit, a plain RuntimeError. The other two are the messages
# CUDA and cuBLAS give when a kernel or a handle cannot allocate.
OUT_OF_MEMORY = [
    "CUDA out of memory. Tried to allocate 200.00 MiB. GPU 0 has a total capacity of 15.92 GiB",
    "Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling "
    "NumCudaDevices() that might have already set an error? Error 2: out of memory",
    "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported at some "
    "other API call, so the stacktrace below might be incorrect.",
    "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
]
OTHER_RUNTIME_ERRORS = [
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
    "mat1 and mat2 shapes cannot be multiplied (4x8 and 16x8)",
    "Expected all tensors to be on the same device, but found at least two devices",
    "",
]


@pytest.mark.parametrize("message", OUT_OF_MEMORY)
def test_a_plain_runtime_error_that_says_out_of_memory_or_alloc_failed_is_an_allocation_failure(
    message: str,
) -> None:
    assert is_allocation_failure(RuntimeError(message))


def test_the_allocators_own_error_is_one_whatever_its_text() -> None:
    assert is_allocation_failure(torch.cuda.OutOfMemoryError())
    assert is_allocation_failure(torch.cuda.OutOfMemoryError("anything"))


@pytest.mark.parametrize("message", OTHER_RUNTIME_ERRORS)
def test_every_other_runtime_error_is_not(message: str) -> None:
    assert not is_allocation_failure(RuntimeError(message))


@pytest.mark.parametrize(
    "error", [ValueError("out of memory"), KeyError("ALLOC_FAILED"), OSError("out of memory")]
)
def test_only_a_runtime_error_can_be_one(error: BaseException) -> None:
    assert not is_allocation_failure(error)


def test_the_message_match_ignores_case() -> None:
    assert is_allocation_failure(RuntimeError("CUDA Error: Out Of Memory"))
    assert is_allocation_failure(RuntimeError("cublas_status_alloc_failed"))


def scorer_of(root: Path) -> NetworkScorer:
    found = make_worker(root)._scorer
    assert isinstance(found, NetworkScorer)
    return found


def collated(kind: str, tmp_path: Path) -> Batch:
    scorer = scorer_of(tmp_path)
    query = battle_query() if kind == "battle" else preview_query()
    return collate([scorer._encoders[kind].encode(encoder_input(query))])


@pytest.mark.parametrize("kind", ["battle", "preview"])
def test_batch_to_moves_every_tensor_and_keeps_every_other_field(kind: str, tmp_path: Path) -> None:
    batch = collated(kind, tmp_path)
    moved = batch_to(batch, torch.device("meta"))
    assert moved is not batch
    seen = 0
    for field in dataclasses.fields(batch):
        before, after = getattr(batch, field.name), getattr(moved, field.name)
        if isinstance(before, torch.Tensor):
            seen += 1
            assert after.device.type == "meta", field.name
            assert after.shape == before.shape, field.name
            assert after.dtype == before.dtype, field.name
            assert before.device.type == "cpu"  # the original is untouched
        else:
            assert after == before, field.name  # the kind, and a None stays None
    assert seen >= 10


def test_batch_to_on_the_cpu_is_an_equal_copy_of_the_fields(tmp_path: Path) -> None:
    batch = collated("battle", tmp_path)
    moved = batch_to(batch, torch.device("cpu"))
    for field in dataclasses.fields(batch):
        before, after = getattr(batch, field.name), getattr(moved, field.name)
        if isinstance(before, torch.Tensor):
            assert torch.equal(before, after), field.name


class Recorder:
    """Stands in for the three torch calls ``configure_gpu_inference`` makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(
            torch,
            "use_deterministic_algorithms",
            lambda *a, **k: self.calls.append(("deterministic", a, k)),
        )
        monkeypatch.setattr(
            torch, "set_num_threads", lambda *a, **k: self.calls.append(("threads", a, k))
        )


def test_configure_gpu_inference_sets_the_process_up_when_cuda_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "x")  # restored afterwards, whatever was there
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    recorder = Recorder(monkeypatch, available=True)
    configure_gpu_inference()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert recorder.calls == [
        ("deterministic", (True,), {"warn_only": True}),  # warn-only: GPU results are not claimed
        ("threads", (1,), {}),  # the CPU path a worker falls back to keeps one thread
    ]


def test_configure_gpu_inference_keeps_a_workspace_the_environment_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    Recorder(monkeypatch, available=True)
    configure_gpu_inference()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_configure_gpu_inference_refuses_a_torch_without_a_usable_device_and_sets_nothing_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "x")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    recorder = Recorder(monkeypatch, available=False)
    with pytest.raises(DeviceError, match="--inference-cpu"):
        configure_gpu_inference()
    assert recorder.calls == []
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_a_failing_cache_release_does_not_turn_a_fallback_into_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an allocation failure the cache release is best effort (R-5: a crowded card)."""
    placement = InferencePlacement(torch.device("cpu"))
    placement.adopt(scorer_of(tmp_path)._networks)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def refuse() -> None:
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(torch.cuda, "empty_cache", refuse)
    calls: list[torch.device] = []

    def call(network: Any, device: torch.device) -> str:
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate`")
        return "answered"

    assert placement.run("battle", call) == "answered"  # the retry on the CPU path
    assert not placement.on_device
    assert placement.stats()["allocation_fallbacks"] == 1


def test_the_device_copy_is_not_there_on_the_cpu_path(tmp_path: Path) -> None:
    placement = InferencePlacement(torch.device("cpu"), forced_cpu=True)
    placement.adopt(scorer_of(tmp_path)._networks)
    assert not placement.on_device
    with pytest.raises(RuntimeError, match="not on the device"):
        placement.device_network("battle")
