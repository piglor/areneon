"""Deterministic CPU inference (ADR-010 §2, batch invariance).

Evaluation and sealed runs score one query per forward pass on CPU with
deterministic algorithms enabled [S13] and one intra-op thread, so float
accumulation does not depend on which other battles were waiting or on how
many cores the machine has. Training self-play may batch and may use more
threads: it does not call this. A GPU-inference worker of training self-play (ADR-030
step 4) calls ``configure_gpu_inference`` instead.
"""

from __future__ import annotations

import os

import torch


def configure_cpu_inference() -> None:
    """Make this process's torch deterministic for evaluation; call once at start-up."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class DeviceError(RuntimeError):
    """The worker was asked for a device this torch cannot use."""


def configure_gpu_inference() -> None:
    """Set a GPU-inference worker up (ADR-030 step 4, training self-play only); call once.

    It refuses a torch with no usable CUDA device (the shipped closure's CPU torch has none: the
    GPU worker runs from the unshipped ``ml/gpu-env`` environment), and never replaces the GPU by
    the CPU here; the rollback flag starts the worker on its CPU path instead. cuBLAS gets a fixed
    workspace and deterministic algorithms are asked for where torch has them (warn-only, as the
    CUDA learner does), but GPU results are not claimed bit-reproducible, nor equal to the CPU's.
    The CPU path a worker falls back to keeps one intra-op thread.
    """
    if not torch.cuda.is_available():
        raise DeviceError(
            "the worker was asked for cuda, but this torch has no usable CUDA device: start it "
            "from the ml/gpu-env Python, or roll back with --inference-cpu"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
