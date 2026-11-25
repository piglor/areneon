"""Where a GPU-inference worker runs its batched forward passes (ADR-030 step 4, #1458).

Training self-play only: a run whose config says ``generation.inference.device`` starts its play
workers from the unshipped ``ml/gpu-env`` environment, and their scorer runs the networks on that
device through an ``InferencePlacement``. Evaluation and sealed paths never build one (they score
one query per pass on the CPU, and their batch-1 clients refuse such a worker at the handshake).

The placement keeps the networks as they were loaded (on the CPU) and works on a copy on the
device. It fails closed to the CPU: an allocation failure, while copying the networks or in any
forward pass, drops the device copies, empties the device cache and answers that pass and every
later one of the session with the CPU networks (one fallback per worker, counted). The rollback
flag (``forced_cpu``) starts on the CPU networks and never touches the device. An allocation
failure on the CPU path is a defect and propagates. GPU and CPU results are not guaranteed equal.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import torch

from areneon_ml.model.network import PolicyNetwork

T = TypeVar("T")

#: What an allocation failure is: torch's out-of-memory error (a ``RuntimeError``), which the CPU
#: build defines too.
ALLOCATION_ERRORS: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)

#: The device a GPU-inference setting names in the handshake, the policy id and run metrics.
STATED_DEVICE = "cuda"


def _copy_to(network: PolicyNetwork, device: torch.device) -> PolicyNetwork:
    placed = copy.deepcopy(network).to(device)
    placed.eval()
    return placed


def batch_to(batch: Any, device: torch.device) -> Any:
    """A copy of a collated ``Batch`` with every tensor field on ``device`` (others as they are)."""
    changes = {
        field.name: value.to(device)
        for field in dataclasses.fields(batch)
        if isinstance(value := getattr(batch, field.name), torch.Tensor)
    }
    return dataclasses.replace(batch, **changes)


class InferencePlacement:
    """The device copies of a worker's networks, with the fail-closed CPU path and its counts.

    ``device`` is where the copies go (``cuda`` in a real worker; a test gives the CPU, which
    exercises the same path without a GPU). ``copy_to`` makes a device copy of one network (a test
    seam: the default deep-copies it and moves the copy).
    """

    def __init__(
        self,
        device: torch.device,
        forced_cpu: bool = False,
        copy_to: Callable[[PolicyNetwork, torch.device], PolicyNetwork] = _copy_to,
    ) -> None:
        self._device = device
        self._forced = forced_cpu
        self._copy_to = copy_to
        self._cpu: dict[str, PolicyNetwork] = {}
        self._placed: dict[str, PolicyNetwork] | None = None
        self._device_batches = 0
        self._cpu_batches = 0
        self._fallbacks = 0

    @property
    def on_device(self) -> bool:
        """Whether the next pass runs on the device (False after a fallback or when forced)."""
        return self._placed is not None

    def adopt(self, networks: Mapping[str, PolicyNetwork]) -> None:
        """Take the loaded (CPU) networks and copy them to the device, unless forced to the CPU."""
        self._cpu = dict(networks)
        if self._forced:
            return
        try:
            self._placed = {
                role: self._copy_to(net, self._device) for role, net in networks.items()
            }
        except ALLOCATION_ERRORS:
            self._fall_back()

    def device_network(self, role: str) -> PolicyNetwork:
        """The device copy of ``role``'s network (a test injects a failure into it)."""
        if self._placed is None:
            raise RuntimeError("the networks are not on the device")
        return self._placed[role]

    def cpu_network(self, role: str) -> PolicyNetwork:
        """``role``'s network as it was loaded: the CPU path."""
        return self._cpu[role]

    def run(self, role: str, call: Callable[[PolicyNetwork, torch.device], T]) -> T:
        """One forward pass: ``call(network, device)`` on the device, else on the CPU path.

        ``call`` moves its inputs to ``device`` and must return results that no longer depend on
        the device (plain Python values), so a retry on the CPU gives the whole result again.
        """
        if self._placed is not None:
            try:
                found = call(self._placed[role], self._device)
            except ALLOCATION_ERRORS:
                self._fall_back()
            else:
                self._device_batches += 1
                return found
        found = call(self._cpu[role], torch.device("cpu"))
        self._cpu_batches += 1
        return found

    def _fall_back(self) -> None:
        self._placed = None
        self._fallbacks += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stats(self) -> dict[str, Any]:
        """The counts a worker leaves for the run metrics (``inference_device``)."""
        return {
            "device": STATED_DEVICE,
            "device_batches": self._device_batches,
            "cpu_batches": self._cpu_batches,
            "allocation_fallbacks": self._fallbacks,
            "forced": self._forced,
        }
