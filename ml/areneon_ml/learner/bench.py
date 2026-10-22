"""Learner throughput on one device (ADR-010 §4): decisions and optimiser steps per second.

The run config decides the device, the model and the PPO settings, so a
measurement is of exactly what a training run would do; the decisions come from
a recorded TrainingSet stream. One untimed pass warms the device up (allocator,
kernels, autotuning), then ``repeats`` timed passes run over the same stream.
The numbers are measurements, not results of a run: they depend on the machine
and are not claimed to repeat, and the networks are the seeded untrained pair.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.trainer import Learner, TrainReport, configure_learner
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, init_network


def device_facts(device: torch.device) -> dict[str, Any]:
    """What a measurement needs to say about where it ran."""
    facts: dict[str, Any] = {
        "device": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        facts |= {
            "device_name": torch.cuda.get_device_name(device),
            "capability": f"sm_{major}{minor}",
            "arch_list": torch.cuda.get_arch_list(),
        }
    else:
        facts |= {"device_name": platform.processor() or platform.machine()}
    return facts


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class Recorded:
    """A TrainingSet stream and its manifest."""

    stream: str | Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Timing:
    seconds: float
    decisions: int
    optimiser_steps: int
    peak_memory_bytes: int | None


def time_passes(
    train: Callable[[], TrainReport],
    device: torch.device,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """Run ``train`` once untimed, then ``repeats`` times between two readings of ``clock``.

    On CUDA the clock is read only after the device has finished, and the peak memory of the
    timed passes is reported.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    train()  # warm-up, not timed
    _synchronise(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = clock()
    reports = [train() for _ in range(repeats)]
    _synchronise(device)
    seconds = clock() - started
    return Timing(
        seconds,
        sum(r.steps for report in reports for r in report.roles.values()),
        sum(r.optimiser_steps for report in reports for r in report.roles.values()),
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )


def benchmark(
    config: RunConfig,
    game: GameData,
    recorded: Recorded,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time ``repeats`` PPO generations over the recorded stream on the config's device."""
    device = configure_learner(config)
    encoders = {role: Encoder(game, config.model.encoder) for role in ROLES}
    manifest = recorded.manifest
    found = trajectories(recorded.stream, manifest, encoders, manifest["behaviour_policy_id"])
    seeds = {"preview": config.init_preview, "battle": config.init_battle}
    networks = {role: init_network(role, game, config.model, seeds[role]) for role in ROLES}
    learner = Learner(config, networks, device)
    timing = time_passes(lambda: learner.train(found), device, repeats, clock)
    return {
        **device_facts(device),
        "repeats": repeats,
        "decisions": timing.decisions,
        "optimiser_steps": timing.optimiser_steps,
        "seconds": timing.seconds,
        "decisions_per_second": timing.decisions / timing.seconds,
        "optimiser_steps_per_second": timing.optimiser_steps / timing.seconds,
        "peak_memory_bytes": timing.peak_memory_bytes,
        "run_config_hash": config.hash,
        "minibatch_size": config.ppo.minibatch_size,
        "d_model": config.model.d_model,
        "n_layers": config.model.n_layers,
    }
