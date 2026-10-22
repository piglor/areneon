"""The run-metrics record of one generation (ADR-010 erratum A1, item 3).

Wall-clock time, throughput and peak accelerator memory differ on every run, so they cannot live in
a file the generation manifest hashes: a CPU rerun must stay byte-identical. They go to
``<run-dir>/metrics/g<NNNNNN>.json``, written by the driver after the generation commits, outside
every manifest and hash. Nothing in the loop, the loader or the learner reads the file; only the
run dashboard does, and it lists the file's SHA-256 among its inputs. A crash between the commit and
the write loses that generation's timings and no resume writes them again (the dashboard says
``not recorded``); the file has no tamper evidence of its own.

The record is the ``RunMetrics`` contract (integers only). A rate is the count times 3,600,000
divided by the phase's milliseconds, truncated, and 0 when the phase took no time. A CUDA
learner's generation that trains nothing (generation 0) records a peak of 0. A failed write
raises out of ``Loop.run`` after the commit: the generation is whole and a resume continues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, validate

from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout

METRICS_DIR = "metrics"
PHASES = ("play", "write", "load", "train", "seal")
_MS_PER_HOUR = 3_600_000


class MetricsError(ValueError):
    """A RunMetrics record breaks the contract or a reader rule."""


def metrics_path(run_dir: Path, generation: int) -> Path:
    return run_dir / METRICS_DIR / f"{layout.generation_name(generation)}.json"


def per_hour(count: int, ms: int) -> int:
    """``count`` events in ``ms`` milliseconds as whole events per hour (0 if no time passed)."""
    return count * _MS_PER_HOUR // ms if ms > 0 else 0


@dataclass(frozen=True)
class Counts:
    """What a generation processed: games, decisions trained on, optimiser steps taken."""

    games: int = 0
    decisions: int = 0
    optimiser_steps: int = 0


def build_metrics(
    config: RunConfig,
    generation: int,
    phase_ms: dict[str, int],
    counts: Counts,
    peak_memory: int | None,
) -> dict[str, Any]:
    """The RunMetrics value for one committed generation."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_config_hash": config.hash,
        "generation": generation,
        "phase_ms": {name: phase_ms.get(name, 0) for name in PHASES},
        "counts": {
            "games": counts.games,
            "decisions": counts.decisions,
            "optimiser_steps": counts.optimiser_steps,
        },
        "per_hour": {
            "games": per_hour(counts.games, phase_ms.get("play", 0)),
            "decisions": per_hour(counts.decisions, phase_ms.get("train", 0)),
            "optimiser_steps": per_hour(counts.optimiser_steps, phase_ms.get("train", 0)),
        },
        "peak_accelerator_memory_bytes": (peak_memory or 0) if config.device == "cuda" else None,
        "device": config.device,
        "threads": config.threads,
    }
    validate("run-metrics", record)
    return record


def write_metrics(run_dir: Path, record: dict[str, Any]) -> Path:
    """Write ``metrics/g<NNNNNN>.json`` (canonical JSON) so it holds all of the record or none."""
    path = metrics_path(run_dir, record["generation"])
    fresh = not path.parent.is_dir()
    path.parent.mkdir(exist_ok=True)
    if fresh:
        layout.sync_dir(run_dir)
    layout.write_atomic(path, jcs(record) + b"\n")
    return path


def read_run_metrics(raw: object, config: RunConfig, generation: int) -> dict[str, Any]:
    """Validate ``raw`` against the contract, then tie it to the run, generation and its rates."""
    try:
        validate("run-metrics", raw)
    except ContractError as exc:
        raise MetricsError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    if raw["run_config_hash"] != config.hash:
        raise MetricsError("run_config_hash is not the config's hash")
    if raw["generation"] != generation:
        raise MetricsError("the record is another generation's")
    phases, counts, rates = raw["phase_ms"], raw["counts"], raw["per_hour"]
    expected = {
        "games": per_hour(counts["games"], phases["play"]),
        "decisions": per_hour(counts["decisions"], phases["train"]),
        "optimiser_steps": per_hour(counts["optimiser_steps"], phases["train"]),
    }
    if rates != expected:
        raise MetricsError("per_hour is not the count over the phase's time")
    if raw["device"] != config.device or raw["threads"] != config.threads:
        raise MetricsError("device and threads are not the config's")
    if (raw["device"] == "cpu") != (raw["peak_accelerator_memory_bytes"] is None):
        raise MetricsError("peak accelerator memory is null exactly on a CPU learner")
    return raw
