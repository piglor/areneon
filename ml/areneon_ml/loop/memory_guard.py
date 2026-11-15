"""An operational memory guard for a generation's play phase (#1394).

``--game-processes S`` starts ``S`` Node processes, each with ``--policy-workers K`` torch workers
(for a population run, per member that plays in its slice). At 9 processes a 4-member population
generation was OOM-killed on a 23 GB machine. Before play the driver compares what the request
needs with what the machine has and refuses, with a fixed message that names the largest ``S``
that fits. It never lowers ``S`` itself: the operator chooses. Nothing here is in a run config or
a hash, and nothing it reads or decides reaches a generation's files.

The estimate per game process is learned from the newest ``metrics/g*.json`` that reports both
``peak_rss_bytes.game_process`` and ``peak_rss_bytes.policy_worker`` (the largest of each, so it
is conservative), else the documented default (docs/training-run.md, sizing a run to a machine).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30
#: Without a measurement: a Node game process (0.6 GB with ``--node-heap-mb 640``) ...
DEFAULT_GAME_PROCESS_BYTES = 6 * GIB // 10
#: ... plus 0.43 GB for each policy worker it starts.
DEFAULT_POLICY_WORKER_BYTES = 43 * GIB // 100
#: Kept free for the driver, its learner and the load phase (docs: plan 1.5 to 2 GB).
DRIVER_RESERVE_BYTES = 2 * GIB
MEMINFO = Path("/proc/meminfo")


class MemoryGuardError(RuntimeError):
    """The requested game processes do not fit the memory available."""


@dataclass(frozen=True)
class Estimate:
    per_process_bytes: int
    learned: bool


def available_bytes(meminfo: Path = MEMINFO) -> int | None:
    """``MemAvailable`` plus ``SwapFree`` in bytes; ``None`` where ``meminfo`` cannot be read."""
    try:
        text = meminfo.read_text(encoding="ascii")
    except (OSError, ValueError):
        return None
    found: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if name in ("MemAvailable", "SwapFree") and parts and parts[0].isdigit():
            found[name] = int(parts[0]) * 1024
    if "MemAvailable" not in found:
        return None
    return found["MemAvailable"] + found.get("SwapFree", 0)


def _learned_peaks(metrics_dir: Path) -> tuple[int, int] | None:
    """The (game process, policy worker) peaks of the newest metrics file that has both."""
    for path in sorted(metrics_dir.glob("g*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        peaks = record.get("peak_rss_bytes") if isinstance(record, dict) else None
        if not isinstance(peaks, dict):
            continue
        game, worker = peaks.get("game_process"), peaks.get("policy_worker")
        if (
            isinstance(game, int)
            and isinstance(worker, int)
            and not isinstance(game, bool)
            and not isinstance(worker, bool)
            and game > 0
            and worker > 0
        ):
            return game, worker
    return None


def estimate(metrics_dir: Path, policy_workers: int, members: int) -> Estimate:
    """Bytes one game process needs: itself plus ``policy_workers`` workers per member."""
    peaks = _learned_peaks(metrics_dir)
    game, worker = peaks or (DEFAULT_GAME_PROCESS_BYTES, DEFAULT_POLICY_WORKER_BYTES)
    return Estimate(game + worker * policy_workers * members, peaks is not None)


def largest_fit(available: int, per_process: int) -> int:
    return max(0, (available - DRIVER_RESERVE_BYTES) // per_process)


def check(
    *,
    game_processes: int,
    policy_workers: int,
    members: int,
    metrics_dir: Path,
    meminfo: Path = MEMINFO,
) -> None:
    """Raise :class:`MemoryGuardError` when ``game_processes`` do not fit; else return."""
    available = available_bytes(meminfo)
    if available is None:
        return  # no /proc/meminfo (not Linux): nothing to compare with
    found = estimate(metrics_dir, policy_workers, members)
    fit = largest_fit(available, found.per_process_bytes)
    if game_processes <= fit:
        return
    source = "measured in the last generation" if found.learned else "default"
    raise MemoryGuardError(
        f"memory guard: {game_processes} game processes x {found.per_process_bytes >> 20} MB "
        f"({source}; {policy_workers} policy workers x {members} members each) do not fit "
        f"{available >> 20} MB available (MemAvailable + swap, less "
        f"{DRIVER_RESERVE_BYTES >> 20} MB for the driver): the largest "
        f"--game-processes that fits is {fit}; lower it, or pass --no-memory-guard"
    )
