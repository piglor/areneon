"""The loop share ``L`` of a profiled generation (ADR-028 section 1, #1191).

``L = c * play_wall / T``: the part of a generation's time an on-device loop could replace.
``T`` is the generation's whole time (play + load + train + write + seal, from ``phase_ms``);
``c`` is the CPU share of ``{E_sim, E_judge, O_obs, W_enc, W_fwd}`` among all the CPU the game
processes and the policy workers spent while playing. Play may be latency-bound rather than
CPU-bound, so ``c`` is computed twice and the smaller is used:

* from CPU time: ``(E_sim + E_judge + O_obs) + (W_enc + W_fwd CPU)`` over the game processes' CPU
  (less their pool load and merge, which are not play) plus the workers' CPU. The game side
  figures are wall time of code that runs to completion without waiting, which is its CPU time
  up to scheduling; the worker side figures are thread CPU time;
* from per-request wall attribution: ``(E_sim + E_judge + O_obs + IPC wait)`` over the same plus
  ``E_probe`` (the time a request spent in the probe, the feature builder and the scripted
  baselines). The IPC wait contains the worker's own time, so the workers' figures are not added.

``E_probe`` is never part of ``L`` (ADR-016: the probe stays on the pinned engine). Everything
is integers: shares and ``L`` in parts per million, truncated. The command reads the metrics
files only and writes nothing. A generation without a ``profile`` is reported as missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, validate

from areneon_ml.loop.metrics import METRICS_DIR, MetricsError, check_profile

_PPM = 1_000_000


def _share(part: int, whole: int) -> int:
    return min(_PPM, part * _PPM // whole) if whole > 0 else 0


def loop_share(record: dict[str, Any]) -> dict[str, Any]:
    """The breakdown and ``L`` of one validated ``RunMetrics`` record that carries a profile."""
    profile = record["profile"]
    games, workers, load = profile["game_processes"], profile["policy_workers"], profile["load"]
    phases = record["phase_ms"]
    total = sum(phases.values())
    play = phases["play"]
    on_loop_game = games["e_sim_ms"] + games["e_judge_ms"] + games["o_obs_ms"]
    worker_loop_cpu = workers["w_enc_cpu_ms"] + workers["w_fwd_cpu_ms"]
    game_play_cpu = max(games["cpu_ms"] - games["pool_load_ms"] - games["merge_ms"], on_loop_game)
    c_cpu = _share(on_loop_game + worker_loop_cpu, game_play_cpu + workers["cpu_ms"])
    c_wall = _share(
        on_loop_game + games["ipc_wait_ms"],
        on_loop_game + games["ipc_wait_ms"] + games["e_probe_ms"],
    )
    c = min(c_cpu, c_wall)
    return {
        "generation": record["generation"],
        "run_config_hash": record["run_config_hash"],
        "t_ms": total,
        "phase_ms": phases,
        "decisions": profile["decisions"],
        "c_cpu_ppm": c_cpu,
        "c_wall_ppm": c_wall,
        "c_ppm": c,
        "c_source": "cpu" if c_cpu <= c_wall else "wall",
        "l_ppm": c * play // total if total > 0 else 0,
        "play_cpu_ms": {
            "e_sim": games["e_sim_ms"],
            "e_judge": games["e_judge_ms"],
            "o_obs": games["o_obs_ms"],
            "e_probe": games["e_probe_ms"],
            "w_enc": workers["w_enc_cpu_ms"],
            "w_fwd": workers["w_fwd_cpu_ms"],
            "game_processes_cpu": games["cpu_ms"],
            "policy_workers_cpu": workers["cpu_ms"],
        },
        "play_wall_ms": {
            "ipc_wait": games["ipc_wait_ms"],
            "w_enc": workers["w_enc_ms"],
            "w_fwd": workers["w_fwd_ms"],
        },
        "serial_ms": {"pool_load": games["pool_load_ms"], "merge": games["merge_ms"]},
        "load_ms": {k: load[f"{k}_ms"] for k in ("validate", "reduce", "encode")},
        "load_cpu_ms": {k: load[f"{k}_cpu_ms"] for k in ("validate", "reduce", "encode")},
        "processes": {"game": games["processes"], "workers": workers["workers"]},
    }


def run_loop_share(run_dir: Path, generations: tuple[int, ...] | None = None) -> dict[str, Any]:
    """``loop_share`` for each profiled generation of a run (or the named ones), oldest first."""
    folder = run_dir / METRICS_DIR
    if not folder.is_dir():
        raise MetricsError("the run has no metrics directory")
    wanted = sorted(set(generations)) if generations else None
    found: list[dict[str, Any]] = []
    missing: list[int] = []
    for path in sorted(folder.glob("g*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        generation = raw.get("generation") if isinstance(raw, dict) else None
        if not isinstance(generation, int) or (wanted is not None and generation not in wanted):
            continue
        try:
            validate("run-metrics", raw)
        except ContractError as exc:
            raise MetricsError(f"{path.name}: {exc}") from exc
        if "profile" not in raw:
            if generation > 0:  # the seeded pair plays nothing, so it has nothing to time
                missing.append(generation)
            continue
        check_profile(raw["profile"])
        found.append(loop_share(raw))
    return {"run_dir": str(run_dir), "generations": found, "without_profile": missing}
