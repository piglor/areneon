"""Shared pieces of the report tests: the committed fixture run, copies, and a progress root."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.league import league_game_seed
from areneon_ml.loop.play import PlayRequest
from areneon_ml.progress.run import progress_generation

from tests.ml.inspector.support import FIXTURE, edit
from tests.ml.loop.support import FakePlayer, make_loop
from tests.ml.progress.support import FakeProgressPlayer, committed_manifest, dataset_config

HOSTILE_CAUSE = "cause</script><img src=x onerror=alert(1)> \"q\" 's' & <!-- " + chr(0x2028)

TRAIN_REPORT = {
    "battle": {
        "approx_kl": 0.02,
        "clip_fraction": 0.2,
        "entropy": 3.5,
        "optimiser_steps": 12,
        "policy_loss": -0.01,
        "steps": 100,
        "value_loss": 0.04,
    },
    "preview": {
        "approx_kl": 0.05,
        "clip_fraction": 0.4,
        "entropy": 5.7,
        "optimiser_steps": 3,
        "policy_loss": -0.11,
        "steps": 20,
        "value_loss": 0.25,
    },
}


def fixture_copy(tmp_path: Path, *, train: bool = False, forfeit: bool = False) -> Path:
    """A writable copy of the inspector's fixture run, optionally with train reports."""
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run, ignore=shutil.ignore_patterns("pool.json", "README.md"))
    if train:
        for n in (1, 2):
            path = layout.generation_dir(run, n)
            layout.write_json(path / "train-report.json", _train(n))
            _reseal(run, n)
    if forfeit:
        make_forfeit(run, 1)
    return run


def _train(n: int) -> dict[str, Any]:
    out = json.loads(json.dumps(TRAIN_REPORT))
    out["battle"]["entropy"] = 3.5 - n / 10
    result: dict[str, Any] = out
    return result


def _reseal(run: Path, n: int) -> None:
    path = layout.generation_dir(run, n)
    manifest = layout.read_json(path / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(path)
    layout.write_json(path / layout.MANIFEST, manifest)


def make_forfeit(run: Path, n: int) -> None:
    """Game 0 of generation ``n`` becomes a forfeit of p1 with a hostile cause."""
    path = layout.generation_dir(run, n)
    gen = layout.read_json(path / "generation.json")
    row = gen["games"][0]
    row.update(outcome="forfeit", winner="p2", turns=2)
    row["forfeit"] = {"loser": "p1", "cause": HOSTILE_CAUSE, "request_seq": 1}
    gen["forfeits"] = 1
    gen["episode_hashes"] = gen["episode_hashes"][2:]
    layout.write_json(path / "generation.json", gen)
    league = layout.read_json(path / "league-run.json")
    league.update(forfeits=1, played=league["played"] - 1, episodes=len(gen["episode_hashes"]))
    layout.write_json(path / "league-run.json", league)
    _reseal(run, n)


class RichPlayer(FakePlayer):
    """The loop's fake simulator side, with the plan and rows of a real generation.

    Every generation plays two self-play games, one against each scripted baseline and, from
    generation 2 on, a forfeited game; the files have the real loop's shapes and keys.
    """

    def play(self, request: PlayRequest) -> None:
        super().play(request)
        config = self.config
        assert config is not None
        g = request.generation
        pair = request.policy_id
        kinds = [("self", 0), ("self", 1), ("self", 2), ("random", 3), ("actor", 4)]
        if g >= 2:
            kinds.append(("self", 5))
        plan, rows = [], []
        for n, (kind, _) in enumerate(kinds):
            seed = league_game_seed(config.run_key, g, n)
            seat = "both" if kind == "self" else "p1" if kind == "random" else "p2"
            opponent: Any = (
                "self"
                if kind == "self"
                else {
                    "actor": f"{kind}-rule" if kind == "actor" else "random-legal",
                    "policy_id": "actor-rule@v1" if kind == "actor" else "random-legal@v1",
                }
            )
            plan.append(
                {
                    "baseline_seed": n,
                    "game": n,
                    "opponent": opponent,
                    "policy_seat": seat,
                    "seed": seed,
                    "teams": {"p1": n, "p2": n + 1},
                }
            )
            name = pair if kind == "self" else opponent["policy_id"]
            row: dict[str, Any] = {"game": n, "opponent": name, "seed": seed}
            if n == 5:
                row.update(outcome="forfeit", turns=2, winner="p2")
                row["forfeit"] = {"loser": "p1", "cause": "latency_overrun", "request_seq": 3}
            else:
                row.update(outcome="played", turns=4 + n, winner=[None, "p1", "p2", "p1", "p1"][n])
            rows.append(row)
        forfeits = sum(1 for r in rows if r["outcome"] == "forfeit")
        played = len(rows) - forfeits
        hashes = sorted(hashlib.sha256(f"e{g}-{i}".encode()).hexdigest() for i in range(2 * played))
        out = request.out_dir
        summary = layout.read_json(out / "generation.json")
        summary.update(games=rows, forfeits=forfeits, episode_hashes=hashes)
        if forfeits:
            summary["forfeit_causes"] = {"latency_overrun": forfeits}
        layout.write_json(out / "generation.json", summary)
        record = layout.read_json(out / "league-run.json")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["opponent"]] = counts.get(r["opponent"], 0) + 1
        record.update(
            plan_hash=sha256_hex(jcs(plan)),
            games=len(rows),
            played=played,
            forfeits=forfeits,
            episodes=2 * played,
            opponents=[{"opponent_policy_id": k, "games": v} for k, v in sorted(counts.items())],
        )
        if forfeits:
            record["forfeit_causes"] = [{"cause": "latency_overrun", "count": forfeits}]
        layout.write_json(out / "league-run.json", record)
        layout.write_json(out / "plan.json", plan)


def make_run(run_dir: Path, generations: int = 3, config: RunConfig | None = None) -> RunConfig:
    """A committed run with realistic plans, whose config names a team dataset."""
    config = config or dataset_config()
    make_loop(run_dir, config, RichPlayer(config=config)).run(generations)
    return config


def progress_root(run: Path, root: Path, generations: tuple[int, ...] = (0, 1, 2)) -> Path:
    """A progress root for a run made by ``make_run``, written by the real command."""
    player = FakeProgressPlayer(_config(run), committed_manifest(run, 0))
    for g in generations:
        progress_generation(run, g, root, player, games=2, record_games=1)
    return root


def _config(run: Path) -> Any:
    return read_run_config(layout.read_json(run / layout.CONFIG))


__all__ = ["FIXTURE", "edit", "fixture_copy", "make_run", "progress_root"]
