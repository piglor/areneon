"""Shared pieces of the progress tests: a run with a team dataset and a stand-in simulator side.

``FakeProgressPlayer`` plays no battle. It writes what the real simulator side writes
(``progress.json`` and the stored games), built from the contract corpus's samples, for the
frozen pair and the games the request names, so the command, the checks on the files and the
commit protocol run for real while the engine does not.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.records import policy_id
from areneon_ml.progress.records import log_hash, record_hash
from areneon_ml.progress.run import ProgressRequest
from areneon_ml.progress.seeds import progress_game_seed

from tests.ml.learner.support import edited_config
from tests.ml.loop.support import FakePlayer, make_loop, reseal

DATASET = (
    Path(__file__).resolve().parents[3]
    / "fixtures/teams/gen9championsvgc2026regmc/datasets/fixtures-v1"
)
SAMPLES = Path(__file__).resolve().parents[3] / "fixtures" / "contracts" / "v1"
BASELINES = (("actor-rule", "actor-rule@v1"), ("random-legal", "random-legal@v1"))
MODES = ("argmax", "sample")


def dataset_config() -> RunConfig:
    """The tiny config, naming a team dataset (the Node side reads it; Python only checks it)."""

    def edit(raw: dict[str, Any]) -> None:
        raw["team_dataset"] = {"path": str(DATASET), "dataset_hash": "5" * 64}

    return edited_config(edit)


def sample(name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text())
    return found


def make_run(run_dir: Path, generations: int = 2, config: RunConfig | None = None) -> RunConfig:
    """A committed run (the loop with its fake simulator side) whose config names a team dataset."""
    config = config or dataset_config()
    make_loop(run_dir, config, FakePlayer(config=config)).run(generations)
    return config


def game_file(  # noqa: PLR0913, PLR0917
    config: RunConfig, request: ProgressRequest, match_id: str, game: int, pair: str, opponent: str
) -> dict[str, Any]:
    """A stored game: the corpus's played game under this run, match and game index."""
    stored = copy.deepcopy(sample("progress-game-record/valid/played"))
    stored["run_id"] = config.run_key
    stored["generation"] = request.generation
    stored["match_id"] = match_id
    stored["game"] = game
    stored["seed"] = progress_game_seed(config.run_key, game)
    stored["policies"] = {"p1": pair, "p2": opponent}
    return stored


def progress_files(
    config: RunConfig,
    request: ProgressRequest,
    manifest: dict[str, Any],
    games: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The record and the stored games the simulator side would write for ``request``."""
    games = request.games if games is None else games
    stored: dict[str, dict[str, Any]] = {}
    matches = []
    rows = []
    for _actor, baseline in BASELINES:
        for mode in MODES:
            match_id = f"{_actor}-{mode}"
            pair = policy_id(request.preview, request.battle, mode)
            hashes = []
            for game in range(request.record_games):
                found = game_file(config, request, match_id, game, pair, baseline)
                found["omniscient_log_sha256"] = log_hash(found["omniscient_log"])
                stored[f"{match_id}-{game}.json"] = found
                hashes.append(record_hash(found))
            for game in range(games):
                outcome = (
                    {"kind": "played", "winner": "candidate", "turns": 5}
                    if game % 2 == 0
                    else {"kind": "played", "winner": "opponent", "turns": 3}
                )
                rows.append(
                    {
                        "match_id": match_id,
                        "game": game,
                        "pair": game,
                        "candidate_seat": "p1" if game % 2 == 0 else "p2",
                        "outcome": outcome,
                    }
                )
            wins = (games + 1) // 2
            matches.append(
                {
                    "match_id": match_id,
                    "policy_id": pair,
                    "opponent": {"kind": "baseline", "policy_id": baseline},
                    "mode": mode,
                    "games": games,
                    "wins": wins,
                    "ties": 0,
                    "losses": games - wins,
                    "turn_cap_ties": 0,
                    "turns_total": 5 * wins + 3 * (games - wins),
                    "forfeits": [],
                    "game_record_hashes": sorted(hashes),
                }
            )
    record = {
        "schema_version": 1,
        "data_class": "synthetic",
        "run_id": config.run_key,
        "run_config_hash": config.hash,
        "generation": request.generation,
        "generation_manifest_sha256": request.manifest_sha256,
        "checkpoints": {"preview": request.preview, "battle": request.battle},
        "version_manifest": manifest,
        "teams": {"dataset_hash": "5" * 64, "split": "validation", "pairs_used": games},
        "plan": {
            "seed_derivation": "progress-seed/1",
            "pairing": "progress-pairing/1",
            "games_per_match": games,
            "recorded_games_per_match": request.record_games,
            "max_turns": config.generation.max_turns,
            "latency_budget_ms": {
                "preview_ms": config.generation.preview_ms,
                "battle_ms": config.generation.battle_ms,
            },
        },
        "matches": sorted(matches, key=lambda m: str(m["match_id"])),
        "games": rows,
    }
    return record, stored


Change = Callable[[Path, ProgressRequest], None]


class FakeProgressPlayer:
    """The simulator side of the progress evaluation, without a simulator."""

    def __init__(
        self, config: RunConfig, manifest: dict[str, Any], change: Change | None = None
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.change = change
        self.requests: list[ProgressRequest] = []

    def progress(self, request: ProgressRequest) -> dict[str, Any]:
        self.requests.append(request)
        record, stored = progress_files(self.config, request, self.manifest)
        out = request.out_dir
        layout.write_json(out / "progress.json", record)
        for name, found in stored.items():
            layout.write_json(out / "games" / name, found)
        if self.change is not None:
            self.change(out, request)
        return {"wall_ms": {"fake": 1}}


def committed_manifest(run_dir: Path, generation: int) -> dict[str, Any]:
    found: dict[str, Any] = layout.read_json(
        layout.generation_dir(run_dir, generation) / "records" / "battle.json"
    )["manifest"]
    return found


__all__ = ["FakeProgressPlayer", "committed_manifest", "jcs", "make_run", "reseal"]
