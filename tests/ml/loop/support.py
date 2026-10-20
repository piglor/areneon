"""Shared pieces of the loop tests: a stand-in simulator side and run helpers.

``FakePlayer`` plays no battle. It writes what the real simulator side writes (the
loader's stream and TrainingSet, the per-game summary) from the committed random-play
channels, for the policy the driver names, so the driver, the learner and the commit
protocol run for real while the engine does not.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Loop
from areneon_ml.loop.league import league_game_seed
from areneon_ml.loop.play import PlayRequest

from tests.ml.learner.support import episode_lines, stream_of, tiny_config
from tests.ml.model.support import game, sample

MANIFEST = sample("version-manifest/valid/golden.json")


def league_files(
    config: RunConfig, generation: int, pair: str, games: int = 2
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A match plan of ``games`` self-play games and the LeagueRunRecord that names it."""
    plan = [
        {"game": n, "seed": league_game_seed(config.run_key, generation, n), "opponent": "self"}
        for n in range(games)
    ]
    record = {
        "schema_version": 1,
        "run_id": config.run_key,
        "run_config_hash": config.hash,
        "generation": generation,
        "seed_derivation": "league-seed/1",
        "plan_hash": sha256_hex(jcs(plan)),
        "pair_policy_id": pair,
        "games": games,
        "played": games,
        "forfeits": 0,
        "episodes": 2 * games,
        "opponents": [{"opponent_policy_id": pair, "games": games}],
    }
    return plan, record


class FakePlayer:
    """The simulator side without a simulator."""

    def __init__(
        self, change: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    ) -> None:
        self.change = change
        self.requests: list[PlayRequest] = []

    def manifest(self, format_id: str) -> dict[str, Any]:
        assert format_id == tiny_config().format_id
        return dict(MANIFEST)

    def play(self, request: PlayRequest) -> None:
        self.requests.append(request)
        generation = request.generation
        picker = random.Random(generation)
        examples = [e for player in ("p1", "p2") for e in episode_lines(generation, player, picker)]
        text, training_set = stream_of(examples, request.policy_id)
        summary = {
            "schema_version": 1,
            "generation": generation,
            "games": [],
            "forfeits": 0,
            "episode_hashes": training_set["episode_hashes"],
            "training_set_hash": sha256_hex(jcs(training_set)),
            "examples": training_set["examples"],
            "manifest": dict(MANIFEST),
            "team_hashes": sorted(
                hashlib.sha256(f"team-{generation}-{n}".encode()).hexdigest() for n in range(2)
            ),
        }
        if self.change is not None:
            self.change(summary, training_set)
        out = request.out_dir
        plan, record = league_files(tiny_config(), generation, request.policy_id)
        layout.write_json(out / "plan.json", plan)
        layout.write_json(out / "league-run.json", record)
        (out / "stream.jsonl").write_text(text, encoding="utf-8")
        layout.write_json(out / "training-set.json", training_set)
        layout.write_json(out / "generation.json", summary)


class Killed(Exception):
    """Raised by a stage hook to stop a run where a SIGKILL would."""


def killer(generation: int, stage: str) -> Callable[[int, str], None]:
    def hook(g: int, s: str) -> None:
        if (g, s) == (generation, stage):
            raise Killed

    return hook


def make_loop(
    run_dir: Path,
    config: RunConfig | None = None,
    player: FakePlayer | None = None,
    on_stage: Callable[[int, str], None] | None = None,
) -> Loop:
    return Loop(config or tiny_config(), run_dir, game(), player or FakePlayer(), on_stage)


def manifests(run_dir: Path) -> dict[str, bytes]:
    """Every committed generation's manifest bytes, by directory name."""
    base = layout.generations_dir(run_dir)
    return {
        p.name: (p / layout.MANIFEST).read_bytes()
        for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    }


def reseal(run_dir: Path, generation: int) -> None:
    """Recompute a committed generation's manifest after a test edited a file in it."""
    path = layout.generation_dir(run_dir, generation)
    manifest = json.loads((path / layout.MANIFEST).read_text())
    manifest["files"] = layout.hash_tree(path)
    layout.write_json(path / layout.MANIFEST, manifest)
