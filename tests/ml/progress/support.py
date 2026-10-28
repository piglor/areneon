"""Shared pieces of the progress tests: a run with a team dataset and a stand-in simulator side.

``FakeProgressPlayer`` plays no battle. It writes what the real simulator side writes
(``progress.json`` and the stored games), built from the contract corpus's samples, for the
frozen pair and the games the request names, so the command, the checks on the files and the
commit protocol run for real while the engine does not.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.progress.records import (
    RANDOM_PREVIEW_POLICY,
    log_hash,
    record_hash,
    sum_baseline_counts,
)
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
DAMAGE = ("damage-aware", "damage-aware@v1")
#: The stand-in baseline identity hash of the damage-aware matches.
IDENTITY = "9" * 64


def baseline_counts(stored: dict[str, Any], seat: str, game: int) -> dict[str, int]:
    """The stand-in per-game counts of a damage-aware seat: its decisions are the stored game's."""
    answered = sum(1 for d in stored["decisions"] if d["player"] == seat)
    preview = min(1, answered)
    return {
        "decisions": answered,
        "preview_decisions": preview,
        "scored_decisions": answered - preview,
        "queries": 3 + game,
        "probe_runs": 2,
        "unsupported_queries": game % 2,
        "reader_fallbacks": 0,
        "assumed_set_fallbacks": 0,
        "epsilon_plays": 0,
    }


def dataset_config(
    run_key: str | None = None,
    seeds: tuple[int, int] | None = None,
    dataset_hash: str = "5" * 64,
    learner_seed: int | None = None,
) -> RunConfig:
    """The tiny config, naming a team dataset (the Node side reads it; Python only checks it).

    ``run_key`` and ``seeds`` (the two initialisation seeds) make another run's config."""

    def edit(raw: dict[str, Any]) -> None:
        raw["team_dataset"] = {"path": str(DATASET), "dataset_hash": dataset_hash}
        if run_key is not None:
            raw["seeds"]["run_key"] = run_key
        if seeds is not None:
            raw["seeds"]["init_preview"], raw["seeds"]["init_battle"] = seeds
        if learner_seed is not None:
            raw["seeds"]["learner"] = learner_seed

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
    stored["pair"] = game // 2  # every pair is played from both seats (version 2)
    stored["seed"] = progress_game_seed(request.evaluation_key or config.run_key, game // 2)
    if request.evaluation_key is not None:
        stored["evaluation_key"] = request.evaluation_key
    stored["policies"] = {"p1": pair, "p2": opponent}
    return stored


ABLATION_RESULTS: dict[str, Callable[[int], str]] = {
    # The stand-in's outcome of game ``g`` per Preview variant: learned splits every pair, a
    # random Preview loses every game and generation 0's wins every game.
    "learned": lambda g: "candidate" if g % 2 == 0 else "opponent",
    "random": lambda g: "opponent",
    "g0": lambda g: "candidate",
}


def add_ablation(  # noqa: PLR0913, PLR0917
    config: RunConfig,
    request: ProgressRequest,
    games: int,
    matches: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    stored: dict[str, dict[str, Any]],
) -> None:
    """The six matches of a Preview ablation: three Previews against the two baselines, argmax."""
    spec = request.ablation
    assert spec is not None
    previews: dict[str, tuple[str, dict[str, Any]]] = {
        "learned": (request.preview, {"kind": "learned"}),
        "random": (RANDOM_PREVIEW_POLICY, {"kind": "random"}),
        "g0": (
            spec.preview,
            {"kind": "generation", "generation": spec.generation, "checkpoint": spec.preview},
        ),
    }
    for variant, (preview_id, candidate_preview) in previews.items():
        mine = policy_id(preview_id, request.battle, "argmax", features=features_hash(config))
        for actor, baseline in BASELINES:
            match_id = f"preview-{variant}-{actor}-argmax"
            hashes = []
            for game in range(min(request.record_games, games)):
                found = game_file(config, request, match_id, game, mine, baseline)
                found["omniscient_log_sha256"] = log_hash(found["omniscient_log"])
                stored[f"{match_id}-{game}.json"] = found
                hashes.append(record_hash(found))
            wins = 0
            for game in range(games):
                winner = ABLATION_RESULTS[variant](game)
                wins += winner == "candidate"
                rows.append(
                    {
                        "match_id": match_id,
                        "game": game,
                        "pair": game // 2,
                        "candidate_seat": "p1" if game % 2 == 0 else "p2",
                        "outcome": {"kind": "played", "winner": winner, "turns": 4},
                    }
                )
            matches.append(
                {
                    "match_id": match_id,
                    "policy_id": mine,
                    "opponent": {"kind": "baseline", "policy_id": baseline},
                    "mode": "argmax",
                    "candidate_preview": candidate_preview,
                    "games": games,
                    "wins": wins,
                    "ties": 0,
                    "losses": games - wins,
                    "turn_cap_ties": 0,
                    "turns_total": 4 * games,
                    "forfeits": [],
                    "game_record_hashes": sorted(hashes),
                }
            )


def progress_files(  # noqa: PLR0912, PLR0915
    config: RunConfig,
    request: ProgressRequest,
    manifest: dict[str, Any],
    games: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The record and the stored games the simulator side would write for ``request``."""
    games = request.games if games is None else games
    stored: dict[str, dict[str, Any]] = {}
    matches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    opponents: list[tuple[str, str, dict[str, Any] | None]] = [
        (actor, baseline, None) for actor, baseline in BASELINES
    ]
    if request.damage_aware_budget_ms is not None:
        opponents.append((*DAMAGE, None))
    if request.previous is not None:
        opponents.append(("previous", "", {}))
    anchors = {a.generation: a for a in request.anchors}
    for generation in sorted(anchors):
        opponents.append((f"anchor-g{generation}", "", {"anchor": generation}))
    if request.versus is not None:
        opponents.append(("run", "", {"run": True}))
    if request.ablation is not None:
        opponents = []
        add_ablation(config, request, games, matches, rows, stored)
    for _actor, baseline, previous in opponents:
        for mode in MODES:
            match_id = f"{_actor}-{mode}"
            pair = policy_id(request.preview, request.battle, mode, features=features_hash(config))
            other = baseline
            opponent: dict[str, Any] = {"kind": "baseline", "policy_id": baseline}
            count = request.anchor_games if previous and "anchor" in previous else games
            if previous is not None and "run" in previous and request.versus is not None:
                count = request.versus.games
                versus = request.versus
                other = versus.policy_id_sample if mode == "sample" else versus.policy_id_argmax
                opponent = {
                    "kind": "run",
                    "run_id": versus.run_id,
                    "run_config_hash": versus.run_config_hash,
                    "generation": versus.generation,
                    "policy_id": other,
                    "checkpoints": {"preview": versus.preview, "battle": versus.battle},
                }
            elif previous is not None and "anchor" in previous:
                found_anchor = anchors[previous["anchor"]]
                other = policy_id(
                    found_anchor.preview, found_anchor.battle, mode, features=features_hash(config)
                )
                opponent = {
                    "kind": "anchor",
                    "generation": found_anchor.generation,
                    "policy_id": other,
                    "checkpoints": {"preview": found_anchor.preview, "battle": found_anchor.battle},
                }
            elif previous is not None and request.previous is not None:
                other = policy_id(
                    request.previous.preview,
                    request.previous.battle,
                    mode,
                    features=features_hash(config),
                )
                opponent = {
                    "kind": "previous",
                    "policy_id": other,
                    "checkpoints": {
                        "preview": request.previous.preview,
                        "battle": request.previous.battle,
                    },
                }
            hashes = []
            damage = baseline == DAMAGE[1]
            per_game: list[dict[str, int]] = []
            for game in range(count):
                seat = "p2" if game % 2 == 0 else "p1"
                if damage:
                    sampled = game_file(config, request, match_id, game, pair, other)
                    per_game.append(baseline_counts(sampled, seat, game))
                if game >= min(request.record_games, count):
                    continue
                found = game_file(config, request, match_id, game, pair, other)
                if damage:
                    # The baseline holds the seat the candidate does not.
                    found["policies"] = (
                        {"p1": other, "p2": pair} if seat == "p1" else found["policies"]
                    )
                    found["baseline"] = {
                        "seat": seat,
                        "identity_hash": IDENTITY,
                        "counts": baseline_counts(found, seat, game),
                    }
                found["omniscient_log_sha256"] = log_hash(found["omniscient_log"])
                stored[f"{match_id}-{game}.json"] = found
                hashes.append(record_hash(found))
            for game in range(count):
                outcome = (
                    {"kind": "played", "winner": "candidate", "turns": 5}
                    if game % 2 == 0
                    else {"kind": "played", "winner": "opponent", "turns": 3}
                )
                rows.append(
                    {
                        "match_id": match_id,
                        "game": game,
                        "pair": game // 2,
                        "candidate_seat": "p1" if game % 2 == 0 else "p2",
                        "outcome": outcome,
                        **({"baseline_counts": per_game[game]} if damage else {}),
                    }
                )
            wins = (count + 1) // 2
            matches.append(
                {
                    "match_id": match_id,
                    "policy_id": pair,
                    "opponent": opponent,
                    "mode": mode,
                    "games": count,
                    "wins": wins,
                    "ties": 0,
                    "losses": count - wins,
                    "turn_cap_ties": 0,
                    "turns_total": 5 * wins + 3 * (count - wins),
                    "forfeits": [],
                    "game_record_hashes": sorted(hashes),
                    **(
                        {
                            "baseline": {
                                "identity_hash": IDENTITY,
                                "counts": sum_baseline_counts(per_game),
                            }
                        }
                        if damage
                        else {}
                    ),
                }
            )
    run_games = 0 if request.versus is None else request.versus.games
    record = {
        "schema_version": 2,
        "data_class": "synthetic",
        "run_id": config.run_key,
        "run_config_hash": config.hash,
        "generation": request.generation,
        "generation_manifest_sha256": request.manifest_sha256,
        "checkpoints": {"preview": request.preview, "battle": request.battle},
        "version_manifest": manifest,
        "teams": {
            "dataset_hash": "5" * 64,
            **(
                {"evaluation_dataset_hash": request.eval_dataset.dataset_hash}
                if request.eval_dataset is not None
                else {}
            ),
            "split": "validation",
            "pairs_used": max(games, request.anchor_games, run_games) // 2,
        },
        "plan": {
            "seed_derivation": "progress-seed/1",
            "pairing": "progress-pairing/1",
            "games_per_match": games,
            "recorded_games_per_match": request.record_games,
            **({"anchor_games_per_match": request.anchor_games} if anchors else {}),
            **({"run_games_per_match": run_games} if request.versus is not None else {}),
            **(
                {"evaluation_key": request.evaluation_key}
                if request.evaluation_key is not None
                else {}
            ),
            "max_turns": config.generation.max_turns,
            "latency_budget_ms": {
                "preview_ms": config.generation.preview_ms,
                "battle_ms": config.generation.battle_ms,
            },
        },
        "matches": sorted(matches, key=lambda m: str(m["match_id"])),
        "games": sorted(rows, key=lambda r: (str(r["match_id"]), int(str(r["game"])))),
    }
    return record, stored


Change = Callable[[Path, ProgressRequest], None]


class FakeProgressPlayer:
    """The simulator side of the progress evaluation, without a simulator."""

    def __init__(
        self,
        config: RunConfig,
        manifest: dict[str, Any],
        change: Change | None = None,
        *,
        previous: bool = True,
    ) -> None:
        self.previous = previous  # False plays like slice 4: baselines only
        self.config = config
        self.manifest = manifest
        self.change = change
        self.requests: list[ProgressRequest] = []

    def progress(self, request: ProgressRequest) -> dict[str, Any]:
        self.requests.append(request)
        files = request if self.previous else dataclasses.replace(request, previous=None)
        record, stored = progress_files(self.config, files, self.manifest)
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


def downgrade_to_v1(generation_dir: Path) -> None:
    """Rewrite a committed baseline-only version 2 generation as a version 1 record (one game per
    pair, one seed per game) and reseal its manifest: the shape a record written before ADR-017
    erratum E2 has. Every game keeps its stand-in outcome; ``pair`` is the game index."""
    record = layout.read_json(generation_dir / "progress.json")
    assert record["schema_version"] == 2
    record["schema_version"] = 1
    record["teams"]["pairs_used"] = record["plan"]["games_per_match"]
    for row in record["games"]:
        row["pair"] = row["game"]
    keep: dict[str, str] = {}
    for path in sorted((generation_dir / "games").glob("*.json")):
        stored = layout.read_json(path)
        before = record_hash(stored)
        stored.pop("pair", None)
        stored["seed"] = progress_game_seed(stored["run_id"], stored["game"])
        layout.write_json(path, stored)
        keep[before] = record_hash(stored)
    for match in record["matches"]:
        match["game_record_hashes"] = sorted(keep[h] for h in match["game_record_hashes"])
    layout.write_json(generation_dir / "progress.json", record)
    manifest = layout.read_json(generation_dir / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(generation_dir)
    layout.write_json(generation_dir / layout.MANIFEST, manifest)


__all__ = [
    "FakeProgressPlayer",
    "committed_manifest",
    "downgrade_to_v1",
    "jcs",
    "make_run",
    "reseal",
]
