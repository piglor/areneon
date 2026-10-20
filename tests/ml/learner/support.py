"""Shared inputs of the learner tests: the tiny run config and learner-ready streams.

The streams are built from the committed random-play channels. Every example is
a valid ``training-example`` (the learner reads them through ``read_stream``);
the candidates are the test helper's syntactic joint choices, which is all the
encoder needs, and the chosen candidate is drawn from a seeded generator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable
from functools import cache
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.network import ROLES, PolicyNetwork, init_network

from tests.ml.model.support import (
    RANDOM,
    SAMPLES,
    candidates_for,
    game,
    request_points,
    sample,
)

POLICY = "policy:" + "ab" * 32


@cache
def _tiny() -> RunConfig:
    return read_run_config(
        json.loads((SAMPLES / "training-run-config/valid/tiny.json").read_text())
    )


def tiny_config(**ppo: Any) -> RunConfig:
    """The tiny sample config, with PPO ``*_micro`` fields replaced by ``ppo``."""
    raw = copy.deepcopy(_tiny().raw)
    raw["ppo"].update(ppo)
    return read_run_config(raw)


def edited_config(edit: Callable[[dict[str, Any]], None]) -> RunConfig:
    """The tiny sample config after ``edit`` changed a copy of its record."""
    raw = copy.deepcopy(_tiny().raw)
    edit(raw)
    return read_run_config(raw)


def networks(config: RunConfig) -> dict[str, PolicyNetwork]:
    seeds = {"preview": config.init_preview, "battle": config.init_battle}
    return {role: init_network(role, game(), config.model, seeds[role]) for role in ROLES}


def learner(config: RunConfig | None = None) -> Learner:
    config = config or tiny_config()
    return Learner(config, networks(config), torch.device("cpu"))


def episode_lines(
    index: int, player: str, picker: random.Random, outcome: str | None = None
) -> list[dict[str, Any]]:
    """The examples of one random-corpus battle from ``player``'s seat."""
    battle = json.loads(RANDOM[index].read_text())
    lines = battle["channels"][player]["lines"]
    if outcome is None:
        outcome = (
            "tie" if battle["winner"] is None else "win" if battle["winner"] == player else "loss"
        )
    episode = hashlib.sha256(f"{RANDOM[index].name}/{player}".encode()).hexdigest()
    team = {"schema_version": 1, "data_class": "synthetic", "team": battle["teams"][player]}
    base = sample("training-example/valid/preview.json")
    out = []
    for position, body in request_points(lines):
        seq = sum(1 for line in lines[: position + 1] if line.startswith("|request|"))
        kind = "preview" if body.get("teamPreview") else "battle"
        candidates = candidates_for(body)
        example = copy.deepcopy(base)
        example["episode_hash"] = episode
        example["observation"].update(
            {
                "kind": kind,
                "player": player,
                "log": lines[: position + 1],
                "team": team,
                "identity": {
                    "request_seq": seq,
                    "request_hash": hashlib.sha256(str(seq).encode()).hexdigest(),
                },
            }
        )
        example["candidates"] = candidates
        example["choice"] = picker.choice(candidates)["choice"]
        example["outcome"] = outcome
        out.append(example)
    return out


def make_stream(
    episodes: list[tuple[int, str]] | None = None, seed: int = 0, policy: str = POLICY
) -> tuple[str, dict[str, Any]]:
    """A learner-ready stream and its manifest, Episodes ordered by hash."""
    picker = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for index, player in episodes or [(0, "p1"), (0, "p2"), (1, "p1")]:
        examples += episode_lines(index, player, picker)
    return stream_of(examples, policy)


def stream_of(examples: list[dict[str, Any]], policy: str = POLICY) -> tuple[str, dict[str, Any]]:
    # Episodes in hash order, each Episode's decisions in request order (the loader's order).
    ordered = sorted(
        examples, key=lambda e: (e["episode_hash"], e["observation"]["identity"]["request_seq"])
    )
    text = "".join(jcs(example).decode("utf-8") + "\n" for example in ordered)
    manifest = {
        "schema_version": 1,
        "data_class": "synthetic",
        "episode_hashes": sorted({e["episode_hash"] for e in ordered}),
        "evaluation_manifest_hashes": [],
        "learner_ready": True,
        "behaviour_policy_id": policy,
        "examples": len(ordered),
        "stream_sha256": sha256_hex(text.encode("utf-8")),
    }
    return text, manifest
