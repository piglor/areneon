"""Shared inputs of the model tests: the golden format's game data and real request points."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from areneon_ml.model.encoder import Encoder, EncoderConfig
from areneon_ml.model.gamedata import GameData

ROOT = Path(__file__).resolve().parents[3]
FORMAT_DIR = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc"
SAMPLES = ROOT / "fixtures" / "contracts" / "v1"
A2_DIR = ROOT / "fixtures" / "a2" / "gen9championsvgc2026regmc"
RANDOM = sorted((FORMAT_DIR / "random").glob("b*.json"))

# A small model: the sizes are configuration, the tests run fast on CPU.
TINY_ENCODER = EncoderConfig(volatile_buckets=4, condition_buckets=4)


@cache
def table() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FORMAT_DIR / "engine-data.json").read_text())
    return loaded


@cache
def game() -> GameData:
    return GameData.from_table(table())


@cache
def encoder() -> Encoder:
    return Encoder(game(), TINY_ENCODER)


def sample(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SAMPLES / name).read_text())
    return loaded


def observation(
    player: str, lines: list[str], team: list[dict[str, Any]], kind: str
) -> dict[str, Any]:
    """A minimal observation: only the fields the encoder may read."""
    return {"kind": kind, "player": player, "log": lines, "team": {"team": team}}


def request_points(lines: list[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """(index, body) of every request that asks for a decision."""
    for index, line in enumerate(lines):
        if line.startswith("|request|"):
            body = json.loads(line[len("|request|") :])
            if not body.get("wait"):
                yield index, body


def _slot_options(body: dict[str, Any], slot: int) -> list[str]:
    force = body.get("forceSwitch")
    active = body.get("active")
    options = ["pass"] if force and not force[slot] else []
    if force and force[slot]:
        listed = len(body["side"]["pokemon"])
        options += [f"switch {n}" for n in range(3, listed + 1)]
    elif active:
        count = len(active[slot]["moves"])
        for n in range(1, count + 1):
            options += [f"move {n}", *(f"move {n} {t}" for t in (1, 2, -1, -2))]
        options += [f"move 1 {t} mega" for t in (1, -1)]
        options += [f"switch {n}" for n in range(3, len(body["side"]["pokemon"]) + 1)]
    return options or ["pass"]


def candidates_for(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Syntactic joint choices for a request: a test input only, not a legality oracle."""
    if body.get("teamPreview"):
        orders = ["team 1234", "team 6543", "team 3126"]
    else:
        orders = [
            f"{a}, {b}"
            for a, b in itertools.product(_slot_options(body, 0)[:9], _slot_options(body, 1)[:9])
        ]
    return [{"choice": order, "pending": number % 3 == 0} for number, order in enumerate(orders)]
