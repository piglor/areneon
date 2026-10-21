"""The training pool, listed by the loop's own TypeScript pool function and read back strictly.

``packages/team-dataset/scripts/print-pool.ts`` rebuilds the pool exactly as the loop's play
script does (the dataset reader and the ``excluded_abilities`` filter, run by Node) and prints it
as one canonical JSON line. This module runs it and reads that line with closed key sets.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_ml.inspector.readers import (
    InspectError,
    array,
    closed,
    hash64,
    integer,
    text,
)

#: The six stat keys of the engine's stat table (Stat Points in the Champions mod).
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
SCRIPT = "packages/team-dataset/scripts/print-pool.ts"
LISTING_KEYS: tuple[set[str], set[str]] = (
    {"schema_version", "source", "dataset_hash", "records_before", "excluded_abilities", "teams"},
    set(),
)
TEAM_KEYS: tuple[set[str], set[str]] = (
    {"hash", "canonical_team_hash", "labels", "split", "paste", "sets"},
    set(),
)
SET_KEYS: tuple[set[str], set[str]] = (
    {
        "name",
        "species",
        "item",
        "ability",
        "nature",
        "gender",
        "level",
        "moves",
        "evs",
        "mega_stone",
    },
    set(),
)


@dataclass(frozen=True)
class PoolSet:
    name: str
    species: str
    item: str
    ability: str
    nature: str
    gender: str
    level: int
    moves: tuple[str, ...]
    evs: tuple[tuple[str, int], ...]
    mega_stone: bool


@dataclass(frozen=True)
class PoolTeam:
    hash: str
    canonical_team_hash: str | None
    labels: tuple[tuple[str, str], ...]
    split: str | None
    paste: str
    sets: tuple[PoolSet, ...]


@dataclass(frozen=True)
class Pool:
    source: str
    dataset_hash: str | None
    records_before: int
    excluded_abilities: tuple[str, ...]
    teams: tuple[PoolTeam, ...]


def _read_set(raw: object, what: str) -> PoolSet:
    body = closed(raw, SET_KEYS, what)
    if not isinstance(body["mega_stone"], bool):
        raise InspectError(f"{what} mega_stone is not a boolean")
    evs = closed(body["evs"], (set(STATS), set()), f"{what} evs")
    return PoolSet(
        text(body["name"], f"{what} name"),
        text(body["species"], f"{what} species"),
        text(body["item"], f"{what} item"),
        text(body["ability"], f"{what} ability"),
        text(body["nature"], f"{what} nature"),
        text(body["gender"], f"{what} gender"),
        integer(body["level"], f"{what} level"),
        tuple(text(m, f"{what} move") for m in array(body["moves"], f"{what} moves")),
        tuple((k, integer(v, f"{what} evs {k}")) for k, v in evs.items()),
        body["mega_stone"],
    )


def read_pool(raw: object) -> Pool:
    body = closed(raw, LISTING_KEYS, "the pool listing")
    if body["schema_version"] != 1 or body["source"] not in ("dataset", "interim"):
        raise InspectError("the pool listing is not schema 1 of a dataset or interim pool")
    dataset_hash = body["dataset_hash"]
    if (body["source"] == "dataset") != (dataset_hash is not None):
        raise InspectError("only a dataset pool has a dataset hash")
    if dataset_hash is not None:
        hash64(dataset_hash, "dataset_hash")
    teams = []
    for index, item in enumerate(array(body["teams"], "pool teams")):
        what = f"pool team {index}"
        team = closed(item, TEAM_KEYS, what)
        canonical = team["canonical_team_hash"]
        if canonical is not None:
            hash64(canonical, f"{what} canonical hash")
        labels = []
        for label in array(team["labels"], f"{what} labels"):
            pair = closed(label, ({"label_id", "value"}, set()), f"{what} label")
            labels.append(
                (
                    text(pair["label_id"], f"{what} label_id"),
                    text(pair["value"], f"{what} label value"),
                )
            )
        split = team["split"]
        sets = tuple(_read_set(s, f"{what} set") for s in array(team["sets"], f"{what} sets"))
        teams.append(
            PoolTeam(
                hash64(team["hash"], f"{what} hash"),
                canonical,
                tuple(labels),
                None if split is None else text(split, f"{what} split"),
                text(team["paste"], f"{what} paste"),
                sets,
            )
        )
    hashes = [t.hash for t in teams]
    if hashes != sorted(set(hashes)) or not hashes:
        raise InspectError("the pool is empty or not sorted and unique by team hash")
    return Pool(
        body["source"],
        dataset_hash,
        integer(body["records_before"], "records_before"),
        tuple(
            text(a, "excluded ability")
            for a in array(body["excluded_abilities"], "excluded_abilities")
        ),
        tuple(teams),
    )


def load_pool(repo: Path, node: str, config_path: Path) -> Pool:
    """Run the pool script; the current directory is the base of a relative dataset path."""
    command = [node, str(repo / SCRIPT), "--config", str(config_path)]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL
        )
    except OSError as exc:
        raise InspectError(f"cannot start {node}: {exc}") from exc
    if done.returncode != 0:
        raise InspectError(f"{SCRIPT} failed (exit {done.returncode}): {done.stderr[-2000:]}")
    return read_pool_text(done.stdout)


def read_pool_text(output: str) -> Pool:
    try:
        raw: Any = json.loads(output)
    except json.JSONDecodeError as exc:
        raise InspectError("the pool script did not print JSON") from exc
    return read_pool(raw)


__all__ = ["Pool", "PoolSet", "PoolTeam", "load_pool", "read_pool", "read_pool_text"]
