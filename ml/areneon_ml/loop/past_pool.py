"""The frozen past pool of a population run (ADR-021 sections 3 and 4, slice 10).

Pure functions over plain data (no I/O, no torch, no floats); :mod:`areneon_ml.loop.population`
reads and writes the files. The TypeScript twin is ``packages/showdown-bridge/src/past-pool.ts``
(the plan draws the opponents there); both are held to the same vectors.

**Membership.** A snapshot ``m<k>@g<G>`` is member ``k``'s frozen pair at the end of generation
``G``. Every ``past_pool.snapshot_every`` generations (generation 0 included: the seeded or
cloned start pairs) each member's pair joins the pool. The pool is capped at ``max_size``, oldest
evicted first, but generation-0 pairs are never evicted (so the pool can exceed the cap by exactly
those). Eviction is therefore a function of the committed generation number alone
(:func:`pool_keys`).

**The committed file.** ``past-pool.json`` sits in every committed generation directory (written
only when the config has a past-pool share). The file of generation ``G`` holds the pool the next
generation plays, with each snapshot's checkpoints and policy id, and each member's pair-score
counts against each pool snapshot over the rating window ending at ``G``. The plan of generation
``G + 1`` reads nothing else about the pool, so it is a pure function of committed bytes and a
killed generation replays the same plan. The counts come from the same window rows as
``ladder.json``; they are a separate file because the ladder record can only list a snapshot that
already has a frozen strength, while the weights need the counts of young ones too.

**PFSP f_hard.** ``w = max(floor_ppm, (10^6 - s)^p // 10^(6 (p - 1)))`` with ``s`` the member's
mean pair score against the snapshot (``score_sum_ppm // pairs``); a snapshot with no games gets
10^6, the largest weight the rule gives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from areneon_ml.learner.config import RunConfig
from areneon_ml.loop.population_seed import population_draw

FORMAT = "past-pool/1"
FILE = "past-pool.json"
#: ``opponent-draw`` indices of the past pool, clear of the ladder's ``0 .. pairs - 1``.
SELECT_BASE = 1_000_000
ASSIGN_BASE = 2_000_000
FULL_PPM = 1_000_000
BUCKETS = 5
SCORE_STEP_PPM = 250_000


class PastPoolError(ValueError):
    """The past pool or its file breaks a rule."""


def _fail(message: str) -> None:
    raise PastPoolError(message)


@dataclass(frozen=True)
class Snapshot:
    """One frozen pair in the pool."""

    member: str
    generation: int
    policy_id: str
    preview: str
    battle: str

    @property
    def id(self) -> str:
        return snapshot_id(self.member, self.generation)

    def to_json(self) -> dict[str, Any]:
        return {
            "battle": self.battle,
            "generation": self.generation,
            "id": self.id,
            "member": self.member,
            "policy_id": self.policy_id,
            "preview": self.preview,
        }


def snapshot_id(member: str, generation: int) -> str:
    return f"{member}@g{generation}"


def member_index(member: str) -> int:
    return int(member[1:])


@dataclass(frozen=True)
class PastPool:
    """A validated ``past-pool.json``."""

    generation: int
    snapshots: tuple[Snapshot, ...]
    cells: dict[tuple[str, str], tuple[int, int, int, int, int]]


def settings(config: RunConfig) -> dict[str, int]:
    found: dict[str, int] = config.raw["population"]["past_pool"]
    return found


def enabled(config: RunConfig) -> bool:
    """The run plays a past pool: its config has a non-zero past-pool share."""
    return bool(config.raw.get("population")) and (
        config.raw["population"]["shares"]["past_pool_ppm"] != 0
    )


def pool_keys(
    members: int, snapshot_every: int, max_size: int, generation: int
) -> list[tuple[int, int]]:
    """``(generation, member index)`` of the pool after generation ``generation`` is committed.

    Snapshots are taken at every multiple of ``snapshot_every`` up to ``generation``; the newest
    ``max_size - (generation-0 pairs)`` of the others stay (oldest evicted first) and the
    generation-0 pairs are never evicted.
    """
    taken = [
        (g, k) for g in range(0, generation + 1) if g % snapshot_every == 0 for k in range(members)
    ]
    zero = [key for key in taken if key[0] == 0]
    rest = [key for key in taken if key[0] != 0]
    room = max(0, max_size - len(zero))
    return zero + (rest[-room:] if room else [])


def next_snapshots(
    config: RunConfig,
    previous: Sequence[Snapshot],
    generation: int,
    frozen: Mapping[str, Snapshot],
) -> tuple[Snapshot, ...]:
    """The pool after ``generation``: the previous pool, less evictions, plus new snapshots.

    ``frozen`` maps each member to its pair at the end of ``generation`` as a snapshot (used only
    when ``generation`` is a snapshot generation).
    """
    population = config.raw["population"]
    s = settings(config)
    keys = pool_keys(population["members"], s["snapshot_every"], s["max_size"], generation)
    held = {(p.generation, member_index(p.member)): p for p in previous}
    if generation % s["snapshot_every"] == 0:
        for member, snapshot in frozen.items():
            if snapshot.generation != generation or snapshot.member != member:
                _fail("a new snapshot must be its member's pair at the generation being sealed")
            held[(generation, member_index(member))] = snapshot
    missing = [key for key in keys if key not in held]
    if missing:
        lost = snapshot_id(f"m{missing[0][1]}", missing[0][0])
        _fail(f"the pool lacks the committed snapshot {lost}")
    return tuple(held[key] for key in keys)


# -- PFSP -------------------------------------------------------------------------------------


def pfsp_weight(floor_ppm: int, power: int, scores: Sequence[int] | None) -> int:
    """The ``f_hard`` weight in integers (see the module docstring)."""
    pairs = sum(scores) if scores is not None else 0
    if scores is None or pairs < 1:
        return FULL_PPM
    total = SCORE_STEP_PPM * sum(i * c for i, c in enumerate(scores))
    s = total // pairs
    weight: int = (FULL_PPM - s) ** power // 10 ** (6 * (power - 1))
    return max(floor_ppm, weight)


def pfsp_weights(
    config: RunConfig,
    pool: PastPool,
    member: str,
    candidates: Sequence[Snapshot],
) -> list[int]:
    s = settings(config)
    return [
        pfsp_weight(s["floor_ppm"], s["pfsp_power"], pool.cells.get((member, c.id)))
        for c in candidates
    ]


def _pick(weights: Sequence[int], draw: int) -> int:
    total = sum(weights)
    equal = total == 0
    at = draw % (len(weights) if equal else total)
    for index, weight in enumerate(weights):
        at -= 1 if equal else weight
        if at < 0:
            return index
    return len(weights) - 1


def draw_past(  # noqa: PLR0913, PLR0917
    config: RunConfig,
    generation: int,
    member: str,
    pairs: int,
    current_policy_id: str,
    pool: PastPool,
) -> list[Snapshot]:
    """The snapshot of each of ``member``'s past pairs (the twin of ``drawPast``)."""
    candidates = [c for c in pool.snapshots if c.policy_id != current_policy_id]
    if not candidates or pairs == 0:
        return []
    weights = pfsp_weights(config, pool, member, candidates)
    run_key = config.run_key
    wanted = min(settings(config)["max_past_per_member"], len(candidates), pairs)
    remaining = list(range(len(candidates)))
    chosen: list[int] = []
    for j in range(wanted):
        at = _pick(
            [weights[i] for i in remaining],
            population_draw(run_key, generation, member, "opponent-draw", SELECT_BASE + j),
        )
        chosen.append(remaining.pop(at))
    chosen.sort()
    chosen_weights = [weights[i] for i in chosen]
    return [
        candidates[
            chosen[
                _pick(
                    chosen_weights,
                    population_draw(run_key, generation, member, "opponent-draw", ASSIGN_BASE + i),
                )
            ]
        ]
        for i in range(pairs)
    ]


# -- the file ---------------------------------------------------------------------------------


def build_file(
    config: RunConfig,
    generation: int,
    snapshots: Iterable[Snapshot],
    cells: Mapping[tuple[str, str], Sequence[int]],
) -> dict[str, Any]:
    """The ``past-pool.json`` of a committed generation."""
    ordered = sorted(snapshots, key=lambda p: (p.generation, member_index(p.member)))
    ids = {p.id for p in ordered}
    return {
        "cells": [
            {"member": m, "scores": list(scores), "snapshot": s}
            for (m, s), scores in sorted(
                cells.items(), key=lambda kv: (member_index(kv[0][0]), kv[0][1])
            )
            if s in ids and sum(scores) > 0
        ],
        "format": FORMAT,
        "generation": generation,
        "run_config_hash": config.hash,
        "run_id": config.run_key,
        "schema_version": 1,
        "snapshots": [p.to_json() for p in ordered],
    }


def read_file(raw: object, config: RunConfig) -> PastPool:
    """Check a ``past-pool.json`` against the config (the rules ``readPastPool`` checks)."""
    if not isinstance(raw, dict) or sorted(raw) != [
        "cells",
        "format",
        "generation",
        "run_config_hash",
        "run_id",
        "schema_version",
        "snapshots",
    ]:
        _fail("past-pool.json is not a past-pool file")
    assert isinstance(raw, dict)
    if raw["schema_version"] != 1 or raw["format"] != FORMAT:
        _fail("past-pool.json is not a version 1 past-pool file")
    if raw["run_id"] != config.run_key or raw["run_config_hash"] != config.hash:
        _fail("past-pool.json belongs to another run or config")
    population = config.raw["population"]
    s = settings(config)
    generation = raw["generation"]
    if not isinstance(generation, int) or generation < 0:
        _fail("past-pool.json: generation must be a non-negative integer")
    snapshots = []
    for entry in raw["snapshots"]:
        snap = Snapshot(
            entry["member"],
            entry["generation"],
            entry["policy_id"],
            entry["preview"],
            entry["battle"],
        )
        if entry != snap.to_json() or member_index(snap.member) >= population["members"]:
            _fail("past-pool.json: a snapshot is malformed")
        snapshots.append(snap)
    expected = pool_keys(population["members"], s["snapshot_every"], s["max_size"], generation)
    if [(p.generation, member_index(p.member)) for p in snapshots] != expected:
        _fail("past-pool.json: the snapshots are not the pool the config gives at that generation")
    ids = {p.id for p in snapshots}
    cells: dict[tuple[str, str], tuple[int, int, int, int, int]] = {}
    last: tuple[int, str] | None = None
    for cell in raw["cells"]:
        scores = cell["scores"]
        key = (member_index(cell["member"]), cell["snapshot"])
        if (
            sorted(cell) != ["member", "scores", "snapshot"]
            or cell["snapshot"] not in ids
            or member_index(cell["member"]) >= population["members"]
            or len(scores) != BUCKETS
            or any(not isinstance(n, int) or n < 0 for n in scores)
            or sum(scores) < 1
            or (last is not None and not last < key)
        ):
            _fail("past-pool.json: a cell is malformed, out of order or names an unknown player")
        last = key
        cells[(cell["member"], cell["snapshot"])] = (
            scores[0],
            scores[1],
            scores[2],
            scores[3],
            scores[4],
        )
    return PastPool(generation, tuple(snapshots), cells)
