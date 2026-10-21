"""Training-game replays: the two players' Episodes of one game, read and checked (ADR-017 4a).

An Episode holds ONE player's own channel. A game's replay is its two Episodes side by side, joined
by battle seed and seat (never by file order); nothing of either channel's hidden half is filled
in. Every refusal is an :class:`InspectError`:

* the run is not a public, synthetic one, or a generation's manifest does not match its files;
* ``plan.json``, ``generation.json`` and ``league-run.json`` disagree, or the game is not planned;
* the game was forfeited (it has no Episodes, so no replay);
* an Episode is not a valid synthetic league Episode of this run (the contract schema, the labels,
  the file's own hash), holds a ``|split|`` marker, a channel that is not protocol lines, or a
  decision for a request the channel does not have;
* a game has not exactly one Episode per seat, or an Episode's result differs from the file's.

The team hashes are not checked against the pool here (that needs Node); ``inspect`` does it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, validate

from areneon_ml.inspector.model import (
    _episodes_by_seed,
    read_run_dir_config,
    refuse_non_public,
)
from areneon_ml.inspector.readers import (
    SEATS,
    GameRow,
    InspectError,
    PlanEntry,
    read_generation,
    read_plan,
)
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.league import LeagueRecordError, check_generation_plan, read_league_run_record

_NON_ID = re.compile(r"[^a-z0-9]")
#: A page name is built only from ids that match this (ADR-017 4c).
GAME_NAME = "game-{game:06d}"
GAME_FILE = re.compile(r"game-(\d{6})\.html")
GEN_DIR = re.compile(r"g(\d{6})")


@dataclass(frozen=True)
class Decision:
    choice: str
    request_seq: int
    policy: str


@dataclass(frozen=True)
class SeatView:
    """One player's own side of a game: its channel, team and decisions."""

    seat: str
    #: Who played the seat: the frozen pair, or the scripted baseline's policy id.
    played_by: str
    lines: tuple[str, ...]
    team: tuple[dict[str, Any], ...]
    decisions: tuple[Decision, ...]
    winner: str | None
    turns: int


@dataclass(frozen=True)
class GameReplay:
    run_key: str
    generation: int
    plan: PlanEntry
    row: GameRow
    pair_policy_id: str
    views: tuple[SeatView, ...]


def to_id(text: str) -> str:
    return _NON_ID.sub("", text.lower())


def _played_by(plan: PlanEntry, seat: str, pair: str) -> str:
    return plan.opponent if plan.policy_seat not in ("both", seat) else pair


def _channel(episode: dict[str, Any], name: str) -> tuple[str, ...]:
    lines = episode["channel"]
    if not all(isinstance(x, str) and x.startswith("|") for x in lines):
        raise InspectError(f"episodes/{name}: its channel holds a line that is not protocol")
    if any(x.startswith("|split|") for x in lines):
        raise InspectError(f"episodes/{name}: its channel holds a |split| marker (a secret half)")
    return tuple(lines)


def _decisions(episode: dict[str, Any], lines: tuple[str, ...], name: str) -> tuple[Decision, ...]:
    requests = sum(1 for x in lines if x.startswith("|request|"))
    out = []
    for item in episode["decisions"]:
        seq = item["identity"]["request_seq"]
        if not 1 <= seq <= requests:
            raise InspectError(f"episodes/{name}: a decision answers a request the channel lacks")
        out.append(Decision(item["choice"], seq, item["source_policy_id"]))
    if len({d.request_seq for d in out}) != len(out):
        raise InspectError(f"episodes/{name}: two decisions answer one request")
    return tuple(out)


def read_seat(path: Path, run_key: str, plan: PlanEntry, pair: str, row: GameRow) -> SeatView:
    """One Episode file, validated against the contract and the game it belongs to."""
    try:
        episode = json.loads(path.read_bytes())
        validate("episode", episode)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        raise InspectError(f"episodes/{path.name} is not a valid Episode: {exc}") from exc
    seat = episode["player"]
    lines = _channel(episode, path.name)
    result = episode["result"]
    if result.get("winner") != row.winner or result.get("turns") != row.turns:
        raise InspectError(f"episodes/{path.name}: its result differs from generation.json")
    return SeatView(
        seat,
        _played_by(plan, seat, pair),
        lines,
        tuple(episode["team"]["team"]),
        _decisions(episode, lines, path.name),
        result["winner"],
        result["turns"],
    )


def _generation_files(
    run_dir: Path, generation: int, config: RunConfig
) -> tuple[Path, tuple[PlanEntry, ...], Any, str]:
    if generation < 1:
        raise InspectError("generation 0 (the seeded pair) plays no games")
    path = layout.generation_dir(run_dir, generation)
    try:
        layout.verify_generation(path, generation)
        plan_raw = layout.read_json(path / "plan.json")
        record = read_league_run_record(layout.read_json(path / "league-run.json"), config)
        check_generation_plan(record, plan_raw, config)
        gen_file = read_generation(layout.read_json(path / "generation.json"), generation)
    except (layout.RunDirError, LeagueRecordError, OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"g{generation:06d}: {exc}") from exc
    plan = read_plan(plan_raw)
    if len(plan) != len(gen_file.games):
        raise InspectError(f"g{generation:06d}: plan.json and generation.json differ in game count")
    return path, plan, gen_file, record.pair_policy_id


def read_games(
    run_dir: Path, generation: int, games: list[int] | None
) -> tuple[RunConfig, list[GameReplay]]:
    """The replays of ``games`` (None: every played game) of one generation, read and checked."""
    config = read_run_dir_config(run_dir)
    refuse_non_public(config)
    path, plan, gen_file, pair = _generation_files(run_dir, generation, config)
    wanted = list(range(len(plan))) if games is None else games
    for index in wanted:
        if not 0 <= index < len(plan):
            raise InspectError(f"g{generation:06d} has no game {index} (it has {len(plan)})")
    by_seed = _episodes_by_seed(path, gen_file.episode_hashes, config.run_key)
    out = []
    for index in wanted:
        entry, row = plan[index], gen_file.games[index]
        if row.seed != entry.seed or row.opponent != (
            pair if entry.opponent == "self" else entry.opponent
        ):
            raise InspectError(
                f"g{generation:06d} game {index}: plan.json and generation.json differ"
            )
        if row.outcome == "forfeit":
            if games is None:
                continue
            raise InspectError(f"g{generation:06d} game {index} was forfeited: it has no replay")
        files = by_seed.get(row.seed, {})
        if sorted(files) != sorted(SEATS):
            raise InspectError(f"g{generation:06d} game {index}: it needs one Episode per seat")
        views = tuple(read_seat(files[seat], config.run_key, entry, pair, row) for seat in SEATS)
        out.append(GameReplay(config.run_key, generation, entry, row, pair, views))
    return config, out
