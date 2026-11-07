"""Population-game replays: each seat's own Episode, read and checked (ADR-021 section 8).

The reading rules are the single lineage's (:mod:`.replay`): an Episode holds ONE player's own
channel, a game's replay is its two Episodes side by side, no hidden half is filled in, a
``|split|`` marker or a channel that is not protocol is refused. What a population adds is the
join: the two halves of a team pair share a battle seed, so a seat's Episode is found by seed, seat,
own team (the pool resolves the plan's index) and the policy that played the seat, in the directory
of the member that owns the seat (:mod:`.population_model`). The seat is labelled with who held it:
a learning member, a frozen past-pool snapshot or a scripted baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from areneon_contracts.contracts import ContractError, validate

from areneon_ml.inspector.model import refuse_non_public
from areneon_ml.inspector.pool import Pool
from areneon_ml.inspector.population_model import (
    _Episode,
    _episodes,
    _members_of,
    _past_file,
    _policies,
    read_population_config,
)
from areneon_ml.inspector.population_readers import (
    PopGameRow,
    PopPlanEntry,
    read_population_generation,
    read_population_plan,
)
from areneon_ml.inspector.readers import SEATS, GameRow, InspectError, PlanEntry
from areneon_ml.inspector.replay import (
    GameReplay,
    PopulationInfo,
    SeatView,
    _channel,
    _decisions,
)
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.league import (
    LeagueRecordError,
    check_population_plan,
    read_league_run_record_v2,
)


def _read_seat(
    path: Path, entry: PopPlanEntry, seat: str, row_winner: str | None, turns: int
) -> SeatView:
    """One Episode file, validated against the contract and the game it belongs to."""
    try:
        episode = json.loads(path.read_bytes())
        validate("episode", episode)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        raise InspectError(f"episodes/{path.name} is not a valid Episode: {exc}") from exc
    if episode["player"] != seat:
        raise InspectError(f"episodes/{path.name}: it is not the {seat} seat's Episode")
    lines = _channel(episode, path.name)
    result = episode["result"]
    if result.get("winner") != row_winner or result.get("turns") != turns:
        raise InspectError(f"episodes/{path.name}: its result differs from generation.json")
    held = entry.seats[seat]
    return SeatView(
        seat,
        held.label,
        lines,
        tuple(episode["team"]["team"]),
        _decisions(episode, lines, path.name),
        result["winner"],
        result["turns"],
    )


def _agree(where: str, entry: PopPlanEntry, row: PopGameRow) -> dict[str, str]:
    """The seat labels, once plan.json and generation.json tell one story about the game."""
    labels = {s: entry.seats[s].label for s in SEATS}
    if (row.seed, row.pair, row.half, row.kind, row.member, row.seats) != (
        entry.seed,
        entry.pair,
        entry.half,
        entry.kind,
        entry.member,
        labels,
    ):
        raise InspectError(f"{where}: plan.json and generation.json differ")
    return labels


def _views(  # noqa: PLR0913, PLR0917
    where: str,
    entry: PopPlanEntry,
    row: PopGameRow,
    policies: dict[str, str],
    found: dict[tuple[str, str, str, str], _Episode],
    pool: Pool,
) -> tuple[SeatView, ...]:
    """Each seat's Episode, found by seed, seat, own team and the policy that played the seat."""
    views = []
    for seat in SEATS:
        if entry.teams[seat] >= len(pool.teams):
            raise InspectError(f"{where}: a team index is outside the pool")
        key = (row.seed, seat, pool.teams[entry.teams[seat]].hash, policies[seat])
        episode = found.get(key)
        if episode is None:
            raise InspectError(f"{where}: it needs one Episode per seat")
        views.append(_read_seat(episode.path, entry, seat, row.winner, row.turns))
    return tuple(views)


def read_population_games(
    run_dir: Path,
    generation: int,
    games: list[int] | None,
    pool: Pool,
    member: str | None = None,
) -> tuple[RunConfig, list[GameReplay]]:
    """The replays of ``games`` (None: every played game) of one population generation.

    ``member`` keeps the games in which that learning member holds a seat. The pool resolves each
    seat's team, which tells the two teams of a self-play pair (swapped between its halves) apart.
    """
    config = read_population_config(run_dir)
    refuse_non_public(config)
    members = _members_of(config)
    if member is not None and member not in members:
        raise InspectError(f"the run has no member {member}")
    if generation < 1:
        raise InspectError("generation 0 (the seeded pairs) plays no games")
    path = layout.generation_dir(run_dir, generation)
    try:
        layout.verify_generation(path, generation)
        plan_raw = layout.read_json(path / "plan.json")
        record = read_league_run_record_v2(layout.read_json(path / "league-run.json"), config)
        check_population_plan(record, plan_raw, config)
        gen_file = read_population_generation(
            layout.read_json(path / "generation.json"), generation
        )
        previous = layout.generation_dir(run_dir, generation - 1)
        layout.verify_generation(previous, generation - 1)
        played_pool = _past_file(previous, generation - 1, config)
    except (layout.RunDirError, LeagueRecordError, OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"g{generation:06d}: {exc}") from exc
    plan = read_population_plan(plan_raw)
    if record.generation != generation or len(plan) != len(gen_file.games):
        raise InspectError(f"g{generation:06d}: plan.json and generation.json differ in game count")
    wanted = list(range(len(plan))) if games is None else games
    for index in wanted:
        if not 0 <= index < len(plan):
            raise InspectError(f"g{generation:06d} has no game {index} (it has {len(plan)})")
    found = _episodes(path, gen_file.members, config)
    replays = []
    for index in wanted:
        entry, row = plan[index], gen_file.games[index]
        where = f"g{generation:06d} game {index}"
        labels = _agree(where, entry, row)
        if member is not None and member not in labels.values():
            continue
        if row.outcome == "forfeit":
            if games is None:
                continue
            raise InspectError(f"{where} was forfeited: it has no replay")
        policies = _policies(entry, record, played_pool)
        views = _views(where, entry, row, policies, found, pool)
        replays.append(_game(config, generation, entry, row, views))
    return config, replays


def _game(
    config: RunConfig,
    generation: int,
    entry: PopPlanEntry,
    row: PopGameRow,
    views: tuple[SeatView, ...],
) -> GameReplay:
    """A population game in the shape the replay page reads."""
    shell = PlanEntry(entry.game, entry.seed, row.opponent, None, "both", entry.teams)
    plain = GameRow(row.game, row.seed, row.opponent, row.outcome, row.winner, row.turns, None)
    holders = {s: (entry.seats[s].kind, entry.seats[s].label) for s in SEATS}
    info = PopulationInfo(entry.kind, entry.member, entry.pair, entry.half, holders)
    return GameReplay(config.run_key, generation, shell, plain, "", views, info)
