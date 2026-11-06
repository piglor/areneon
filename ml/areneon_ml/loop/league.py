"""The LeagueRunRecord reader and the league seed check (ADR-009 §3, ADR-010 §4).

The contract fixes ``seed_derivation`` to ``league-seed/1``, so a record of another version
fails the schema.

``league-seed/1`` makes every battle seed from the run key, the generation and
the game index: SHA-256 over the JCS of ``{run_key, generation, game, purpose}``,
the first 16 bytes as ``sodium,<32 hex>``. The TS side derives them and writes
the match plan; this module derives them again, independently, so the loop
refuses a generation whose plan carries a seed the run key does not give.
A seed is the simulator side's value only: it is never passed to a policy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.learner.config import RunConfig

VERSION_2 = 2


class LeagueRecordError(ValueError):
    """A LeagueRunRecord or a match plan breaks a rule."""


def _seed_of(run_key: str, generation: int, game: int) -> str:
    body = {"run_key": run_key, "generation": generation, "game": game, "purpose": "league-game"}
    return "sodium," + str(sha256_hex(jcs(body)))[:32]


def league_game_seed(run_key: str, generation: int, game: int) -> str:
    """The battle seed of one league game (``league-seed/1``)."""
    if generation < 1 or game < 0:
        raise ValueError("a generation counts from 1 and a game from 0")
    return _seed_of(run_key, generation, game)


def calibration_game_seed(run_key: str, pair: int) -> str:
    """The battle seed of calibration pair ``pair``: ``league-seed/1`` at generation 0.

    No training plan uses generation 0, so a calibration pair never shares a seed with a training
    pair (ADR-021 section 4; the twin of the TS ``calibrationGameStreams``).
    """
    if pair < 0:
        raise ValueError("a pair counts from 0")
    return _seed_of(run_key, 0, pair)


@dataclass(frozen=True)
class LeagueRunRecord:
    """A validated record of one generation."""

    run_id: str
    run_config_hash: str
    generation: int
    plan_hash: str
    pair_policy_id: str
    games: int
    played: int
    forfeits: int
    episodes: int
    opponents: tuple[tuple[str, int], ...]
    forfeit_causes: tuple[tuple[str, int], ...]
    raw: dict[str, Any]


def read_league_run_record(raw: object, config: RunConfig) -> LeagueRunRecord:
    """Validate ``raw`` against the contract, the reader rules and the run config."""
    try:
        validate("league-run-record", raw)
    except ContractError as exc:
        raise LeagueRecordError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    if raw["played"] + raw["forfeits"] != raw["games"]:
        raise LeagueRecordError("played + forfeits must equal games")
    if raw["episodes"] != 2 * raw["played"]:
        raise LeagueRecordError("episodes must be two per played game")
    opponents = tuple((o["opponent_policy_id"], o["games"]) for o in raw["opponents"])
    if not all(a[0] < b[0] for a, b in itertools.pairwise(opponents)):
        raise LeagueRecordError("opponents must be sorted by opponent_policy_id, each once")
    if sum(games for _, games in opponents) != raw["games"]:
        raise LeagueRecordError("the opponents' games must sum to games")
    causes = tuple((c["cause"], c["count"]) for c in raw.get("forfeit_causes", []))
    if not all(a[0] < b[0] for a, b in itertools.pairwise(causes)):
        raise LeagueRecordError("forfeit_causes must be sorted by cause, each once")
    if "forfeit_causes" in raw and sum(count for _, count in causes) != raw["forfeits"]:
        raise LeagueRecordError("the forfeit causes' counts must sum to forfeits")
    _check_baselines(raw)
    if raw["run_id"] != config.run_key:
        raise LeagueRecordError(f"run_id {raw['run_id']} is not the config's run key")
    if raw["run_config_hash"] != config.hash:
        raise LeagueRecordError("run_config_hash is not the config's hash")
    expected = sorted(
        (b.policy_id, b.games)
        for b in config.generation.baselines
        if b.actor == "damage-aware" and b.games > 0
    )
    actual = sorted((e["policy_id"], e["games"]) for e in raw.get("baselines", []))
    if expected != actual:
        raise LeagueRecordError(
            "baselines must list exactly the config's damage-aware baselines, with their games"
        )
    return LeagueRunRecord(
        run_id=raw["run_id"],
        run_config_hash=raw["run_config_hash"],
        generation=raw["generation"],
        plan_hash=raw["plan_hash"],
        pair_policy_id=raw["pair_policy_id"],
        games=raw["games"],
        played=raw["played"],
        forfeits=raw["forfeits"],
        episodes=raw["episodes"],
        opponents=opponents,
        forfeit_causes=causes,
        raw=raw,
    )


def _check_baselines(raw: dict[str, Any], what: str = "") -> None:
    """The ``baselines`` rules (ADR-018): sorted, each id once, games and counts agree.

    ``what`` is ``"a member's "`` when ``raw`` is a version 2 member entry."""
    baselines = raw.get("baselines")
    if baselines is None:
        return
    ids = [b["policy_id"] for b in baselines]
    if not all(a < b for a, b in itertools.pairwise(ids)):
        raise LeagueRecordError(f"{what}baselines must be sorted by policy_id, each once")
    games_by_opponent = {
        o["opponent_policy_id"]: o["games"] for o in raw["opponents"] if "member" not in o
    }
    for entry in baselines:
        if games_by_opponent.get(entry["policy_id"]) != entry["games"]:
            label = f"{what}baseline" if what else "a baseline"
            raise LeagueRecordError(f"{label}'s games must equal its opponent entry's games")
        counts = entry["counts"]
        queries = counts["queries"]
        if (
            counts["probe_runs"] > queries
            or counts["unsupported_queries"] > queries
            or counts.get("probe_refusals", 0) > queries
        ):
            raise LeagueRecordError(
                "a baseline's probe runs, unsupported queries and probe refusals"
                " cannot exceed its queries"
            )
        if counts["preview_decisions"] + counts["scored_decisions"] > counts["decisions"]:
            raise LeagueRecordError(
                "a baseline's preview and scored decisions cannot exceed its decisions"
            )


@dataclass(frozen=True)
class MemberEntry:
    """One member or exploiter seat of a version 2 record."""

    member: str
    pair_policy_id: str
    preview: str
    battle: str
    games: int
    played: int
    forfeits: int
    episodes: int
    opponents: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LeagueRunRecordV2:
    """A validated version 2 record of one population generation (ADR-021 section 8)."""

    run_id: str
    run_config_hash: str
    generation: int
    plan_hash: str
    games: int
    played: int
    forfeits: int
    episodes: int
    copy_decision_hash: str | None
    members: tuple[MemberEntry, ...]
    forfeit_causes: tuple[tuple[str, int], ...]
    raw: dict[str, Any]


def _member_key(member: str) -> tuple[int, int]:
    return (0 if member.startswith("m") else 1, int(member[1:]))


def _check_causes(causes: list[dict[str, Any]] | None, forfeits: int, what: str) -> None:
    """Forfeit causes: present exactly with forfeits, sorted, unique, summing to them."""
    if forfeits == 0:
        if causes is not None:
            raise LeagueRecordError(f"{what} forfeit_causes must be absent without forfeits")
        return
    if causes is None:
        raise LeagueRecordError(f"{what} forfeit_causes must be present with forfeits")
    names = [c["cause"] for c in causes]
    if not all(a < b for a, b in itertools.pairwise(names)):
        raise LeagueRecordError(f"{what} forfeit_causes must be sorted by cause, each once")
    if sum(c["count"] for c in causes) != forfeits:
        raise LeagueRecordError(f"{what} forfeit causes' counts must sum to forfeits")


def _opponent_key(o: dict[str, Any]) -> tuple[str, str]:
    return (o["opponent_policy_id"], o.get("member", ""))


def _check_member(entry: dict[str, Any]) -> None:
    """One member's own rules: counts, opponents, own-seat Episodes, causes, baselines."""
    if entry["played"] + entry["forfeits"] != entry["games"]:
        raise LeagueRecordError("a member's played + forfeits must equal games")
    ops = entry["opponents"]
    if not all(_opponent_key(a) < _opponent_key(b) for a, b in itertools.pairwise(ops)):
        raise LeagueRecordError(
            "a member's opponents must be sorted by opponent_policy_id and member, each once"
        )
    if any(o["played"] > o["games"] for o in ops):
        raise LeagueRecordError("an opponent's played cannot exceed its games")
    if sum(o["games"] for o in ops) != entry["games"]:
        raise LeagueRecordError("a member's opponents' games must sum to its games")
    if sum(o["played"] for o in ops) != entry["played"]:
        raise LeagueRecordError("a member's opponents' played counts must sum to its played")
    self_played = sum(o["played"] for o in ops if o.get("member") == entry["member"])
    if entry["episodes"] != entry["played"] + self_played:
        raise LeagueRecordError(
            "a member's episodes must be its played games plus its played self-play games"
        )
    _check_causes(entry.get("forfeit_causes"), entry["forfeits"], "a member's")
    _check_baselines(entry, "a member's ")


def _check_member_links(members: list[dict[str, Any]]) -> None:
    """Members name each other consistently: the pair id, and equal counts from both sides."""
    by_id = {m["member"]: m for m in members}
    for entry in members:
        for o in entry["opponents"]:
            if "member" not in o:
                continue
            other = by_id.get(o["member"])
            if other is None:
                raise LeagueRecordError("an opponent names a member the record does not hold")
            if o["opponent_policy_id"] != other["pair_policy_id"]:
                raise LeagueRecordError(
                    "an opponent's policy id must be the named member's pair_policy_id"
                )
            if other is entry:
                continue
            back = next((p for p in other["opponents"] if p.get("member") == entry["member"]), None)
            if back is None or (back["games"], back["played"]) != (o["games"], o["played"]):
                raise LeagueRecordError(
                    "two members' entries about each other must agree in games and played"
                )


def _check_totals(raw: dict[str, Any], members: list[dict[str, Any]]) -> None:
    """The root counts distinct games: the members' sums less each member-vs-member game once."""
    games = sum(m["games"] for m in members)
    played = sum(m["played"] for m in members)
    for entry in members:
        for o in entry["opponents"]:
            other = o.get("member")
            if (
                other is not None
                and other != entry["member"]
                and _member_key(other) > _member_key(entry["member"])
            ):
                games -= o["games"]
                played -= o["played"]
    if (raw["games"], raw["played"]) != (games, played):
        raise LeagueRecordError(
            "the root totals must count distinct games:"
            " the members' sums less the games between two members"
        )
    if raw["played"] + raw["forfeits"] != raw["games"]:
        raise LeagueRecordError("the record's played + forfeits must equal games")
    if raw["episodes"] != 2 * raw["played"]:
        raise LeagueRecordError("the record's episodes must be two per played game")
    _check_causes(raw.get("forfeit_causes"), raw["forfeits"], "the record's")
    bounds: dict[str, tuple[int, int]] = {}
    for entry in members:
        for c in entry.get("forfeit_causes", []):
            largest, total = bounds.get(c["cause"], (0, 0))
            bounds[c["cause"]] = (max(largest, c["count"]), total + c["count"])
    listed = {c["cause"]: c["count"] for c in raw.get("forfeit_causes", [])}
    if listed.keys() != bounds.keys() or any(
        not bounds[cause][0] <= count <= bounds[cause][1] for cause, count in listed.items()
    ):
        raise LeagueRecordError(
            "each root forfeit cause must be one a member has,"
            " between the largest member count and the sum over the members"
        )


def read_league_run_record_v2(raw: object, config: RunConfig) -> LeagueRunRecordV2:
    """Validate a version 2 record against the contract, the reader rules and the run config.

    The config's ``population`` is read as data (``members`` and ``exploiters``): the record
    must hold exactly that many ``m<k>`` and ``x<k>`` entries, numbered from 0 without gaps."""
    try:
        validate("league-run-record-2", raw)
    except ContractError as exc:
        raise LeagueRecordError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    members: list[dict[str, Any]] = raw["members"]
    keys = [_member_key(m["member"]) for m in members]
    if not all(a < b for a, b in itertools.pairwise(keys)):
        raise LeagueRecordError("members must be sorted (m<k> by k, then x<k> by k), each once")
    for entry in members:
        _check_member(entry)
    _check_member_links(members)
    _check_totals(raw, members)
    if raw["run_id"] != config.run_key:
        raise LeagueRecordError(f"run_id {raw['run_id']} is not the config's run key")
    if raw["run_config_hash"] != config.hash:
        raise LeagueRecordError("run_config_hash is not the config's hash")
    population = config.raw.get("population")
    if not isinstance(population, dict):
        raise LeagueRecordError("a version 2 record belongs to a run whose config has a population")
    learning = sum(1 for m in members if m["member"].startswith("m"))
    if learning != population["members"] or len(members) - learning != population["exploiters"]:
        raise LeagueRecordError(
            "the members and exploiters must be the config's population.members"
            " and population.exploiters"
        )
    for index, entry in enumerate(members):
        position = index if index < learning else index - learning
        if entry["member"] != f"{entry['member'][0]}{position}":
            raise LeagueRecordError("member ids must be m0.. and x0.. without gaps")
    return LeagueRunRecordV2(
        run_id=raw["run_id"],
        run_config_hash=raw["run_config_hash"],
        generation=raw["generation"],
        plan_hash=raw["plan_hash"],
        games=raw["games"],
        played=raw["played"],
        forfeits=raw["forfeits"],
        episodes=raw["episodes"],
        copy_decision_hash=raw["copy_decision_hash"],
        members=tuple(
            MemberEntry(
                member=m["member"],
                pair_policy_id=m["pair_policy_id"],
                preview=m["checkpoints"]["preview"],
                battle=m["checkpoints"]["battle"],
                games=m["games"],
                played=m["played"],
                forfeits=m["forfeits"],
                episodes=m["episodes"],
                opponents=tuple((o["opponent_policy_id"], o["games"]) for o in m["opponents"]),
            )
            for m in members
        ),
        forfeit_causes=tuple((c["cause"], c["count"]) for c in raw.get("forfeit_causes", [])),
        raw=raw,
    )


def read_any_league_run_record(
    raw: object, config: RunConfig
) -> LeagueRunRecord | LeagueRunRecordV2:
    """Read a record of either version, by its ``schema_version``; another version is refused."""
    version = raw.get("schema_version") if isinstance(raw, dict) else None
    if version == 1:
        return read_league_run_record(raw, config)
    if version == VERSION_2:
        return read_league_run_record_v2(raw, config)
    raise LeagueRecordError("a LeagueRunRecord has schema_version 1 or 2")


def check_generation_plan(
    record: LeagueRunRecord | LeagueRunRecordV2, plan: object, config: RunConfig
) -> None:
    """The plan is the record's plan, and every game's seed is the one the run key gives."""
    if not isinstance(plan, list):
        raise LeagueRecordError("the match plan is not a list")
    if sha256_hex(jcs(plan)) != record.plan_hash:
        raise LeagueRecordError("the match plan differs from the plan_hash of the record")
    if len(plan) != record.games:
        raise LeagueRecordError("the match plan has another number of games than the record")
    for index, entry in enumerate(plan):
        expected = league_game_seed(config.run_key, record.generation, index)
        if (
            not isinstance(entry, dict)
            or entry.get("game") != index
            or entry.get("seed") != expected
        ):
            raise LeagueRecordError(
                f"game {index} does not carry the seed the run key gives it ({expected})"
            )


def check_population_plan(record: LeagueRunRecordV2, plan: object, config: RunConfig) -> None:
    """A population plan is the record's plan, and every game carries its pair's seed.

    A population game is half of a team pair played twice on one battle seed, so game ``i`` is in
    pair ``i // 2`` and its seed is ``league-seed/1`` at the PAIR's index (ADR-021 section 5). The
    record's root counts every distinct game, which is the plan's length.
    """
    if not isinstance(plan, list):
        raise LeagueRecordError("the match plan is not a list")
    if sha256_hex(jcs(plan)) != record.plan_hash:
        raise LeagueRecordError("the match plan differs from the plan_hash of the record")
    if len(plan) != record.games:
        raise LeagueRecordError("the match plan has another number of games than the record")
    for index, entry in enumerate(plan):
        expected = league_game_seed(config.run_key, record.generation, index // 2)
        if (
            not isinstance(entry, dict)
            or entry.get("game") != index
            or entry.get("pair") != index // 2
            or entry.get("half") != index % 2
            or entry.get("seed") != expected
        ):
            raise LeagueRecordError(
                f"game {index} does not carry the seed the run key gives its pair ({expected})"
            )
