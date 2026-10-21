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


class LeagueRecordError(ValueError):
    """A LeagueRunRecord or a match plan breaks a rule."""


def league_game_seed(run_key: str, generation: int, game: int) -> str:
    """The battle seed of one league game (``league-seed/1``)."""
    if generation < 1 or game < 0:
        raise ValueError("a generation counts from 1 and a game from 0")
    body = {"run_key": run_key, "generation": generation, "game": game, "purpose": "league-game"}
    return "sodium," + sha256_hex(jcs(body))[:32]


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
    if causes and sum(count for _, count in causes) != raw["forfeits"]:
        raise LeagueRecordError("the forfeit causes' counts must sum to forfeits")
    if raw["run_id"] != config.run_key:
        raise LeagueRecordError(f"run_id {raw['run_id']} is not the config's run key")
    if raw["run_config_hash"] != config.hash:
        raise LeagueRecordError("run_config_hash is not the config's hash")
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


def check_generation_plan(record: LeagueRunRecord, plan: object, config: RunConfig) -> None:
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
