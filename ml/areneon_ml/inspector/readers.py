"""Strict readers of the loop's own files (ADR-017 section 7, slice 1).

``plan.json``, ``generation.json`` and ``team-set.json`` have no schema of their own, so the
inspector reads them with closed key sets: a key it does not know, or a missing one, is an
:class:`InspectError`, never a silent misreading. ``tests/ml/inspector/test_readers.py`` pins the
key sets against the files a real (tiny) loop wrote, so a change to the loop's writer fails there.
``run-config.json`` and ``league-run.json`` go through the contract readers the loop itself uses.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

HEX64 = re.compile(r"[0-9a-f]{64}")
SEED = re.compile(r"sodium,[0-9a-f]{32}")
SEATS = ("p1", "p2")

#: Closed key sets of the loop's files: (required, optional).
PLAN_KEYS = (
    frozenset({"baseline_seed", "game", "opponent", "policy_seat", "seed", "teams"}),
    frozenset[str](),
)
BASELINE_KEYS = (frozenset({"actor", "policy_id"}), frozenset[str]())
GAME_KEYS = (
    frozenset({"game", "opponent", "outcome", "seed", "turns", "winner"}),
    frozenset({"forfeit", "feature_belief"}),
)
BELIEF_KEYS = (
    frozenset({"decisions", "worlds_sampled", "redraws", "contradicted_worlds", "fallbacks"}),
    frozenset[str](),
)
FORFEIT_KEYS = (frozenset({"loser", "cause", "request_seq"}), frozenset({"detail"}))
GENERATION_KEYS = (
    frozenset(
        {
            "schema_version",
            "generation",
            "games",
            "forfeits",
            "episode_hashes",
            "training_set_hash",
            "examples",
            "manifest",
            "team_hashes",
        }
    ),
    frozenset({"forfeit_causes"}),
)
VERSION_MANIFEST_KEYS = (
    frozenset({"format_id", "format_rules_hash", "regulation", "ruleset_hash", "showdown_commit"}),
    frozenset[str](),
)
TEAM_SET_KEYS = (frozenset({"schema_version", "teams"}), frozenset[str]())


class InspectError(ValueError):
    """A run directory, a pool or a page cannot be inspected: the message says why."""


def closed(
    value: object, keys: tuple[Collection[str], Collection[str]], what: str
) -> dict[str, Any]:
    """``value`` as an object with the required keys and no key outside the optional ones."""
    required, optional = keys
    if not isinstance(value, dict):
        raise InspectError(f"{what} is not an object")
    missing = sorted(set(required) - set(value))
    extra = sorted(set(value) - set(required) - set(optional))
    if missing:
        raise InspectError(f"{what} lacks {', '.join(missing)}")
    if extra:
        raise InspectError(f"{what} has keys this inspector does not know: {', '.join(extra)}")
    return value


def integer(value: object, what: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InspectError(f"{what} is not an integer of at least {minimum}")
    return value


def text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise InspectError(f"{what} is not a string")
    return value


def hash64(value: object, what: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise InspectError(f"{what} is not a SHA-256 digest")
    return value


def array(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise InspectError(f"{what} is not a list")
    return value


@dataclass(frozen=True)
class Forfeit:
    loser: str
    cause: str
    request_seq: int
    detail: str | None


@dataclass(frozen=True)
class PlanEntry:
    game: int
    seed: str
    #: ``"self"`` or the baseline's policy id.
    opponent: str
    actor: str | None
    policy_seat: str
    teams: dict[str, int]


@dataclass(frozen=True)
class GameRow:
    game: int
    seed: str
    opponent: str
    outcome: str
    winner: str | None
    turns: int
    forfeit: Forfeit | None


@dataclass(frozen=True)
class GenerationFile:
    generation: int
    games: tuple[GameRow, ...]
    forfeits: int
    forfeit_causes: dict[str, int] | None
    examples: int
    episode_hashes: tuple[str, ...]
    team_hashes: tuple[str, ...]
    manifest: dict[str, str]


def read_plan(raw: object) -> tuple[PlanEntry, ...]:
    entries = []
    for index, item in enumerate(array(raw, "plan.json")):
        what = f"plan.json game {index}"
        entry = closed(item, PLAN_KEYS, what)
        opponent = entry["opponent"]
        actor = None
        if opponent != "self":
            baseline = closed(opponent, BASELINE_KEYS, f"{what} opponent")
            actor = text(baseline["actor"], f"{what} opponent actor")
            opponent = text(baseline["policy_id"], f"{what} opponent policy_id")
        seat = entry["policy_seat"]
        if seat not in ("both", *SEATS):
            raise InspectError(f"{what} policy_seat is not both, p1 or p2")
        if (opponent == "self") != (seat == "both"):
            raise InspectError(f"{what}: only a self-play game has both seats played by the pair")
        teams = closed(entry["teams"], (set(SEATS), set()), f"{what} teams")
        if integer(entry["game"], f"{what} game") != index:
            raise InspectError(f"{what} is numbered {entry['game']}")
        integer(entry["baseline_seed"], f"{what} baseline_seed")
        if SEED.fullmatch(text(entry["seed"], f"{what} seed")) is None:
            raise InspectError(f"{what} seed is not a sodium seed")
        entries.append(
            PlanEntry(
                index,
                entry["seed"],
                opponent,
                actor,
                seat,
                {s: integer(teams[s], f"{what} teams {s}") for s in SEATS},
            )
        )
    if len({e.seed for e in entries}) != len(entries):
        raise InspectError("plan.json repeats a seed")
    return tuple(entries)


def _read_game(item: object, index: int) -> GameRow:
    what = f"generation.json game {index}"
    row = closed(item, GAME_KEYS, what)
    if integer(row["game"], f"{what} game") != index:
        raise InspectError(f"{what} is numbered {row['game']}")
    if row["outcome"] not in ("played", "forfeit"):
        raise InspectError(f"{what} outcome is not played or forfeit")
    if row["winner"] not in (*SEATS, None):
        raise InspectError(f"{what} winner is not p1, p2 or null")
    forfeit = None
    if "forfeit" in row:
        if row["outcome"] != "forfeit":
            raise InspectError(f"{what} has a forfeit record but was played")
        raw = closed(row["forfeit"], FORFEIT_KEYS, f"{what} forfeit")
        if raw["loser"] not in SEATS:
            raise InspectError(f"{what} forfeit loser is not p1 or p2")
        detail = raw.get("detail")
        forfeit = Forfeit(
            raw["loser"],
            text(raw["cause"], f"{what} forfeit cause"),
            integer(raw["request_seq"], f"{what} forfeit request_seq"),
            None if detail is None else text(detail, f"{what} forfeit detail"),
        )
    if "feature_belief" in row:
        # What the belief did for the pair's seats (#1271): checked, never shown.
        belief = closed(row["feature_belief"], BELIEF_KEYS, f"{what} feature_belief")
        for key in ("decisions", "worlds_sampled", "redraws", "contradicted_worlds"):
            integer(belief[key], f"{what} feature_belief {key}")
        fallbacks = belief["fallbacks"]
        if not isinstance(fallbacks, dict):
            raise InspectError(f"{what} feature_belief fallbacks is not an object")
        for cause, count in fallbacks.items():
            integer(count, f"{what} feature_belief fallback {text(cause, 'cause')}")
    return GameRow(
        index,
        text(row["seed"], f"{what} seed"),
        text(row["opponent"], f"{what} opponent"),
        row["outcome"],
        row["winner"],
        integer(row["turns"], f"{what} turns"),
        forfeit,
    )


def read_generation(raw: object, generation: int) -> GenerationFile:
    body = closed(raw, GENERATION_KEYS, "generation.json")
    if body["schema_version"] != 1 or body["generation"] != generation:
        raise InspectError(f"generation.json is not schema 1 of generation {generation}")
    games = tuple(_read_game(g, i) for i, g in enumerate(array(body["games"], "games")))
    causes = body.get("forfeit_causes")
    if causes is not None:
        if not isinstance(causes, dict):
            raise InspectError("forfeit_causes is not an object")
        causes = {
            text(k, "forfeit_causes key"): integer(v, "forfeit_causes value")
            for k, v in causes.items()
        }
    for name in ("episode_hashes", "team_hashes"):
        for item in array(body[name], name):
            hash64(item, f"{name} entry")
    manifest = closed(body["manifest"], VERSION_MANIFEST_KEYS, "generation.json manifest")
    hash64(body["training_set_hash"], "training_set_hash")
    return GenerationFile(
        generation,
        games,
        integer(body["forfeits"], "forfeits"),
        causes,
        integer(body["examples"], "examples"),
        tuple(body["episode_hashes"]),
        tuple(body["team_hashes"]),
        {k: text(v, f"manifest {k}") for k, v in manifest.items()},
    )


def read_team_set(raw: object) -> tuple[str, ...]:
    body = closed(raw, TEAM_SET_KEYS, "team-set.json")
    if body["schema_version"] != 1:
        raise InspectError("team-set.json is not schema 1")
    teams = array(body["teams"], "team-set teams")
    for item in teams:
        hash64(item, "team-set entry")
    if teams != sorted(set(teams)):
        raise InspectError("team-set.json is not sorted and unique")
    return tuple(teams)
