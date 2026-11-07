"""Strict readers of a population run's own files (ADR-021 sections 2, 7 and 8, slice 12b).

``plan.json`` and ``generation.json`` of a population generation have no schema of their own, so
the inspector reads them with closed key sets, as :mod:`areneon_ml.inspector.readers` does for a
single lineage: a key it does not know, or a missing one, is an :class:`InspectError`.
``tests/ml/inspector/test_population_readers.py`` pins the key sets against the files a real tiny
population loop wrote (``fixtures/loop/population-run``). ``league-run.json`` (version 2),
``ladder.json``, ``past-pool.json`` and a member's checkpoint records go through the loop's own
readers, which tie them to the run config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from areneon_ml.inspector.readers import (
    SEATS,
    SEED,
    Forfeit,
    InspectError,
    array,
    closed,
    hash64,
    integer,
    text,
)

SCHEMA_VERSION = 2
KINDS = ("ladder", "self", "past", "anchor")
MEMBER = re.compile(r"m(0|[1-9][0-9]?)")
SNAPSHOT = re.compile(r"(m(?:0|[1-9][0-9]?))@g(0|[1-9][0-9]*)")

PLAN_KEYS = (
    frozenset(
        {"baseline_seed", "game", "half", "kind", "member", "pair", "seats", "seed", "teams"}
    ),
    frozenset[str](),
)
GAME_KEYS = (
    frozenset(
        {
            "game",
            "half",
            "kind",
            "member",
            "opponent",
            "outcome",
            "pair",
            "seats",
            "seed",
            "turns",
            "winner",
        }
    ),
    frozenset({"forfeit"}),
)
FORFEIT_KEYS = (frozenset({"loser", "cause", "request_seq", "detail"}), frozenset[str]())
GENERATION_KEYS = (
    frozenset(
        {"schema_version", "generation", "games", "forfeits", "members", "manifest", "team_hashes"}
    ),
    frozenset({"forfeit_causes"}),
)
MEMBER_KEYS = (
    frozenset({"member", "episode_hashes", "training_set_hash", "examples"}),
    frozenset[str](),
)
VERSION_MANIFEST_KEYS = (
    frozenset({"format_id", "format_rules_hash", "regulation", "ruleset_hash", "showdown_commit"}),
    frozenset[str](),
)
MEMBER_SEAT_KEYS = (frozenset({"member"}), frozenset[str]())
SNAPSHOT_SEAT_KEYS = (frozenset({"snapshot", "policy_id"}), frozenset[str]())
BASELINE_SEAT_KEYS = (frozenset({"actor", "policy_id"}), frozenset[str]())


@dataclass(frozen=True)
class PopSeat:
    """What takes a seat of a planned game: a learning member, a past-pool snapshot or a baseline.

    ``label`` is the member id (``m0``), the snapshot id (``m1@g0``) or the baseline's policy id;
    ``policy_id`` is None for a member (its pair is the one ``league-run.json`` names).
    """

    kind: str
    label: str
    policy_id: str | None


@dataclass(frozen=True)
class PopPlanEntry:
    game: int
    pair: int
    half: int
    seed: str
    kind: str
    member: str
    seats: dict[str, PopSeat]
    teams: dict[str, int]


@dataclass(frozen=True)
class PopGameRow:
    game: int
    pair: int
    half: int
    seed: str
    kind: str
    member: str
    #: The other seat: the member's own id for self-play, else a member, snapshot or baseline id.
    opponent: str
    seats: dict[str, str]
    outcome: str
    winner: str | None
    turns: int
    forfeit: Forfeit | None


@dataclass(frozen=True)
class PopMemberFile:
    member: str
    episode_hashes: tuple[str, ...]
    training_set_hash: str
    examples: int


@dataclass(frozen=True)
class PopGenerationFile:
    generation: int
    games: tuple[PopGameRow, ...]
    forfeits: int
    forfeit_causes: dict[str, int] | None
    members: tuple[PopMemberFile, ...]
    team_hashes: tuple[str, ...]
    manifest: dict[str, str]


def _seat(raw: object, what: str) -> PopSeat:
    if isinstance(raw, dict) and "member" in raw:
        body = closed(raw, MEMBER_SEAT_KEYS, what)
        member = text(body["member"], f"{what} member")
        if MEMBER.fullmatch(member) is None:
            raise InspectError(f"{what} names no learning member")
        return PopSeat("member", member, None)
    if isinstance(raw, dict) and "snapshot" in raw:
        body = closed(raw, SNAPSHOT_SEAT_KEYS, what)
        snapshot = text(body["snapshot"], f"{what} snapshot")
        if SNAPSHOT.fullmatch(snapshot) is None:
            raise InspectError(f"{what} names no past-pool snapshot")
        return PopSeat("snapshot", snapshot, text(body["policy_id"], f"{what} policy_id"))
    body = closed(raw, BASELINE_SEAT_KEYS, what)
    policy = text(body["policy_id"], f"{what} policy_id")
    text(body["actor"], f"{what} actor")
    return PopSeat("baseline", policy, policy)


def read_population_plan(raw: object) -> tuple[PopPlanEntry, ...]:
    entries = []
    for index, item in enumerate(array(raw, "plan.json")):
        what = f"plan.json game {index}"
        entry = closed(item, PLAN_KEYS, what)
        if integer(entry["game"], f"{what} game") != index:
            raise InspectError(f"{what} is numbered {entry['game']}")
        pair, half = integer(entry["pair"], f"{what} pair"), integer(entry["half"], f"{what} half")
        if half not in (0, 1) or index != 2 * pair + half:
            raise InspectError(f"{what} is not half {half} of pair {pair}")
        if entry["kind"] not in KINDS:
            raise InspectError(f"{what} kind is not one of {', '.join(KINDS)}")
        member = text(entry["member"], f"{what} member")
        if MEMBER.fullmatch(member) is None:
            raise InspectError(f"{what} member names no learning member")
        integer(entry["baseline_seed"], f"{what} baseline_seed")
        if SEED.fullmatch(text(entry["seed"], f"{what} seed")) is None:
            raise InspectError(f"{what} seed is not a sodium seed")
        seats = closed(entry["seats"], (set(SEATS), set()), f"{what} seats")
        teams = closed(entry["teams"], (set(SEATS), set()), f"{what} teams")
        entries.append(
            PopPlanEntry(
                index,
                pair,
                half,
                entry["seed"],
                entry["kind"],
                member,
                {s: _seat(seats[s], f"{what} seat {s}") for s in SEATS},
                {s: integer(teams[s], f"{what} teams {s}") for s in SEATS},
            )
        )
    return tuple(entries)


def _game(item: object, index: int) -> PopGameRow:
    what = f"generation.json game {index}"
    row = closed(item, GAME_KEYS, what)
    if integer(row["game"], f"{what} game") != index:
        raise InspectError(f"{what} is numbered {row['game']}")
    if row["outcome"] not in ("played", "forfeit"):
        raise InspectError(f"{what} outcome is not played or forfeit")
    if row["winner"] not in (*SEATS, None):
        raise InspectError(f"{what} winner is not p1, p2 or null")
    if row["kind"] not in KINDS:
        raise InspectError(f"{what} kind is not one of {', '.join(KINDS)}")
    seats = closed(row["seats"], (set(SEATS), set()), f"{what} seats")
    forfeit = None
    if "forfeit" in row:
        if row["outcome"] != "forfeit":
            raise InspectError(f"{what} has a forfeit record but was played")
        raw = closed(row["forfeit"], FORFEIT_KEYS, f"{what} forfeit")
        if raw["loser"] not in SEATS:
            raise InspectError(f"{what} forfeit loser is not p1 or p2")
        forfeit = Forfeit(
            raw["loser"],
            text(raw["cause"], f"{what} forfeit cause"),
            integer(raw["request_seq"], f"{what} forfeit request_seq"),
            text(raw["detail"], f"{what} forfeit detail"),
        )
    return PopGameRow(
        index,
        integer(row["pair"], f"{what} pair"),
        integer(row["half"], f"{what} half"),
        text(row["seed"], f"{what} seed"),
        row["kind"],
        text(row["member"], f"{what} member"),
        text(row["opponent"], f"{what} opponent"),
        {s: text(seats[s], f"{what} seat {s}") for s in SEATS},
        row["outcome"],
        row["winner"],
        integer(row["turns"], f"{what} turns"),
        forfeit,
    )


def read_population_generation(raw: object, generation: int) -> PopGenerationFile:
    body = closed(raw, GENERATION_KEYS, "generation.json")
    if body["schema_version"] != SCHEMA_VERSION or body["generation"] != generation:
        raise InspectError(f"generation.json is not schema 2 of generation {generation}")
    games = tuple(_game(g, i) for i, g in enumerate(array(body["games"], "games")))
    causes = body.get("forfeit_causes")
    if causes is not None:
        if not isinstance(causes, dict):
            raise InspectError("forfeit_causes is not an object")
        causes = {
            text(k, "forfeit_causes key"): integer(v, "forfeit_causes value")
            for k, v in causes.items()
        }
    members = []
    for item in array(body["members"], "members"):
        entry = closed(item, MEMBER_KEYS, "generation.json member")
        member = text(entry["member"], "member")
        if MEMBER.fullmatch(member) is None:
            raise InspectError("generation.json names a member that is no learning member")
        hashes = array(entry["episode_hashes"], f"{member} episode_hashes")
        for h in hashes:
            hash64(h, f"{member} episode_hashes entry")
        if hashes != sorted(set(hashes)):
            raise InspectError(f"{member}'s episode_hashes are not sorted and unique")
        members.append(
            PopMemberFile(
                member,
                tuple(hashes),
                hash64(entry["training_set_hash"], f"{member} training_set_hash"),
                integer(entry["examples"], f"{member} examples"),
            )
        )
    for name in ("team_hashes",):
        for item in array(body[name], name):
            hash64(item, f"{name} entry")
    manifest = closed(body["manifest"], VERSION_MANIFEST_KEYS, "generation.json manifest")
    return PopGenerationFile(
        generation,
        games,
        integer(body["forfeits"], "forfeits"),
        causes,
        tuple(members),
        tuple(body["team_hashes"]),
        {k: text(v, f"manifest {k}") for k, v in manifest.items()},
    )


def snapshot_parts(snapshot: str) -> tuple[str, int]:
    """(member, generation) of a past-pool snapshot id such as ``m1@g0``."""
    found = SNAPSHOT.fullmatch(snapshot)
    if found is None:
        raise InspectError("a past-pool snapshot id is <member>@g<generation>")
    return found[1], int(found[2])


def manifest_member(manifest: dict[str, Any], member: str) -> dict[str, Any]:
    """The sealed facts the generation manifest holds for one member."""
    members = manifest.get("members")
    entry = members.get(member) if isinstance(members, dict) else None
    if not isinstance(entry, dict):
        raise InspectError(f"the generation manifest names no facts for {member}")
    return entry
