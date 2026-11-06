"""Anchor calibration and the per-generation LadderRecord (ADR-021 sections 4 and 8, slice 7).

Pure functions over plain data (no I/O, no torch, no floats); :mod:`areneon_ml.loop.population`
reads and writes the files.

**Calibration (generation 0).** The anchors are the config's baselines among ``random-legal@v1``,
``actor-rule@v1`` and ``damage-aware@v1`` that play at least 2 games per generation; ``actor-
rule@v1`` must be one. The simulator side plays every pair of anchors for
``rating.calibration_pairs`` team pairs, each twice with the seats swapped (``calibration.json``).
:func:`read_calibration` checks the games against the plan they must be (seeds from ``league-
seed/1`` at generation 0), :func:`fit_anchors` fits the other anchors against ``actor-rule@v1`` =
10^9 and :func:`anchors_document` is the ``anchors.json`` that generation 0 commits. Those
strengths are frozen for the run: every later ``ladder.json`` repeats them, and
:func:`check_anchors` refuses a record whose anchors differ.

**Pair scores.** A team pair is played twice on one battle seed with the policies' seats swapped
(ADR-021 section 5), so a subject's pair score is the mean of its two games' scores. A game scores
1 for a win, 1/2 for a tie (a turn-cap tie has no winner), 0 for a loss, and a forfeit is a scored
loss for the side that failed (the row's ``winner`` is the other side). In half units a game is 0,
1 or 2, so the sum of the two is the bucket 0..4 of the record's ``Cell.scores`` (0, 250000, ...,
1000000 ppm).

**Window and copies.** A record at generation ``g`` fits the results of generations ``max(1, g -
window + 1) .. g``. A copy decided in generation ``c`` is applied by the driver at the start of
generation ``c + 1`` (:mod:`areneon_ml.loop.selection`), but the games of ``c + 1`` are still played
by the loser's old pair. So every result involving that slot (as the member, as an opposing member,
and its anchor games) from generations ``<= c + 1`` is stale and dropped (:func:`window_results`)
from the records of generation ``c + 2`` on. The record of generation ``c + 1`` itself rates the
pair that played it: it drops ``<= c`` and keeps ``c + 1`` (dropping that too would leave the slot
without a result, and the record needs every member's anchor cells).

**Bootstrap labels.** Every bootstrap of generation ``g`` draws ``population_draw(run_key, g,
"m0", "bootstrap", lane << 32 | counter)``: lane 0 for the strengths, ``1 + 64 i + j`` for the
cyclic-triple edge of members ``i < j``, ``2^20`` (``rating.LANE_COPY``) for the copy test. One seed
label, distinct lanes per use (the slice-6 rule).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import rating, selection
from areneon_ml.loop.ladder import read_ladder_record
from areneon_ml.loop.league import calibration_game_seed
from areneon_ml.loop.past_pool import Snapshot, member_index
from areneon_ml.loop.rating import BootstrapPlan, BootstrapSeed, ResultCell

#: Anchor policy ids and the baseline actor each must be configured with.
ANCHOR_ACTORS = {
    "actor-rule@v1": "actor-rule",
    "damage-aware@v1": "damage-aware",
    "random-legal@v1": "random-legal",
}
REFERENCE = rating.REFERENCE_ANCHOR
CALIBRATION_FORMAT = "population-calibration/1"
CALIBRATION_FILE = "calibration.json"
ANCHORS_FILE = "anchors.json"
LADDER_FILE = "ladder.json"
#: The label every bootstrap of a generation is seeded under; uses are separated by lane.
BOOTSTRAP_LABEL = "m0"
#: Bootstrap lanes of the snapshots rated at entry (slice 10): ``LANE_SNAPSHOT + 16 * generation +
#: member index``, below the copy test's lane, clear of the strengths' lane and the edge lanes.
LANE_SNAPSHOT = 1 << 16
_BUCKETS = 5
MIN_ANCHOR_GAMES = 2
_HASH_LENGTH = 64
_HEX64 = frozenset("0123456789abcdef")


#: Kept here for the callers that read a member's starting settings with the ladder code.
member_settings = selection.member_settings


class LadderBuildError(ValueError):
    """Calibration or ladder inputs break a rule."""


def _fail(message: str) -> None:
    raise LadderBuildError(message)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == _HASH_LENGTH and set(value) <= _HEX64


def anchor_ids(config: RunConfig) -> list[str]:
    """The run's anchors, sorted: baselines of at least 2 games with a known id and actor."""
    found = {
        b["policy_id"]
        for b in config.raw["generation"]["baseline_games"]
        if b["games"] >= MIN_ANCHOR_GAMES and ANCHOR_ACTORS.get(b["policy_id"]) == b["actor"]
    }
    if REFERENCE not in found:
        _fail(
            "population: baseline_games must list actor-rule@v1, the ladder's reference anchor, "
            "with at least 2 games"
        )
    return sorted(found)


def _plan_pairs(ids: Sequence[str], pairs: int) -> list[tuple[str, str]]:
    """The anchor pair (a, b) of each calibration pair, in plan order."""
    out: list[tuple[str, str]] = []
    for a, b in itertools.combinations(ids, 2):
        out.extend([(a, b)] * pairs)
    return out


# -- pair scores ---------------------------------------------------------------------------------


def _half_score(row: Mapping[str, Any], subject: str) -> int:
    """The subject's score of one game in half units: 2 a win, 1 a tie, 0 a loss."""
    seats = row["seats"]
    winner = row["winner"]
    if winner is None:
        return 1
    if seats["p1"] == subject and seats["p2"] == subject:
        _fail("a game against itself has no subject")
    return 2 if seats[winner] == subject else 0


def pair_buckets(
    rows: Iterable[Mapping[str, Any]], subject_of: Any
) -> list[tuple[Mapping[str, Any], int]]:
    """``(first row of the pair, bucket)`` for each team pair of ``rows``.

    ``subject_of(row)`` names the id whose score is counted. A pair is two rows with the same
    ``pair`` and ``half`` 0 then 1; anything else is refused.
    """
    by_pair: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair"], []).append(row)
    out = []
    for pair, held in sorted(by_pair.items()):
        halves = sorted(held, key=lambda r: r["half"])
        if [r["half"] for r in halves] != [0, 1]:
            _fail(f"pair {pair} does not hold exactly its two games")
        subject = subject_of(halves[0])
        if subject_of(halves[1]) != subject:
            _fail(f"pair {pair}'s two games name different subjects")
        out.append((halves[0], _half_score(halves[0], subject) + _half_score(halves[1], subject)))
    return out


def _tally(buckets: Iterable[int]) -> list[int]:
    scores = [0] * _BUCKETS
    for bucket in buckets:
        scores[bucket] += 1
    return scores


# -- calibration ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """A checked ``calibration.json``: the anchors' pair-score cells and the teams it drew."""

    anchors: tuple[str, ...]
    cells: tuple[ResultCell, ...]
    team_hashes: tuple[str, ...]


def read_calibration(raw: object, config: RunConfig) -> Calibration:
    """Check the simulator side's calibration games against the plan they must be."""
    if not isinstance(raw, dict):
        _fail("calibration.json is not an object")
    assert isinstance(raw, dict)
    if raw.get("schema_version") != 1 or raw.get("format") != CALIBRATION_FORMAT:
        _fail("calibration.json is not a version 1 population calibration")
    if raw.get("run_key") != config.run_key or raw.get("run_config_hash") != config.hash:
        _fail("calibration.json belongs to another run or config")
    ids = anchor_ids(config)
    pairs = config.raw["population"]["rating"]["calibration_pairs"]
    if raw.get("calibration_pairs") != pairs or raw.get("anchors") != ids:
        _fail("calibration.json names other anchors or another number of pairs than the config")
    games = raw.get("games")
    plan = _plan_pairs(ids, pairs)
    if not isinstance(games, list) or len(games) != 2 * len(plan):
        _fail("calibration.json lists another number of games than the plan")
    assert isinstance(games, list)
    for index, row in enumerate(games):
        pair, half = divmod(index, 2)
        a, b = plan[pair]
        seats = {"p1": a, "p2": b} if half == 0 else {"p1": b, "p2": a}
        if (
            not isinstance(row, dict)
            or row.get("game") != index
            or row.get("pair") != pair
            or row.get("half") != half
            or row.get("seed") != calibration_game_seed(config.run_key, pair)
            or row.get("a") != a
            or row.get("b") != b
            or row.get("seats") != seats
            or row.get("outcome") not in ("played", "forfeit")
            or row.get("winner") not in ("p1", "p2", None)
            or not _forfeit_fits(row)
        ):
            _fail(f"calibration game {index} is not the game the plan asks for")
    teams: Any = raw.get("team_hashes")
    if (
        not isinstance(teams, list)
        or teams != sorted(set(teams))
        or any(not _is_hash(t) for t in teams)
    ):
        _fail("calibration.json's team_hashes must be sorted unique SHA-256 hex")
    cells = []
    for (a, b), buckets in _group_by_anchor_pair(games).items():
        cells.append(ResultCell(a, b, _scores_tuple(_tally(buckets))))
    return Calibration(tuple(ids), tuple(cells), tuple(teams))


_ROW_KEYS = frozenset(
    ("game", "pair", "half", "seed", "a", "b", "seats", "outcome", "winner", "turns", "forfeit")
)


def _forfeit_fits(row: Mapping[str, Any]) -> bool:
    """A row holds the simulator side's keys only, and a forfeit's winner is the other seat."""
    if not set(row) <= _ROW_KEYS or not isinstance(row.get("turns"), int):
        return False
    held = row.get("forfeit")
    if row["outcome"] == "played":
        return held is None
    if not isinstance(held, dict) or set(held) != {"loser", "cause"}:
        return False
    return (
        held["loser"] in ("p1", "p2")
        and isinstance(held["cause"], str)
        and row["winner"]
        == {
            "p1": "p2",
            "p2": "p1",
        }.get(held["loser"])
    )


def _group_by_anchor_pair(games: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in games:
        grouped.setdefault((row["a"], row["b"]), []).append(row)
    return {
        key: [bucket for _, bucket in pair_buckets(rows, lambda r: r["a"])]
        for key, rows in grouped.items()
    }


def _scores_tuple(scores: list[int]) -> tuple[int, int, int, int, int]:
    a, b, c, d, e = scores
    return a, b, c, d, e


def fit_anchors(calibration: Calibration) -> dict[str, int]:
    """Anchor strengths on the ladder's scale: ``actor-rule@v1`` is 10^9, the rest are fitted."""
    reference = {REFERENCE: rating.SCALE}
    rated = [a for a in calibration.anchors if a != REFERENCE]
    fitted = rating.fit_strengths(calibration.cells, rated, reference) if rated else {}
    return dict(sorted({**reference, **fitted}.items()))


def anchors_document(
    config: RunConfig, calibration: Calibration, strengths: Mapping[str, int]
) -> dict[str, Any]:
    """The ``anchors.json`` of generation 0: frozen strengths and their calibration cells."""
    cells = []
    for cell in sorted(calibration.cells, key=lambda c: (c.player, c.opponent)):
        pairs, total = rating.cell_totals(cell.scores)
        cells.append(
            {
                "anchor_id": cell.player,
                "opponent_id": cell.opponent,
                "scores": list(cell.scores),
                "pairs": pairs,
                "score_sum_ppm": total,
            }
        )
    return {
        "schema_version": 1,
        "run_id": config.run_key,
        "run_config_hash": config.hash,
        "calibration_pairs": config.raw["population"]["rating"]["calibration_pairs"],
        "anchors": [{"anchor_id": a, "strength": s} for a, s in strengths.items()],
        "cells": cells,
    }


def read_anchors(raw: object, config: RunConfig) -> dict[str, int]:
    """The frozen strengths of a committed ``anchors.json``, once it is this run's."""
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        _fail("anchors.json is not a version 1 anchor record")
    assert isinstance(raw, dict)
    if raw.get("run_id") != config.run_key or raw.get("run_config_hash") != config.hash:
        _fail("anchors.json belongs to another run or config")
    held: Any = raw.get("anchors")
    if not isinstance(held, list) or [a.get("anchor_id") for a in held] != anchor_ids(config):
        _fail("anchors.json names other anchors than the config")
    strengths = {a["anchor_id"]: a["strength"] for a in held}
    if strengths.get(REFERENCE) != rating.SCALE:
        _fail("anchors.json: the reference anchor must be at exactly 1000000000")
    return strengths


def json_hash(raw: object) -> str:
    """SHA-256 of the JCS bytes of a record, the hash a manifest names it by."""
    return str(sha256_hex(jcs(raw)))


def check_anchors(record: Mapping[str, Any], strengths: Mapping[str, int]) -> None:
    """A ladder record repeats the frozen anchor strengths exactly."""
    found = {a["anchor_id"]: a["strength"] for a in record["anchors"]}
    if found != dict(strengths):
        _fail("anchor strengths are frozen at generation 0: the record's differ from anchors.json")


# -- the ladder ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRows:
    """One generation's game rows (``generation.json``) and the slots its copy replaced."""

    generation: int
    rows: Sequence[Mapping[str, Any]]
    copied: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowResult:
    generation: int
    member: str
    #: ``member`` or ``anchor``
    kind: str
    opponent: str
    bucket: int


def _generation_results(
    entry: GenerationRows,
    anchors: Iterable[str],
    members: frozenset[str],
    snapshots: frozenset[str] = frozenset(),
) -> list[WindowResult]:
    known = set(anchors)
    found: list[WindowResult] = []
    for kind, wanted in (("ladder", "member"), ("anchor", "anchor"), ("past", "snapshot")):
        rows = [r for r in entry.rows if r["kind"] == kind]
        for first, bucket in pair_buckets(rows, lambda r: r["member"]):
            member, opponent = first["member"], first["opponent"]
            if member not in members:
                _fail(f"generation {entry.generation}: {member} is not a member of the run")
            if wanted == "member" and (opponent not in members or opponent == member):
                _fail(f"generation {entry.generation}: a ladder pair names {opponent} as opponent")
            if wanted == "anchor" and opponent not in known:
                continue  # a baseline that is not an anchor is played but not rated
            if wanted == "snapshot" and opponent not in snapshots:
                continue  # a snapshot the pool has since evicted: its old results are dropped
            found.append(WindowResult(entry.generation, member, wanted, opponent, bucket))
    return found


def window_results(  # noqa: PLR0913, PLR0917
    history: Sequence[GenerationRows],
    generation: int,
    window: int,
    anchors: Iterable[str],
    members: Sequence[str],
    snapshots: Iterable[str] = (),
) -> list[WindowResult]:
    """The window's pair results, without any result of a slot from before a copy.

    ``history`` holds the committed generations of the window and the current one, in any order;
    ``copied`` names the loser slots of the copy each decided. A result of generation ``t`` that
    involves a slot (as the member or as an opposing member) copied in generation ``c`` is
    dropped when ``t <= c + 1``, except that the record of generation ``c + 1`` itself keeps the
    results of ``c + 1`` (the old pair played them and the record rates that pair); the cut is
    ``min(c + 1, generation - 1)``.
    """
    first = max(1, generation - window + 1)
    cut: dict[str, int] = {}
    for entry in history:
        for slot in entry.copied:
            cut[slot] = max(cut.get(slot, 0), min(entry.generation + 1, generation - 1))
    seen = [entry.generation for entry in history]
    if len(set(seen)) != len(seen) or any(not first <= g <= generation for g in seen):
        _fail("the history must hold each generation of the window once")
    out: list[WindowResult] = []
    labels = frozenset(members)
    pooled = frozenset(snapshots)
    for entry in sorted(history, key=lambda e: e.generation):
        for result in _generation_results(entry, anchors, labels, pooled):
            stale = result.generation <= cut.get(result.member, 0) or (
                result.kind == "member" and result.generation <= cut.get(result.opponent, 0)
            )
            if not stale:
                out.append(result)
    return out


Table = dict[tuple[str, str, str], list[int]]


def result_table(results: Iterable[WindowResult]) -> Table:
    """Pair-score counts by (member, opponent kind, id); member pairs in both directions."""
    table: Table = {}
    for r in results:
        table.setdefault((r.member, r.kind, r.opponent), [0] * _BUCKETS)[r.bucket] += 1
        if r.kind == "member":
            table.setdefault((r.opponent, "member", r.member), [0] * _BUCKETS)[
                _BUCKETS - 1 - r.bucket
            ] += 1
    return dict(sorted(table.items()))


def result_cells(table: Table) -> list[ResultCell]:
    """The cells the rating fits: one direction per member pair, every anchor cell."""
    cells = []
    for (member, kind, opponent), scores in table.items():
        if kind == "member" and opponent < member:
            continue
        cells.append(ResultCell(member, opponent, _scores_tuple(scores)))
    return cells


def snapshot_cells(
    config: RunConfig,
    generation: int,
    history: Sequence[GenerationRows],
    anchors: Mapping[str, int],
    pool_ids: Sequence[str],
) -> dict[tuple[str, str], list[int]]:
    """Each member's pair-score counts against each snapshot id, for planning ``generation + 1``.

    The window and the copy cut are those of the record of ``generation + 1`` over the copies
    decided up to ``generation - 1`` (``history`` holds no copy decided in ``generation`` itself):
    a slot copied in ``c`` plays the pair frozen at the end of ``c + 1`` as a new lineage, so its
    results against snapshots up to ``c + 1`` are dropped and it meets them afresh.
    """
    population = config.raw["population"]
    members = [f"m{k}" for k in range(population["members"])]
    window = population["rating"]["window"]
    first = max(1, generation + 2 - window)
    kept = [entry for entry in history if entry.generation >= first]
    table = result_table(
        window_results(kept, generation + 1, window, list(anchors), members, pool_ids)
    )
    return {(m, i): scores for (m, kind, i), scores in table.items() if kind == "snapshot"}


def _rate_snapshot(
    config: RunConfig,
    generation: int,
    snapshot: Snapshot,
    cells: Sequence[ResultCell],
    references: Mapping[str, int],
) -> dict[str, Any] | None:
    """A snapshot with ``n_min`` pairs in the window is rated once, against fixed references."""
    population = config.raw["population"]
    if sum(sum(c.scores) for c in cells) < population["selection"]["n_min"]:
        return None
    lane = LANE_SNAPSHOT + 16 * snapshot.generation + member_index(snapshot.member)
    if lane >= rating.LANE_COPY:
        _fail("a snapshot's bootstrap lane would pass the copy test's lane")
    plan = BootstrapPlan(
        BootstrapSeed(config.run_key, generation, BOOTSTRAP_LABEL),
        population["rating"]["bootstrap_resamples"],
        population["selection"]["alpha_ppm"],
    )
    found = rating.bootstrap_strengths(cells, [snapshot.id], references, plan, lane)[snapshot.id]
    return {
        "snapshot_id": snapshot.id,
        "member": snapshot.member,
        "generation": snapshot.generation,
        "strength": {"strength": found.strength, "lower": found.lower, "upper": found.upper},
    }


def build_ladder_record(  # noqa: PLR0913, PLR0917
    config: RunConfig,
    generation: int,
    history: Sequence[GenerationRows],
    anchors: Mapping[str, int],
    previous: Mapping[str, Any] | None = None,
    last_copy: Mapping[str, int] | None = None,
    pool: Sequence[Snapshot] = (),
) -> dict[str, Any]:
    """The generation's ``ladder.json``, with the copy test's outcome (slice 8).

    ``previous`` is the committed record of the generation before: the members' roots and effective
    settings are its state with its copy applied (:func:`selection.slot_state`). ``last_copy`` maps
    a member to the generation of its last copy, from every committed record.

    ``pool`` is the past pool the generation played (empty without a past-pool share). A snapshot
    listed in ``previous`` and still in the pool keeps its strength and bounds untouched and is a
    fixed reference of this record's fit; a pool snapshot with ``selection.n_min`` pairs in the
    window and no strength yet is rated once, against the anchors, the frozen snapshots and the
    members' fitted strengths held fixed, and listed (it joins the fit from the next record on).
    Cells against snapshots without a strength are not in the record (its schema lists a snapshot
    with its strength); ``past-pool.json`` carries them for the matchmaking.
    """
    population = config.raw["population"]
    window = population["rating"]["window"]
    members = [f"m{k}" for k in range(population["members"])]
    pool_ids = [p.id for p in pool]
    results = window_results(history, generation, window, list(anchors), members, pool_ids)
    table = result_table(results)
    for member in members:
        for anchor in anchors:
            if (member, "anchor", anchor) not in table:
                _fail(f"{member} has no result against the anchor {anchor} in the window")
    frozen: dict[str, dict[str, Any]] = {
        s["snapshot_id"]: s
        for s in (previous["snapshots"] if previous is not None else [])
        if s["snapshot_id"] in pool_ids
    }
    references = {**anchors, **{i: s["strength"]["strength"] for i, s in frozen.items()}}
    fit_cells = [
        c
        for c in result_cells(table)
        if c.opponent in anchors or c.opponent in members or c.opponent in frozen
    ]
    plan = BootstrapPlan(
        BootstrapSeed(config.run_key, generation, BOOTSTRAP_LABEL),
        population["rating"]["bootstrap_resamples"],
        population["selection"]["alpha_ppm"],
    )
    strengths = rating.bootstrap_strengths(fit_cells, members, references, plan)
    triples = rating.cyclic_triples(fit_cells, members, plan)
    roots, settings = selection.slot_state(config, previous)
    outcome = selection.decide(
        config,
        generation,
        strengths={m: strengths[m].strength for m in members},
        table=table,
        anchors=sorted(anchors),
        roots=roots,
        settings=settings,
        last_copy=last_copy or {},
        bootstrap=plan,
    )
    listed = dict(frozen)
    fixed = {**references, **{m: strengths[m].strength for m in members}}
    for snapshot in sorted(pool, key=lambda p: p.id):
        if snapshot.id in listed:
            continue
        own = [
            ResultCell(m, snapshot.id, _scores_tuple(table[(m, "snapshot", snapshot.id)]))
            for m in members
            if (m, "snapshot", snapshot.id) in table
        ]
        rated = _rate_snapshot(config, generation, snapshot, own, fixed)
        if rated is not None:
            listed[snapshot.id] = rated
    record: dict[str, Any] = {
        "schema_version": 1,
        "data_class": "synthetic",
        "run_id": config.run_key,
        "run_config_hash": config.hash,
        "generation": generation,
        "window": {
            "size": window,
            "first_generation": max(1, generation - window + 1),
            "last_generation": generation,
        },
        "anchors": [{"anchor_id": a, "strength": s} for a, s in sorted(anchors.items())],
        "members": [
            {
                "member": m,
                "root": roots[m],
                "strength": {
                    "strength": strengths[m].strength,
                    "lower": strengths[m].lower,
                    "upper": strengths[m].upper,
                },
                "settings": settings[m],
            }
            for m in members
        ],
        "snapshots": [listed[i] for i in sorted(listed)],
        "cells": [
            _cell_record(member, kind, opponent, scores)
            for (member, kind, opponent), scores in table.items()
            if kind != "snapshot" or opponent in listed
        ],
        "cyclic_triples": triples,
        "selections": [outcome],
    }
    read_ladder_record(record, previous, config)
    return record


def _cell_record(member: str, kind: str, opponent: str, scores: list[int]) -> dict[str, Any]:
    pairs, total = rating.cell_totals(scores)
    return {
        "member": member,
        "opponent": {"kind": kind, "id": opponent},
        "scores": list(scores),
        "pairs": pairs,
        "score_sum_ppm": total,
    }
