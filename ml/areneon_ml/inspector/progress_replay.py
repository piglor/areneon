"""Progress-game replays: a stored ProgressGameRecord, read and checked (ADR-017 3, 4b, 4c).

A progress game record holds the engine's OMNISCIENT log: both sides' exact values and hidden
information. It is an evaluation artifact for the owner's local use, on a synthetic public-lineage
run. Every refusal is an :class:`InspectError`:

* the progress directory is not a committed ``g<NNNNNN>`` (its ``manifest.json`` does not match its
  files), or its ``progress.json`` / a game file is not a valid record of the contract (a private
  data class is schema-invalid, so a non-synthetic record never gets this far);
* a game file is not the game it is named for, is not one the progress record lists
  (``game_record_hashes``), or its outcome differs from the record's game row;
* ``omniscient_log_sha256`` or a team hash does not recompute (``read_game_record``);
* a played game's log ends differently from its recorded outcome (winner and turn count);
* the decisions do not fit the log (more move decisions than turns, a seat's request numbers not
  rising, a refused choice with nothing to retry).

The omniscient log has no ``|request|`` lines, so the page places each decision by the log's own
structure: the Team Preview choice after ``|teampreview``, the k-th move decision at ``|turn|k``,
and a forced switch at the replacement ``|switch|`` after a faint of that seat (else at the end of
its turn). Request numbers are the seat's own and count wait requests too, so they are shown but
never used to place a decision.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError

from areneon_ml.inspector.model import read_run_dir_config, refuse_non_public
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout
from areneon_ml.progress.records import (
    ProgressRecordError,
    check_game_baseline,
    is_v2,
    preview_variant,
    read_game_record,
    read_progress_record,
    record_hash,
)

SEATS = ("p1", "p2")
GAME_FILE = re.compile(r"([a-z0-9][a-z0-9_-]{0,63})-(\d+)\.json")
_TURN = re.compile(r"\|turn\|(\d+)")
_PLAYER = re.compile(r"\|player\|(p[12])\|([^|]*)")


@dataclass(frozen=True)
class ProgressGame:
    """One stored game with what the progress record says of its match."""

    run_id: str
    generation: int
    match_id: str
    game: int
    mode: str
    opponent_kind: str
    candidate_seat: str
    record: dict[str, Any]
    points: tuple[dict[str, Any], ...]
    #: Version 2 (both seats): the pair this game belongs to and the other game of the pair when
    #: its file is stored; None for a version 1 game.
    pair: int | None = None
    twin: int | None = None
    #: The generation of an anchor match's opponent, and the candidate's Team Preview variant of
    #: an ablation match (``learned``, ``random``, ``g<N>``); None otherwise.
    opponent_generation: int | None = None
    preview_variant: str | None = None
    #: A damage-aware game (ADR-018): the baseline seat, the match's identity hash and that seat's
    #: counts as the record states them; None for every other game.
    baseline: dict[str, Any] | None = None


def _read(path: Path, what: str) -> Any:
    try:
        return layout.read_json(path)
    except (layout.RunDirError, OSError, ValueError) as exc:
        raise InspectError(f"{what} cannot be read: {exc}") from exc


def _kind(candidates: list[dict[str, Any]]) -> str:
    choices = [c["choice"] for c in candidates]
    if choices and all(c.startswith("team ") for c in choices):
        return "preview"
    if any(part.startswith("move ") for c in choices for part in c.split(", ")):
        return "move"
    return "forceSwitch"


def _chains(decisions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """A refused choice and the retry that answered it are one position."""
    out: list[list[dict[str, Any]]] = []
    open_chain: list[dict[str, Any]] = []
    for decision in decisions:
        open_chain.append(decision)
        if "refused" not in decision:
            out.append(open_chain)
            open_chain = []
    if open_chain:
        out.append(open_chain)
    return out


def _turn_lines(lines: list[str]) -> dict[int, int]:
    found: dict[int, int] = {}
    for index, line in enumerate(lines):
        match = _TURN.fullmatch(line)
        if match is not None:
            found.setdefault(int(match[1]), index)
    return found


def _first(lines: list[str], prefix: str) -> int | None:
    return next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)


def _forced_positions(lines: list[str], seat: str, start: int, end: int) -> list[int]:
    """The seat's replacement ``|switch|`` lines in lines ``start..end`` after one of its faints."""
    fainted = next((i for i in range(start, end) if lines[i].startswith(f"|faint|{seat}")), None)
    if fainted is None:
        return []
    return [i for i in range(fainted, end) if lines[i].startswith(f"|switch|{seat}")]


def decision_points(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Each decision as a point with its ``frame`` (lines applied before it), sorted by frame."""
    lines: list[str] = record["omniscient_log"]
    turns = _turn_lines(lines)
    out: list[dict[str, Any]] = []
    for seat in SEATS:
        own = [d for d in record["decisions"] if d["player"] == seat]
        numbers = [d["request_seq"] for d in own]
        if any(a >= b for a, b in itertools.pairwise(numbers)):
            raise InspectError(f"the decisions of {seat} do not answer rising request numbers")
        moves = 0
        forced: dict[int, int] = {}
        for chain in _chains(own):
            kind = _kind(chain[-1]["candidates"])
            if kind == "preview":
                found = _first(lines, "|teampreview")
                frame = 0 if found is None else found + 1
            elif kind == "move":
                moves += 1
                if moves not in turns:
                    raise InspectError(
                        f"{seat} has a move decision for turn {moves}, which the log lacks"
                    )
                frame = turns[moves] + 1
            else:
                start = turns[moves] + 1 if moves in turns else (_first(lines, "|start") or 0)
                end = turns.get(moves + 1, len(lines))
                spots = _forced_positions(lines, seat, start, end)
                at = forced.get(moves, 0)
                forced[moves] = at + 1
                frame = spots[at] if at < len(spots) else end
            out.extend(_point(seat, kind, frame, d, record) for d in chain)
    return tuple(sorted(out, key=lambda point: point["frame"]))


def _point(
    seat: str, kind: str, frame: int, decision: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "seat": seat,
        "kind": kind,
        "frame": frame,
        "request_seq": decision["request_seq"],
        "chosen": decision["chosen"],
        "refused": decision.get("refused"),
        "policy": record["policies"][seat],
        "candidates": [
            {"choice": c["choice"], "pending": c["pending"]} for c in decision["candidates"]
        ],
    }


def log_result(lines: list[str]) -> tuple[str | None, int]:
    """(winner seat or ``tie``, None if the log does not end) and the last turn the log shows."""
    names = {m[1]: m[2] for m in map(_PLAYER.match, lines) if m is not None}
    turns = 0
    winner: str | None = None
    for line in lines:
        turn = _TURN.fullmatch(line)
        if turn is not None:
            turns = int(turn[1])
        elif line == "|tie":
            winner = "tie"
        elif line.startswith("|win|"):
            winner = next((seat for seat, name in names.items() if name == line[5:]), None)
    return winner, turns


def _check_outcome(record: dict[str, Any]) -> None:
    # A forfeit is not checked: the committed forfeit sample's log shows turn 4 while its outcome
    # says 3 turns, so "the log's last turn is at most outcome.turns" does not hold for the corpus.
    outcome = record["outcome"]
    winner, turns = log_result(record["omniscient_log"])
    if outcome["kind"] == "played" and (winner != outcome["winner"] or turns != outcome["turns"]):
        raise InspectError(
            f"{record['match_id']}-{record['game']}: the log ends unlike the recorded outcome"
        )


def seat_outcome(row: dict[str, Any]) -> dict[str, Any]:
    """A game row's outcome (candidate or opponent) as a game record states it (by seat)."""
    seat = row["candidate_seat"]
    side = {"candidate": seat, "opponent": "p2" if seat == "p1" else "p1"}
    outcome: dict[str, Any] = dict(row["outcome"])
    if outcome["kind"] == "played":
        outcome["winner"] = side.get(outcome["winner"], "tie")
    else:
        outcome["loser"] = side[outcome["loser"]]
    return outcome


def _check_game(
    record: dict[str, Any],
    name: str,
    progress: dict[str, Any],
    match: dict[str, Any],
    row: dict[str, Any],
) -> None:
    if (
        record["run_id"] != progress["run_id"]
        or record["generation"] != progress["generation"]
        or f"{record['match_id']}-{record['game']}.json" != name
        or record["match_id"] != match["match_id"]
        or record["version_manifest"] != progress["version_manifest"]
    ):
        raise InspectError(f"games/{name} is not the game it is named for")
    if record_hash(record) not in match["game_record_hashes"]:
        raise InspectError(f"games/{name} is not listed by progress.json")
    if record["outcome"] != seat_outcome(row):
        raise InspectError(f"games/{name}: its outcome differs from progress.json's game row")
    try:
        check_game_baseline(record, match, row)
    except ProgressRecordError as exc:
        raise InspectError(f"games/{name}: {exc}") from exc


def read_progress_games(
    progress_dir: Path,
    generation: int,
    match: str | None = None,
    games: list[int] | None = None,
    run_dir: Path | None = None,
) -> list[ProgressGame]:
    """The stored games of one committed progress generation (all, or ``match`` / ``games``)."""
    target = progress_dir / layout.generation_name(generation)
    try:
        layout.verify_generation(target, generation)
    except (layout.RunDirError, OSError, ValueError) as exc:
        raise InspectError(f"progress {target.name}: {exc}") from exc
    config = None
    if run_dir is not None:
        config = read_run_dir_config(run_dir)
        refuse_non_public(config)
    try:
        progress = read_progress_record(_read(target / "progress.json", "progress.json"), config)
    except ProgressRecordError as exc:
        raise InspectError(f"progress.json: {exc}") from exc
    if progress["generation"] != generation:
        raise InspectError("progress.json is another generation's")
    matches = {m["match_id"]: m for m in progress["matches"]}
    if match is not None and match not in matches:
        raise InspectError(f"progress {target.name} has no match {match}")
    rows = {(r["match_id"], r["game"]): r for r in progress["games"]}
    both = is_v2(progress)
    out: list[ProgressGame] = []
    # Both games of a version 2 pair are built together, so each page can link to the other.
    asked = None if games is None else {g ^ 1 if both else g for g in games} | set(games)
    for name in sorted(p.name for p in (target / "games").glob("*.json")):
        found = GAME_FILE.fullmatch(name)
        if found is None or found[1] not in matches:
            raise InspectError(f"games/{name} is not a game of a match of progress.json")
        if (match is not None and found[1] != match) or (
            asked is not None and int(found[2]) not in asked
        ):
            continue
        try:
            record = read_game_record(_read(target / "games" / name, f"games/{name}"))
        except (ProgressRecordError, ContractError) as exc:
            raise InspectError(f"games/{name}: {exc}") from exc
        row = rows.get((record["match_id"], record["game"]))
        if row is None:
            raise InspectError(f"games/{name} has no game row in progress.json")
        _check_game(record, name, progress, matches[found[1]], row)
        _check_outcome(record)
        out.append(
            ProgressGame(
                progress["run_id"],
                generation,
                record["match_id"],
                record["game"],
                matches[found[1]]["mode"],
                matches[found[1]]["opponent"]["kind"],
                row["candidate_seat"],
                record,
                decision_points(record),
                record["game"] // 2 if both else None,
                None,
                matches[found[1]]["opponent"].get("generation"),
                (
                    preview_variant(matches[found[1]]["candidate_preview"])
                    if matches[found[1]].get("candidate_preview") is not None
                    else None
                ),
                record.get("baseline"),
            )
        )
    if both:
        stored = {(g.match_id, g.game) for g in out}
        out = [
            dataclasses.replace(g, twin=g.game ^ 1 if (g.match_id, g.game ^ 1) in stored else None)
            for g in out
        ]
    wanted = [g for g in (games or []) if not any(x.game == g for x in out)]
    if wanted or not out:
        raise InspectError(
            "no stored game matches the selection (only the first games of a match are kept)"
        )
    return out
