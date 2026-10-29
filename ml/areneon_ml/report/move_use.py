"""The move-use diagnostic (#1164): what kinds of move a seat chooses, from stored progress games.

A read-only computation over the stored ``ProgressGameRecord``s of a committed progress generation
(a head-to-head match is one of them): each seat's recorded candidates and choices, joined to the
Pokemon in each slot by the omniscient log and the seat's own team. Per match and seat it counts
the slot actions in each move class, and how often a move of a class was among a slot's candidates
and was not chosen. It measures nothing about strength: a policy that plays only for immediate
damage and one that also uses setup, speed control or Protect differ here and are not ranked.

The classes come from the move-class table (``packages/showdown-bridge/src/move-classes.ts``), a
sibling of the engine-data export bound to it by hash: this module holds no move name or list. The
only class it makes itself is ``switch`` (a ``switch N`` slot action of a move request; a forced
replacement after a faint is not a choice between a move and a switch and is not counted).

Reading a candidate: ``move N`` means the N-th move of the Pokemon in that slot at that point of
the log. The record keeps no request, so the slot's Pokemon is rebuilt from the log's ``|switch|``,
``|drag|``, ``|replace|``, ``|faint|`` and ``|swap|`` lines and matched to the seat's team by its
display name. A slot action is *unresolved* (counted apart, never guessed) when its Pokemon cannot
be named from the team, its slot offers moves whose numbers the team's set cannot be (a move number
past the set's four, or the single move 1 a lock, a Struggle or a recharge leaves of a set with
several), or the seat used, in this game, a move its team set does not hold outside a called move
(Transform copies the target's moves). The pinned engine lists a set's moves in the set's order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.inspector.progress_replay import decision_points
from areneon_ml.inspector.readers import InspectError
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.progress.records import (
    ProgressRecordError,
    read_game_record,
    read_progress_record,
    record_hash,
)

LABEL = "descriptive move-use diagnostic: not a strength measure"
#: The classes of the move-class table, then the one this module makes itself.
CLASS_ORDER = (
    "status",
    "lowers_target_stat",
    "raises_own_stat",
    "side_condition",
    "field_condition",
    "protection",
    "switch",
)
SWITCH = "switch"
SEATS = ("p1", "p2")
_IDENT = re.compile(r"(p[12])([a-z]): (.+)")
_NOT_ID = re.compile(r"[^a-z0-9]")
#: ``|move|IDENT|MOVE|...`` has its tags after index 3; every board line has its ident at index 2.
MOVE_LINE = 3
IDENT_AT = 2
BOARD_LINES = ("switch", "drag", "replace", "faint", "swap")
REPO = Path(__file__).resolve().parents[3]


def to_id(name: str) -> str:
    """The engine's id of a display name: lower case letters and digits."""
    return _NOT_ID.sub("", name.lower())


class MoveUseError(InspectError):
    """The move-class table, or a game, cannot be read for the diagnostic."""


@dataclass(frozen=True)
class MoveClasses:
    """The move-class table, checked against the engine-data table it names."""

    table_sha256: str
    engine_data_hash: str
    #: class name -> the move ids in it (every class of the table; ``switch`` is not one of them)
    classes: dict[str, frozenset[str]]
    #: the engine-data table's move ids, by id (a set's move name resolves through ``to_id``)
    move_ids: frozenset[str]

    def of(self, move_id: str) -> tuple[str, ...]:
        """The classes a move is in, in ``CLASS_ORDER``."""
        return tuple(name for name in CLASS_ORDER[:-1] if move_id in self.classes[name])


def read_move_classes(classes_path: Path, engine_data_path: Path) -> MoveClasses:
    """Read the table and refuse it unless it is for the engine-data table beside it."""
    try:
        raw = json.loads(classes_path.read_text(encoding="utf-8"))
        engine = json.loads(engine_data_path.read_text(encoding="utf-8"))
        validate("move-class-table", raw)
    except (OSError, ValueError, ContractError) as exc:
        raise MoveUseError(f"{classes_path.name} cannot be read: {exc}") from exc
    if raw["engine_data_hash"] != sha256_hex(jcs(engine)):
        raise MoveUseError("the move-class table is for another engine-data table")
    if raw["format_id"] != engine["format_id"] or raw["mod"] != engine["mod"]:
        raise MoveUseError("the move-class table is for another format than the engine-data table")
    ids = frozenset(m["id"] for m in engine["moves"])
    classes: dict[str, frozenset[str]] = {}
    for name in CLASS_ORDER[:-1]:
        listed = raw["classes"][name]
        if listed != sorted(set(listed)) or not set(listed) <= ids:
            raise MoveUseError(f"class {name} is not sorted, unique ids of the engine-data table")
        classes[name] = frozenset(listed)
    if classes["status"] != {m["id"] for m in engine["moves"] if m["category"] == "Status"}:
        raise MoveUseError("the status class is not the Status moves of the engine-data table")
    return MoveClasses(sha256_hex(jcs(raw)), raw["engine_data_hash"], classes, ids)


def class_files(format_id: str) -> tuple[Path, Path]:
    """The committed move-class table and engine-data table of a format."""
    base = REPO / "fixtures" / "showdown" / format_id
    return base / "move-classes.json", base / "engine-data.json"


def committed_move_classes(format_id: str) -> MoveClasses | None:
    """The committed table of a format, or None when the repository holds none for it."""
    classes_path, engine_path = class_files(format_id)
    if not classes_path.is_file():
        return None
    return read_move_classes(classes_path, engine_path)


# -- the board: who stands in each slot, from the log -------------------------------------------


def _display(entry: dict[str, Any]) -> str:
    return str(entry["name"] or entry["species"])


class _Seat:
    """One seat's team: its sets by display name."""

    def __init__(self, team: list[dict[str, Any]]) -> None:
        self.team = team
        self.by_name: dict[str, list[int]] = {}
        for index, entry in enumerate(team):
            self.by_name.setdefault(_display(entry), []).append(index)

    def find(self, name: str) -> dict[str, Any] | None:
        found = self.by_name.get(name, [])
        return self.team[found[0]] if len(found) == 1 else None


def _called(parts: list[str]) -> bool:
    """A ``|move|`` line another effect made (``[from]`` tag), not the Pokemon's own choice."""
    return any(p.startswith("[from]") for p in parts[MOVE_LINE + 1 :])


def _explained(seats: dict[str, _Seat], log: list[str]) -> dict[str, bool]:
    """Per seat: whether every move the seat used (not called by another effect) is in its set."""
    ok = dict.fromkeys(SEATS, True)
    for line in log:
        parts = line.split("|")
        if len(parts) <= MOVE_LINE or parts[1] != "move" or _called(parts):
            continue
        ident = _IDENT.fullmatch(parts[2])
        if ident is None or to_id(parts[3]) == "struggle":
            continue
        entry = seats[ident[1]].find(ident[3])
        if entry is None or to_id(parts[3]) not in {to_id(m) for m in entry["moves"]}:
            ok[ident[1]] = False
    return ok


class _Board:
    """The occupant of each slot, by display name; ``None`` when the slot is empty."""

    def __init__(self) -> None:
        self.slots: dict[tuple[str, int], str | None] = {}

    def apply(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) <= IDENT_AT or parts[1] not in BOARD_LINES:
            return
        ident = _IDENT.fullmatch(parts[2])
        if ident is None:
            return
        slot = (ident[1], ord(ident[2]) - 97)
        if parts[1] == "faint":
            self.slots[slot] = None
        elif parts[1] == "swap":
            other = (ident[1], int(parts[3]))
            self.slots[slot], self.slots[other] = self.slots.get(other), self.slots.get(slot)
        else:
            self.slots[slot] = ident[3]


@dataclass(frozen=True)
class StoredGame:
    """One stored game: its record, where it belongs and its decisions placed in the log."""

    match_id: str
    game: int
    candidate_seat: str
    opponent_kind: str
    record: dict[str, Any]
    points: tuple[dict[str, Any], ...]


# -- one game ------------------------------------------------------------------------------------


@dataclass
class _Count:
    actions: int = 0  # slot actions of this class
    offered: int = 0  # slot decisions that had a candidate of this class
    passed: int = 0  # ... and chose something else


class SeatUse:
    """One seat's tallies, summed over games."""

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        self.slot_actions = 0
        self.unresolved = 0
        self.classes = {name: _Count() for name in CLASS_ORDER}

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "slot_actions": self.slot_actions,
            "unresolved": self.unresolved,
            "classes": {
                name: {"actions": c.actions, "offered": c.offered, "passed": c.passed}
                for name, c in self.classes.items()
            },
        }


def _move_number(part: str) -> int | None:
    words = part.split(" ")
    return int(words[1]) if words[0] == "move" and len(words) > 1 and words[1].isdigit() else None


def _slot_moves(
    options: set[str], entry: dict[str, Any] | None, usable: bool
) -> dict[str, str] | None:
    """The move id each ``move N`` option of one slot names, or None when it cannot be told."""
    numbers = {n for n in map(_move_number, options) if n is not None}
    if not numbers:
        return {}
    moves = [to_id(m) for m in entry["moves"]] if entry is not None else []
    if not usable or not moves or max(numbers) > len(moves) or (numbers == {1} and len(moves) > 1):
        return None
    return {str(n): moves[n - 1] for n in numbers}


def _classes_of(part: str, ids: dict[str, str], classes: MoveClasses) -> tuple[str, ...]:
    """The classes of one slot action: ``switch N`` is the switch class, ``move N`` its move's."""
    if part.startswith("switch "):
        return (SWITCH,)
    number = _move_number(part)
    return () if number is None else classes.of(ids[str(number)])


@dataclass
class _Game:
    """What reading one game's decisions needs: the board, both teams, who the log explains."""

    board: _Board
    seats: dict[str, _Seat]
    explained: dict[str, bool]
    classes: MoveClasses


def _count_decision(use: SeatUse, point: dict[str, Any], game: _Game) -> None:
    seat_id = point["seat"]
    chosen = point["chosen"].split(", ")
    choices = [c["choice"].split(", ") for c in point["candidates"]]
    for slot, part in enumerate(chosen):
        if part == "pass":
            continue
        options = {c[slot] for c in choices if len(c) > slot and c[slot] != "pass"}
        occupant = game.board.slots.get((seat_id, slot))
        entry = None if occupant is None else game.seats[seat_id].find(occupant)
        ids = _slot_moves(options, entry, game.explained[seat_id])
        if ids is None:
            use.unresolved += 1
            continue
        mine = _classes_of(part, ids, game.classes)
        offered = {cls for option in options for cls in _classes_of(option, ids, game.classes)}
        use.slot_actions += 1
        for cls in mine:
            use.classes[cls].actions += 1
        for cls in offered:
            use.classes[cls].offered += 1
            use.classes[cls].passed += cls not in mine


def game_use(game: StoredGame, classes: MoveClasses) -> dict[str, SeatUse]:
    """The tallies of one stored game, by seat: its move requests' final decisions only."""
    record = game.record
    seats = {s: _Seat(record["teams"][s]["team"]) for s in SEATS}
    log: list[str] = record["omniscient_log"]
    explained = _explained(seats, log)
    out = {s: SeatUse(record["policies"][s]) for s in SEATS}
    game_state = _Game(_Board(), seats, explained, classes)
    applied = 0
    for point in game.points:  # sorted by frame
        for line in log[applied : point["frame"]]:
            game_state.board.apply(line)
        applied = max(applied, point["frame"])
        if point["kind"] == "move" and point["refused"] is None:
            _count_decision(out[point["seat"]], point, game_state)
    return out


# -- a progress generation ----------------------------------------------------------------------


class MoveUseTally:
    """The move-use tallies, summed one stored game at a time (no game is kept)."""

    def __init__(self, classes: MoveClasses) -> None:
        self.classes = classes
        self.games = 0
        self._matches: dict[str, dict[str, Any]] = {}

    def add(self, game: StoredGame) -> None:
        used = game_use(game, self.classes)
        mine = game.candidate_seat
        other = "p2" if mine == "p1" else "p1"
        found = self._matches.setdefault(
            game.match_id,
            {
                "match_id": game.match_id,
                "games": 0,
                "opponent_kind": game.opponent_kind,
                "candidate": SeatUse(game.record["policies"][mine]),
                "opponent": SeatUse(game.record["policies"][other]),
            },
        )
        found["games"] += 1
        self.games += 1
        for role, seat in (("candidate", mine), ("opponent", other)):
            _add(found[role], used[seat])

    def result(self) -> dict[str, Any]:
        out = []
        for match_id in sorted(self._matches):  # the sums do not depend on the order of the games
            found = self._matches[match_id]
            opponent = found["opponent"]
            out.append(
                {
                    "match_id": found["match_id"],
                    "games": found["games"],
                    "opponent_kind": found["opponent_kind"],
                    "reference": opponent.policy_id.startswith("damage-aware@"),
                    "candidate": found["candidate"].as_dict(),
                    "opponent": opponent.as_dict(),
                }
            )
        return {
            "label": LABEL,
            "class_table_sha256": self.classes.table_sha256,
            "engine_data_hash": self.classes.engine_data_hash,
            "stored_games": self.games,
            "matches": out,
        }


def move_use(games: Iterable[StoredGame], classes: MoveClasses) -> dict[str, Any]:
    """Per match, the candidate's and the opponent's tallies over the stored games."""
    tally = MoveUseTally(classes)
    for game in games:
        tally.add(game)
    return tally.result()


def _add(total: SeatUse, part: SeatUse) -> None:
    total.slot_actions += part.slot_actions
    total.unresolved += part.unresolved
    for name in CLASS_ORDER:
        total.classes[name].actions += part.classes[name].actions
        total.classes[name].offered += part.classes[name].offered
        total.classes[name].passed += part.classes[name].passed


def share_ppm(part: int, whole: int) -> int | None:
    """``part / whole`` in parts per million, floored; None when there is nothing to divide by."""
    return None if whole == 0 else part * 1_000_000 // whole


def read_stored_games(target: Path, record: dict[str, Any]) -> Iterator[StoredGame]:
    """The stored games of one committed progress generation, tied to its ``progress.json``.

    Each file goes through the game record reader (schema, choices among candidates, hashes and
    seed recomputed), must be listed by its match, and must have a game row that names the
    candidate's seat. The replay pages' stricter checks (outcome, the log's end) are not repeated:
    the counts here need only the decisions, the teams and the log. A generator: one parsed game
    is alive at a time (a generation stores thousands, each holding its whole log), so a caller
    folds a game into its tallies and drops it.
    """
    matches = {m["match_id"]: m for m in record["matches"]}
    rows = {(r["match_id"], r["game"]): r for r in record["games"]}
    for path in sorted((target / "games").glob("*.json")):
        try:
            game = read_game_record(layout.read_json(path))
        except (ProgressRecordError, layout.RunDirError, OSError, ValueError) as exc:
            raise MoveUseError(f"games/{path.name}: {exc}") from exc
        match = matches.get(game["match_id"])
        row = rows.get((game["match_id"], game["game"]))
        if match is None or row is None or record_hash(game) not in match["game_record_hashes"]:
            raise MoveUseError(f"games/{path.name} is not a game progress.json lists")
        yield StoredGame(
            game["match_id"],
            game["game"],
            row["candidate_seat"],
            match["opponent"]["kind"],
            game,
            decision_points(game),
        )


def stored_games(
    progress_dir: Path, generation: int, config: RunConfig, record: dict[str, Any] | None = None
) -> Iterator[StoredGame]:
    """The stored games of one committed progress generation, one at a time.

    The generation's manifest is verified and ``progress.json`` read first (or ``record``, when
    the caller has already read it). Empty when the generation stores no game. The checks run
    when the iteration starts, before the first game is yielded.
    """
    target = progress_dir / layout.generation_name(generation)
    if not any((target / "games").glob("*.json")):
        return iter(())
    try:
        layout.verify_generation(target, generation)
        if record is None:
            record = read_progress_record(layout.read_json(target / "progress.json"), config)
    except (layout.RunDirError, OSError, ValueError) as exc:
        raise MoveUseError(f"progress {target.name}: {exc}") from exc
    return read_stored_games(target, record)


def progress_move_use(
    progress_dir: Path,
    generation: int,
    config: RunConfig,
    record: dict[str, Any] | None = None,
    games: Iterable[StoredGame] | None = None,
) -> dict[str, Any] | None:
    """The diagnostic of one committed progress generation, or None when there is nothing to read.

    None: the generation stores no game, or the repository holds no move-class table for the
    run's format. ``games`` are the generation's stored games when the caller has them
    (``stored_games``; consumed once); otherwise they are read here, one at a time.
    """
    classes = committed_move_classes(config.format_id)
    if classes is None:
        return None
    if games is None:
        games = stored_games(progress_dir, generation, config, record)
    found = move_use(games, classes)
    return found if found["stored_games"] else None
