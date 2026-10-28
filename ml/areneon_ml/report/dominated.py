"""The statically dominated damaging-move rate (ADR-022 measurement M2, #1025).

A read-only computation over the stored ``ProgressGameRecord``s of a committed progress generation
(head-to-head matches included), beside the move-use diagnostic (#1164) and sharing its reading of
the games: each seat's recorded candidates and choices, joined to the Pokemon in each slot by the
omniscient log and the seat's own team. For each slot action that uses a damaging move against the
opposing active Pokemon it computes a static score and asks whether that score is below the
threshold share of the best score among the same Pokemon's enabled damaging moves on any target.

APPROXIMATE STATIC PROXY: NOT A MECHANICS CLAIM, NEVER A GATE OR ADOPTION MEASURE. The score
ignores abilities, items, stats, boosts, weather, accuracy, variable-power moves and the value of
a weaker move (spread against single target, priority, Protect reads). A strong player scores
above zero as well; what carries information is the comparison with another seat on the same
games. It is a secondary diagnostic that decides nothing (ADR-022 Q5).

No game rule is written here (owner rule R7). The base power, category, type and target of each
move, the types of each shown species and the type chart's cells come from the hashed engine-data
table. Every other constant (the value of each ``damage_taken`` code, the same-type and spread
factors, the threshold and which move targets are single, spread or random) is in the reviewed,
versioned config record ``config/diagnostics/<format>/dominated-move.v1.json``, which names the
engine-data table by hash; its own JCS hash is on the page.

Score of a slot action, in exact rationals: base power, times the product over the target's shown
types of the cell value, times the same-type factor when the move's type is one of the user's
shown types. A spread move is the sum over the alive foes, times the spread factor; a random-target
move is the mean over the alive foes. An action is *dominated* when its score is below
``threshold`` times the best score of the slot's enabled damaging moves (each on its best target;
a spread move as its sum) and *immune* when its score is 0 while the best is positive (an immune
action is also dominated). Counted apart and left out of the rates: status moves, an action aimed
at the ally, a damaging move with no base power (variable power), an action the table cannot score
(a species or target it does not know, a fainted target), a slot whose moves cannot be named from
the log and the team (unresolved, as for the move-use diagnostic) and a decision with no
candidates. The shown species is the one the log's ``|switch|``, ``|drag|``, ``|replace|`` and
``|detailschange|`` lines state; a temporary type change is not tracked.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.learner.config import RunConfig
from areneon_ml.progress.stats import wilson
from areneon_ml.report.move_use import (
    _IDENT,
    BOARD_LINES,
    IDENT_AT,
    REPO,
    SEATS,
    MoveUseError,
    StoredGame,
    _Board,
    _explained,
    _move_number,
    _Seat,
    _slot_moves,
    class_files,
    stored_games,
    to_id,
)

LABEL = "approximate static proxy: not a mechanics claim, never a gate or adoption measure"
INTERVAL_LABEL = "binomial (Wilson) 95% interval, not clustered by game"
PPM = 1_000_000
CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "format_id",
        "engine_data_hash",
        "effectiveness_ppm",
        "same_type_ppm",
        "spread_ppm",
        "threshold_ppm",
        "target_kinds",
    }
)
KIND_GROUPS = ("single", "spread", "random")
TARGET_AT = 2  # ``move N T``: the target is the third word
DETAILS_AT = 3  # ``|switch|IDENT|DETAILS|...``


class DominatedError(MoveUseError):
    """The config record or the engine-data table cannot be used for the diagnostic."""


def config_file(format_id: str) -> Path:
    """The committed config record of a format."""
    return REPO / "config" / "diagnostics" / format_id / "dominated-move.v1.json"


@dataclass(frozen=True)
class MoveRow:
    """What the score reads of one move: the engine-data row, nothing else."""

    type_id: str
    status: bool
    base_power: int
    target: str


@dataclass(frozen=True)
class DominatedTable:
    """The reviewed config record read against the engine-data table it names."""

    config_hash: str
    engine_data_hash: str
    effectiveness: tuple[Fraction, ...]
    same_type: Fraction
    spread: Fraction
    threshold: Fraction
    single: frozenset[str]
    spread_kinds: frozenset[str]
    random: frozenset[str]
    moves: dict[str, MoveRow]
    species_types: dict[str, tuple[str, ...]]
    type_position: dict[str, int]
    #: ``chart[defender position][attacker position]``: the engine-data ``damage_taken`` code
    chart: tuple[tuple[int, ...], ...]


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise DominatedError(f"{name} must be an integer from {low} to {high}")
    return value


def read_dominated_table(config_path: Path, engine_data_path: Path) -> DominatedTable:
    """Read the config record and refuse it unless it is for the engine-data table beside it."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        engine = json.loads(engine_data_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DominatedError(f"{config_path.name} cannot be read: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise DominatedError("the dominated-move config has missing or unknown keys")
    if raw["schema_version"] != 1 or raw["kind"] != "dominated-move-config":
        raise DominatedError("the dominated-move config is not a version 1 record")
    if (
        raw["engine_data_hash"] != sha256_hex(jcs(engine))
        or raw["format_id"] != engine["format_id"]
    ):
        raise DominatedError("the dominated-move config is for another engine-data table")
    types = engine["types"]
    chart = tuple(tuple(int(c) for c in t["damage_taken"]) for t in types)
    codes = raw["effectiveness_ppm"]
    if not isinstance(codes, list) or len(codes) <= max(c for row in chart for c in row):
        raise DominatedError("effectiveness_ppm must give a value for every code of the type chart")
    values = tuple(Fraction(_int(c, "effectiveness_ppm", 0, 100 * PPM), PPM) for c in codes)
    kinds = raw["target_kinds"]
    if not isinstance(kinds, dict) or set(kinds) != set(KIND_GROUPS):
        raise DominatedError("target_kinds must name single, spread and random")
    groups = {}
    for name in KIND_GROUPS:
        listed = kinds[name]
        if not isinstance(listed, list) or listed != sorted(set(listed)):
            raise DominatedError(f"target_kinds.{name} must be sorted, unique target kinds")
        if not set(listed) <= set(engine["move_targets"]):
            raise DominatedError(f"target_kinds.{name} names a target the engine-data lacks")
        groups[name] = frozenset(listed)
    if sum(len(g) for g in groups.values()) != len(
        groups["single"] | groups["spread"] | groups["random"]
    ):
        raise DominatedError("a target kind is in two groups")
    moves = {
        m["id"]: MoveRow(m["type"], m["category"] == "Status", int(m["base_power"]), m["target"])
        for m in engine["moves"]
    }
    return DominatedTable(
        sha256_hex(jcs(raw)),
        raw["engine_data_hash"],
        values,
        Fraction(_int(raw["same_type_ppm"], "same_type_ppm", 1, 100 * PPM), PPM),
        Fraction(_int(raw["spread_ppm"], "spread_ppm", 1, 100 * PPM), PPM),
        Fraction(_int(raw["threshold_ppm"], "threshold_ppm", 1, PPM), PPM),
        groups["single"],
        groups["spread"],
        groups["random"],
        moves,
        {s["id"]: tuple(s["types"]) for s in engine["species"]},
        {t["id"]: i for i, t in enumerate(types)},
        chart,
    )


def committed_dominated_table(format_id: str) -> DominatedTable | None:
    """The committed table of a format, or None when the repository holds no config for it."""
    path = config_file(format_id)
    if not path.is_file():
        return None
    return read_dominated_table(path, class_files(format_id)[1])


# -- the board: who stands in each slot, and what it shows ------------------------------------


class _Shown(_Board):
    """``_Board`` that also keeps the species each slot's occupant is shown as."""

    def __init__(self) -> None:
        super().__init__()
        self.species: dict[tuple[str, int], str | None] = {}

    def apply(self, line: str) -> None:
        super().apply(line)
        parts = line.split("|")
        if len(parts) <= IDENT_AT or parts[1] not in (*BOARD_LINES, "detailschange"):
            return
        ident = _IDENT.fullmatch(parts[2])
        if ident is None:
            return
        slot = (ident[1], ord(ident[2]) - 97)
        kind = parts[1]
        if kind == "faint":
            self.species[slot] = None
        elif kind == "swap":
            other = (ident[1], int(parts[3]))
            self.species[slot], self.species[other] = (
                self.species.get(other),
                self.species.get(slot),
            )
        elif len(parts) > DETAILS_AT:
            self.species[slot] = to_id(parts[DETAILS_AT].split(",")[0])


# -- one game ---------------------------------------------------------------------------------


class DominatedUse:
    """One seat's tallies, summed over games."""

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        self.scored = 0  # damaging-move slot actions with a static score
        self.dominated = 0  # ... below the threshold share of the best (immune ones included)
        self.immune = 0  # ... with score 0 while a move with a positive score was enabled
        self.status = 0  # status-move slot actions
        self.ally_target = 0  # damaging moves aimed at the ally
        self.no_power = 0  # damaging-category moves with no base power
        self.unscored = 0  # actions the table cannot score
        self.unresolved = 0  # slots whose moves cannot be named
        self.no_candidates = 0  # decisions with an empty candidate list

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "scored": self.scored,
            "dominated": self.dominated,
            "immune": self.immune,
            "status": self.status,
            "ally_target": self.ally_target,
            "no_power": self.no_power,
            "unscored": self.unscored,
            "unresolved": self.unresolved,
            "no_candidates": self.no_candidates,
            "dominated_interval": interval_of(self.dominated, self.scored),
            "immune_interval": interval_of(self.immune, self.scored),
        }

    def add(self, other: DominatedUse) -> None:
        for name in COUNTS:
            setattr(self, name, getattr(self, name) + getattr(other, name))


COUNTS = (
    "scored",
    "dominated",
    "immune",
    "status",
    "ally_target",
    "no_power",
    "unscored",
    "unresolved",
    "no_candidates",
)


def interval_of(part: int, whole: int) -> dict[str, int] | None:
    """The share and its Wilson interval in ppm (derived on display), None with no trials."""
    if whole == 0:
        return None
    found = wilson(part, 0, whole)
    return {"ppm": found.score_ppm, "low_ppm": found.low_ppm, "high_ppm": found.high_ppm}


@dataclass
class _Context:
    board: _Shown
    seats: dict[str, _Seat]
    explained: dict[str, bool]
    table: DominatedTable


def _foes(ctx: _Context, seat: str) -> list[tuple[str, ...] | None]:
    """The opposing slots' shown types, one entry per alive foe (None: species unknown)."""
    other = "p2" if seat == "p1" else "p1"
    out: list[tuple[str, ...] | None] = []
    for slot in range(2):
        if ctx.board.slots.get((other, slot)) is None:
            continue
        shown = ctx.board.species.get((other, slot))
        out.append(None if shown is None else ctx.table.species_types.get(shown))
    return out


def _target_factor(table: DominatedTable, move: MoveRow, foe: tuple[str, ...]) -> Fraction:
    attacker = table.type_position[move.type_id]
    factor = Fraction(1)
    for kind in foe:
        factor *= table.effectiveness[table.chart[table.type_position[kind]][attacker]]
    return factor


def _scores(
    table: DominatedTable, move: MoveRow, user: tuple[str, ...], foes: list[tuple[str, ...] | None]
) -> list[Fraction] | None:
    """The move's score on each alive foe (single), or [total] (spread, random); None: unscored."""
    if any(f is None for f in foes) or not foes:
        return None
    known = [f for f in foes if f is not None]
    base = Fraction(move.base_power)
    if move.type_id in user:
        base *= table.same_type
    each = [base * _target_factor(table, move, f) for f in known]
    if move.target in table.spread_kinds:
        return [sum(each, Fraction(0)) * table.spread]
    if move.target in table.random:
        return [sum(each, Fraction(0)) / len(each)]
    return each


def _count_slot(
    use: DominatedUse, ctx: _Context, spot: tuple[str, int], part: str, options: set[str]
) -> None:
    seat = spot[0]
    occupant = ctx.board.slots.get(spot)
    entry = None if occupant is None else ctx.seats[seat].find(occupant)
    ids = _slot_moves(options, entry, ctx.explained[seat])
    number = _move_number(part)
    if ids is None:
        use.unresolved += 1
        return
    table = ctx.table
    move = table.moves[ids[str(number)]]
    if move.status:
        use.status += 1
        return
    words = part.split(" ")
    aimed = (
        int(words[TARGET_AT])
        if len(words) > TARGET_AT and words[TARGET_AT].lstrip("-").isdigit()
        else None
    )
    if aimed is not None and aimed < 0:
        use.ally_target += 1
        return
    if move.base_power == 0:
        use.no_power += 1
        return
    shown = ctx.board.species.get(spot)
    user = table.species_types.get(shown) if shown is not None else None
    foes = _foes(ctx, seat)
    chosen = _chosen_score(ctx, spot, move, user, aimed)
    enabled = {ids[str(n)] for n in map(_move_number, options) if n is not None}
    best = _best_score(table, user, foes, enabled)
    if chosen is None or best is None:
        use.unscored += 1
        return
    use.scored += 1
    if chosen < table.threshold * best:
        use.dominated += 1
        use.immune += chosen == 0


def _chosen_score(
    ctx: _Context,
    spot: tuple[str, int],
    move: MoveRow,
    user: tuple[str, ...] | None,
    aimed: int | None,
) -> Fraction | None:
    table = ctx.table
    if user is None or move.target not in table.single | table.spread_kinds | table.random:
        return None
    foes = _foes(ctx, spot[0])
    if move.target in table.single:
        picked: list[tuple[str, ...] | None]
        if aimed is None:
            picked = foes if len(foes) == 1 else [None]
        else:
            at = ("p2" if spot[0] == "p1" else "p1", aimed - 1)
            shown = ctx.board.species.get(at) if ctx.board.slots.get(at) is not None else None
            picked = [None] if shown is None else [table.species_types.get(shown)]
        foes = picked
    scores = _scores(table, move, user, foes)
    return None if scores is None else scores[0]


def _best_score(
    table: DominatedTable,
    user: tuple[str, ...] | None,
    foes: list[tuple[str, ...] | None],
    enabled: set[str],
) -> Fraction | None:
    if user is None:
        return None
    best: Fraction | None = None
    for move_id in sorted(enabled):
        move = table.moves[move_id]
        if move.status or move.base_power == 0:
            continue
        if move.target not in table.single | table.spread_kinds | table.random:
            continue
        scores = _scores(table, move, user, foes)
        if scores is not None and (best is None or max(scores) > best):
            best = max(scores)
    return best


def game_dominated(game: StoredGame, table: DominatedTable) -> dict[str, DominatedUse]:
    """The tallies of one stored game, by seat: its move requests' final decisions only."""
    record = game.record
    seats = {s: _Seat(record["teams"][s]["team"]) for s in SEATS}
    log: list[str] = record["omniscient_log"]
    ctx = _Context(_Shown(), seats, _explained(seats, log), table)
    out = {s: DominatedUse(record["policies"][s]) for s in SEATS}
    applied = 0
    for point in game.points:  # sorted by frame
        for line in log[applied : point["frame"]]:
            ctx.board.apply(line)
        applied = max(applied, point["frame"])
        if point["kind"] != "move" or point["refused"] is not None:
            continue
        use = out[point["seat"]]
        if not point["candidates"]:
            use.no_candidates += 1
            continue
        choices = [c["choice"].split(", ") for c in point["candidates"]]
        for slot, part in enumerate(point["chosen"].split(", ")):
            if part != "pass" and not part.startswith("switch"):
                options = {c[slot] for c in choices if len(c) > slot and c[slot] != "pass"}
                _count_slot(use, ctx, (point["seat"], slot), part, options)
    return out


# -- a progress generation --------------------------------------------------------------------


def dominated_rates(games: Sequence[StoredGame], table: DominatedTable) -> dict[str, Any]:
    """Per match, the candidate's and the opponent's tallies over the stored games."""
    matches: dict[str, dict[str, Any]] = {}
    for game in sorted(games, key=lambda g: (g.match_id, g.game)):
        used = game_dominated(game, table)
        mine = game.candidate_seat
        other = "p2" if mine == "p1" else "p1"
        found = matches.setdefault(
            game.match_id,
            {
                "games": 0,
                "opponent_kind": game.opponent_kind,
                "candidate": DominatedUse(game.record["policies"][mine]),
                "opponent": DominatedUse(game.record["policies"][other]),
            },
        )
        found["games"] += 1
        found["candidate"].add(used[mine])
        found["opponent"].add(used[other])
    return {
        "label": LABEL,
        "interval_label": INTERVAL_LABEL,
        "config_hash": table.config_hash,
        "engine_data_hash": table.engine_data_hash,
        "stored_games": len(games),
        "matches": [
            {
                "match_id": match_id,
                "games": found["games"],
                "opponent_kind": found["opponent_kind"],
                "reference": found["opponent"].policy_id.startswith("damage-aware@"),
                "candidate": found["candidate"].as_dict(),
                "opponent": found["opponent"].as_dict(),
            }
            for match_id, found in matches.items()
        ],
    }


def progress_dominated(
    progress_dir: Path,
    generation: int,
    config: RunConfig,
    record: dict[str, Any] | None = None,
    games: Sequence[StoredGame] | None = None,
) -> dict[str, Any] | None:
    """The dominated-move rates of one committed progress generation, or None with nothing to read.

    None: the generation stores no game, or the repository holds no config record for the run's
    format. ``games`` are the stored games when the caller has read them (``stored_games``).
    """
    table = committed_dominated_table(config.format_id)
    if table is None:
        return None
    if games is None:
        games = stored_games(progress_dir, generation, config, record)
    return dominated_rates(games, table) if games else None
