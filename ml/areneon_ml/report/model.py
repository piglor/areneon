"""What a run's files say, read strictly and read-only (ADR-017 sections 4 and 7, slice 6).

``read_report_model`` reads a run directory, and a progress root when there is one, and returns
plain numbers. It never reads Episode content: an Episode is only counted, through
``generation.json``. Every committed generation's ``manifest.json`` is checked against its files
before anything in it is used; the loop's files are read with closed key sets (an unknown or
missing key is a :class:`ReportError`), the contract readers the loop and the progress command use
check the records, and a run whose lineage is private, or whose checkpoint records or progress
records are not synthetic, is refused. A progress record is tied to the generation it names by the
SHA-256 of that generation's manifest.

The page's numbers are these integers (and the training diagnostics' floats, shown as they are
stored); nothing is estimated. A fact the files do not hold (wall-clock time, a forfeit's cause in
a generation written before causes were recorded) is ``None`` or ``"not recorded"``, never zero.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import sha256_hex

from areneon_ml.checkpoint.record import RecordError, read_record
from areneon_ml.inspector.model import generation_numbers, read_run_dir_config, refuse_non_public
from areneon_ml.inspector.readers import (
    InspectError,
    PlanEntry,
    closed,
    integer,
    read_generation,
    read_plan,
)
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecord,
    check_generation_plan,
    read_league_run_record,
)
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import summary_of

NOT_RECORDED = "not recorded"
ROLES = ("battle", "preview")
METRICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
STEPS_KEYS = ("steps", "optimiser_steps")
TRAIN_ROLE_KEYS = (frozenset({*METRICS, *STEPS_KEYS}), frozenset[str]())
TRAIN_KEYS = (frozenset(ROLES), frozenset[str]())
_PROGRESS_DIR = re.compile(r"g(\d{6})")


class ReportError(InspectError):
    """A run directory or progress root cannot be reported on: the message says why."""


@dataclass(frozen=True)
class Tally:
    """Counts of the games of one opponent kind in one generation.

    For self-play ``a`` is the games p1 won and ``b`` the games p2 won; against a scripted baseline
    ``a`` is the games the frozen pair won and ``b`` the games the baseline won.
    """

    games: int = 0
    a: int = 0
    b: int = 0
    ties: int = 0
    forfeits: int = 0
    turns: int = 0  # the turns of the played games

    @property
    def played(self) -> int:
        return self.games - self.forfeits


@dataclass(frozen=True)
class TrainFacts:
    """``train-report.json``: per role the diagnostics as stored."""

    roles: dict[str, dict[str, float]]


@dataclass(frozen=True)
class GenerationFacts:
    number: int
    manifest_sha256: str
    pair_policy_id: str | None
    games: int | None
    played: int | None
    forfeits: int | None
    episodes: int | None
    examples: int | None
    self_play: Tally | None
    baselines: dict[str, Tally]
    #: Forfeits by cause over all games (``NOT_RECORDED`` for a forfeit written without a cause);
    #: None for a generation without games.
    forfeit_causes: dict[str, int] | None
    train: TrainFacts | None


@dataclass(frozen=True)
class MatchFacts:
    match_id: str
    mode: str
    opponent_kind: str
    games: int
    wins: int
    ties: int
    losses: int
    score_ppm: int
    low_ppm: int
    high_ppm: int
    turns_total: int
    turn_cap_ties: int
    forfeits: tuple[tuple[str, str, int], ...]
    seen_label: str | None


@dataclass(frozen=True)
class ProgressFacts:
    generation: int
    games_per_match: int
    recorded_games_per_match: int
    max_turns: int
    matches: tuple[MatchFacts, ...]


@dataclass(frozen=True)
class ReportModel:
    run_key: str
    lineage_id: str
    format_id: str
    config_hash: str
    generations: tuple[GenerationFacts, ...]
    progress: dict[int, ProgressFacts]
    progress_root: bool
    inputs: dict[str, str] = field(default_factory=dict)


def _json(path: Path) -> Any:
    try:
        return layout.read_json(path)
    except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"{path.parent.name}/{path.name}: {exc}") from exc


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ReportError(f"{what} is not a finite number")
    return float(value)


def read_train_report(raw: object) -> TrainFacts:
    """``train-report.json`` with closed key sets: both roles, each with exactly the known keys."""
    body = closed(raw, TRAIN_KEYS, "train-report.json")
    roles: dict[str, dict[str, float]] = {}
    for role in ROLES:
        values = closed(body[role], TRAIN_ROLE_KEYS, f"train-report.json {role}")
        roles[role] = {m: _number(values[m], f"{role} {m}") for m in METRICS}
        for key in STEPS_KEYS:
            roles[role][key] = integer(values[key], f"{role} {key}")
    return TrainFacts(roles)


def _tallies(
    games: tuple[Any, ...], plan: tuple[PlanEntry, ...]
) -> tuple[Tally, dict[str, Tally], dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    causes: Counter[str] = Counter()
    for row, entry in zip(games, plan, strict=True):
        kind = "self-play" if entry.opponent == "self" else entry.opponent
        tally = counts.setdefault(kind, Counter())
        tally["games"] += 1
        if row.outcome == "forfeit":
            tally["forfeits"] += 1
            causes[NOT_RECORDED if row.forfeit is None else row.forfeit.cause] += 1
            continue
        tally["turns"] += row.turns
        if row.winner is None:
            tally["ties"] += 1
        elif kind == "self-play":
            tally["a" if row.winner == "p1" else "b"] += 1
        else:
            tally["a" if row.winner == entry.policy_seat else "b"] += 1
    made = {k: Tally(**dict(v)) for k, v in counts.items()}
    return made.pop("self-play", Tally()), dict(sorted(made.items())), dict(sorted(causes.items()))


def _played_generation(
    number: int, path: Path, config: RunConfig, manifest_sha: str, inputs: dict[str, str]
) -> GenerationFacts:
    plan_raw = _json(path / "plan.json")
    try:
        record: LeagueRunRecord = read_league_run_record(_json(path / "league-run.json"), config)
        check_generation_plan(record, plan_raw, config)
    except LeagueRecordError as exc:
        raise ReportError(f"g{number:06d}: {exc}") from exc
    if record.generation != number:
        raise ReportError(f"g{number:06d}: its league-run.json is another generation's")
    file = read_generation(_json(path / "generation.json"), number)
    plan = read_plan(plan_raw)
    if len(file.games) != len(plan):
        raise ReportError(f"g{number:06d}: generation.json and plan.json differ in games")
    for row, entry in zip(file.games, plan, strict=True):
        if row.seed != entry.seed:
            raise ReportError(f"g{number:06d} game {row.game}: the seed differs between files")
    played = sum(1 for g in file.games if g.outcome == "played")
    if (file.forfeits, record.played, record.forfeits, record.games) != (
        len(file.games) - played,
        played,
        len(file.games) - played,
        len(file.games),
    ):
        raise ReportError(f"g{number:06d}: the game counts of its files differ")
    if record.episodes != len(file.episode_hashes):
        raise ReportError(f"g{number:06d}: the Episode counts of its files differ")
    self_play, baselines, causes = _tallies(file.games, plan)
    if record.raw.get("forfeit_causes") is not None and dict(record.forfeit_causes) != {
        c: n for c, n in causes.items() if c != NOT_RECORDED
    }:
        raise ReportError(f"g{number:06d}: forfeit_causes are not the causes of the game rows")
    report = path / "train-report.json"
    train = read_train_report(_json(report)) if report.exists() else None
    for name in ("plan.json", "generation.json", "league-run.json", "train-report.json"):
        if (path / name).exists():
            inputs[f"generations/g{number:06d}/{name}"] = sha256_hex((path / name).read_bytes())
    return GenerationFacts(
        number,
        manifest_sha,
        record.pair_policy_id,
        record.games,
        record.played,
        record.forfeits,
        record.episodes,
        file.examples,
        self_play,
        baselines,
        causes,
        train,
    )


def _check_synthetic(path: Path, number: int) -> None:
    for role in ROLES:
        found = path / "records" / f"{role}.json"
        if not found.exists():
            continue
        try:
            record = read_record(layout.read_json(found))
        except (RecordError, layout.RunDirError, OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"g{number:06d}: records/{role}.json: {exc}") from exc
        if record.data_class != "synthetic":
            raise ReportError("only synthetic, public-lineage runs are reported")


def _progress_facts(record: dict[str, Any]) -> ProgressFacts:
    by_id = {s["match_id"]: s for s in summary_of(record)}
    matches = []
    for match in record["matches"]:
        summary = by_id[match["match_id"]]
        matches.append(
            MatchFacts(
                match["match_id"],
                match["mode"],
                match["opponent"]["kind"],
                match["games"],
                match["wins"],
                match["ties"],
                match["losses"],
                summary["score_ppm"],
                summary["wilson_low_ppm"],
                summary["wilson_high_ppm"],
                match["turns_total"],
                match["turn_cap_ties"],
                tuple((f["side"], f["cause"], f["count"]) for f in match["forfeits"]),
                summary.get("comparison"),
            )
        )
    plan = record["plan"]
    return ProgressFacts(
        record["generation"],
        plan["games_per_match"],
        plan["recorded_games_per_match"],
        plan["max_turns"],
        tuple(matches),
    )


def read_progress_root(
    root: Path, config: RunConfig, manifest_shas: dict[int, str], inputs: dict[str, str]
) -> dict[int, ProgressFacts]:
    """Every committed generation of a progress root, verified against its manifest and the run."""
    found: dict[int, ProgressFacts] = {}
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue  # the lock and unfinished partials
        match = _PROGRESS_DIR.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise ReportError(f"{path.name} is not a generation of a progress root")
        number = int(match[1])
        if number not in manifest_shas:
            raise ReportError(f"{path.name}: the run has no such generation")
        try:
            manifest = layout.verify_generation(path, number)
            record = read_progress_record(_json(path / "progress.json"), config)
        except (layout.RunDirError, OSError, json.JSONDecodeError, ProgressRecordError) as exc:
            raise ReportError(f"{path.name}: {exc}") from exc
        if manifest.get("run_id") != config.run_key or record["generation"] != number:
            raise ReportError(f"{path.name}: it belongs to another run or generation")
        if record["generation_manifest_sha256"] != manifest_shas[number]:
            raise ReportError(f"{path.name}: its record is not of the run's committed generation")
        for name in ("progress.json", layout.MANIFEST):
            inputs[f"progress/{path.name}/{name}"] = sha256_hex((path / name).read_bytes())
        found[number] = _progress_facts(record)
    return found


def read_report_model(run_dir: Path, progress_dir: Path | None) -> ReportModel:
    """Read ``run_dir`` (and ``progress_dir`` if given) read-only; refuse anything inconsistent."""
    if progress_dir is not None and not progress_dir.is_dir():
        raise ReportError("the progress directory does not exist")
    config = read_run_dir_config(run_dir)
    refuse_non_public(config)
    inputs: dict[str, str] = {layout.CONFIG: sha256_hex((run_dir / layout.CONFIG).read_bytes())}
    generations = []
    shas: dict[int, str] = {}
    for number in generation_numbers(run_dir):
        path = layout.generation_dir(run_dir, number)
        try:
            layout.verify_generation(path, number)
        except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
            raise ReportError(str(exc)) from exc
        _check_synthetic(path, number)
        shas[number] = sha256_hex((path / layout.MANIFEST).read_bytes())
        inputs[f"generations/g{number:06d}/{layout.MANIFEST}"] = shas[number]
        if number == 0:
            generations.append(
                GenerationFacts(
                    0, shas[0], None, None, None, None, None, None, None, {}, None, None
                )
            )
        else:
            generations.append(_played_generation(number, path, config, shas[number], inputs))
    progress = (
        {} if progress_dir is None else read_progress_root(progress_dir, config, shas, inputs)
    )
    return ReportModel(
        config.run_key,
        config.lineage_id,
        config.format_id,
        config.hash,
        tuple(generations),
        progress,
        progress_dir is not None,
        dict(sorted(inputs.items())),
    )
