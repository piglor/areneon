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
stored); nothing is estimated. A fact the files do not hold (a generation's wall-clock time and
throughput when ``metrics/g<NNNNNN>.json`` is missing, a forfeit's cause in a generation written
before causes were recorded) is ``None`` or ``"not recorded"``, never zero. The metrics files
(ADR-010 erratum A1) sit outside every manifest and hash: they are read through the RunMetrics
reader, tied to the run and the generation, and their SHA-256 is listed among the page's inputs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
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
from areneon_ml.loop.metrics import METRICS_DIR, MetricsError, metrics_path, read_run_metrics
from areneon_ml.progress.ablation import ablation_summary
from areneon_ml.progress.records import (
    ProgressRecordError,
    preview_variant,
    read_progress_record,
)
from areneon_ml.progress.run import summary_of
from areneon_ml.report.move_use import class_files, progress_move_use

NOT_RECORDED = "not recorded"
ROLES = ("battle", "preview")
METRICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
STEPS_KEYS = ("steps", "optimiser_steps")
KL_STOP_KEYS = ("epochs_run", "kl_stopped")  # only when ppo.target_kl_micro is set (ADR-010 A1)
VALUE_WARMUP_KEY = "value_warmup"  # only on a value warm-up update (ADR-019 section 4)
ANCHOR_KEYS = ("anchor_kl", "anchor_beta_micro")  # only when ppo.anchor is set (ADR-019 section 4)
TRAIN_ROLE_KEYS = (
    frozenset({*METRICS, *STEPS_KEYS}),
    frozenset({*KL_STOP_KEYS, VALUE_WARMUP_KEY, *ANCHOR_KEYS}),
)
TRAIN_KEYS = (frozenset(ROLES), frozenset[str]())
_PROGRESS_DIR = re.compile(r"g(\d{6})")
_METRICS_FILE = re.compile(r"g(\d{6})\.json")


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
    # per role, (epochs run, whether the KL stop fired); a role is absent when the run has no target
    kl_stops: dict[str, tuple[int, bool]] = field(default_factory=dict)
    # per role, (mean anchor KL, beta in millionths); a role is absent when the run has no anchor
    anchors: dict[str, tuple[float, int]] = field(default_factory=dict)


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
    #: The RunMetrics record as stored; None when the generation has no metrics file.
    metrics: dict[str, Any] | None = None


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
    #: The candidate's wins, ties and losses by seat it sat in: (seat, wins, ties, losses).
    seats: tuple[tuple[str, int, int, int], ...] = ()
    #: "wilson" (a version 1 record, per game) or "pairs" (version 2, bootstrap over pairs).
    interval_kind: str = "wilson"
    #: Version 2 only: the pairs played (two games each).
    pairs: int = 0
    #: Anchor and previous-generation matches of a version 2 record: the pair score minus 1/2 as
    #: (mean, low, high) in parts per million (the page shows the anchors'); None for the others.
    diff: tuple[int, int, int] | None = None
    #: The generation of an anchor match's opponent; None for every other match.
    anchor_generation: int | None = None
    #: The candidate's Team Preview variant of a Preview ablation match (``learned``, ``random``
    #: or ``g<N>``); None for every other match.
    variant: str | None = None
    #: A damage-aware match (ADR-018): the baseline identity hash and the counts summed over its
    #: games (integers, as the record states them); None for every other match.
    baseline: dict[str, Any] | None = None
    #: A cross-run match (``run-<mode>``): the other run's key and generation; None otherwise.
    versus: tuple[str, int] | None = None


@dataclass(frozen=True)
class ProgressFacts:
    generation: int
    games_per_match: int
    recorded_games_per_match: int
    max_turns: int
    matches: tuple[MatchFacts, ...]
    #: Games in each anchor match (version 2 records that have anchors); None otherwise.
    anchor_games: int | None = None
    #: The Preview ablation (``ablation_summary``: integers only); empty without ablation matches.
    ablation: dict[str, Any] = field(default_factory=dict)
    #: The move-use diagnostic (#1164) over the generation's stored games, as ``move_use`` returns
    #: it; None when the generation stores no game or the repository has no move-class table.
    move_use: dict[str, Any] | None = None
    #: The hash of the other verified dataset the validation teams came from (the eval-dataset
    #: option, ADR-020 E-17.1); None: the validation split of the run's own dataset.
    evaluation_dataset: str | None = None


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
    #: The number of looks at validation results logged so far (None: no progress root given).
    looks: int | None = None


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
    stops: dict[str, tuple[int, bool]] = {}
    anchors: dict[str, tuple[float, int]] = {}
    for role in ROLES:
        values = closed(body[role], TRAIN_ROLE_KEYS, f"train-report.json {role}")
        roles[role] = {m: _number(values[m], f"{role} {m}") for m in METRICS}
        for key in STEPS_KEYS:
            roles[role][key] = integer(values[key], f"{role} {key}")
        if set(KL_STOP_KEYS) & set(values):
            fired = values.get("kl_stopped")
            if set(KL_STOP_KEYS) - set(values) or not isinstance(fired, bool):
                raise ReportError(f"{role} epochs_run and kl_stopped come together, a boolean stop")
            stops[role] = (integer(values["epochs_run"], f"{role} epochs_run", 1), fired)
        if VALUE_WARMUP_KEY in values and values[VALUE_WARMUP_KEY] is not True:
            raise ReportError(f"{role} value_warmup is true when present (ADR-019 section 4)")
        if set(ANCHOR_KEYS) & set(values):
            if set(ANCHOR_KEYS) - set(values):
                raise ReportError(f"{role} anchor_kl and anchor_beta_micro come together")
            kl = _number(values["anchor_kl"], f"{role} anchor_kl")
            if kl < 0:
                raise ReportError(f"{role} anchor_kl is a KL divergence and cannot be negative")
            anchors[role] = (kl, integer(values["anchor_beta_micro"], f"{role} anchor_beta_micro"))
    return TrainFacts(roles, stops, anchors)


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


def _seat_counts(
    rows: list[dict[str, Any]], match_id: str
) -> tuple[tuple[str, int, int, int], ...]:
    """The candidate's results by the seat it sat in, from the match's game rows."""
    out = []
    for seat in ("p1", "p2"):
        won = tied = lost = 0
        for row in rows:
            if row["match_id"] != match_id or row["candidate_seat"] != seat:
                continue
            outcome = row["outcome"]
            who = outcome["loser"] if outcome["kind"] == "forfeit" else outcome["winner"]
            if outcome["kind"] == "forfeit":
                won += who == "opponent"
                lost += who == "candidate"
            else:
                won += who == "candidate"
                tied += who == "tie"
                lost += who == "opponent"
        out.append((seat, won, tied, lost))
    return tuple(out)


def _progress_facts(record: dict[str, Any], move_use: dict[str, Any] | None) -> ProgressFacts:
    by_id = {s["match_id"]: s for s in summary_of(record)}
    matches = []
    for match in record["matches"]:
        summary = by_id[match["match_id"]]
        pairs = "interval_low_ppm" in summary
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
                summary["interval_low_ppm" if pairs else "wilson_low_ppm"],
                summary["interval_high_ppm" if pairs else "wilson_high_ppm"],
                match["turns_total"],
                match["turn_cap_ties"],
                tuple((f["side"], f["cause"], f["count"]) for f in match["forfeits"]),
                summary.get("comparison"),
                _seat_counts(record["games"], match["match_id"]),
                "pairs" if pairs else "wilson",
                summary.get("pairs", 0),
                (
                    (summary["diff_mean_ppm"], summary["diff_low_ppm"], summary["diff_high_ppm"])
                    if "diff_mean_ppm" in summary
                    else None
                ),
                match["opponent"]["generation"] if match["opponent"]["kind"] == "anchor" else None,
                (
                    preview_variant(match["candidate_preview"])
                    if match.get("candidate_preview") is not None
                    else None
                ),
                match.get("baseline"),
                (
                    (match["opponent"]["run_id"], match["opponent"]["generation"])
                    if match["opponent"]["kind"] == "run"
                    else None
                ),
            )
        )
    plan = record["plan"]
    return ProgressFacts(
        record["generation"],
        plan["games_per_match"],
        plan["recorded_games_per_match"],
        plan["max_turns"],
        tuple(matches),
        plan.get("anchor_games_per_match"),
        ablation_summary(record),
        move_use,
        record["teams"].get("evaluation_dataset_hash"),
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
        used = progress_move_use(root, number, config, record)
        if used is not None:
            for file in class_files(config.format_id):
                inputs[f"fixtures/showdown/{config.format_id}/{file.name}"] = sha256_hex(
                    file.read_bytes()
                )
        found[number] = _progress_facts(record, used)
    return found


def read_metrics(
    run_dir: Path, config: RunConfig, numbers: list[int], inputs: dict[str, str]
) -> dict[int, dict[str, Any]]:
    """The run-metrics record of each generation that has one; none is estimated for the rest.

    The directory may be absent (a run from before the records) or lack a generation (a crash
    between its commit and the write). A file for a generation the run does not hold, a file that
    is not named like a record or one that breaks the contract is refused.
    """
    base = run_dir / METRICS_DIR
    if not base.exists():
        return {}
    if base.is_symlink() or not base.is_dir():
        raise ReportError(f"{METRICS_DIR} is not a directory of the run")
    found: dict[int, dict[str, Any]] = {}
    for path in sorted(base.iterdir()):
        if path.name.startswith("."):
            continue  # a write that was interrupted
        match = _METRICS_FILE.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise ReportError(f"{METRICS_DIR}/{path.name} is not a metrics file of this run")
        number = int(match[1])
        if number not in numbers:
            raise ReportError(f"{METRICS_DIR}/{path.name}: the run has no such generation")
        try:
            found[number] = read_run_metrics(_json(path), config, number)
        except MetricsError as exc:
            raise ReportError(f"{METRICS_DIR}/{path.name}: {exc}") from exc
        inputs[f"{METRICS_DIR}/{path.name}"] = sha256_hex(
            metrics_path(run_dir, number).read_bytes()
        )
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
    metrics = read_metrics(run_dir, config, sorted(shas), inputs)
    progress = (
        {} if progress_dir is None else read_progress_root(progress_dir, config, shas, inputs)
    )
    return ReportModel(
        config.run_key,
        config.lineage_id,
        config.format_id,
        config.hash,
        tuple(replace(g, metrics=metrics.get(g.number)) for g in generations),
        progress,
        progress_dir is not None,
        dict(sorted(inputs.items())),
    )
