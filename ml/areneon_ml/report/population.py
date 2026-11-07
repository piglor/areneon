"""What a population run's files say (ADR-021 sections 4, 8 and 11, slice 11).

``read_population_model`` reads the generations of a run whose config has ``population``, after
each generation's manifest check: the version 2 ``league-run.json`` through its contract reader,
``plan.json`` against the record, ``generation.json`` (schema 2) with closed key sets,
``ladder.json`` through ``read_ladder_record`` (tied to the config, the previous record, the
frozen ``anchors.json`` of generation 0 and the manifest's ``ladder_hash``), the optional
``past-pool.json``, and every member's ``train-report.json``. It refuses a ladder record that does
not tie to the config or the manifest. Nothing is estimated: the page's numbers are these files'
integers, and a fact a file does not hold is ``None``.

A quarantined member (ADR-021 section 12) stores ``null`` for a non-finite number in its
``train-report.json`` and a ``null`` ``training_set_hash``: both are read as what they are, a value
that does not exist, never as zero.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, sha256_hex

from areneon_ml.checkpoint.record import RecordError, read_record
from areneon_ml.inspector.readers import InspectError, closed, integer
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import ladder_write, layout, past_pool
from areneon_ml.loop.ladder import LadderRecordError, read_ladder_record
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecordV2,
    check_population_plan,
    read_league_run_record_v2,
)
from areneon_ml.loop.past_pool import PastPoolError
from areneon_ml.loop.selection import quarantined
from areneon_ml.report.model import (
    METRICS,
    ROLES,
    GenerationFacts,
    ReportError,
    TrainFacts,
    read_train_report,
)

MEMBERS_METRICS_DIR = "members"
GENERATION_SCHEMA = 2
_MEMBER = re.compile(r"m(0|[1-9][0-9]?)")
GENERATION_KEYS = (
    frozenset({"forfeits", "games", "generation", "manifest", "members", "schema_version"}),
    frozenset({"team_hashes", "forfeit_causes"}),
)
GENERATION_MEMBER_KEYS = (
    frozenset({"episode_hashes", "examples", "member", "training_set_hash"}),
    frozenset[str](),
)
#: The float keys of a train report role: ``health.sanitize`` stores ``null`` for a non-finite one.
FLOAT_KEYS = (*METRICS, "explained_variance", "anchor_kl", "search_kl_before", "search_kl_after")
CALIBRATION_FLOATS = ("mean_value", "mean_score")


@dataclass(frozen=True)
class MemberGeneration:
    """One learning member in one generation, as the generation's files state it."""

    member: str
    games: int
    played: int
    forfeits: int
    episodes: int
    #: Forfeits by cause, sorted by cause.
    forfeit_causes: tuple[tuple[str, int], ...]
    #: ``train-report.json``; None when the member has none (never the case in a committed run).
    train: TrainFacts | None
    #: Per role, the report keys stored as ``null``: a quarantined member's non-finite numbers.
    nulls: tuple[tuple[str, tuple[str, ...]], ...]
    #: None for a member whose update was discarded (its manifest names no training set).
    training_set_hash: str | None
    examples: int


@dataclass(frozen=True)
class PopulationGeneration:
    number: int
    #: The validated ladder record; None for generation 0, which only freezes the anchors.
    ladder: dict[str, Any] | None
    league: LeagueRunRecordV2 | None
    members: tuple[MemberGeneration, ...]
    #: The snapshots in the past pool this generation plays (None without a past-pool share).
    pool_snapshots: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PopulationFacts:
    members: tuple[str, ...]
    #: The frozen anchor strengths of generation 0 (scale 10^9), sorted by anchor id.
    anchors: tuple[tuple[str, int], ...]
    generations: tuple[PopulationGeneration, ...]
    #: ``metrics/members/g<NNNNNN>.json``: generation -> member -> (games, decisions, steps).
    member_counts: dict[int, dict[str, tuple[int, int, int]]] = field(default_factory=dict)


def _json(path: Path) -> Any:
    try:
        return layout.read_json(path)
    except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"{path.parent.name}/{path.name}: {exc}") from exc


def is_population(config: RunConfig) -> bool:
    return isinstance(config.raw.get("population"), dict)


def read_member_train_report(
    raw: object,
) -> tuple[TrainFacts, tuple[tuple[str, tuple[str, ...]], ...]]:
    """A member's train report, with ``null`` accepted wherever ``health.sanitize`` writes one.

    ``sanitize`` replaces every non-finite float by ``null``: the metrics, the explained
    variance, the anchor and search KLs and the means of the value calibration bins. Integers and
    booleans are never null. The facts read a ``null`` as 0.0 and the second value lists, per
    role, the keys that were ``null`` (``value_calibration`` once for its bins), so the page can
    say "not finite" and never show the zero. The caller accepts a non-empty list only for a
    quarantined member.
    """
    if not isinstance(raw, dict):
        raise ReportError("train-report.json is not an object")
    cleaned: dict[str, Any] = {}
    nulls: list[tuple[str, tuple[str, ...]]] = []
    for role in ROLES:
        values = raw.get(role)
        if not isinstance(values, dict):
            cleaned[role] = values  # the strict reader names the problem
            continue
        found = {k for k in FLOAT_KEYS if k in values and values[k] is None}
        out = {k: (0.0 if k in found else v) for k, v in values.items()}
        bins = values.get("value_calibration")
        if isinstance(bins, list):
            fixed = []
            for item in bins:
                entry = item
                if isinstance(entry, dict):
                    nulled = [k for k in CALIBRATION_FLOATS if k in entry and entry[k] is None]
                    if nulled:
                        found.add("value_calibration")
                    entry = {k: (0.0 if k in nulled else v) for k, v in entry.items()}
                fixed.append(entry)
            out["value_calibration"] = fixed
        cleaned[role] = out
        if found:
            nulls.append((role, tuple(sorted(found))))
    for key, value in raw.items():
        if key not in ROLES:
            cleaned[key] = value
    return read_train_report(cleaned), tuple(nulls)


def _read_generation_file(raw: object, number: int, names: list[str]) -> list[dict[str, Any]]:
    body = closed(raw, GENERATION_KEYS, "generation.json")
    if body["schema_version"] != GENERATION_SCHEMA or body["generation"] != number:
        raise ReportError(f"g{number:06d}: generation.json is not this generation's version 2 file")
    members = body["members"]
    if not isinstance(members, list) or [m.get("member") for m in members] != names:
        raise ReportError(f"g{number:06d}: generation.json names other members than the run has")
    out = []
    for entry in members:
        found = closed(entry, GENERATION_MEMBER_KEYS, "generation.json member")
        integer(found["examples"], "generation.json examples")
        if found["training_set_hash"] is not None and not isinstance(
            found["training_set_hash"], str
        ):
            raise ReportError(f"g{number:06d}: a training_set_hash is a string or null")
        out.append(found)
    return out


def _tie_ladder(  # noqa: PLR0913, PLR0917
    raw: object,
    previous: object | None,
    config: RunConfig,
    manifest: dict[str, Any],
    anchors: dict[str, int],
    number: int,
) -> dict[str, Any]:
    try:
        record = read_ladder_record(raw, previous, config)
        ladder_write.check_anchors(record.raw, anchors)
    except (LadderRecordError, ladder_write.LadderBuildError) as exc:
        raise ReportError(f"g{number:06d}: ladder.json: {exc}") from exc
    if record.generation != number:
        raise ReportError(f"g{number:06d}: its ladder.json is another generation's")
    if manifest.get("ladder_hash") != ladder_write.json_hash(record.raw):
        raise ReportError(f"g{number:06d}: its ladder.json is not the one its manifest names")
    return record.raw


def _check_records(path: Path, number: int, labels: list[str]) -> None:
    for label in labels:
        for role in ROLES:
            found = path / "members" / label / "records" / f"{role}.json"
            if not found.exists():
                continue
            try:
                record = read_record(layout.read_json(found))
            except (
                RecordError,
                ContractError,
                layout.RunDirError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise ReportError(f"g{number:06d}: {label} records/{role}.json: {exc}") from exc
            if record.data_class != "synthetic":
                raise ReportError("only synthetic, public-lineage runs are reported")


def _past_pool(
    path: Path, number: int, config: RunConfig, manifest: dict[str, Any], inputs: dict[str, str]
) -> tuple[str, ...] | None:
    wanted = manifest.get("past_pool_hash")
    file = path / past_pool.FILE
    if wanted is None:
        if file.exists():
            raise ReportError(f"g{number:06d}: a past-pool.json its manifest does not name")
        return None
    raw = _json(file)
    if ladder_write.json_hash(raw) != wanted:
        raise ReportError(f"g{number:06d}: its past-pool.json is not the one its manifest names")
    try:
        found = past_pool.read_file(raw, config)
    except PastPoolError as exc:
        raise ReportError(f"g{number:06d}: past-pool.json: {exc}") from exc
    inputs[f"generations/g{number:06d}/{past_pool.FILE}"] = sha256_hex(file.read_bytes())
    return tuple(s.id for s in found.snapshots)


def _read_generation(  # noqa: PLR0913, PLR0917
    number: int,
    path: Path,
    config: RunConfig,
    manifest: dict[str, Any],
    anchors: dict[str, int],
    anchors_digest: str,
    previous: dict[str, Any] | None,
    before: dict[str, Any],
    inputs: dict[str, str],
) -> PopulationGeneration:
    count = config.raw["population"]["members"]
    names = [f"m{k}" for k in range(count)]
    held = manifest.get("members")
    if not isinstance(held, dict) or list(held) != names:
        raise ReportError(
            f"g{number:06d}: its manifest does not hold exactly {names[0]}..{names[-1]}"
        )
    plan_raw = _json(path / "plan.json")
    try:
        league = read_league_run_record_v2(_json(path / "league-run.json"), config)
        check_population_plan(league, plan_raw, config)
    except LeagueRecordError as exc:
        raise ReportError(f"g{number:06d}: {exc}") from exc
    if league.generation != number:
        raise ReportError(f"g{number:06d}: its league-run.json is another generation's")
    if manifest.get("anchors_hash") != anchors_digest:
        raise ReportError(
            f"g{number:06d}: its manifest names other anchor strengths than generation 0 froze"
        )
    file_members = _read_generation_file(_json(path / "generation.json"), number, names)
    ladder = _tie_ladder(
        _json(path / ladder_write.LADDER_FILE), previous, config, manifest, anchors, number
    )
    _check_records(path, number, names)
    stopped_members = set(quarantined(ladder))
    by_member = {m["member"]: m for m in league.raw["members"]}
    rows = []
    for entry, label in zip(file_members, names, strict=True):
        stated = by_member[label]
        # A member whose update was discarded (quarantine) has a null manifest hash while the
        # simulator side's generation.json still names the set it played.
        sealed = held[label].get("training_set_hash")
        stopped = label in stopped_members
        if (sealed is None) != stopped:
            raise ReportError(
                f"g{number:06d}: {label}'s manifest training set is null exactly when the ladder "
                "record quarantines it"
            )
        if sealed is not None and entry["training_set_hash"] != sealed:
            raise ReportError(f"g{number:06d}: {label}'s training set differs between its files")
        # The games were played by the pair frozen at the end of the generation before.
        if stated["checkpoints"] != before.get(label, {}).get("checkpoints"):
            raise ReportError(
                f"g{number:06d}: {label}'s pair is not the one the generation before froze"
            )
        report = path / "members" / label / "train-report.json"
        try:
            train, nulls = (
                (None, ()) if not report.exists() else read_member_train_report(_json(report))
            )
        except InspectError as exc:
            raise ReportError(f"g{number:06d}: {label}: {exc}") from exc
        if nulls and not stopped:
            raise ReportError(
                f"g{number:06d}: {label}'s train report holds a null but the ladder record "
                "does not quarantine it"
            )
        if report.exists():
            inputs[f"generations/g{number:06d}/members/{label}/train-report.json"] = sha256_hex(
                report.read_bytes()
            )
        rows.append(
            MemberGeneration(
                label,
                stated["games"],
                stated["played"],
                stated["forfeits"],
                stated["episodes"],
                tuple(sorted((c["cause"], c["count"]) for c in stated.get("forfeit_causes", []))),
                train,
                nulls,
                sealed,
                entry["examples"],
            )
        )
    for name in ("plan.json", "generation.json", "league-run.json", ladder_write.LADDER_FILE):
        inputs[f"generations/g{number:06d}/{name}"] = sha256_hex((path / name).read_bytes())
    pool = _past_pool(path, number, config, manifest, inputs)
    return PopulationGeneration(number, ladder, league, tuple(rows), pool)


def _member_counts(
    run_dir: Path, numbers: list[int], count: int, inputs: dict[str, str]
) -> dict[int, dict[str, tuple[int, int, int]]]:
    base = run_dir / "metrics" / MEMBERS_METRICS_DIR
    if not base.exists():
        return {}
    if base.is_symlink() or not base.is_dir():
        raise ReportError("metrics/members is not a directory of the run")
    out: dict[int, dict[str, tuple[int, int, int]]] = {}
    for path in sorted(base.iterdir()):
        if path.name.startswith("."):
            continue
        found = re.fullmatch(r"g(\d{6})\.json", path.name)
        if found is None or not path.is_file() or path.is_symlink() or int(found[1]) not in numbers:
            raise ReportError(f"metrics/members/{path.name} is not a members file of this run")
        number = int(found[1])
        body = closed(
            _json(path), (frozenset({"generation", "members"}), frozenset[str]()), path.name
        )
        if body["generation"] != number or not isinstance(body["members"], dict):
            raise ReportError(f"metrics/members/{path.name}: it is another generation's")
        names = [f"m{k}" for k in range(count)]
        if list(body["members"]) != names:
            raise ReportError(f"metrics/members/{path.name}: it names other members than the run")
        keys = (frozenset({"games", "decisions", "optimiser_steps"}), frozenset[str]())
        out[number] = {}
        for label, values in body["members"].items():
            v = closed(values, keys, f"{path.name} {label}")
            out[number][label] = tuple(  # type: ignore[assignment]
                integer(v[k], f"{path.name} {label} {k}")
                for k in ("games", "decisions", "optimiser_steps")
            )
        inputs[f"metrics/members/{path.name}"] = sha256_hex(path.read_bytes())
    return out


def read_population_model(
    run_dir: Path,
    config: RunConfig,
    numbers: list[int],
    shas: dict[int, str],
    inputs: dict[str, str],
) -> tuple[tuple[GenerationFacts, ...], PopulationFacts]:
    """The generations of a population run, verified; every refusal is a :class:`ReportError`."""
    try:
        return _read_population_model(run_dir, config, numbers, shas, inputs)
    except ReportError:
        raise
    except InspectError as exc:
        raise ReportError(str(exc)) from exc


def _read_population_model(
    run_dir: Path,
    config: RunConfig,
    numbers: list[int],
    shas: dict[int, str],
    inputs: dict[str, str],
) -> tuple[tuple[GenerationFacts, ...], PopulationFacts]:
    """The generations of a population run, verified; the per-generation totals and the facts."""
    count = config.raw["population"]["members"]
    names = tuple(f"m{k}" for k in range(count))
    zero = layout.generation_dir(run_dir, 0)
    anchors_raw = _json(zero / ladder_write.ANCHORS_FILE)
    try:
        anchors = ladder_write.read_anchors(anchors_raw, config)
    except ladder_write.LadderBuildError as exc:
        raise ReportError(f"g000000: anchors.json: {exc}") from exc
    for name in (ladder_write.ANCHORS_FILE, ladder_write.CALIBRATION_FILE):
        if (zero / name).exists():
            inputs[f"generations/g000000/{name}"] = sha256_hex((zero / name).read_bytes())
    digest = ladder_write.json_hash(anchors_raw)
    first = layout.read_json(zero / layout.MANIFEST)
    if first.get("anchors_hash") != digest:
        raise ReportError("g000000: its manifest names other anchor strengths than anchors.json")
    facts: list[GenerationFacts] = []
    gens: list[PopulationGeneration] = [
        PopulationGeneration(0, None, None, (), _past_pool(zero, 0, config, first, inputs))
    ]
    facts.append(
        GenerationFacts(0, shas[0], None, None, None, None, None, None, None, {}, None, None)
    )
    previous: dict[str, Any] | None = None
    before = first.get("members") if isinstance(first.get("members"), dict) else {}
    for number in numbers[1:]:
        path = layout.generation_dir(run_dir, number)
        manifest = layout.read_json(path / layout.MANIFEST)
        current = _read_generation(
            number, path, config, manifest, anchors, digest, previous, before, inputs
        )
        previous = current.ladder
        before = manifest["members"]
        gens.append(current)
        league = current.league
        assert league is not None  # a played generation always has its record
        causes = dict(league.forfeit_causes)
        facts.append(
            GenerationFacts(
                number,
                shas[number],
                None,
                league.games,
                league.played,
                league.forfeits,
                league.episodes,
                sum(m.examples for m in current.members),
                None,
                {},
                dict(sorted(causes.items())),
                None,
            )
        )
    counts = _member_counts(run_dir, numbers, count, inputs)
    return tuple(facts), PopulationFacts(names, tuple(sorted(anchors.items())), tuple(gens), counts)
