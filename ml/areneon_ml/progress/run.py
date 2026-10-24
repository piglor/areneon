"""One generation's progress evaluation, and the progress root (ADR-017 sections 2 and 3).

``progress_generation`` reads a committed generation of a run directory (read-only: the run's own
lock is never taken, because only committed generations are read), has the simulator side play the
frozen pair against the scripted baselines on the validation teams, and commits the result into the
progress root ``<run-dir>.progress/`` as ``g<NNNNNN>/``: ``progress.json`` (a
ProgressEvaluationRecord), ``games/<match>-<k>.json`` (ProgressGameRecords) and ``manifest.json``
(SHA-256 of every other file), written last. The directory is built in ``.g<NNNNNN>.partial``,
fsynced, and made real by one ``os.replace``. A committed directory is verified and never rewritten;
a failed evaluation commits nothing and removes its partial directory.

This is a progress view, never the sealed Training gate: nothing here is read back by the loop.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.play import NodePlayer, PlayError
from areneon_ml.loop.records import policy_id
from areneon_ml.progress.ablation import ablation_summary, expected_ablation_matches
from areneon_ml.progress.records import (
    DAMAGE_AWARE_LABEL,
    RANDOM_PREVIEW_POLICY,
    SCHEMA_V2,
    ProgressRecordError,
    check_game_baseline,
    is_damage_aware,
    is_v2,
    read_game_record,
    read_progress_record,
    record_hash,
)
from areneon_ml.progress.seeds import check_not_training_seeds
from areneon_ml.progress.stats import (
    CONDITIONAL_LABEL,
    PAIR_INTERVAL_LABEL,
    PPM,
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
    wilson,
)

PROGRESS = "progress.json"
GAMES = "games"
SCRIPT = "packages/team-dataset/scripts/progress.ts"
#: Games per baseline match (even: every pair is played from both seats, ADR-017 erratum E2).
DEFAULT_GAMES = 100
DEFAULT_RECORD_GAMES = 8
#: Games per anchor match (even): each of this many / 2 validation pairs from both seats.
DEFAULT_ANCHOR_GAMES = 200
MIN_GAMES = 2
#: ``--all`` evaluates the committed generations whose number is a multiple of this, 0 included.
SCHEDULE_EVERY = 5
#: The working budget for one damage-aware match, in minutes (ADR-018 section 6).
DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES = 30
#: The baseline's policy id (the TS side derives it from ``damageAwarePolicyId("v1")``).
DAMAGE_AWARE_POLICY = "damage-aware@v1"
#: The damage-aware baseline's match ids.
DAMAGE_AWARE_MATCHES = ("damage-aware-argmax", "damage-aware-sample")
#: Says what a ``previous-*`` row is: the pair's own lineage, so it can cycle (ADR-017 section 2).
SEEN_LABEL = "seen, same-lineage comparison: it can cycle"


class ProgressError(ValueError):
    """The progress evaluation cannot be made or its files disagree: refused, nothing committed."""


@dataclass(frozen=True)
class PreviousPair:
    """The previous generation's frozen pair, the opponent of the ``previous-*`` matches."""

    preview: str
    battle: str
    policy_id_sample: str
    policy_id_argmax: str


@dataclass(frozen=True)
class AnchorPair:
    """A fixed earlier generation's frozen pair, the opponent of its ``anchor-g<N>-*`` matches."""

    generation: int
    preview: str
    battle: str
    policy_id_sample: str
    policy_id_argmax: str


@dataclass(frozen=True)
class AblationSpec:
    """The Preview ablation (#940): this generation's Battle network with its own
    learned Preview, a uniformly random Preview and an earlier generation's Preview."""

    generation: int
    preview: str
    policy_id_argmax: str
    random_policy_id: str


@dataclass(frozen=True)
class ProgressRequest:
    """What the simulator side needs to play one generation's progress evaluation."""

    generation: int
    preview: str
    battle: str
    policy_id_sample: str
    policy_id_argmax: str
    manifest_sha256: str
    team_set: Path
    games: int
    record_games: int
    out_dir: Path
    #: None at generation 0, which has no previous generation (no ``previous-*`` matches then).
    previous: PreviousPair | None = None
    #: Fixed earlier generations to play (each earlier than ``generation``), both seats per pair.
    anchors: tuple[AnchorPair, ...] = ()
    #: Games in each anchor match (even); 0 with no anchors.
    anchor_games: int = 0
    #: The Preview ablation, played instead of the usual matches (no previous, no anchors).
    ablation: AblationSpec | None = None
    #: The most wall-clock milliseconds one damage-aware match may take; None plays no damage-aware
    #: match (ADR-018 slice 4: opt-in, and a match over the budget fails the evaluation).
    damage_aware_budget_ms: int | None = None


class ProgressPlayer(Protocol):
    def progress(self, request: ProgressRequest) -> dict[str, Any]: ...


class NodeProgressPlayer(NodePlayer):
    """Runs ``packages/team-dataset/scripts/progress.ts`` as a child process."""

    def progress(self, request: ProgressRequest) -> dict[str, Any]:
        previous = request.previous
        more = (
            ()
            if previous is None
            else (
                *("--previous-preview", previous.preview),
                *("--previous-battle", previous.battle),
                *("--previous-policy-id-sample", previous.policy_id_sample),
                *("--previous-policy-id-argmax", previous.policy_id_argmax),
            )
        )
        anchors = tuple(
            arg
            for a in request.anchors
            for arg in (
                "--anchor",
                f"{a.generation},{a.preview},{a.battle},{a.policy_id_sample},{a.policy_id_argmax}",
            )
        ) + (("--anchor-games", str(request.anchor_games)) if request.anchors else ())
        ablation = (
            ()
            if request.ablation is None
            else (
                "--ablation-preview",
                f"{request.ablation.generation},{request.ablation.preview},"
                f"{request.ablation.policy_id_argmax}",
                *("--ablation-random-policy-id", request.ablation.random_policy_id),
            )
        )
        damage = (
            ()
            if request.damage_aware_budget_ms is None
            else ("--damage-aware-budget-ms", str(request.damage_aware_budget_ms))
        )
        out = self._run(
            SCRIPT,
            *("--config", str(self.config_path)),
            *("--generation", str(request.generation)),
            *("--preview", request.preview),
            *("--battle", request.battle),
            *("--policy-id-sample", request.policy_id_sample),
            *("--policy-id-argmax", request.policy_id_argmax),
            *("--generation-manifest-sha256", request.manifest_sha256),
            *("--team-set", str(request.team_set)),
            *("--run-dir", str(self.run_dir)),
            *("--engine-data", str(self.engine_data)),
            *("--encoder-version", self.encoder_version),
            *("--worker-python", self.worker_python),
            *("--games", str(request.games)),
            *("--record-games", str(request.record_games)),
            *("--out", str(request.out_dir)),
            *more,
            *anchors,
            *ablation,
            *damage,
        )
        try:
            found = json.loads(out)
        except json.JSONDecodeError as exc:
            raise PlayError(f"{SCRIPT} did not print a JSON line") from exc
        if not isinstance(found, dict):
            raise PlayError(f"{SCRIPT} did not print an object")
        return found


def progress_root(run_dir: Path, progress_dir: Path | None) -> Path:
    """The progress root: ``--progress-dir``, else ``<run-dir>.progress``; never inside the run."""
    run = run_dir.resolve()
    root = (
        progress_dir if progress_dir is not None else run.with_name(run.name + ".progress")
    ).resolve()
    if root == run or run in root.parents:
        raise ProgressError("the progress root must not be inside the run directory")
    if root in run.parents:
        raise ProgressError("the progress root must not contain the run directory")
    return root


def progress_dir_of(root: Path, generation: int) -> Path:
    return root / layout.generation_name(generation)


def _partial(root: Path, generation: int) -> Path:
    return root / f".{layout.generation_name(generation)}.partial"


@contextmanager
def progress_lock(root: Path) -> Iterator[None]:
    """Hold ``<root>/.lock``: two progress commands never run on one root at once."""
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if created:
        layout.sync_dir(root)
        layout.sync_dir(root.parent)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProgressError(f"another progress command holds {root / '.lock'}") from exc
        yield


def max_pairs(config: RunConfig) -> int:
    """The most disjoint validation pairs the run's dataset can give: half its validation teams.

    Read from the dataset's manifest before anything starts. It is an upper bound (the Node side
    also drops teams with an excluded ability and refuses exactly), so a refusal here is certain.
    """
    path = Path(config.raw["team_dataset"]["path"]) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        teams = next(s["teams"] for s in manifest["splits"] if s["split"] == "validation")
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise ProgressError(f"cannot read the validation split's size from {path}") from exc
    return int(teams) // 2


def _read_config(run_dir: Path) -> RunConfig:
    stored = layout.read_json(run_dir / layout.CONFIG)
    config = read_run_config(stored)
    return config


@dataclass(frozen=True)
class _Source:
    """What a committed generation says: its manifest and hash, the frozen pair's ids."""

    generation: int
    manifest: dict[str, Any]
    manifest_sha256: str
    preview: str
    battle: str
    team_set: Path
    version_manifest: dict[str, Any]


def _source(run_dir: Path, config: RunConfig, generation: int) -> _Source:
    """Verify the committed generation and its records; refuse a non-synthetic run."""
    if not config.team_dataset:
        raise ProgressError(
            "the run config names no team dataset: the interim pool has no validation split"
        )
    newest = layout.latest_complete(run_dir)
    if newest is None or not 0 <= generation <= newest.generation:
        raise ProgressError(f"generation {generation} is not committed in {run_dir}")
    path = layout.generation_dir(run_dir, generation)
    facts = (
        newest.manifest
        if newest.generation == generation
        else layout.verify_generation(path, generation)
    )
    if facts.get("run_config_hash") != config.hash:
        raise ProgressError("the generation was committed under another run config")
    checkpoints = facts.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"preview", "battle"}:
        raise ProgressError("the generation's manifest names no frozen pair")
    version_manifest: dict[str, Any] = {}
    for role in ("preview", "battle"):
        record = read_record(layout.read_json(path / "records" / f"{role}.json"))
        version_manifest = record.manifest
        if record.data_class != "synthetic":
            raise ProgressError("a progress evaluation runs on synthetic, public-lineage runs only")
        if record.checkpoint_id != checkpoints[role] or record.role != role:
            raise ProgressError(f"records/{role}.json names another checkpoint than the manifest")
    team_set = newest.path / "team-set.json"
    return _Source(
        generation,
        facts,
        sha256_hex((path / layout.MANIFEST).read_bytes()),
        checkpoints["preview"],
        checkpoints["battle"],
        team_set,
        version_manifest,
    )


def _check_previous(record: dict[str, Any], previous: PreviousPair | None, *, legacy: bool) -> None:
    """The ``previous-*`` matches are exactly the previous generation's frozen pair in both modes.

    They are opt-in (ADR-017 E2). ``previous`` is None when none were asked for (always at
    generation 0): the record then holds none. Otherwise there are two (sample and argmax),
    naming the checkpoints of the generation before; ``legacy`` is a committed record, which may
    hold none (committed without them), but never a wrong or half set.
    """
    found = {m["match_id"]: m for m in record["matches"] if m["opponent"]["kind"] == "previous"}
    if previous is None:
        if found:
            raise ProgressError("this evaluation asks for no previous-generation matches")
        return
    if not found and legacy:
        return
    if set(found) != {"previous-argmax", "previous-sample"}:
        raise ProgressError("previous-generation matches are previous-sample and previous-argmax")
    ids = {"sample": previous.policy_id_sample, "argmax": previous.policy_id_argmax}
    for match in found.values():
        opponent = match["opponent"]
        if (
            match["match_id"] != f"previous-{match['mode']}"
            or opponent["checkpoints"] != {"preview": previous.preview, "battle": previous.battle}
            or opponent["policy_id"] != ids[match["mode"]]
        ):
            raise ProgressError(
                f"{match['match_id']}: the opponent is not the previous generation's frozen pair"
            )


def _check_damage_aware(record: dict[str, Any], asked: bool, *, legacy: bool) -> None:
    """The damage-aware matches are exactly the two asked for (a fresh evaluation), each naming the
    baseline's policy id ``damage-aware@v1``, and share one identity hash (the reader checks it).

    ``legacy`` is a committed record, which may hold none (committed without them), never a wrong
    or half set.
    """
    found = {
        m["match_id"]: m
        for m in record["matches"]
        if m["opponent"]["kind"] == "baseline" and is_damage_aware(m["opponent"]["policy_id"])
    }
    if not asked and not legacy:
        if found:
            raise ProgressError("this evaluation asks for no damage-aware matches")
        return
    if not found and legacy:
        return
    if set(found) != set(DAMAGE_AWARE_MATCHES):
        raise ProgressError("damage-aware matches are damage-aware-argmax and damage-aware-sample")
    for match in found.values():
        if match["match_id"] != f"damage-aware-{match['mode']}" or match["opponent"] != {
            "kind": "baseline",
            "policy_id": DAMAGE_AWARE_POLICY,
        }:
            raise ProgressError(f"{match['match_id']}: the opponent is not {DAMAGE_AWARE_POLICY}")


def _damage_aware_status(record: dict[str, Any]) -> str:
    """Says what the record holds of the damage-aware matches, in words for the operator."""
    if any(m.get("baseline") is not None for m in record["matches"]):
        return "played"
    return (
        "none: damage-aware matches are opt-in (--damage-aware); a committed generation is never "
        "rewritten, so evaluate into a new --progress-dir to add them"
    )


def _damage_budget_ms(minutes: int | None, preview_ablation: bool) -> int | None:
    """The wall-clock budget of one damage-aware match in milliseconds, or None for no match."""
    if minutes is None:
        return None
    if minutes < 1 or preview_ablation:
        raise ProgressError(
            "--damage-aware-budget-minutes must be a whole number of minutes of 1 or more, "
            "and the Preview ablation plays no damage-aware match"
        )
    return minutes * 60_000


def _previous_pair(run_dir: Path, config: RunConfig, generation: int) -> PreviousPair | None:
    """The frozen pair of generation ``generation - 1`` (verified), or None at generation 0."""
    if generation == 0:
        return None
    before = _source(run_dir, config, generation - 1)
    return PreviousPair(
        before.preview,
        before.battle,
        policy_id(before.preview, before.battle, "sample"),
        policy_id(before.preview, before.battle, "argmax"),
    )


def _anchor_pair(run_dir: Path, config: RunConfig, generation: int) -> AnchorPair:
    """The frozen pair of a committed generation (verified), as an opponent."""
    found = _source(run_dir, config, generation)
    return AnchorPair(
        generation,
        found.preview,
        found.battle,
        policy_id(found.preview, found.battle, "sample"),
        policy_id(found.preview, found.battle, "argmax"),
    )


def _check_anchors(
    record: dict[str, Any], anchors: Sequence[AnchorPair] | None, run_dir: Path, config: RunConfig
) -> None:
    """Every anchor match plays exactly its generation's frozen pair, read from its own records.

    ``anchors`` is the set asked for (a fresh evaluation: all of them, both modes); None checks
    only what the record holds (a committed directory may hold any subset of earlier anchors).
    """
    found: dict[int, dict[str, dict[str, Any]]] = {}
    for match in record["matches"]:
        if match["opponent"]["kind"] == "anchor":
            found.setdefault(match["opponent"]["generation"], {})[match["mode"]] = match
    if anchors is not None and set(found) != {a.generation for a in anchors}:
        raise ProgressError("the record's anchor matches are not the anchors asked for")
    for generation, modes in found.items():
        if set(modes) != {"argmax", "sample"}:
            raise ProgressError(f"anchor g{generation}: both sample and argmax must be played")
        pair = _anchor_pair(run_dir, config, generation)
        ids = {"sample": pair.policy_id_sample, "argmax": pair.policy_id_argmax}
        for mode, match in modes.items():
            opponent = match["opponent"]
            if (
                opponent["checkpoints"] != {"preview": pair.preview, "battle": pair.battle}
                or opponent["policy_id"] != ids[mode]
            ):
                raise ProgressError(
                    f"{match['match_id']}: the opponent is not generation {generation}'s pair"
                )


def _verify_output(  # noqa: PLR0913, PLR0917
    work: Path,
    config: RunConfig,
    source: _Source,
    games: int,
    record_games: int,
    previous: PreviousPair | None,
    anchors: Sequence[AnchorPair],
    run_dir: Path,
    anchor_games: int,
    ablation: AblationSpec | None = None,
    damage_aware: bool = False,
) -> dict[str, Any]:
    """The simulator side's files are the record's: read both kinds, check every hash and name."""
    record = read_progress_record(layout.read_json(work / PROGRESS), config)
    plan = record["plan"]
    if (
        record["generation"] != source.generation
        or record["generation_manifest_sha256"] != source.manifest_sha256
        or record["checkpoints"] != {"preview": source.preview, "battle": source.battle}
        or plan["games_per_match"] != games
        or plan["recorded_games_per_match"] != record_games
        or record["version_manifest"] != source.version_manifest
    ):
        raise ProgressError("progress.json is not the record asked for")
    if record["schema_version"] != SCHEMA_V2:
        raise ProgressError("a new evaluation is a schema version 2 record")
    _check_previous(record, previous, legacy=False)
    _check_anchors(record, anchors, run_dir, config)
    if plan.get("anchor_games_per_match", 0) != anchor_games:
        raise ProgressError("progress.json is not the record asked for")
    _check_ablation(record, ablation, run_dir, config, fresh=True)
    _check_damage_aware(record, damage_aware, legacy=False)
    rows = {(r["match_id"], r["game"]): r for r in record["games"]}
    expected = set()
    for match in record["matches"]:
        listed = []
        for game in range(min(record_games, match["games"])):
            name = f"{match['match_id']}-{game}.json"
            expected.add(name)
            stored = read_game_record(layout.read_json(work / GAMES / name))
            if (
                stored["run_id"] != record["run_id"]
                or stored["generation"] != source.generation
                or stored["match_id"] != match["match_id"]
                or stored["game"] != game
                or stored.get("pair") != game // 2
                or stored["version_manifest"] != record["version_manifest"]
            ):
                raise ProgressError(f"games/{name} is not the game it is named for")
            try:
                check_game_baseline(stored, match, rows[(match["match_id"], game)])
            except ProgressRecordError as exc:
                raise ProgressError(f"games/{name}: {exc}") from exc
            listed.append(record_hash(stored))
        if sorted(listed) != match["game_record_hashes"]:
            raise ProgressError(
                f"{match['match_id']}: the stored games differ from game_record_hashes"
            )
    found = {p.name for p in (work / GAMES).glob("*")} if (work / GAMES).is_dir() else set()
    if found != expected:
        raise ProgressError("games/ holds other files than the record lists")
    return record


def _check_ablation(
    record: dict[str, Any],
    ablation: AblationSpec | None,
    run_dir: Path,
    config: RunConfig,
    *,
    fresh: bool,
) -> None:
    """The Preview ablation matches are exactly the six asked for (a fresh evaluation), and an
    earlier generation's Preview is that generation's own, read from its records."""
    found = [m for m in record["matches"] if m.get("candidate_preview") is not None]
    if fresh:
        asked = [] if ablation is None else expected_ablation_matches(ablation)
        if sorted(m["match_id"] for m in found) != sorted(asked) or (
            ablation is not None and len(found) != len(record["matches"])
        ):
            raise ProgressError("the record's ablation matches are not the ones asked for")
    for match in found:
        preview = match["candidate_preview"]
        if preview["kind"] == "generation":
            theirs = _source(run_dir, config, preview["generation"]).preview
            if preview["checkpoint"] != theirs:
                raise ProgressError(
                    f"{match['match_id']}: the Preview is not generation "
                    f"{preview['generation']}'s own"
                )


def summary_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Per match: the counts and the derived score with its interval (never stored).

    A version 1 record (one game per pair) gets the per-game Wilson interval. A version 2 record
    plays every pair from both seats, so its games are not independent: it gets the pair-level
    percentile bootstrap of ADR-017 erratum E2 instead (``pairs``, ``interval_low_ppm``,
    ``interval_high_ppm``, no Wilson keys), seeded from the run key; a match against an anchor
    or the previous generation also carries ``diff_*``, the pair score minus 1/2.
    """
    out = []
    pairs_version = record["schema_version"] == SCHEMA_V2
    for match in record["matches"]:
        if pairs_version:
            stats = _pair_stats(record, match)
        else:
            interval = wilson(match["wins"], match["ties"], match["games"])
            stats = {
                "score_ppm": interval.score_ppm,
                "wilson_low_ppm": interval.low_ppm,
                "wilson_high_ppm": interval.high_ppm,
                "interval": CONDITIONAL_LABEL,
            }
        out.append(
            {
                "match_id": match["match_id"],
                "games": match["games"],
                "wins": match["wins"],
                "ties": match["ties"],
                "losses": match["losses"],
            }
            | stats
            | {
                "opponent": match["opponent"]["kind"],
                "forfeits": sum(f["count"] for f in match["forfeits"]),
                "turn_cap_ties": match["turn_cap_ties"],
            }
            | (
                {"comparison": SEEN_LABEL}
                if match["opponent"]["kind"] in ("previous", "anchor")
                else {}
            )
            | (
                {"comparison": DAMAGE_AWARE_LABEL, "baseline": match["baseline"]}
                if "baseline" in match
                else {}
            )
        )
    return out


def _pair_stats(record: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    scores = match_pair_scores(record, match["match_id"])
    interval = pair_interval(
        scores,
        bootstrap_stream_key(record["run_id"], record["generation"], f"match:{match['match_id']}"),
    )
    out: dict[str, Any] = {
        "score_ppm": (2 * match["wins"] + match["ties"]) * PPM // (2 * match["games"]),
        "pairs": interval.pairs,
        "interval_low_ppm": interval.low_ppm,
        "interval_high_ppm": interval.high_ppm,
        "interval": PAIR_INTERVAL_LABEL,
    }
    if match["opponent"]["kind"] in ("previous", "anchor"):
        # pair score minus 1/2: the same resamples, shifted, so the bounds shift exactly
        out |= {
            "diff_mean_ppm": interval.mean_ppm - PPM // 2,
            "diff_low_ppm": interval.low_ppm - PPM // 2,
            "diff_high_ppm": interval.high_ppm - PPM // 2,
        }
    return out


def verify_progress_dir(
    target: Path, config: RunConfig, run_dir: Path, generation: int
) -> dict[str, Any]:
    """A committed progress directory, checked against its manifest and the run."""
    manifest = layout.verify_generation(target, generation)
    source = _source(run_dir, config, generation)
    record = read_progress_record(layout.read_json(target / PROGRESS), config)
    if (
        record["generation_manifest_sha256"] != source.manifest_sha256
        or record["generation"] != generation
    ):
        raise ProgressError(f"{target.name}: the record is not of the committed generation")
    # Previous-generation matches are optional (a record may hold them or not).
    _check_previous(record, _previous_pair(run_dir, config, generation), legacy=True)
    _check_anchors(record, None, run_dir, config)
    _check_ablation(record, None, run_dir, config, fresh=False)
    # The damage-aware matches are optional (a record may hold them or not), never half a set.
    _check_damage_aware(record, False, legacy=True)
    rows = {(r["match_id"], r["game"]): r for r in record["games"]}
    for match in record["matches"]:
        for name in sorted(p.name for p in (target / GAMES).glob(f"{match['match_id']}-*.json")):
            stored = read_game_record(layout.read_json(target / GAMES / name))
            if is_v2(record) and stored.get("pair") != stored["game"] // 2:
                raise ProgressError(f"games/{name}: a both-seats game must name its pair")
            if is_v2(record):
                try:
                    check_game_baseline(stored, match, rows[(match["match_id"], stored["game"])])
                except (ProgressRecordError, KeyError) as exc:
                    raise ProgressError(f"games/{name}: {exc}") from exc
    if manifest.get("run_id") != config.run_key:
        raise ProgressError(f"{target.name}: its manifest names another run")
    return record


def progress_generation(  # noqa: PLR0913, PLR0915, PLR0917
    run_dir: Path,
    generation: int,
    root: Path,
    player: ProgressPlayer,
    games: int = DEFAULT_GAMES,
    record_games: int = DEFAULT_RECORD_GAMES,
    anchors: Sequence[int] = (),
    anchor_games: int = DEFAULT_ANCHOR_GAMES,
    previous_matches: bool = False,
    preview_ablation: bool = False,
    damage_aware_budget_minutes: int | None = None,
) -> dict[str, Any]:
    """Evaluate one generation into ``root``; a committed directory is verified and kept.

    ``preview_ablation`` plays the Team Preview ablation (#940) instead of the usual
    matches: this generation's Battle network with its own Preview, a random Preview and
    generation 0's Preview, argmax, against both baselines. It is played alone (no anchors, no
    previous-generation matches) and needs a generation of 1 or more.

    Every pair is played from both seats (version 2 records), so ``games`` is even. ``anchors``
    are fixed earlier generations to play as well (one that is not earlier than ``generation`` is
    left out: a generation is never its own anchor); ``anchor_games`` games per anchor match.
    The previous-generation matches are played only with ``previous_matches`` (and never at
    generation 0, which has no previous generation).
    """
    if games < MIN_GAMES or games % 2 or record_games < 0 or record_games > games:
        raise ProgressError(
            "--games must be an even number of at least 2 (every pair is played from both seats) "
            "and --record-games between 0 and --games"
        )
    if any(a < 0 for a in anchors):
        raise ProgressError("--anchors must be generation numbers of 0 or more")
    wanted = sorted({a for a in anchors if a < generation})
    dropped = sorted({a for a in anchors if a >= generation})
    if wanted and (anchor_games < MIN_GAMES or anchor_games % 2):
        raise ProgressError("--anchor-games must be an even number of at least 2")
    if preview_ablation and (anchors or previous_matches or generation < 1):
        raise ProgressError(
            "the Preview ablation is played alone, for a generation of 1 or more: "
            "no --anchors and no --previous-matches"
        )
    if preview_ablation:
        wanted, dropped = [], []
    damage_budget_ms = _damage_budget_ms(damage_aware_budget_minutes, preview_ablation)
    config = _read_config(run_dir)
    with progress_lock(root):
        target = progress_dir_of(root, generation)
        for stale in sorted(root.glob(".g*.partial")):
            shutil.rmtree(stale)
        if target.exists():
            record = verify_progress_dir(target, config, run_dir, generation)
            if preview_ablation != bool(ablation_summary(record)):
                raise ProgressError(
                    f"{target.name} already holds a "
                    f"{'normal' if preview_ablation else 'Preview ablation'} record: "
                    "use a progress directory of its own for the other kind"
                )
            return {
                "generation": generation,
                "skipped": True,
                "previous": _previous_status(record, generation),
                "anchors": _anchor_status(record, wanted, dropped),
                "damage_aware": _damage_aware_status(record),
                "matches": summary_of(record),
            } | _ablation_part(record)
        source = _source(run_dir, config, generation)
        previous = _previous_pair(run_dir, config, generation) if previous_matches else None
        ablation = _ablation_spec(run_dir, config, source) if preview_ablation else None
        anchor_pairs = tuple(_anchor_pair(run_dir, config, a) for a in wanted)
        played_anchor_games = anchor_games if wanted else 0
        needed = max(games, played_anchor_games) // 2
        limit = max_pairs(config)
        if needed > limit:
            raise ProgressError(
                f"--games {games} / --anchor-games {played_anchor_games} need {needed} disjoint "
                f"validation team pairs: the dataset gives at most {limit}"
            )
        try:
            check_not_training_seeds(run_dir, config.run_key, needed)
        except ValueError as exc:
            raise ProgressError(str(exc)) from exc
        teams = layout.read_json(source.team_set)
        if sha256_hex(jcs(teams)) != layout.read_json(source.team_set.parent / layout.MANIFEST).get(
            "team_set_manifest_hash"
        ):
            raise ProgressError("team-set.json differs from its generation's manifest")
        work = _partial(root, generation)
        work.mkdir()
        try:
            facts = player.progress(
                ProgressRequest(
                    generation=generation,
                    preview=source.preview,
                    battle=source.battle,
                    policy_id_sample=policy_id(source.preview, source.battle, "sample"),
                    policy_id_argmax=policy_id(source.preview, source.battle, "argmax"),
                    manifest_sha256=source.manifest_sha256,
                    team_set=source.team_set,
                    games=games,
                    record_games=record_games,
                    out_dir=work,
                    previous=previous,
                    anchors=anchor_pairs,
                    anchor_games=played_anchor_games,
                    ablation=ablation,
                    damage_aware_budget_ms=damage_budget_ms,
                )
            )
            record = _verify_output(
                work,
                config,
                source,
                games,
                record_games,
                previous,
                anchor_pairs,
                run_dir,
                played_anchor_games,
                ablation,
                damage_aware_budget_minutes is not None,
            )
            layout.fsync_tree(work)
            layout.write_json(
                work / layout.MANIFEST,
                {
                    "schema_version": 1,
                    "generation": generation,
                    "run_id": config.run_key,
                    "files": layout.hash_tree(work),
                },
            )
            layout.sync_dir(work)
            os.replace(work, target)
            layout.sync_dir(root)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return {
            "generation": generation,
            "skipped": False,
            "previous": _previous_status(record, generation),
            "anchors": _anchor_status(record, wanted, dropped),
            "damage_aware": _damage_aware_status(record),
            "wall_ms": facts.get("wall_ms", {}),
            "game_ms": facts.get("game_ms", {}),
            "matches": summary_of(record),
        } | _ablation_part(record)


def _ablation_spec(run_dir: Path, config: RunConfig, source: _Source) -> AblationSpec:
    """Generation 0's Preview (verified) with this generation's Battle, and the random Preview."""
    first = _source(run_dir, config, 0)
    return AblationSpec(
        0,
        first.preview,
        policy_id(first.preview, source.battle, "argmax"),
        policy_id(RANDOM_PREVIEW_POLICY, source.battle, "argmax"),
    )


def _ablation_part(record: dict[str, Any]) -> dict[str, Any]:
    """The Preview ablation's scores and paired differences, when the record is one."""
    found = ablation_summary(record)
    return {"ablation": found} if found else {}


def _previous_status(record: dict[str, Any], generation: int) -> str:
    """Says what the record holds of the previous-generation matches, in words for the operator."""
    if generation == 0:
        return "none: generation 0 has no previous generation"
    if any(m["opponent"]["kind"] == "previous" for m in record["matches"]):
        return "played"
    return (
        "none: previous-generation matches are opt-in (--previous-matches); a committed "
        "generation is never rewritten, so evaluate into a new --progress-dir to add them"
    )


def _anchor_status(
    record: dict[str, Any], wanted: Sequence[int], skipped: Sequence[int] = ()
) -> str:
    """Says which anchors the record holds, and which asked for are missing."""
    held = sorted(
        {
            m["opponent"]["generation"]
            for m in record["matches"]
            if m["opponent"]["kind"] == "anchor"
        }
    )
    missing = [a for a in wanted if a not in held]
    text = "played: " + ", ".join(f"g{a}" for a in held) if held else "none requested or played"
    if missing:
        text += (
            "; missing: "
            + ", ".join(f"g{a}" for a in missing)
            + " (committed without them; evaluate into a new --progress-dir to add them)"
        )
    if skipped:
        text += "; skipped (not earlier than this generation): " + ", ".join(
            f"g{a}" for a in skipped
        )
    return text


def committed_generations(run_dir: Path) -> list[int]:
    newest = layout.latest_complete(run_dir)
    return [] if newest is None else list(range(newest.generation + 1))


def scheduled_generations(run_dir: Path) -> list[int]:
    """The generations ``--all`` evaluates: multiples of ``SCHEDULE_EVERY``, 0 included."""
    return [g for g in committed_generations(run_dir) if g % SCHEDULE_EVERY == 0]


__all__ = [
    "DAMAGE_AWARE_LABEL",
    "DEFAULT_ANCHOR_GAMES",
    "DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES",
    "SCHEDULE_EVERY",
    "AblationSpec",
    "AnchorPair",
    "NodeProgressPlayer",
    "ProgressError",
    "ProgressPlayer",
    "ProgressRecordError",
    "ProgressRequest",
    "committed_generations",
    "progress_generation",
    "progress_root",
    "scheduled_generations",
    "summary_of",
    "verify_progress_dir",
]
