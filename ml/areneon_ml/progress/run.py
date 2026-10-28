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

from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.driver import sources_registry
from areneon_ml.loop.play import NodePlayer, PlayError
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.model.network import ROLES
from areneon_ml.progress import evaluation_key as ekey
from areneon_ml.progress.ablation import ablation_summary, expected_ablation_matches
from areneon_ml.progress.eval_dataset import EvalDataset, EvalDatasetError, validation_teams
from areneon_ml.progress.records import (
    CONTROL_LABEL,
    DAMAGE_AWARE_LABEL,
    RANDOM_PREVIEW_POLICY,
    SCHEMA_V2,
    SEARCH_EVIDENCE_NOTE,
    SEARCH_LABEL,
    SEARCH_PREFIX,
    SEARCH_PRIOR_LABEL,
    ProgressRecordError,
    check_game_baseline,
    check_game_search,
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
#: The versions of the baseline an evaluation can play (#1183); v1 is the default and keeps every
#: record's bytes.
DAMAGE_AWARE_VERSIONS = ("v1", "v3")
#: The damage-aware baseline's match ids.
DAMAGE_AWARE_MATCHES = ("damage-aware-argmax", "damage-aware-sample")


def damage_aware_policy(version: str) -> str:
    """The policy id of the baseline version (the TS side: ``damageAwarePolicyId(version)``)."""
    if version not in DAMAGE_AWARE_VERSIONS:
        raise ProgressError(
            f"--damage-aware-version must be one of {', '.join(DAMAGE_AWARE_VERSIONS)}"
        )
    return f"damage-aware@{version}"


def damage_aware_matches(version: str) -> tuple[str, str]:
    """The match ids of a version: v1's are the original ones, v3's carry the version, so they
    cannot collide (TS ``damageAwareMatchPrefix``)."""
    prefix = "damage-aware" if version == "v1" else f"damage-aware-{version}"
    return (f"{prefix}-argmax", f"{prefix}-sample")


#: Says what a ``previous-*`` row is: the pair's own lineage, so it can cycle (ADR-017 section 2).
SEEN_LABEL = "seen, same-lineage comparison: it can cycle"
#: Says what a ``run-*`` row is: another run's frozen pair (not held-out evidence, not a gate).
CROSS_RUN_LABEL = "cross-run comparison"
#: Games per cross-run match (even): each of this many / 2 validation pairs from both seats.
DEFAULT_RUN_GAMES = 200


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
class VersusSpec:
    """Another run's committed generation to play against (named by its run directory)."""

    run_dir: Path
    generation: int
    games: int
    #: The other run's encoder version (from its own config): its worker is checked against it.
    encoder_version: str


@dataclass(frozen=True)
class RunPair:
    """Another run's frozen pair, verified, the opponent of the ``run-*`` matches."""

    run_id: str
    run_config_hash: str
    generation: int
    preview: str
    battle: str
    policy_id_sample: str
    policy_id_argmax: str
    run_dir: Path
    team_set: Path
    encoder_version: str
    games: int


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
    #: Which baseline record plays in those matches (``v1`` or ``v3``, #1183).
    damage_aware_version: str = "v1"
    #: Another run's frozen pair, played as ``run-sample`` and ``run-argmax``.
    versus: RunPair | None = None
    #: Another verified dataset whose validation split supplies the teams (ADR-020 E-17.1).
    eval_dataset: EvalDataset | None = None
    #: Replaces the run key in ``progress-pairing/1`` and ``progress-seed/1`` (#1177).
    evaluation_key: str | None = None
    #: The decision-time search as the candidate (ADR-026): the pair in this one play mode
    #: (``argmax`` or ``sample``) with the committed ``rerank-1`` config, against the baselines and
    #: the control ``self-no-search``; its matches are ``search-<opponent>-<mode>``.
    search_mode: str | None = None


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
            else (
                "--damage-aware-budget-ms",
                str(request.damage_aware_budget_ms),
                # v1 passes nothing, so a v1 evaluation runs the command it always did.
                *(
                    ()
                    if request.damage_aware_version == "v1"
                    else ("--damage-aware-version", request.damage_aware_version)
                ),
            )
        )
        versus = request.versus
        cross = (
            ()
            if versus is None
            else (
                "--versus",
                f"{versus.run_id},{versus.run_config_hash},{versus.generation},{versus.preview},"
                f"{versus.battle},{versus.policy_id_sample},{versus.policy_id_argmax}",
                *("--versus-dir", str(versus.run_dir)),
                *("--versus-encoder-version", versus.encoder_version),
                *("--versus-team-set", str(versus.team_set)),
                *("--run-games", str(versus.games)),
            )
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
            *cross,
            *(() if request.eval_dataset is None else request.eval_dataset.node_args()),
            *ekey.node_args(request.evaluation_key),
            *(() if request.search_mode is None else ("--search-mode", request.search_mode)),
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


def max_pairs(config: RunConfig, eval_dataset: EvalDataset | None = None) -> int:
    """The most disjoint validation pairs the dataset can give: half its validation teams.

    The dataset is the evaluation dataset when one is given, else the run's own. Read from the
    dataset's manifest before anything starts. It is an upper bound (the Node side also drops
    teams with an excluded ability and refuses exactly), so a refusal here is certain.
    """
    directory = eval_dataset.path if eval_dataset else Path(config.raw["team_dataset"]["path"])
    try:
        return validation_teams(directory) // 2
    except EvalDatasetError as exc:
        raise ProgressError(str(exc)) from exc


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
    #: ``private`` when either checkpoint record has private ancestry (only with ``allow_private``).
    data_class: str = "synthetic"


def _source(
    run_dir: Path,
    config: RunConfig,
    generation: int,
    *,
    allow_private: bool = False,
    needs_dataset: bool = True,
) -> _Source:
    """Verify the committed generation and its records; refuse a non-synthetic run.

    The private test (ADR-015 erratum E1) reads a pair with ``allow_private`` and, for a source
    other than the public dataset, ``needs_dataset=False``; it learns the pair's class from
    ``data_class`` and decides where that class may be tested.
    """
    if needs_dataset and not config.team_dataset:
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
    data_class = "synthetic"
    for role in ("preview", "battle"):
        record = read_record(layout.read_json(path / "records" / f"{role}.json"))
        version_manifest = record.manifest
        if record.data_class != "synthetic":
            if not allow_private:
                raise ProgressError(
                    "a progress evaluation runs on synthetic, public-lineage runs only"
                )
            data_class = "private"
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
        data_class,
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


def _is_search(record: dict[str, Any]) -> bool:
    """Whether the record is a search record: the search candidate's matches, alone."""
    return any("search" in m for m in record["matches"])


def _bare(match: dict[str, Any]) -> str:
    """The match id without the search candidate's ``search-`` prefix."""
    name: str = match["match_id"]
    return name.removeprefix(SEARCH_PREFIX) if "search" in match else name


def _modes(record: dict[str, Any]) -> set[str]:
    """The modes a record's matches are played in: the one search mode, else both."""
    return {m["mode"] for m in record["matches"]} if _is_search(record) else {"argmax", "sample"}


def _check_search(record: dict[str, Any], asked: str | None, *, legacy: bool) -> None:
    """A search record (ADR-026 slice P) holds the search candidate's matches alone, in the one
    mode asked for, with the control ``search-self-no-search-<mode>`` (the policy ids and the
    control's pair are tied to the run by the record reader).

    ``asked`` is the mode, or None for no search; ``legacy`` is a committed record (any one mode).
    """
    searched = _is_search(record)
    if asked is None and not legacy:
        if searched:
            raise ProgressError("this evaluation asks for no search")
        return
    if not searched:
        if asked is not None:
            raise ProgressError("the record holds no search matches")
        return
    modes = {m["mode"] for m in record["matches"]}
    if len(modes) != 1 or (asked is not None and modes != {asked}):
        raise ProgressError("a search record plays the one mode asked for")
    mode = next(iter(modes))
    controls = [
        m["match_id"] for m in record["matches"] if m["opponent"]["kind"] == "self-no-search"
    ]
    if controls != [f"{SEARCH_PREFIX}self-no-search-{mode}"]:
        raise ProgressError(f"a search record holds the control match search-self-no-search-{mode}")


def _check_damage_aware(record: dict[str, Any], asked: str | None, *, legacy: bool) -> None:
    """The damage-aware matches are exactly the two asked for (a fresh evaluation), each naming the
    asked version's policy id (``damage-aware@v1`` or ``@v3``), and share one identity hash (the
    reader checks it).

    ``asked`` is the version, or None for none. ``legacy`` is a committed record, which may hold
    none (committed without them) or one version's two matches, never a wrong or half set.
    """
    found = {
        _bare(m): m
        for m in record["matches"]
        if m["opponent"]["kind"] == "baseline" and is_damage_aware(m["opponent"]["policy_id"])
    }
    if asked is None and not legacy:
        if found:
            raise ProgressError("this evaluation asks for no damage-aware matches")
        return
    if not found and legacy:
        return
    version = asked
    if version is None:
        ids = {m["opponent"]["policy_id"] for m in found.values()}
        version = next((v for v in DAMAGE_AWARE_VERSIONS if {damage_aware_policy(v)} == ids), None)
        if version is None:
            raise ProgressError("damage-aware matches of a record name one known baseline version")
    policy = damage_aware_policy(version)
    modes = _modes(record)
    wanted = [w for w in damage_aware_matches(version) if w.rsplit("-", 1)[1] in modes]
    if set(found) != set(wanted):
        raise ProgressError(f"damage-aware matches are {' and '.join(wanted)}")
    for match in found.values():
        if _bare(match) != f"{wanted[0].rsplit('-', 1)[0]}-{match['mode']}" or match[
            "opponent"
        ] != {"kind": "baseline", "policy_id": policy}:
            raise ProgressError(f"{match['match_id']}: the opponent is not {policy}")


def _damage_aware_status(record: dict[str, Any]) -> str:
    """Says what the record holds of the damage-aware matches, in words for the operator."""
    if any(m.get("baseline") is not None for m in record["matches"]):
        return "played"
    return (
        "none: damage-aware matches are opt-in (--damage-aware); a committed generation is never "
        "rewritten, so evaluate into a new --progress-dir to add them"
    )


def _damage_budget_ms(minutes: int | None, preview_ablation: bool, version: str) -> int | None:
    """The wall-clock budget of one damage-aware match in milliseconds, or None for no match.

    An unknown baseline version is refused here, before any work.
    """
    damage_aware_policy(version)
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
        policy_id(before.preview, before.battle, "sample", features=features_hash(config)),
        policy_id(before.preview, before.battle, "argmax", features=features_hash(config)),
    )


def _anchor_pair(run_dir: Path, config: RunConfig, generation: int) -> AnchorPair:
    """The frozen pair of a committed generation (verified), as an opponent."""
    found = _source(run_dir, config, generation)
    return AnchorPair(
        generation,
        found.preview,
        found.battle,
        policy_id(found.preview, found.battle, "sample", features=features_hash(config)),
        policy_id(found.preview, found.battle, "argmax", features=features_hash(config)),
    )


def _dataset_of(config: RunConfig) -> dict[str, Any]:
    found = config.raw.get("team_dataset")
    if not isinstance(found, dict) or "dataset_hash" not in found:
        raise ProgressError("the run config names no team dataset with a validation split")
    return found


def _versus_pair(run_dir: Path, config: RunConfig, spec: VersusSpec) -> RunPair:
    """The other run's frozen pair, verified; refuses a run that cannot be compared.

    Both runs must name the same format and the same team dataset (so the validation split
    is one and the same), each must have committed the generation, the records must be
    synthetic, and the run keys, run directories and checkpoints must differ. The other run
    is only read (its lock is never taken).
    """
    other_dir = spec.run_dir.resolve()
    if other_dir == run_dir.resolve():
        raise ProgressError("the other run is the evaluated run itself")
    other = _read_config(other_dir)
    if other.run_key == config.run_key:
        raise ProgressError("the other run has the same run key as the evaluated one")
    if other.format_id != config.format_id:
        raise ProgressError("the two runs name different formats")
    if _dataset_of(other)["dataset_hash"] != _dataset_of(config)["dataset_hash"]:
        raise ProgressError("the two runs name different team datasets: no common validation split")
    found = _source(other_dir, other, spec.generation)
    return RunPair(
        other.run_key,
        other.hash,
        spec.generation,
        found.preview,
        found.battle,
        policy_id(found.preview, found.battle, "sample", features=features_hash(other)),
        policy_id(found.preview, found.battle, "argmax", features=features_hash(other)),
        other_dir,
        found.team_set,
        spec.encoder_version,
        spec.games,
    )


def _run_records(run_dir: Path) -> list[CheckpointRecord]:
    newest = layout.latest_complete(run_dir)
    if newest is None:
        raise ProgressError(f"{run_dir} holds no committed generation")
    return [
        read_record(
            layout.read_json(layout.generation_dir(run_dir, g) / "records" / f"{role}.json")
        )
        for g in range(newest.generation + 1)
        for role in ROLES
    ]


RELATED_LABEL = "related lineage: shared start"


def _merged_records(
    mine: list[CheckpointRecord], theirs: list[CheckpointRecord]
) -> list[CheckpointRecord]:
    """Both runs' records, one per checkpoint id: runs with equal initial seeds hold the same
    seeded generation-0 checkpoint, which is one node of the graph. A shared id must have the
    same parent in both runs (a shared start, never a shared later history)."""
    found = {r.checkpoint_id: r for r in mine}
    for record in theirs:
        before = found.get(record.checkpoint_id)
        if before is not None and before.parent != record.parent:
            raise ProgressError("the two runs hold one checkpoint with different ancestry")
        found.setdefault(record.checkpoint_id, record)
    return list(found.values())


def versus_relation(run_dir: Path, source: _Source, pair: RunPair) -> dict[str, str]:
    """How the two compared pairs relate, from the C1/C6 checker given both runs' records.

    The comparison is refused when the two frozen pairs share a checkpoint, or when one of
    them is in the other's ancestry (that would be a pair against its own past: self-play).
    A shared older ancestor (the seeded generation-0 pair of runs started from equal seeds, or
    any earlier common generation) is a relation to label, not a refusal: ``independent
    lineage`` when the ancestries are disjoint (C6), else ``related lineage: shared start``.
    Every committed generation of both runs is verified by ``sources_registry``.
    """
    candidate = [source.preview, source.battle]
    opponent = [pair.preview, pair.battle]
    if set(candidate) & set(opponent):
        raise ProgressError("the two compared pairs share a checkpoint: not two different pairs")
    registry: dict[str, list[str]] = {}
    for part in (sources_registry(run_dir), sources_registry(pair.run_dir)):
        for key, generators in part.items():
            registry[key] = sorted({*registry.get(key, []), *generators})
    graph = ProvenanceGraph(
        _merged_records(_run_records(run_dir), _run_records(pair.run_dir)), registry
    )
    if set(opponent) & graph.ancestry(candidate).checkpoints or set(candidate) & (
        graph.ancestry(opponent).checkpoints
    ):
        raise ProgressError("one compared pair is in the other's ancestry: a pair against its past")
    verdict = graph.classify(candidate, opponent)
    independence = graph.independent(candidate, opponent)
    return {
        "label": "independent lineage" if independence.independent else RELATED_LABEL,
        "reason": verdict.reason,
    }


def _check_versus(record: dict[str, Any], versus: RunPair | None, *, legacy: bool) -> None:
    """The ``run-*`` matches are exactly the other run's frozen pair in both modes (opt-in)."""
    found = {_bare(m): m for m in record["matches"] if m["opponent"]["kind"] == "run"}
    if versus is None:
        if found and not legacy:
            raise ProgressError("this evaluation asks for no cross-run matches")
        return
    if not found and legacy:
        return
    wanted_runs = {f"run-{mode}" for mode in _modes(record)}
    if set(found) != wanted_runs:
        raise ProgressError(
            f"cross-run matches are {' and '.join(sorted(wanted_runs, reverse=True))}"
        )
    if record["plan"].get("run_games_per_match") != versus.games:
        raise ProgressError("progress.json is not the record asked for")
    ids = {"sample": versus.policy_id_sample, "argmax": versus.policy_id_argmax}
    for match in found.values():
        opponent = match["opponent"]
        if opponent != {
            "kind": "run",
            "run_id": versus.run_id,
            "run_config_hash": versus.run_config_hash,
            "generation": versus.generation,
            "policy_id": ids[match["mode"]],
            "checkpoints": {"preview": versus.preview, "battle": versus.battle},
        }:
            raise ProgressError(f"{match['match_id']}: the opponent is not the other run's pair")


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
    damage_aware: str | None = None,
    versus: RunPair | None = None,
    eval_dataset: EvalDataset | None = None,
    evaluation_key: str | None = None,
    search_mode: str | None = None,
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
    _check_search(record, search_mode, legacy=False)
    _check_damage_aware(record, damage_aware, legacy=False)
    _check_versus(record, versus, legacy=False)
    _check_eval_dataset(record, eval_dataset, config, strict=True)
    _check_evaluation_key(record, evaluation_key, strict=True)
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
                or stored.get("evaluation_key") != evaluation_key
                or stored["version_manifest"] != record["version_manifest"]
            ):
                raise ProgressError(f"games/{name} is not the game it is named for")
            try:
                check_game_baseline(stored, match, rows[(match["match_id"], game)])
                check_game_search(stored, match, rows[(match["match_id"], game)])
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


def _check_eval_dataset(
    record: dict[str, Any],
    eval_dataset: EvalDataset | None,
    config: RunConfig,
    *,
    strict: bool,
) -> None:
    """The record names the evaluation dataset asked for (ADR-020 E-17.1), else the run's own.

    ``strict`` also refuses a record measured on another dataset when none was asked for (a
    committed directory is never rewritten, so the other kind needs a progress directory of its
    own); a lenient check (reading a committed root) only ties a dataset that was asked for.
    """
    teams = record["teams"]
    found = teams.get("evaluation_dataset_hash")
    if eval_dataset is None:
        if strict and found is not None:
            raise ProgressError(
                "the record was measured on another dataset's validation split: pass the same "
                "--eval-dataset-dir and --eval-dataset-hash, or use a progress directory of its own"
            )
        return
    if found != eval_dataset.dataset_hash:
        raise ProgressError(
            "the record's teams are not the evaluation dataset asked for: use a progress "
            "directory of its own for another dataset"
        )
    if teams["dataset_hash"] != _dataset_of(config)["dataset_hash"]:
        raise ProgressError("the record's dataset_hash is not the run's own dataset")


def _check_evaluation_key(
    record: dict[str, Any], evaluation_key: str | None, *, strict: bool
) -> None:
    """The record names the evaluation key asked for (#1177), else keys by its run key.

    ``strict`` also refuses a record keyed by an evaluation key when none was asked for (a
    committed directory is never rewritten, so another key needs a progress directory of its
    own); a lenient check (reading a committed root) only ties a key that was asked for.
    """
    found = record["plan"].get("evaluation_key")
    if evaluation_key is None:
        if strict and found is not None:
            raise ProgressError(
                "the record was keyed by an evaluation key: pass the same --evaluation-key, "
                "or use a progress directory of its own"
            )
        return
    if found != evaluation_key:
        raise ProgressError(
            "the record's plan is not keyed by the evaluation key asked for: use a progress "
            "directory of its own for another key"
        )


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

    A cross-run match is labelled ``cross-run comparison``.

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
                else {"comparison": CROSS_RUN_LABEL}
                if match["opponent"]["kind"] == "run"
                else {}
            )
            | (
                {"comparison": DAMAGE_AWARE_LABEL, "baseline": match["baseline"]}
                if "baseline" in match
                else {}
            )
            | (
                {"comparison": CONTROL_LABEL, "opponent_search": match["opponent_search"]}
                if match["opponent"]["kind"] == "self-no-search"
                else {}
            )
            | (
                {
                    "search": {
                        "label": SEARCH_LABEL,
                        "prior_label": SEARCH_PRIOR_LABEL,
                        "note": SEARCH_EVIDENCE_NOTE,
                        "identity_hash": match["search"]["identity_hash"],
                        "counts": match["search"]["counts"],
                    }
                }
                if "search" in match
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
    if match["opponent"]["kind"] in ("previous", "anchor", "run", "self-no-search"):
        # pair score minus 1/2: the same resamples, shifted, so the bounds shift exactly
        out |= {
            "diff_mean_ppm": interval.mean_ppm - PPM // 2,
            "diff_low_ppm": interval.low_ppm - PPM // 2,
            "diff_high_ppm": interval.high_ppm - PPM // 2,
        }
    return out


def verify_progress_dir(  # noqa: PLR0913, PLR0917
    target: Path,
    config: RunConfig,
    run_dir: Path,
    generation: int,
    versus: RunPair | None = None,
    eval_dataset: EvalDataset | None = None,
    *,
    strict_dataset: bool = False,
    evaluation_key: str | None = None,
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
    _check_search(record, None, legacy=True)
    # The damage-aware matches are optional (a record may hold them or not), never half a set.
    _check_damage_aware(record, None, legacy=True)
    _check_versus(record, versus, legacy=True)
    _check_eval_dataset(record, eval_dataset, config, strict=strict_dataset)
    _check_evaluation_key(record, evaluation_key, strict=strict_dataset)
    rows = {(r["match_id"], r["game"]): r for r in record["games"]}
    for match in record["matches"]:
        for name in sorted(p.name for p in (target / GAMES).glob(f"{match['match_id']}-*.json")):
            stored = read_game_record(layout.read_json(target / GAMES / name))
            if stored.get("evaluation_key") != record["plan"].get("evaluation_key"):
                raise ProgressError(f"games/{name}: its evaluation key is not the record's")
            if is_v2(record) and stored.get("pair") != stored["game"] // 2:
                raise ProgressError(f"games/{name}: a both-seats game must name its pair")
            if is_v2(record):
                try:
                    check_game_baseline(stored, match, rows[(match["match_id"], stored["game"])])
                    check_game_search(stored, match, rows[(match["match_id"], stored["game"])])
                except (ProgressRecordError, KeyError) as exc:
                    raise ProgressError(f"games/{name}: {exc}") from exc
    if manifest.get("run_id") != config.run_key:
        raise ProgressError(f"{target.name}: its manifest names another run")
    return record


def _check_asked(
    generation: int,
    sizes: tuple[int, int, int],
    asked: tuple[Sequence[int], bool, bool, VersusSpec | None],
    search_mode: str | None = None,
) -> tuple[list[int], list[int]]:
    """Refuse a request that cannot be played; returns the anchors played and left out."""
    games, record_games, anchor_games = sizes
    anchors, previous_matches, preview_ablation, versus = asked
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
    if search_mode is not None and search_mode not in ("argmax", "sample"):
        raise ProgressError("--search-mode must be argmax or sample")
    if search_mode is not None and (preview_ablation or anchors or previous_matches):
        raise ProgressError(
            "the search candidate is evaluated alone, in one mode: no --preview-ablation, "
            "no --anchors and no --previous-matches (the baselines, the control and "
            "--versus-run-dir are played)"
        )
    if versus is not None and (versus.games < MIN_GAMES or versus.games % 2):
        raise ProgressError("--run-games must be an even number of at least 2")
    if versus is not None and preview_ablation:
        raise ProgressError("the Preview ablation is played alone (no other run)")
    if preview_ablation and (anchors or previous_matches or generation < 1):
        raise ProgressError(
            "the Preview ablation is played alone, for a generation of 1 or more: "
            "no --anchors and no --previous-matches"
        )
    return ([], []) if preview_ablation else (wanted, dropped)


def progress_generation(  # noqa: PLR0913, PLR0917
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
    versus: VersusSpec | None = None,
    eval_dataset: EvalDataset | None = None,
    evaluation_key: str | None = None,
    damage_aware_version: str = "v1",
    search_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate one generation into ``root``; a committed directory is verified and kept.

    ``search_mode`` (ADR-026 slice P) makes the candidate the frozen pair played with decision-time
    search (the committed ``rerank-1`` config, one play mode) instead of the pair itself: its
    matches are ``search-<opponent>-<mode>`` against the baselines, ``--versus`` and the control
    ``self-no-search`` (the same pair, same mode, no search). The record is a search record and
    holds nothing else (use a progress directory of its own); each match carries the search
    identity hash and counts. Labelled "learned pair with decision-time search (one-ply re-rank)";
    search results are never held-out evidence.

    ``eval_dataset`` measures on the validation split of another verified dataset than the run's
    own (ADR-020 E-17.1; the same refusals, plus: the dataset is not the run's own training
    dataset, and its hash is the one given). The record names it
    (``teams.evaluation_dataset_hash``).

    ``evaluation_key`` replaces the run key in ``progress-pairing/1`` and ``progress-seed/1`` for
    every match (#1177), so another run evaluated with the same key plays the same pairs, seats,
    seeds and baseline streams. The record names it (``plan.evaluation_key``); a key equal to
    the run's own key is refused (omit it), and the committed-plan seed check applies to it.

    ``versus`` also plays another run's committed frozen pair (``run-sample``, ``run-argmax``,
    ``versus.games`` games each, every pair from both seats; opt-in, off by default). The other
    run is only read. The result then carries ``head_to_head``: its lineage relation to the
    evaluated pair from the C1/C6 checker. Use a progress directory of its own: a committed
    generation is never rewritten.

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
    wanted, dropped = _check_asked(
        generation,
        (games, record_games, anchor_games),
        (anchors, previous_matches, preview_ablation, versus),
        search_mode,
    )
    damage_budget_ms = _damage_budget_ms(
        damage_aware_budget_minutes, preview_ablation, damage_aware_version
    )
    config = _read_config(run_dir)
    try:
        evaluation_key = ekey.check(evaluation_key, config.run_key)
    except ekey.EvaluationKeyError as exc:
        raise ProgressError(str(exc)) from exc
    if (
        eval_dataset is not None
        and config.team_dataset
        and eval_dataset.dataset_hash == _dataset_of(config)["dataset_hash"]
    ):
        raise ProgressError(
            "--eval-dataset-hash is the run's own training dataset: leave the option out "
            "to evaluate on its validation split"
        )
    if versus is not None:
        progress_root(versus.run_dir, root)  # the root is never inside or above the other run
    with progress_lock(root):
        target = progress_dir_of(root, generation)
        for stale in sorted(root.glob(".g*.partial")):
            shutil.rmtree(stale)
        if target.exists():
            return _skipped(
                target,
                run_dir,
                config,
                generation,
                (wanted, dropped),
                preview_ablation,
                versus,
                eval_dataset,
                evaluation_key,
                search_mode,
            )
        source = _source(run_dir, config, generation)
        previous = _previous_pair(run_dir, config, generation) if previous_matches else None
        ablation = _ablation_spec(run_dir, config, source) if preview_ablation else None
        anchor_pairs = tuple(_anchor_pair(run_dir, config, a) for a in wanted)
        played_anchor_games = anchor_games if wanted else 0
        other_pair, relation = _open_versus(run_dir, config, source, versus)
        played_run_games = 0 if versus is None else versus.games
        needed = max(games, played_anchor_games, played_run_games) // 2
        limit = max_pairs(config, eval_dataset)
        if needed > limit:
            raise ProgressError(
                f"--games {games} / --anchor-games {played_anchor_games} / --run-games "
                f"{played_run_games} need {needed} disjoint "
                f"validation team pairs: the dataset gives at most {limit}"
            )
        try:
            plan_key = evaluation_key or config.run_key
            check_not_training_seeds(run_dir, plan_key, needed)
            if other_pair is not None:
                check_not_training_seeds(other_pair.run_dir, plan_key, needed)
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
                    policy_id_sample=policy_id(
                        source.preview, source.battle, "sample", features=features_hash(config)
                    ),
                    policy_id_argmax=policy_id(
                        source.preview, source.battle, "argmax", features=features_hash(config)
                    ),
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
                    damage_aware_version=damage_aware_version,
                    versus=other_pair,
                    eval_dataset=eval_dataset,
                    evaluation_key=evaluation_key,
                    search_mode=search_mode,
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
                damage_aware_version if damage_aware_budget_minutes is not None else None,
                other_pair,
                eval_dataset,
                evaluation_key,
                search_mode,
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
        return (
            {
                "generation": generation,
                "skipped": False,
                "previous": _previous_status(record, generation),
                "anchors": _anchor_status(record, wanted, dropped),
                "damage_aware": _damage_aware_status(record),
                "wall_ms": facts.get("wall_ms", {}),
                "game_ms": facts.get("game_ms", {}),
                "matches": summary_of(record),
            }
            | _ablation_part(record)
            | _versus_part(record, relation)
        )


def _skipped(  # noqa: PLR0913, PLR0917
    target: Path,
    run_dir: Path,
    config: RunConfig,
    generation: int,
    anchor_status: tuple[Sequence[int], Sequence[int]],
    preview_ablation: bool,
    versus: VersusSpec | None,
    eval_dataset: EvalDataset | None,
    evaluation_key: str | None = None,
    search_mode: str | None = None,
) -> dict[str, Any]:
    """A committed directory is verified and reported, never rewritten."""
    pair = None if versus is None else _versus_pair(run_dir, config, versus)
    record = verify_progress_dir(
        target,
        config,
        run_dir,
        generation,
        pair,
        eval_dataset,
        strict_dataset=True,
        evaluation_key=evaluation_key,
    )
    held = None
    if pair is not None and any(m["opponent"]["kind"] == "run" for m in record["matches"]):
        held = versus_relation(run_dir, _source(run_dir, config, generation), pair)
    if (search_mode is not None) != _is_search(record) or (
        search_mode is not None and {m["mode"] for m in record["matches"]} != {search_mode}
    ):
        raise ProgressError(
            f"{target.name} already holds "
            f"{'a search record of another kind' if _is_search(record) else 'a normal record'}: "
            "use a progress directory of its own for the other kind"
        )
    if preview_ablation != bool(ablation_summary(record)):
        raise ProgressError(
            f"{target.name} already holds a "
            f"{'normal' if preview_ablation else 'Preview ablation'} record: "
            "use a progress directory of its own for the other kind"
        )
    return (
        {
            "generation": generation,
            "skipped": True,
            "previous": _previous_status(record, generation),
            "anchors": _anchor_status(record, *anchor_status),
            "damage_aware": _damage_aware_status(record),
            "matches": summary_of(record),
        }
        | _ablation_part(record)
        | _versus_part(record, held)
    )


def _open_versus(
    run_dir: Path, config: RunConfig, source: _Source, versus: VersusSpec | None
) -> tuple[RunPair | None, dict[str, str] | None]:
    """The other run's pair and its lineage relation, with its team set checked; (None, None)
    without ``versus``."""
    if versus is None:
        return None, None
    pair = _versus_pair(run_dir, config, versus)
    relation = versus_relation(run_dir, source, pair)
    teams = layout.read_json(pair.team_set)
    manifest = layout.read_json(pair.team_set.parent / layout.MANIFEST)
    if sha256_hex(jcs(teams)) != manifest.get("team_set_manifest_hash"):
        raise ProgressError("the other run's team-set.json differs from its manifest")
    return pair, relation


def _ablation_spec(run_dir: Path, config: RunConfig, source: _Source) -> AblationSpec:
    """Generation 0's Preview (verified) with this generation's Battle, and the random Preview."""
    first = _source(run_dir, config, 0)
    return AblationSpec(
        0,
        first.preview,
        policy_id(first.preview, source.battle, "argmax", features=features_hash(config)),
        policy_id(RANDOM_PREVIEW_POLICY, source.battle, "argmax", features=features_hash(config)),
    )


def _versus_part(record: dict[str, Any], relation: dict[str, str] | None = None) -> dict[str, Any]:
    """The head-to-head header: the other run, the games, the label and the lineage relation."""
    found = [m for m in record["matches"] if m["opponent"]["kind"] == "run"]
    if not found:
        return {}
    opponent = found[0]["opponent"]
    return {
        "head_to_head": {
            "label": CROSS_RUN_LABEL,
            "run_id": opponent["run_id"],
            "generation": opponent["generation"],
            "games_per_match": record["plan"]["run_games_per_match"],
            "lineage": relation,
        }
    }


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
    "CROSS_RUN_LABEL",
    "DAMAGE_AWARE_LABEL",
    "DAMAGE_AWARE_VERSIONS",
    "DEFAULT_ANCHOR_GAMES",
    "DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES",
    "DEFAULT_RUN_GAMES",
    "RELATED_LABEL",
    "SCHEDULE_EVERY",
    "AblationSpec",
    "AnchorPair",
    "NodeProgressPlayer",
    "ProgressError",
    "ProgressPlayer",
    "ProgressRecordError",
    "ProgressRequest",
    "RunPair",
    "VersusSpec",
    "committed_generations",
    "progress_generation",
    "progress_root",
    "scheduled_generations",
    "summary_of",
    "verify_progress_dir",
    "versus_relation",
]
