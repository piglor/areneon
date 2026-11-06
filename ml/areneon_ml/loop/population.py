"""The population loop (ADR-021 sections 2, 7 and 8, slice 5): M learning members, one ladder.

``python -m areneon_ml.loop run`` on a config with ``population`` runs ``PopulationLoop``. Each
generation the simulator side (``play-generation.ts``, slice 4) plays every member's frozen pair
against the others, against itself and against the scripted baselines, and writes one stream and
one TrainingSet per member (``members/m<k>/``). The driver then trains the members in index order,
one learner at a time, each with its own networks, Adam state, generator and settings, and freezes
every member's new pair. The generation is one directory, committed by one rename after a
``manifest.json`` that covers every file (``layout.py``), so the crash and resume claims of the
single-lineage loop hold unchanged. Without ``population`` nothing here runs and ``driver.Loop``
behaves and writes exactly as before.

Layout of ``generations/g<NNNNNN>/``: the simulator side's ``plan.json``, ``league-run.json``
(version 2) and ``generation.json`` (version 2); ``team-set.json`` (the cumulative team set of the
run); and per member ``members/m<k>/`` with the member's ``episodes/``, ``stream.jsonl``,
``training-set.json`` (from the simulator side) and ``train-report.json``,
``learner-<g>.safetensors|json`` and ``records/{battle,preview}.json`` (from the driver).

Seeds (``population-seed/1``, draws reduced modulo 2^63 to the seed range of the networks):
member ``m<k>``'s Preview network is seeded from ``member-init`` index 0 of generation 0, its
Battle network from index 1 and its learner generator from ``member-learner`` index 0. So members
start from different random pairs. A ``warm_start`` with ``population`` names the Battle role
alone: every member starts with that cloned Battle network (one separate copy each) and its own
seeded Preview, so the pair ids are distinct with overwhelming probability; ``_initial`` asserts it
(the play side refuses two members with the same pair). Both config readers refuse a two-role warm
start with ``population``.

Device residency: every member's learner (networks, anchors, Adam moments) is built and kept on
the CPU; ``_train_member`` brings one learner to the device, trains it and parks it again, so one
learner at a time sits on the GPU. Moving tensors changes no value, so the learner state hash and
the checkpoints are the bytes they would be with everything resident.

Records are the ordinary ``CheckpointRecord``: ``lineage_id`` is ``<lineage>.m<k>``, ``parent``
the member's previous checkpoint of the same role, and ``league_opponents`` the frozen checkpoints
of every OTHER member whose pair it played (and was played against) in the generation. A shared
warm-start Battle checkpoint keeps its clone record at generation 0 for every member (a
checkpoint has one record).

A member whose games all forfeited has no Episode of its own, and the play side then fails the
whole generation (fail closed, nothing is committed); the driver does not work around it.

Run metrics sum the phases and counts of all members (the ``RunMetrics`` contract has no member
keys); the per-member counts go to ``metrics/members/g<NNNNNN>.json``, outside every hash like
the metrics.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id, write_checkpoint
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import MICRO, PPOSettings, RunConfig
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.parallel import fit_processes
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.learner.trainer import Learner, TrainReport, configure_learner
from areneon_ml.loop import ladder_write, layout, rating
from areneon_ml.loop.driver import LoopError, StageHook, Timing
from areneon_ml.loop.ladder import LadderRecordError, read_ladder_record
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecordV2,
    check_population_plan,
    read_league_run_record_v2,
)
from areneon_ml.loop.metrics import (
    Counts,
    Peaks,
    build_metrics,
    driver_peak_rss_bytes,
    metrics_path,
    write_metrics,
)
from areneon_ml.loop.play import (
    MemberPair,
    PopulationCalibrationRequest,
    PopulationPlayer,
    PopulationPlayRequest,
)
from areneon_ml.loop.population_seed import population_draw
from areneon_ml.loop.records import (
    RecordFacts,
    checkpoint_record,
    features_hash,
    policy_id,
    team_set_hash,
    team_set_manifest,
)
from areneon_ml.loop.warm_start import WarmPair, WarmStartError, load_warm_start
from areneon_ml.model.encoder import Encoder, encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import MAX_SEED, ROLES, PolicyNetwork, init_network

#: Index of the ``member-init`` draw that seeds a member's Preview and Battle networks.
INIT_INDEX = {"preview": 0, "battle": 1}
#: The ``schema_version`` of the population ``generation.json`` the simulator side writes.
SUMMARY_VERSION = 2


#: A learning member's label (an exploiter seat is ``x<k>``).
_LEARNER = re.compile(r"m(0|[1-9][0-9]?)")
#: Where an idle learner lives.
HOST = torch.device("cpu")


def move_learner(learner: Learner, device: torch.device) -> None:
    """Move a learner's networks, frozen anchors and Adam moments to ``device``, in place.

    Parameters keep their identity (``Module.to`` swaps the data), so the optimisers still hold
    them. Only the moment tensors (``exp_avg`` and the like) move; Adam's 0-dimensional ``step``
    counters stay where the optimiser keeps them. No value changes.
    """
    for network in (*learner.networks.values(), *learner.anchors.values()):
        network.to(device)
    for optimiser in {id(o): o for o in learner.optimisers.values()}.values():
        for state in optimiser.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    state[key] = value.to(device)
    learner.device = device


def bring_up(learner: Learner, device: torch.device) -> None:
    """Make ``learner`` resident on the training device (a test replaces this)."""
    move_learner(learner, device)


def park(learner: Learner) -> None:
    """Return ``learner`` to the host (a test replaces this)."""
    move_learner(learner, HOST)


def member_label(index: int) -> str:
    return f"m{index}"


def member_dir(generation_path: Path, member: str) -> Path:
    return generation_path / "members" / member


def _seed(config: RunConfig, member: str, purpose: Any, index: int) -> int:
    """One ``population-seed/1`` draw (generation 0), reduced to the networks' seed range."""
    return population_draw(config.run_key, 0, member, purpose, index) & MAX_SEED


def member_ppo(config: RunConfig, index: int) -> PPOSettings:
    """The member's PPO settings: the run's, with its ``member_ppo`` starting overrides applied."""
    ppo = config.ppo
    population = config.raw["population"]
    found = [e for e in population.get("member_ppo", []) if e["member"] == index]
    if not found:
        return ppo
    over = found[0]
    changes: dict[str, Any] = {}
    if "learning_rate_micro" in over:
        changes["learning_rate"] = over["learning_rate_micro"] / MICRO
        if ppo.schedule is not None:
            changes["schedule"] = dataclasses.replace(
                ppo.schedule, start_micro=over["learning_rate_micro"]
            )
    if "entropy_coef_micro" in over:
        changes["entropy_coef"] = over["entropy_coef_micro"] / MICRO
        if ppo.entropy_schedule is not None:
            changes["entropy_schedule"] = dataclasses.replace(
                ppo.entropy_schedule, start_micro=over["entropy_coef_micro"]
            )
    if "epochs" in over:
        changes["epochs"] = over["epochs"]
    if "clip_micro" in over:
        changes["clip"] = over["clip_micro"] / MICRO
    if "target_kl_micro" in over:
        changes["target_kl"] = over["target_kl_micro"] / MICRO
    return dataclasses.replace(ppo, **changes)


def member_config(config: RunConfig, index: int) -> RunConfig:
    """The config one member's learner is built from.

    Only the seeds and PPO starting values differ; ``raw`` and ``hash`` stay the run's, so every
    record and learner-state sidecar names the run's config hash.
    """
    member = member_label(index)
    return dataclasses.replace(
        config,
        init_preview=_seed(config, member, "member-init", INIT_INDEX["preview"]),
        init_battle=_seed(config, member, "member-init", INIT_INDEX["battle"]),
        learner_seed=_seed(config, member, "member-learner", 0),
        ppo=member_ppo(config, index),
    )


@dataclass(frozen=True)
class MemberFrozen:
    """One member's committed pair and the hashes recorded with it."""

    member: str
    checkpoints: dict[str, str]
    learner_state_hash: str
    training_set_hash: str | None


@dataclass(frozen=True)
class PopulationFrozen:
    """A committed population generation."""

    generation: int
    members: tuple[MemberFrozen, ...]
    team_hashes: tuple[str, ...]
    temperature_ppm: int | None = None

    def policy_id(self, member: MemberFrozen) -> str:
        return policy_id(
            member.checkpoints["preview"],
            member.checkpoints["battle"],
            temperature_ppm=self.temperature_ppm,
        )


@dataclass
class PopulationRunResult:
    """What one call of ``PopulationLoop.run`` did."""

    resumed_from: int | None
    discarded: list[str]
    generations: list[PopulationFrozen] = field(default_factory=list)
    reports: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self, final: PopulationFrozen) -> dict[str, Any]:
        return {
            "resumed_from": self.resumed_from,
            "discarded": self.discarded,
            "generation": final.generation,
            "members": {
                m.member: {
                    "checkpoints": m.checkpoints,
                    "learner_state_hash": m.learner_state_hash,
                    "policy_id": final.policy_id(m),
                }
                for m in final.members
            },
            "played": [g.generation for g in self.generations],
        }


@dataclass(frozen=True)
class MemberSealing:
    """What one member's checkpoints were built from, for its records."""

    parents: dict[str, str] | None
    training_set_hash: str | None
    league_opponents: tuple[str, ...] = ()
    #: Clone records to write as they are, for the warm-started roles only.
    records: dict[str, CheckpointRecord] | None = None


def _learner_files(generation: int) -> tuple[str, str]:
    return f"learner-{generation}.safetensors", f"learner-{generation}.json"


class PopulationLoop:
    """One run directory, one population config, one simulator side."""

    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        game: GameData,
        player: PopulationPlayer,
        on_stage: StageHook | None = None,
    ) -> None:
        if not config.population:
            raise LoopError("this driver runs a population config; use Loop for a single lineage")
        if config.raw["population"]["exploiters"] != 0:
            raise LoopError("population: exploiter seats are refused until they are built")
        if config.raw["population"]["shares"]["past_pool_ppm"] != 0:
            raise LoopError("population: a non-zero past_pool_ppm is refused until the past pool")
        if features_hash(config) is not None:
            raise LoopError("population: a run that names candidate features is refused")
        try:
            self.anchor_ids = ladder_write.anchor_ids(config)
        except ladder_write.LadderBuildError as exc:
            raise LoopError(str(exc)) from exc
        self.config = config
        self.run_dir = run_dir
        self.game = game
        self.player = player
        self.members = config.raw["population"]["members"]
        #: The frozen anchor strengths (``anchors.json`` of generation 0); set by ``run``.
        self._anchors: dict[str, int] = {}
        self._anchors_digest = ""
        #: Operational: worker processes that check and encode a member's stream (#1069).
        self.load_processes = 1
        self.load_memory_mb: int | None = None
        self._on_stage = on_stage or (lambda _generation, _stage: None)
        self.clock: Callable[[], float] = time.monotonic
        self.rss_source: Callable[[], int] = driver_peak_rss_bytes
        self.device = configure_learner(config)
        self.encoder_version = encoder_version(game, config.model.encoder)
        self._encoders = {role: Encoder(game, config.model.encoder) for role in ROLES}
        self._learners: dict[str, Learner] = {}
        self._frozen: PopulationFrozen | None = None

    @property
    def frozen(self) -> PopulationFrozen:
        if self._frozen is None:
            raise LoopError("the loop has not run")
        return self._frozen

    # -- the run directory ---------------------------------------------------

    def _bind_config(self) -> None:
        path = self.run_dir / layout.CONFIG
        if path.exists():
            stored = layout.read_json(path)
            if sha256_hex(jcs(stored)) != self.config.hash:
                raise LoopError("this run directory belongs to another TrainingRunConfig")
        else:
            layout.write_atomic(path, jcs(self.config.raw) + b"\n")

    def _labels(self) -> list[str]:
        return [member_label(k) for k in range(self.members)]

    def _warm_pair(self, warm_ids: Any) -> WarmPair:
        try:
            return load_warm_start(
                self.config, warm_ids, self.game, self.run_dir, self.encoder_version
            )
        except WarmStartError as exc:
            raise LoopError(f"warm_start: {exc}") from exc

    def _new_learner(self, index: int, warm: WarmPair | None) -> Learner:
        """A member's learner: seeded networks, the warm-started roles replaced by their copy."""
        cfg = member_config(self.config, index)
        seeds = {"preview": cfg.init_preview, "battle": cfg.init_battle}
        built = {
            role: warm.networks[role]
            if warm is not None and role in warm.networks
            else init_network(role, self.game, cfg.model, seeds[role])
            for role in ROLES
        }
        anchors: dict[str, PolicyNetwork] | None = None
        if self.config.ppo.anchor is not None:
            assert self.config.warm_start is not None  # the reader requires warm_start
            anchors = self._warm_pair(self.config.warm_start).networks
        # Built and kept on the host; `_train_member` brings it to the device while it trains.
        return Learner(cfg, built, HOST, anchors)

    # -- run -----------------------------------------------------------------

    def run(self, generations: int) -> PopulationRunResult:
        """Continue the run until ``generations`` are committed; returns what this call did."""
        if generations < 0:
            raise LoopError("the number of generations cannot be negative")
        with layout.run_lock(self.run_dir):
            return self._run_locked(generations)

    def _run_locked(self, generations: int) -> PopulationRunResult:
        layout.ensure_run_dirs(self.run_dir)
        self._bind_config()
        discarded = layout.discard_partials(self.run_dir)
        layout.discard_stale_checkpoint_files(self.run_dir)
        newest = layout.latest_complete(self.run_dir)
        result = PopulationRunResult(None if newest is None else newest.generation, discarded)
        self._frozen = self._initial() if newest is None else self._resume(newest)
        while self._frozen.generation < generations:
            self._frozen = self._generation(self._frozen, result)
            result.generations.append(self._frozen)
        return result

    # -- generation 0 --------------------------------------------------------

    def _initial(self) -> PopulationFrozen:
        sealings: dict[str, MemberSealing] = {}
        warm_ids = self.config.warm_start
        for k, label in enumerate(self._labels()):
            # Each member holds its own copy of the cloned Battle network (Battle-only start).
            warm = None if warm_ids is None else self._warm_pair(warm_ids)
            self._learners[label] = self._new_learner(k, warm)
            sealings[label] = MemberSealing(
                None, None, records=None if warm is None else warm.records
            )
        ids = [
            {r: checkpoint_id(checkpoint_bytes(learner.networks[r], self.game)) for r in ROLES}
            for learner in self._learners.values()
        ]
        pairs = [policy_id(found["preview"], found["battle"]) for found in ids]
        if len(set(pairs)) != len(pairs):
            raise LoopError("population: two members start from the same frozen pair")
        work = layout.partial_dir(self.run_dir, 0)
        work.mkdir(parents=True)
        manifest = self.player.manifest(self.config.format_id)
        timing = Timing(self.clock)
        # The anchor calibration (ADR-021 section 4): the anchors play each other once, their
        # strengths are fitted against actor-rule@v1 = 10^9 and frozen in this generation's
        # anchors.json.
        with timing.phase("play"):
            self.player.calibrate_population(PopulationCalibrationRequest(work))
        self._on_stage(0, "calibrated")
        try:
            calibration = ladder_write.read_calibration(
                layout.read_json(work / ladder_write.CALIBRATION_FILE), self.config
            )
            strengths = ladder_write.fit_anchors(calibration)
        except (ladder_write.LadderBuildError, rating.RatingError) as exc:
            raise LoopError(f"calibration.json: {exc}") from exc
        document = ladder_write.anchors_document(self.config, calibration, strengths)
        layout.write_json(work / ladder_write.ANCHORS_FILE, document)
        self._anchors = strengths
        self._anchors_digest = ladder_write.json_hash(document)
        return self._seal(
            0,
            work,
            sealings,
            calibration.team_hashes,
            manifest,
            timing,
            {"anchors_hash": self._anchors_digest},
        )

    # -- resume --------------------------------------------------------------

    def _resume(self, newest: layout.Committed) -> PopulationFrozen:
        facts = newest.manifest
        try:
            held = facts["members"]
            teams_hash = facts["team_set_manifest_hash"]
        except KeyError as exc:
            raise LoopError(f"generation {newest.generation}: its manifest lacks {exc}") from exc
        if not isinstance(held, dict) or sorted(held) != sorted(self._labels()):
            raise LoopError(
                f"generation {newest.generation}: its members are not the config's population"
            )
        teams = layout.read_json(newest.path / "team-set.json")
        if team_set_hash(teams) != teams_hash:
            raise LoopError(f"generation {newest.generation}: its team-set manifest differs")
        warm_ids = self.config.warm_start
        warm_zero = newest.generation == 0 and warm_ids is not None
        members: list[MemberFrozen] = []
        for k, label in enumerate(self._labels()):
            entry = held[label]
            try:
                checkpoints = dict(entry["checkpoints"])
                state_hash = entry["learner_state_hash"]
                ts_hash = entry["training_set_hash"]
            except KeyError as exc:
                raise LoopError(
                    f"generation {newest.generation}: {label}'s manifest entry lacks {exc}"
                ) from exc
            learner = self._new_learner(k, None)  # the state restored below replaces the weights
            path = member_dir(newest.path, label)
            tensors_name, sidecar_name = _learner_files(newest.generation)
            restore_state(
                learner,
                (path / tensors_name).read_bytes(),
                (path / sidecar_name).read_bytes(),
                state_hash,
            )
            if learner.generation != newest.generation:
                raise LoopError(
                    f"{label}: the learner state is of another generation than its directory"
                )
            records = {role: self._read_record(path, role) for role in ROLES}
            warm_roles = frozenset(warm_ids or ()) if warm_zero else frozenset()
            if (
                warm_ids is not None
                and warm_zero
                and {r: records[r] for r in warm_roles} != self._warm_pair(warm_ids).records
            ):
                # Generation 0 of a warm-started run: the records are the clone records.
                raise LoopError(
                    f"generation 0: {label}'s records are not the warm-start clone records"
                )
            self._check_member(
                newest.generation,
                label,
                learner,
                records,
                checkpoints,
                state_hash,
                ts_hash,
                warm_roles,
            )
            self._learners[label] = learner
            members.append(MemberFrozen(label, checkpoints, state_hash, ts_hash))
        self._anchors = self._load_anchors(newest)
        if newest.generation >= 1:
            self._check_ladder(newest.generation, newest.manifest)
        return PopulationFrozen(
            newest.generation,
            tuple(members),
            tuple(teams["teams"]),
            self.config.generation.temperature_ppm,
        )

    # -- the ladder files ----------------------------------------------------

    def _committed_json(self, generation: int, name: str) -> Any:
        """A file of a committed generation, once it matches that generation's manifest."""
        path = layout.generation_dir(self.run_dir, generation)
        manifest = layout.read_json(path / layout.MANIFEST)
        held = manifest.get("files", {}).get(name) if isinstance(manifest, dict) else None
        if held is None or not (path / name).is_file() or layout.sha256_file(path / name) != held:
            raise LoopError(f"generation {generation}: {name} differs from its manifest")
        return layout.read_json(path / name)

    def _load_anchors(self, newest: layout.Committed) -> dict[str, int]:
        """The frozen strengths of generation 0, tied to the newest generation's manifest."""
        raw = self._committed_json(0, ladder_write.ANCHORS_FILE)
        self._anchors_digest = ladder_write.json_hash(raw)
        if newest.manifest.get("anchors_hash") != self._anchors_digest:
            raise LoopError(
                f"generation {newest.generation}: its manifest names other anchor strengths "
                "than generation 0 froze"
            )
        try:
            return ladder_write.read_anchors(raw, self.config)
        except ladder_write.LadderBuildError as exc:
            raise LoopError(f"generation 0: {exc}") from exc

    def _read_ladder(
        self, generation: int, manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The committed ``ladder.json`` of ``generation``, read with the frozen anchors."""
        raw = self._committed_json(generation, ladder_write.LADDER_FILE)
        if manifest is not None and manifest.get("ladder_hash") != ladder_write.json_hash(raw):
            raise LoopError(
                f"generation {generation}: its ladder.json is not the one its manifest names"
            )
        before = (
            None
            if generation == 1
            else self._committed_json(generation - 1, ladder_write.LADDER_FILE)
        )
        try:
            read_ladder_record(raw, before, self.config)
            ladder_write.check_anchors(raw, self._anchors)
        except (LadderRecordError, ladder_write.LadderBuildError) as exc:
            raise LoopError(f"generation {generation}: ladder.json: {exc}") from exc
        assert isinstance(raw, dict)
        return raw

    def _check_ladder(self, generation: int, manifest: dict[str, Any]) -> None:
        self._read_ladder(generation, manifest)

    def _check_member(  # noqa: PLR0913, PLR0917
        self,
        generation: int,
        label: str,
        learner: Learner,
        records: dict[str, CheckpointRecord],
        checkpoints: dict[str, str],
        state_hash: str,
        ts_hash: str | None,
        warm_roles: frozenset[str],
    ) -> None:
        for role in ROLES:
            found = checkpoint_bytes(learner.networks[role], self.game)
            record = records[role]
            if checkpoint_id(found) != checkpoints.get(role):
                raise LoopError(
                    f"generation {generation}: {label}'s {role} weights differ from its checkpoint"
                )
            if record.checkpoint_id != checkpoints[role]:
                raise LoopError(
                    f"generation {generation}: {label}'s {role} record names another checkpoint"
                )
            write_checkpoint(self.run_dir, found)  # the store may have lost it; weights verified
            if role in warm_roles:
                continue
            if record.training_set_hashes != (() if ts_hash is None else (ts_hash,)):
                raise LoopError(
                    f"generation {generation}: {label}'s {role} record's training sets "
                    "differ from its manifest"
                )
            if (
                record.learner_state_hash != state_hash
                or record.run_config_hash != self.config.hash
                or record.lineage_id != f"{self.config.lineage_id}.{label}"
            ):
                raise LoopError(
                    f"generation {generation}: {label}'s {role} record disagrees with the state"
                )

    def _read_record(self, path: Path, role: str) -> CheckpointRecord:
        record = read_record(layout.read_json(path / "records" / f"{role}.json"))
        if record.role != role:
            raise LoopError(f"records/{role}.json holds a {record.role} record")
        return record

    # -- generation g --------------------------------------------------------

    def _generation(
        self, previous: PopulationFrozen, result: PopulationRunResult
    ) -> PopulationFrozen:
        generation = previous.generation + 1
        # The plan of this generation reads nothing but the committed ladder.json of the one before
        # (slice 10 draws the opponents' weights from it); a missing or forged one stops here.
        before = self._read_ladder(previous.generation) if previous.generation >= 1 else None
        work = layout.partial_dir(self.run_dir, generation)
        work.mkdir(parents=True)
        timing = Timing(self.clock)
        pairs = tuple(
            MemberPair(
                m.member,
                previous.policy_id(m),
                m.checkpoints["preview"],
                m.checkpoints["battle"],
            )
            for m in previous.members
        )
        with timing.phase("play"):
            self.player.play_population(PopulationPlayRequest(generation, pairs, work))
        timing.play_rss = dict(getattr(self.player, "resource_usage", None) or {})
        self._on_stage(generation, "played")
        with timing.phase("load"):
            summary = layout.read_json(work / "generation.json")
            if (
                summary.get("schema_version") != SUMMARY_VERSION
                or summary.get("generation") != generation
            ):
                raise LoopError("the simulator side played another generation than asked")
            record = self._check_league_run(work, generation, pairs, summary)
        member_counts: dict[str, dict[str, int]] = {}
        reports: dict[str, dict[str, Any]] = {}
        sealings: dict[str, MemberSealing] = {}
        by_label = {m.member: m for m in previous.members}
        for entry in record.members:
            label = entry.member
            held = self._summary_member(summary, label)
            report = self._train_member(
                work, generation, label, held, previous.policy_id(by_label[label]), timing
            )
            roles = report.roles.values()
            member_counts[label] = {
                "games": entry.games,
                "decisions": sum(r.steps for r in roles),
                "optimiser_steps": sum(r.optimiser_steps for r in roles),
            }
            reports[label] = report.as_dict()
            self._on_stage(generation, f"trained:{label}")
            # league_opponents: only opponents the record says were met in a played game.
            others = {
                by_label[o_member].checkpoints[role]
                for (o_member, played) in _ladder_opponents(record.raw, label)
                if played > 0
                for role in ROLES
            }
            sealings[label] = MemberSealing(
                by_label[label].checkpoints,
                held["training_set_hash"],
                tuple(sorted(others)),
            )
        timing.counts = Counts(
            record.games,
            sum(c["decisions"] for c in member_counts.values()),
            sum(c["optimiser_steps"] for c in member_counts.values()),
        )
        result.reports[generation] = reports
        self._on_stage(generation, "trained")
        teams = tuple(sorted(set(previous.team_hashes) | set(summary["team_hashes"])))
        with timing.phase("write"):
            ladder = self._write_ladder(work, generation, summary["games"], before)
        self._on_stage(generation, "rated")
        frozen = self._seal(
            generation,
            work,
            sealings,
            teams,
            summary["manifest"],
            timing,
            {
                "anchors_hash": self._anchors_digest,
                "ladder_hash": ladder_write.json_hash(ladder),
            },
        )
        self._write_member_counts(generation, member_counts)
        return frozen

    def _write_ladder(
        self,
        work: Path,
        generation: int,
        rows: list[dict[str, Any]],
        before: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fit the window, count the cycles and write ``ladder.json`` (selection: none)."""
        window = self.config.raw["population"]["rating"]["window"]
        history: list[ladder_write.GenerationRows] = []
        for g in range(max(1, generation - window + 1), generation):
            first = self._read_ladder(g)["selections"][0]
            games = self._committed_json(g, "generation.json")["games"]
            copied = (first["loser"],) if first["kind"] == "copy" else ()
            history.append(ladder_write.GenerationRows(g, games, copied))
        history.append(ladder_write.GenerationRows(generation, rows))
        try:
            record = ladder_write.build_ladder_record(
                self.config, generation, history, self._anchors, before
            )
        except (LadderRecordError, ladder_write.LadderBuildError, rating.RatingError) as exc:
            raise LoopError(f"generation {generation}: ladder.json: {exc}") from exc
        layout.write_json(work / ladder_write.LADDER_FILE, record)
        return record

    def _train_member(  # noqa: PLR0913, PLR0917
        self,
        work: Path,
        generation: int,
        label: str,
        held: dict[str, Any],
        pair: str,
        timing: Timing,
    ) -> TrainReport:
        """Load one member's own stream, train its learner on it and write its train report."""
        learner = self._learners[label]
        path = member_dir(work, label)
        with timing.phase("load"):
            training_set = layout.read_json(path / "training-set.json")
            if sha256_hex(jcs(training_set)) != held.get("training_set_hash"):
                raise LoopError(
                    f"{label}: the TrainingSet differs from the hash the simulator side reported"
                )
            found = trajectories(
                path / "stream.jsonl",
                training_set,
                self._encoders,
                pair,
                fit_processes(self.load_processes, self.load_memory_mb),
            )
        bring_up(learner, self.device)
        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            with timing.phase("train"):
                report = learner.train(found)
            del found
            if self.device.type == "cuda":
                peak = int(torch.cuda.max_memory_allocated(self.device))
                reserved = int(torch.cuda.max_memory_reserved(self.device))
                timing.peak_memory = max(timing.peak_memory or 0, peak)
                timing.peak_reserved = max(timing.peak_reserved or 0, reserved)
        finally:
            park(learner)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        with timing.phase("write"):
            layout.write_json(path / "train-report.json", report.as_dict())
        return report

    @staticmethod
    def _summary_member(summary: dict[str, Any], label: str) -> dict[str, Any]:
        found = [m for m in summary.get("members", []) if m.get("member") == label]
        if len(found) != 1:
            raise LoopError(f"generation.json does not hold exactly one entry for {label}")
        entry: dict[str, Any] = found[0]
        return entry

    def _check_league_run(
        self,
        work: Path,
        generation: int,
        pairs: tuple[MemberPair, ...],
        summary: dict[str, Any],
    ) -> LeagueRunRecordV2:
        """The version 2 record is this run's and generation's and names the pairs that played."""
        try:
            record = read_league_run_record_v2(
                layout.read_json(work / "league-run.json"), self.config
            )
            check_population_plan(record, layout.read_json(work / "plan.json"), self.config)
        except LeagueRecordError as exc:
            raise LoopError(f"league-run.json: {exc}") from exc
        if record.generation != generation:
            raise LoopError("league-run.json names another generation than the one played")
        learners = [m for m in record.members if _LEARNER.fullmatch(m.member)]
        if [(m.member, m.pair_policy_id, m.preview, m.battle) for m in learners] != [
            (p.member, p.policy_id, p.preview, p.battle) for p in pairs
        ] or len(learners) != len(record.members):
            raise LoopError("league-run.json names other pairs than the ones played")
        if record.forfeits != summary.get("forfeits"):
            raise LoopError("league-run.json counts other forfeits than generation.json")
        if dict(record.forfeit_causes) != (summary.get("forfeit_causes") or {}):
            raise LoopError("league-run.json counts other forfeit causes than generation.json")
        if len(summary.get("games", [])) != record.games:
            raise LoopError("generation.json lists another number of games than league-run.json")
        return record

    def _write_member_counts(self, generation: int, counts: dict[str, dict[str, int]]) -> None:
        """Per-member counts, beside the run metrics and outside every hash."""
        path = (
            metrics_path(self.run_dir, generation).parent
            / "members"
            / (layout.generation_name(generation) + ".json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        layout.write_atomic(path, jcs({"generation": generation, "members": counts}) + b"\n")

    # -- freezing a generation -----------------------------------------------

    def _seal(  # noqa: PLR0913, PLR0917
        self,
        generation: int,
        work: Path,
        sealings: dict[str, MemberSealing],
        teams: tuple[str, ...],
        version_manifest: dict[str, Any],
        timing: Timing,
        facts: dict[str, Any],
    ) -> PopulationFrozen:
        """Store every member's pair, write learner states, records and the team set, and commit."""
        manifest = team_set_manifest(teams)
        manifest_hash = team_set_hash(manifest)
        held: dict[str, dict[str, Any]] = {}
        members: list[MemberFrozen] = []
        with timing.phase("write"):
            layout.write_json(work / "team-set.json", manifest)
            for label in self._labels():
                learner = self._learners[label]
                sealing = sealings[label]
                checkpoints = {
                    role: write_checkpoint(
                        self.run_dir, checkpoint_bytes(learner.networks[role], self.game)
                    )
                    for role in ROLES
                }
                packed = pack_state(learner)
                path = member_dir(work, label)
                tensors_name, sidecar_name = _learner_files(generation)
                layout.write_file(path / tensors_name, packed.tensors)
                layout.write_file(path / sidecar_name, packed.sidecar)
                for role in ROLES:
                    if sealing.records is not None and role in sealing.records:
                        layout.write_json(
                            path / "records" / f"{role}.json", sealing.records[role].to_json()
                        )
                        continue
                    record = checkpoint_record(
                        self.config,
                        RecordFacts(
                            role=role,
                            checkpoint_id=checkpoints[role],
                            parent=None if sealing.parents is None else sealing.parents[role],
                            training_set_hashes=[]
                            if sealing.training_set_hash is None
                            else [sealing.training_set_hash],
                            team_set_manifest_hash=manifest_hash,
                            learner_state_hash=packed.hash,
                            encoder_version=self.encoder_version,
                            manifest=version_manifest,
                            lineage_id=f"{self.config.lineage_id}.{label}",
                            league_opponents=sealing.league_opponents,
                        ),
                    )
                    layout.write_json(path / "records" / f"{role}.json", record.to_json())
                held[label] = {
                    "checkpoints": checkpoints,
                    "learner_state_hash": packed.hash,
                    "training_set_hash": sealing.training_set_hash,
                }
                members.append(
                    MemberFrozen(label, checkpoints, packed.hash, sealing.training_set_hash)
                )
        self._on_stage(generation, "written")
        with timing.phase("seal"):
            layout.commit_generation(
                self.run_dir,
                generation,
                {
                    "members": held,
                    "team_set_manifest_hash": manifest_hash,
                    "run_config_hash": self.config.hash,
                    **facts,
                },
            )
        self._on_stage(generation, "committed")
        write_metrics(
            self.run_dir,
            build_metrics(
                self.config,
                generation,
                timing.phase_ms,
                timing.counts,
                Peaks(
                    timing.peak_memory,
                    timing.peak_reserved,
                    {**timing.play_rss, "driver": self.rss_source()},
                ),
            ),
        )
        return PopulationFrozen(
            generation, tuple(members), teams, self.config.generation.temperature_ppm
        )


def _ladder_opponents(record: dict[str, Any], label: str) -> list[tuple[str, int]]:
    """The other members ``label`` met in the generation: (member, games played), ascending.

    ``played`` is the record's per-opponent count of games that were actually played; a forfeited
    game is counted in ``games`` and in the member's forfeits, not here (``buildLeagueRunRecordV2``
    adds to ``played`` only for a played game). So an opponent with ``played > 0`` really met the
    member, and only those become ``league_opponents``.
    """
    entry = next(m for m in record["members"] if m["member"] == label)
    return sorted(
        (o["member"], o["played"])
        for o in entry["opponents"]
        if "member" in o and o["member"] != label
    )
