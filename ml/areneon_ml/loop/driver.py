"""The generation loop (ADR-010 §1): freeze, play, load, train, freeze.

Generation ``g`` plays the frozen pair ``P_(g-1)`` on A1 (the simulator side,
behind ``GenerationPlayer``), reads back the TrainingSet the training loader
wrote for the pair's own seats, runs PPO epochs on it and freezes ``P_g``. Its
files are built in a ``.partial`` directory and committed by one rename
(``layout.py``); a run that was killed discards the partial generation on its
next start and replays it from the plan, which is a pure function of the run
config. Generation 0 is the seeded random initialisation (ADR-010 §4: no warm
start). Resuming restores the learner from the last committed generation and
refuses unless every hash agrees: the state's, the checkpoints', the records'.

Determinism: with the CPU learner, the same config on the same machine and
software gives the same checkpoints, records and learner-state hashes, whether
or not the run was killed and resumed (tested end to end). Nothing is claimed
across machines or torch builds, and a CUDA learner is not bit-reproducible.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id, write_checkpoint
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.parallel import fit_processes
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.learner.trainer import Learner, configure_learner
from areneon_ml.loop import layout
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecord,
    check_generation_plan,
    read_league_run_record,
)
from areneon_ml.loop.metrics import (
    Counts,
    Peaks,
    build_metrics,
    driver_peak_rss_bytes,
    write_metrics,
)
from areneon_ml.loop.play import GenerationPlayer, PlayRequest
from areneon_ml.loop.records import (
    RecordFacts,
    checkpoint_record,
    policy_id,
    team_set_hash,
    team_set_manifest,
)
from areneon_ml.model.encoder import Encoder, encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, PolicyNetwork, init_network

StageHook = Callable[[int, str], None]
Clock = Callable[[], float]


class LoopError(ValueError):
    """The loop cannot continue: a file or hash disagrees, or the simulator side failed."""


@dataclass(frozen=True)
class Frozen:
    """A committed generation: the frozen pair and the hashes recorded with it."""

    generation: int
    checkpoints: dict[str, str]
    learner_state_hash: str
    training_set_hash: str | None
    team_hashes: tuple[str, ...]

    @property
    def policy_id(self) -> str:
        return policy_id(self.checkpoints["preview"], self.checkpoints["battle"])


@dataclass
class RunResult:
    """What one call of ``Loop.run`` did."""

    resumed_from: int | None
    discarded: list[str]
    generations: list[Frozen] = field(default_factory=list)
    reports: dict[int, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self, final: Frozen) -> dict[str, Any]:
        return {
            "resumed_from": self.resumed_from,
            "discarded": self.discarded,
            "generation": final.generation,
            "checkpoints": final.checkpoints,
            "learner_state_hash": final.learner_state_hash,
            "policy_id": final.policy_id,
            "played": [g.generation for g in self.generations],
        }


@dataclass(frozen=True)
class Sealing:
    """What a generation was built from, for its records."""

    parents: dict[str, str] | None
    training_set_hash: str | None
    teams: tuple[str, ...]
    version_manifest: dict[str, Any]


class Timing:
    """Wall-clock milliseconds per phase of one generation, for its run-metrics record."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.phase_ms: dict[str, int] = {}
        self.counts = Counts()
        self.peak_memory: int | None = None
        self.peak_reserved: int | None = None
        #: Peak resident set (bytes) of the play phase's processes, when the simulator side says.
        self.play_rss: dict[str, int] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            elapsed = int((self._clock() - started) * 1000)
            self.phase_ms[name] = self.phase_ms.get(name, 0) + max(elapsed, 0)


def _learner_files(generation: int) -> tuple[str, str]:
    return f"learner-{generation}.safetensors", f"learner-{generation}.json"


class Loop:
    """One run directory, one config, one simulator side."""

    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        game: GameData,
        player: GenerationPlayer,
        on_stage: StageHook | None = None,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.game = game
        self.player = player
        #: Worker processes that check and encode the generation's stream in the load phase
        #: (#1069). Operational: the trajectories, and so every hash, are the same for any number.
        self.load_processes = 1
        #: A cap on the memory the load workers may add, in MB (``None``: no cap).
        self.load_memory_mb: int | None = None
        self._on_stage = on_stage or (lambda _generation, _stage: None)
        #: Wall-clock source for the run-metrics record; a test replaces it.
        self.clock: Clock = time.monotonic
        #: The driver's peak resident set in bytes, for the run-metrics record; a test replaces it.
        self.rss_source: Callable[[], int] = driver_peak_rss_bytes
        self.device = configure_learner(config)
        self.encoder_version = encoder_version(game, config.model.encoder)
        self._encoders = {role: Encoder(game, config.model.encoder) for role in ROLES}
        self._learner: Learner | None = None
        self._frozen: Frozen | None = None

    # -- the run directory ---------------------------------------------------

    def _bind_config(self) -> None:
        """Write the run config (canonical) or refuse a directory made under another."""
        path = self.run_dir / layout.CONFIG
        if path.exists():
            stored = layout.read_json(path)
            if sha256_hex(jcs(stored)) != self.config.hash:
                raise LoopError("this run directory belongs to another TrainingRunConfig")
        else:
            layout.write_atomic(path, jcs(self.config.raw) + b"\n")

    def _new_learner(self, networks: dict[str, PolicyNetwork] | None = None) -> Learner:
        seeds = {"preview": self.config.init_preview, "battle": self.config.init_battle}
        built = networks or {
            role: init_network(role, self.game, self.config.model, seeds[role]) for role in ROLES
        }
        return Learner(self.config, built, self.device)

    # -- run -----------------------------------------------------------------

    def run(self, generations: int) -> RunResult:
        """Continue the run until ``generations`` are committed; returns what this call did."""
        if generations < 0:
            raise LoopError("the number of generations cannot be negative")
        with layout.run_lock(self.run_dir):
            return self._run_locked(generations)

    def _run_locked(self, generations: int) -> RunResult:
        layout.ensure_run_dirs(self.run_dir)
        self._bind_config()
        discarded = layout.discard_partials(self.run_dir)
        layout.discard_stale_checkpoint_files(self.run_dir)
        newest = layout.latest_complete(self.run_dir)
        result = RunResult(None if newest is None else newest.generation, discarded)
        if newest is None:
            self._frozen = self._initial()
        else:
            self._frozen = self._resume(newest)
        while self._frozen.generation < generations:
            self._frozen = self._generation(self._frozen, result)
            result.generations.append(self._frozen)
        return result

    @property
    def frozen(self) -> Frozen:
        if self._frozen is None:
            raise LoopError("the loop has not run")
        return self._frozen

    # -- generation 0 --------------------------------------------------------

    def _initial(self) -> Frozen:
        self._learner = self._new_learner()
        work = layout.partial_dir(self.run_dir, 0)
        work.mkdir(parents=True)
        manifest = self.player.manifest(self.config.format_id)
        return self._seal(0, work, Sealing(None, None, (), manifest), Timing(self.clock))

    # -- resume --------------------------------------------------------------

    def _resume(self, newest: layout.Committed) -> Frozen:
        facts = newest.manifest
        try:
            checkpoints = dict(facts["checkpoints"])
            state_hash = facts["learner_state_hash"]
            ts_hash = facts["training_set_hash"]
        except KeyError as exc:
            raise LoopError(f"generation {newest.generation}: its manifest lacks {exc}") from exc
        tensors_name, sidecar_name = _learner_files(newest.generation)
        learner = self._new_learner()
        restore_state(
            learner,
            (newest.path / tensors_name).read_bytes(),
            (newest.path / sidecar_name).read_bytes(),
            state_hash,
        )
        if learner.generation != newest.generation:
            raise LoopError("the learner state is of another generation than its directory")
        records = {role: self._read_record(newest.path, role) for role in ROLES}
        for role in ROLES:
            found = checkpoint_bytes(learner.networks[role], self.game)
            record = records[role]
            if checkpoint_id(found) != checkpoints.get(role):
                raise LoopError(
                    f"generation {newest.generation}: the {role} weights differ from its checkpoint"
                )
            if record.checkpoint_id != checkpoints[role]:
                raise LoopError(
                    f"generation {newest.generation}: the {role} record names another checkpoint"
                )
            if record.training_set_hashes != (() if ts_hash is None else (ts_hash,)):
                raise LoopError(
                    f"generation {newest.generation}: the {role} record's training sets "
                    "differ from its manifest"
                )
            # The store may have lost the file (the worker reads it by id): the weights are
            # verified above, so write them again.
            write_checkpoint(self.run_dir, found)
            if (
                record.learner_state_hash != state_hash
                or record.run_config_hash != self.config.hash
            ):
                raise LoopError(
                    f"generation {newest.generation}: the {role} record disagrees with the state"
                )
        teams = layout.read_json(newest.path / "team-set.json")
        if team_set_hash(teams) != facts.get("team_set_manifest_hash"):
            raise LoopError(f"generation {newest.generation}: its team-set manifest differs")
        self._learner = learner
        return Frozen(newest.generation, checkpoints, state_hash, ts_hash, tuple(teams["teams"]))

    def _read_record(self, path: Path, role: str) -> CheckpointRecord:
        record = read_record(layout.read_json(path / "records" / f"{role}.json"))
        if record.role != role:
            raise LoopError(f"records/{role}.json holds a {record.role} record")
        return record

    # -- generation g --------------------------------------------------------

    def _generation(self, previous: Frozen, result: RunResult) -> Frozen:
        learner = self._learner
        assert learner is not None  # set by _initial or _resume
        generation = previous.generation + 1
        work = layout.partial_dir(self.run_dir, generation)
        work.mkdir(parents=True)
        pair = policy_id(previous.checkpoints["preview"], previous.checkpoints["battle"])
        timing = Timing(self.clock)
        with timing.phase("play"):
            self.player.play(
                PlayRequest(
                    generation,
                    pair,
                    previous.checkpoints["preview"],
                    previous.checkpoints["battle"],
                    work,
                )
            )
        timing.play_rss = dict(getattr(self.player, "resource_usage", None) or {})
        self._on_stage(generation, "played")
        with timing.phase("load"):
            summary = layout.read_json(work / "generation.json")
            training_set = layout.read_json(work / "training-set.json")
            if summary.get("generation") != generation:
                raise LoopError("the simulator side played another generation than asked")
            league = self._check_league_run(work, generation, pair, summary)
            if sha256_hex(jcs(training_set)) != summary.get("training_set_hash"):
                raise LoopError("the TrainingSet differs from the hash the simulator side reported")
            # The stream goes in by path: a generation's is over a gigabyte (#998).
            found = trajectories(
                work / "stream.jsonl",
                training_set,
                self._encoders,
                pair,
                fit_processes(self.load_processes, self.load_memory_mb),
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with timing.phase("train"):
            report = learner.train(found)
        if self.device.type == "cuda":
            timing.peak_memory = int(torch.cuda.max_memory_allocated(self.device))
            timing.peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            # The tensors of a generation are on the CPU and only a minibatch is on the device:
            # give the allocator's cache back, so the next phase and any other process on the
            # device start from what is in use (#1069). It changes no value.
            torch.cuda.empty_cache()
        roles = report.roles.values()
        timing.counts = Counts(
            league.games, sum(r.steps for r in roles), sum(r.optimiser_steps for r in roles)
        )
        result.reports[generation] = report.as_dict()
        with timing.phase("write"):
            layout.write_json(work / "train-report.json", report.as_dict())
        self._on_stage(generation, "trained")
        teams = sorted(set(previous.team_hashes) | set(summary["team_hashes"]))
        sealing = Sealing(
            previous.checkpoints, summary["training_set_hash"], tuple(teams), summary["manifest"]
        )
        return self._seal(generation, work, sealing, timing)

    def _check_league_run(
        self, work: Path, generation: int, pair: str, summary: dict[str, Any]
    ) -> LeagueRunRecord:
        """The record is this run's and generation's, and the plan's seeds are the run key's."""
        try:
            record = read_league_run_record(layout.read_json(work / "league-run.json"), self.config)
            check_generation_plan(record, layout.read_json(work / "plan.json"), self.config)
        except LeagueRecordError as exc:
            raise LoopError(f"league-run.json: {exc}") from exc
        if record.generation != generation or record.pair_policy_id != pair:
            raise LoopError("league-run.json names another generation or pair than the one played")
        if record.forfeits != summary.get("forfeits"):
            raise LoopError("league-run.json counts other forfeits than generation.json")
        if dict(record.forfeit_causes) != (summary.get("forfeit_causes") or {}):
            raise LoopError("league-run.json counts other forfeit causes than generation.json")
        return record

    # -- freezing a generation -----------------------------------------------

    def _seal(self, generation: int, work: Path, sealing: Sealing, timing: Timing) -> Frozen:
        """Store the pair, write the learner state, records and team set, and commit.

        The run-metrics record is written last, after the commit and outside every hash
        (ADR-010 erratum A1): a crash before it loses this generation's timings only.
        """
        learner = self._learner
        assert learner is not None
        with timing.phase("write"):
            checkpoints = {
                role: write_checkpoint(
                    self.run_dir, checkpoint_bytes(learner.networks[role], self.game)
                )
                for role in ROLES
            }
            packed = pack_state(learner)
            tensors_name, sidecar_name = _learner_files(generation)
            layout.write_file(work / tensors_name, packed.tensors)
            layout.write_file(work / sidecar_name, packed.sidecar)
            manifest = team_set_manifest(sealing.teams)
            layout.write_json(work / "team-set.json", manifest)
            for role in ROLES:
                record = checkpoint_record(
                    self.config,
                    RecordFacts(
                        role=role,
                        checkpoint_id=checkpoints[role],
                        parent=None if sealing.parents is None else sealing.parents[role],
                        training_set_hashes=[]
                        if sealing.training_set_hash is None
                        else [sealing.training_set_hash],
                        team_set_manifest_hash=team_set_hash(manifest),
                        learner_state_hash=packed.hash,
                        encoder_version=self.encoder_version,
                        manifest=sealing.version_manifest,
                    ),
                )
                layout.write_json(work / "records" / f"{role}.json", record.to_json())
        self._on_stage(generation, "written")
        with timing.phase("seal"):
            layout.commit_generation(
                self.run_dir,
                generation,
                {
                    "checkpoints": checkpoints,
                    "learner_state_hash": packed.hash,
                    "training_set_hash": sealing.training_set_hash,
                    "team_set_manifest_hash": team_set_hash(manifest),
                    "run_config_hash": self.config.hash,
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
        return Frozen(
            generation, checkpoints, packed.hash, sealing.training_set_hash, sealing.teams
        )


def sources_registry(run_dir: Path) -> dict[str, list[str]]:
    """Training-set hash to the checkpoints that generated it (the C1/C6 checker's registry).

    Generation ``g``'s set was played by the pair frozen at ``g - 1``.
    """
    registry: dict[str, list[str]] = {}
    newest = layout.latest_complete(run_dir)
    if newest is None:
        return registry
    previous: dict[str, str] | None = None
    for generation in range(newest.generation + 1):
        facts = layout.verify_generation(layout.generation_dir(run_dir, generation), generation)
        if facts["training_set_hash"] is not None:
            assert previous is not None  # generation 0 has no training set
            registry[facts["training_set_hash"]] = sorted(previous.values())
        previous = dict(facts["checkpoints"])
    return registry


def provenance_graph(run_dir: Path) -> ProvenanceGraph:
    """The C1/C6 provenance graph of every committed generation of a run."""
    newest = layout.latest_complete(run_dir)
    if newest is None:
        raise LoopError("the run directory holds no committed generation")
    records = [
        read_record(
            layout.read_json(layout.generation_dir(run_dir, g) / "records" / f"{role}.json")
        )
        for g in range(newest.generation + 1)
        for role in ROLES
    ]
    return ProvenanceGraph(records, sources_registry(run_dir))


def export_provenance(run_dir: Path, out_dir: Path) -> str:
    """Write ``records/`` and ``sources.json`` into ``out_dir`` for the checkpoint CLI.

    Returns the registry's hash. The registry is derived from the run's own manifests here, so
    it is no longer a hand-supplied input.
    """
    provenance_graph(run_dir)  # refuse a run whose records do not form a graph
    newest = layout.latest_complete(run_dir)
    assert newest is not None
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "records").mkdir(parents=True)
    for generation in range(newest.generation + 1):
        for role in ROLES:
            source = layout.generation_dir(run_dir, generation) / "records" / f"{role}.json"
            record = read_record(layout.read_json(source))
            (out_dir / "records" / f"{record.checkpoint_id}.json").write_bytes(source.read_bytes())
    registry = jcs(sources_registry(run_dir))
    (out_dir / "sources.json").write_bytes(registry)
    return sha256_hex(registry)
