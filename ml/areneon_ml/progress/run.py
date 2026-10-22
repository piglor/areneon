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
from collections.abc import Iterator
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
from areneon_ml.progress.records import (
    ProgressRecordError,
    read_game_record,
    read_progress_record,
    record_hash,
)
from areneon_ml.progress.seeds import check_not_training_seeds
from areneon_ml.progress.stats import CONDITIONAL_LABEL, wilson

PROGRESS = "progress.json"
GAMES = "games"
SCRIPT = "packages/team-dataset/scripts/progress.ts"
DEFAULT_GAMES = 100
DEFAULT_RECORD_GAMES = 8
#: ``--all`` evaluates the committed generations whose number is a multiple of this, 0 included.
SCHEDULE_EVERY = 5
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


def _verify_output(  # noqa: PLR0913, PLR0917
    work: Path,
    config: RunConfig,
    source: _Source,
    games: int,
    record_games: int,
    previous: PreviousPair | None = None,
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
    _check_previous(record, previous, legacy=False)
    expected = set()
    for match in record["matches"]:
        listed = []
        for game in range(record_games):
            name = f"{match['match_id']}-{game}.json"
            expected.add(name)
            stored = read_game_record(layout.read_json(work / GAMES / name))
            if (
                stored["run_id"] != record["run_id"]
                or stored["generation"] != source.generation
                or stored["match_id"] != match["match_id"]
                or stored["game"] != game
                or stored["version_manifest"] != record["version_manifest"]
            ):
                raise ProgressError(f"games/{name} is not the game it is named for")
            listed.append(record_hash(stored))
        if sorted(listed) != match["game_record_hashes"]:
            raise ProgressError(
                f"{match['match_id']}: the stored games differ from game_record_hashes"
            )
    found = {p.name for p in (work / GAMES).glob("*")} if (work / GAMES).is_dir() else set()
    if found != expected:
        raise ProgressError("games/ holds other files than the record lists")
    return record


def summary_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Per match: the counts and the derived score with its Wilson interval (never stored)."""
    out = []
    for match in record["matches"]:
        interval = wilson(match["wins"], match["ties"], match["games"])
        out.append(
            {
                "match_id": match["match_id"],
                "games": match["games"],
                "wins": match["wins"],
                "ties": match["ties"],
                "losses": match["losses"],
                "score_ppm": interval.score_ppm,
                "wilson_low_ppm": interval.low_ppm,
                "wilson_high_ppm": interval.high_ppm,
                "interval": CONDITIONAL_LABEL,
                "opponent": match["opponent"]["kind"],
                "forfeits": sum(f["count"] for f in match["forfeits"]),
                "turn_cap_ties": match["turn_cap_ties"],
            }
            | ({"comparison": SEEN_LABEL} if match["opponent"]["kind"] == "previous" else {})
        )
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
    # A record committed before the previous-generation matches existed (slice 4) stays valid.
    _check_previous(record, _previous_pair(run_dir, config, generation), legacy=True)
    for match in record["matches"]:
        for name in sorted(p.name for p in (target / GAMES).glob(f"{match['match_id']}-*.json")):
            read_game_record(layout.read_json(target / GAMES / name))
    if manifest.get("run_id") != config.run_key:
        raise ProgressError(f"{target.name}: its manifest names another run")
    return record


def progress_generation(  # noqa: PLR0913, PLR0917
    run_dir: Path,
    generation: int,
    root: Path,
    player: ProgressPlayer,
    games: int = DEFAULT_GAMES,
    record_games: int = DEFAULT_RECORD_GAMES,
    previous_matches: bool = False,
) -> dict[str, Any]:
    """Evaluate one generation into ``root``; a committed directory is verified and kept.

    The previous-generation matches are played only with ``previous_matches`` (and never at
    generation 0, which has no previous generation).
    """
    if games < 1 or record_games < 0 or record_games > games:
        raise ProgressError("--games must be positive and --record-games between 0 and --games")
    config = _read_config(run_dir)
    with progress_lock(root):
        target = progress_dir_of(root, generation)
        for stale in sorted(root.glob(".g*.partial")):
            shutil.rmtree(stale)
        if target.exists():
            record = verify_progress_dir(target, config, run_dir, generation)
            return {
                "generation": generation,
                "skipped": True,
                "previous": _previous_status(record, generation),
                "matches": summary_of(record),
            }
        source = _source(run_dir, config, generation)
        previous = _previous_pair(run_dir, config, generation) if previous_matches else None
        limit = max_pairs(config)
        if games > limit:
            raise ProgressError(
                f"--games {games} needs {games} disjoint validation team pairs: "
                f"the dataset gives at most {limit}"
            )
        try:
            check_not_training_seeds(run_dir, config.run_key, games)
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
                )
            )
            record = _verify_output(work, config, source, games, record_games, previous)
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
            "wall_ms": facts.get("wall_ms", {}),
            "matches": summary_of(record),
        }


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


def committed_generations(run_dir: Path) -> list[int]:
    newest = layout.latest_complete(run_dir)
    return [] if newest is None else list(range(newest.generation + 1))


def scheduled_generations(run_dir: Path) -> list[int]:
    """The generations ``--all`` evaluates: multiples of ``SCHEDULE_EVERY``, 0 included."""
    return [g for g in committed_generations(run_dir) if g % SCHEDULE_EVERY == 0]


__all__ = [
    "SCHEDULE_EVERY",
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
