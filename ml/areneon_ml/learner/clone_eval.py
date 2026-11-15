"""``clone-eval``: the group table of any existing clone, read-only (ADR-010 erratum A13.4).

Loads a clone output's Battle checkpoint and ``clone-run-config.json``, re-splits the teacher
stream with the clone's own holdout key and ppm (the held-back Episodes are the ones the clone
held back, whatever block it was trained under) and reports the same per-situation table the
cloning run reports under ``clone.situation_weights``: for ``streak``, ``threatened`` and
``other``, the steps, the UNWEIGHTED mean cross-entropy, the top-1 hits against the teacher and
chance, and on ``streak`` the stalling-move counts of the teacher and of the clone. ``threatened``
is read whenever the clone's feature set carries the threat rows, without weighting anything.
Nothing in the clone directory is changed; CPU only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_ml.checkpoint.files import CheckpointFileError, load_checkpoint
from areneon_ml.learner.clone import (
    CloneError,
    check_teacher,
    evaluate,
    evaluate_groups,
    split_parts,
)
from areneon_ml.learner.clone_config import CloneConfig, read_clone_config
from areneon_ml.learner.config import THREAT_ROW_FEATURE_SETS
from areneon_ml.learner.data import LearnerDataError, trajectories
from areneon_ml.learner.situation import GROUPS, Groups
from areneon_ml.learner.trainer import configure_learner
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES


@dataclass(frozen=True)
class _Cpu:
    device: str
    threads: int


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _threat_rows(config: CloneConfig) -> bool:
    options = config.raw["model"].get("encoder", {}).get("options", {})
    return bool(options.get("candidate_features", {}).get("feature_set") in THREAT_ROW_FEATURE_SETS)


def clone_eval(clone_dir: Path, stream_dir: Path, game: GameData) -> dict[str, Any]:
    """The Battle role's agreement and group tables for the clone in ``clone_dir`` on the
    teacher stream in ``stream_dir`` (``stream.jsonl`` and ``training-set.json``)."""
    config = read_clone_config(_json(clone_dir / "clone-run-config.json"))
    report = _json(clone_dir / "clone-report.json")
    if report["run_config_hash"] != config.hash:
        raise CloneError("the clone's report names another run config than clone-run-config.json")
    manifest = _json(stream_dir / "training-set.json")
    check_teacher(config, manifest)
    wanted = report["checkpoints"]["battle"]
    try:
        data = (clone_dir / "checkpoints" / f"{wanted}.safetensors").read_bytes()
        network = load_checkpoint(data, game, wanted, "battle", configured=config.model)
    except (CheckpointFileError, OSError) as exc:
        raise CloneError(f"the clone's Battle checkpoint cannot be loaded: {exc}") from exc
    device = configure_learner(_Cpu("cpu", config.threads))
    encoders = {role: Encoder(game, config.model.encoder) for role in ROLES}
    try:
        found = trajectories(
            stream_dir / "stream.jsonl",
            manifest,
            encoders,
            config.teacher_policy_id,
            groups=Groups(_threat_rows(config)),
        )
    except LearnerDataError as exc:
        raise CloneError(str(exc)) from exc
    train, held = split_parts(config, found)["battle"]
    size = config.clone.minibatch_size
    out: dict[str, Any] = {
        "schema_version": 1,
        "run_config_hash": config.hash,
        "checkpoint": wanted,
        "threatened_read": _threat_rows(config),
        "train": evaluate(network, train, size, device).as_dict(),
        "train_groups": {
            name: table.as_dict()
            for name, table in evaluate_groups(network, train, size, device).items()
        },
    }
    if held:
        out["holdout"] = evaluate(network, held, size, device).as_dict()
        out["holdout_groups"] = {
            name: table.as_dict()
            for name, table in evaluate_groups(network, held, size, device).items()
        }
    assert tuple(out["train_groups"]) == GROUPS
    return out
