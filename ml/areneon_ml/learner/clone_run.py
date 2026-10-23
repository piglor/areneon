"""The outputs of a cloning run: checkpoints, records, sources registry, state and report.

``write_clone`` turns a ``Cloned`` result into one directory, built beside its final
place and committed by one rename:

* ``checkpoints/<id>.safetensors``: the Preview and Battle networks (the existing writer);
* ``records/<id>.json``: their ``CheckpointRecord`` (``parent`` absent, the teacher set in
  ``training_set_hashes``, ``run_config_hash`` the clone config's hash, ``data_class``
  synthetic), named by checkpoint id so ``python -m areneon_ml.checkpoint classify|independent
  --records records --sources sources.json`` reads the directory as it is;
* ``sources.json``: the registry, the teacher set mapped to ``[]`` (an outside behaviour-cloning
  source, ADR-010 erratum A2);
* ``clone-state.safetensors`` / ``clone-state.json``: the weights, Adam moments and generator
  state of the run and a JCS sidecar, whose hash is the records' ``learner_state_hash``;
* ``clone-report.json``: deterministic values only (loss per epoch, agreement with the teacher);
* ``clone-run-config.json``: the config the hash names.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs

from areneon_ml.checkpoint.files import checkpoint_bytes, pack_safetensors, write_checkpoint
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.clone import Cloned
from areneon_ml.learner.clone_config import CloneConfig
from areneon_ml.learner.state import STATE_FORMAT, gather_tensors, state_hash
from areneon_ml.loop import layout
from areneon_ml.loop.records import team_set_hash, team_set_manifest
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES


class CloneOutputError(ValueError):
    """The output of a cloning run cannot be written where asked."""


@dataclass(frozen=True)
class _Shared:
    """What both records say alike."""

    state: str
    team_hash: str
    version: str
    manifest: dict[str, Any]


def _record(config: CloneConfig, role: str, found: str, shared: _Shared) -> CheckpointRecord:
    return read_record(
        {
            "schema_version": 1,
            "checkpoint_id": found,
            "role": role,
            "lineage_id": config.lineage_id,
            "league_opponents": [],
            "training_set_hashes": [config.teacher_set_hash],
            "team_set_manifest_hash": shared.team_hash,
            "run_config_hash": config.hash,
            "learner_state_hash": shared.state,
            "encoder_version": shared.version,
            "data_class": "synthetic",
            "manifest": shared.manifest,
        }
    )


def write_clone(
    out_dir: Path,
    config: CloneConfig,
    game: GameData,
    cloned: Cloned,
    version_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write the run's outputs to ``out_dir`` (which must not exist) and return the report."""
    if out_dir.exists():
        raise CloneOutputError(f"{out_dir} exists: a cloning run never overwrites")
    work = out_dir.with_name(f".{out_dir.name}.partial")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    tensors, steps = gather_tensors(cloned.networks, cloned.optimisers, cloned.generator)
    sidecar = {
        "schema_version": STATE_FORMAT,
        "kind": "clone",
        "run_config_hash": config.hash,
        "seeds": config.raw["seeds"],
        "clone": config.raw["clone"],
        "optimizer_steps": steps,
        "rng": {"learner": "rng.learner"},
    }
    packed = pack_safetensors(tensors)
    state = state_hash(packed, sidecar)
    ids = {
        role: write_checkpoint(work, checkpoint_bytes(cloned.networks[role], game))
        for role in ROLES
    }
    shared = _Shared(
        state,
        team_set_hash(team_set_manifest(cloned.team_set)),
        encoder_version(game, config.model.encoder),
        version_manifest,
    )
    for role in ROLES:
        record = _record(config, role, ids[role], shared)
        layout.write_json(work / "records" / f"{ids[role]}.json", record.to_json())
    layout.write_file(work / "sources.json", jcs({config.teacher_set_hash: []}))
    layout.write_file(work / "clone-state.safetensors", packed)
    layout.write_file(work / "clone-state.json", jcs(sidecar))
    layout.write_json(work / "clone-run-config.json", config.raw)
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_config_hash": config.hash,
        "teacher_set_hash": config.teacher_set_hash,
        "checkpoints": ids,
        "learner_state_hash": state,
        "roles": {role: cloned.roles[role].as_dict() for role in ROLES},
    }
    layout.write_json(work / "clone-report.json", report)
    layout.sync_dir(work)
    os.replace(work, out_dir)
    layout.sync_dir(out_dir.parent)
    return report
