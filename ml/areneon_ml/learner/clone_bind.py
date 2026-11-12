"""Bind the provenance of a clone directory made before #853 (migration, #1356).

``write_clone`` writes ``provenance.json`` beside ``sources.json``; a directory made before that
has none, and ``import-clone`` refuses it. ``bind_clone`` verifies exactly what ``write_clone`` and
``import_clone`` require, and only then writes the binding:

* one record per role, named by its checkpoint id, each a clone record;
* each checkpoint file hashes to its record's id;
* ``sources.json`` is canonical and maps every teacher set of the records to ``[]``;
* the clone state (``clone-state.safetensors`` and ``.json``) hashes to the records'
  ``learner_state_hash``, and its sidecar names the records' ``run_config_hash``.

Any mismatch is refused and nothing is written. An existing binding is never replaced by other
bytes (identical bytes: a no-op). Like the binding itself this is a self-consistency check of the
directory, not proof against someone who rewrites every file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from areneon_ml.checkpoint.binding import BINDING_NAME, binding_bytes, record_files
from areneon_ml.checkpoint.files import checkpoint_id
from areneon_ml.checkpoint.record import CheckpointRecord, RecordError, read_record
from areneon_ml.learner.state import state_hash
from areneon_ml.loop import layout
from areneon_ml.loop.warm_start import (
    WarmStartError,
    check_clone_record,
    check_sources_registry,
)
from areneon_ml.model.network import ROLES


class CloneBindError(ValueError):
    """The clone directory is not one that may be bound."""


@dataclass(frozen=True)
class Bound:
    """What ``bind_clone`` did: ``written`` is False when the same binding was already there."""

    path: Path
    written: bool


def _records(clone_dir: Path) -> list[CheckpointRecord]:
    found: dict[str, CheckpointRecord] = {}
    for path in sorted((clone_dir / "records").glob("*.json")):
        try:
            record = read_record(json.loads(path.read_bytes()))
        except (OSError, ValueError, RecordError) as exc:
            raise CloneBindError(f"{path.name}: {exc}") from exc
        if path.name != f"{record.checkpoint_id}.json":
            raise CloneBindError(f"{path.name}: a clone record is named by its checkpoint id")
        if record.role in found:
            raise CloneBindError(f"the clone output holds two {record.role} records")
        found[record.role] = record
    if set(found) != set(ROLES):
        raise CloneBindError("the clone output must hold exactly one preview and one battle record")
    return [found[role] for role in ROLES]


def _check_state(clone_dir: Path, records: list[CheckpointRecord]) -> None:
    try:
        tensors = (clone_dir / "clone-state.safetensors").read_bytes()
        sidecar_bytes = (clone_dir / "clone-state.json").read_bytes()
        sidecar = json.loads(sidecar_bytes)
    except (OSError, ValueError) as exc:
        raise CloneBindError(f"the clone state: {exc}") from exc
    if not isinstance(sidecar, dict) or sidecar.get("kind") != "clone":
        raise CloneBindError("clone-state.json is not a clone state sidecar")
    for record in records:
        if record.learner_state_hash != state_hash(tensors, sidecar):
            raise CloneBindError(f"the {record.role} record names another clone state")
        if record.run_config_hash != sidecar.get("run_config_hash"):
            raise CloneBindError(f"the {record.role} record names another run config")


def bind_clone(clone_dir: Path) -> Bound:
    """Verify ``clone_dir`` and write its ``provenance.json`` (see the module docstring)."""
    records = _records(clone_dir)
    for record in records:
        try:
            check_clone_record(record, record.role)
            data = (clone_dir / "checkpoints" / f"{record.checkpoint_id}.safetensors").read_bytes()
        except (WarmStartError, OSError) as exc:
            raise CloneBindError(f"the {record.role} record: {exc}") from exc
        if checkpoint_id(data) != record.checkpoint_id:
            raise CloneBindError(f"the {record.role} checkpoint file does not hash to its id")
    try:
        registry = check_sources_registry(clone_dir, records)
    except WarmStartError as exc:
        raise CloneBindError(str(exc)) from exc
    _check_state(clone_dir, records)
    wanted = binding_bytes(record_files(clone_dir / "records"), registry)
    target = clone_dir / BINDING_NAME
    if target.exists():
        if target.read_bytes() != wanted:
            raise CloneBindError(f"{BINDING_NAME} exists and differs: never overwritten")
        return Bound(target, False)
    layout.write_atomic(target, wanted)
    return Bound(target, True)
