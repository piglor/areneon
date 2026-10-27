"""A warm start: generation 0 from a declared cloned pair (ADR-010 erratum A2, ADR-019 track 1).

``TrainingRunConfig.warm_start`` names two checkpoint ids. They are never fetched: the explicit
command ``import_clone`` copies a cloning run's output (``learner/clone_run.py``: ``checkpoints/``,
``records/``, ``sources.json``) into the run directory, the checkpoints into the store the worker
reads and the records into ``warm-start/records/<role>.json``. At the start of generation 0
``load_warm_start`` then refuses anything that is not a pair this run may start from:

* each file hashes to its id, holds the network of its role, was made for the run's engine-data
  table and encoder version, and holds exactly the run's ``model`` section (and re-serialises to the
  same bytes, so the pair the loop freezes is the pair named);
* each ``CheckpointRecord`` is a clone record: a non-empty ``training_set_hashes`` (the teacher
  set), no ``parent``, no league opponents or exploiter target, ``data_class`` synthetic, the run's
  encoder version and format. Only ADR-019 track 1 is admitted; a record that descends from
  anything, or a private one, is refused.

Generation 0 freezes the pair with no learner step. Its records are the clone records themselves,
byte for byte (a checkpoint has one record, and the provenance graph refuses a second), so the
teacher set sits in the ancestry of every later generation (the C1/C6 checker's outside source).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import CanonicalError, jcs

from areneon_ml.checkpoint.files import (
    CheckpointFileError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    write_checkpoint,
)
from areneon_ml.checkpoint.record import CheckpointRecord, RecordError, read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, PolicyNetwork

WARM_DIR = "warm-start"


class WarmStartError(ValueError):
    """The declared warm-start pair is missing, damaged or not a pair this run may start from."""


@dataclass(frozen=True)
class WarmPair:
    """The loaded pair and the clone records that name it."""

    networks: dict[str, PolicyNetwork]
    records: dict[str, CheckpointRecord]


def records_dir(run_dir: Path) -> Path:
    return run_dir / WARM_DIR / "records"


def record_path(run_dir: Path, role: str) -> Path:
    return records_dir(run_dir) / f"{role}.json"


def check_clone_record(record: CheckpointRecord, role: str) -> None:
    """Refuse a record that is not the record of a cloned checkpoint (ADR-019 track 1)."""
    if record.role != role:
        raise WarmStartError(f"the {role} warm-start record is a {record.role} record")
    if not record.training_set_hashes:
        raise WarmStartError(f"the {role} record names no training set: not a clone record")
    if record.parent is not None or record.exploiter_target is not None or record.league_opponents:
        raise WarmStartError(f"the {role} record descends from another checkpoint: not a clone")
    if record.data_class != "synthetic":
        raise WarmStartError(f"the {role} record is not synthetic: only track 1 is admitted")


def load_warm_start(
    config: RunConfig, game: GameData, run_dir: Path, encoder_version: str
) -> WarmPair:
    """Load and check the pair ``config.warm_start`` names from the run directory's stores."""
    warm = config.warm_start
    if warm is None:
        raise WarmStartError("the run config declares no warm start")
    networks: dict[str, PolicyNetwork] = {}
    records: dict[str, CheckpointRecord] = {}
    for role in ROLES:
        wanted = warm[role]
        store = run_dir / "checkpoints" / f"{wanted}.safetensors"
        try:
            data = store.read_bytes()
        except OSError as exc:
            raise WarmStartError(
                f"the {role} checkpoint {wanted} is not in the run's checkpoint store "
                "(import it with `import-clone`)"
            ) from exc
        try:
            network = load_checkpoint(data, game, wanted, role, configured=config.model)
        except CheckpointFileError as exc:
            raise WarmStartError(f"the {role} warm-start checkpoint: {exc}") from exc
        if network.config != config.model:
            raise WarmStartError(
                f"the {role} warm-start checkpoint was made under another model section"
            )
        if checkpoint_id(checkpoint_bytes(network, game)) != wanted:
            raise WarmStartError(
                f"the {role} warm-start checkpoint does not re-serialise as itself"
            )
        networks[role] = network
        records[role] = _read_clone_record(config, run_dir, role, wanted, encoder_version)
    return WarmPair(networks, records)


def _read_clone_record(
    config: RunConfig, run_dir: Path, role: str, wanted: str, encoder_version: str
) -> CheckpointRecord:
    path = record_path(run_dir, role)
    try:
        record = read_record(layout.read_json(path))
    except (OSError, layout.RunDirError, RecordError, ValueError) as exc:
        raise WarmStartError(f"the {role} warm-start record is missing or invalid: {exc}") from exc
    if record.checkpoint_id != wanted:
        raise WarmStartError(f"the {role} warm-start record names another checkpoint")
    check_clone_record(record, role)
    if record.encoder_version != encoder_version:
        raise WarmStartError(f"the {role} record was made for another encoder version")
    if record.manifest.get("format_id") != config.format_id:
        raise WarmStartError(f"the {role} record was made for another format")
    return record


def import_clone(clone_dir: Path, run_dir: Path) -> dict[str, str]:
    """Copy a cloning run's checkpoints and records into ``run_dir``; return the ids by role.

    The pair to start from is the output's own: one record per role. Only structure is checked
    here (a record reads, is a clone record, its checkpoint file hashes to its id, and the output's
    ``sources.json`` maps every teacher set to ``[]``); the checks against a run config are
    ``load_warm_start``'s, at the start of the run. Nothing is overwritten with different bytes.
    """
    found: dict[str, tuple[CheckpointRecord, bytes, bytes]] = {}
    for path in sorted((clone_dir / "records").glob("*.json")):
        try:
            record = read_record(layout.read_json(path))
            data = (clone_dir / "checkpoints" / f"{record.checkpoint_id}.safetensors").read_bytes()
        except (OSError, layout.RunDirError, RecordError, ValueError) as exc:
            raise WarmStartError(f"{path.name}: {exc}") from exc
        if path.name != f"{record.checkpoint_id}.json":
            raise WarmStartError(f"{path.name}: a clone record is named by its checkpoint id")
        check_clone_record(record, record.role)
        if checkpoint_id(data) != record.checkpoint_id:
            raise WarmStartError(f"{path.name}: its checkpoint file does not hash to its id")
        if record.role in found:
            raise WarmStartError(f"the clone output holds two {record.role} records")
        found[record.role] = (record, data, path.read_bytes())
    if set(found) != set(ROLES):
        raise WarmStartError("the clone output must hold exactly one preview and one battle record")
    _check_sources(clone_dir, [r for r, _, _ in found.values()])
    with layout.run_lock(run_dir):
        layout.ensure_run_dirs(run_dir)
        for role, (_record, data, text) in found.items():
            target = record_path(run_dir, role)
            if target.exists() and target.read_bytes() != text:
                raise WarmStartError(f"{target.name}: the run already holds another {role} record")
            layout.write_atomic(target, text)
            write_checkpoint(run_dir, data)
    return {role: found[role][0].checkpoint_id for role in ROLES}


def _check_sources(clone_dir: Path, records: list[CheckpointRecord]) -> None:
    try:
        sources: Any = json.loads((clone_dir / "sources.json").read_bytes())
        canonical = jcs(sources)
    except (OSError, ValueError, CanonicalError) as exc:
        raise WarmStartError(f"sources.json: {exc}") from exc
    needed = {h for record in records for h in record.training_set_hashes}
    if sources != {h: [] for h in needed} or (clone_dir / "sources.json").read_bytes() != canonical:
        raise WarmStartError("sources.json must map each teacher set of the records to []")
