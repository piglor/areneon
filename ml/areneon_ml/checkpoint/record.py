"""The CheckpointRecord reader (ADR-010 §5, protocol C7).

The contract schema checks the shape; what the profile cannot say is checked
here, as the TS readers do for their records: the hash arrays are sorted and
unique, and no id refers to the record's own checkpoint (a checkpoint cannot
be initialised from, or trained against, itself). A record is content
addressed by the SHA-256 of its JCS bytes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex, validate


class RecordError(ValueError):
    """A CheckpointRecord breaks a rule the contract profile cannot say."""


def _strictly_ascending(values: list[str]) -> bool:
    return all(a < b for a, b in itertools.pairwise(values))


@dataclass(frozen=True)
class CheckpointRecord:
    """A validated record: the provenance of one frozen checkpoint."""

    checkpoint_id: str
    role: str
    lineage_id: str
    parent: str | None
    league_opponents: tuple[str, ...]
    exploiter_target: str | None
    training_set_hashes: tuple[str, ...]
    team_set_manifest_hash: str
    run_config_hash: str
    learner_state_hash: str
    encoder_version: str
    data_class: str
    manifest: dict[str, Any]
    #: ``game_manifests`` (ADR-027 5a), only for a run with ``sheet_regimes``: the version manifests
    #: of the run's formats, sorted by JCS bytes; ``manifest`` is one of them.
    game_manifests: tuple[dict[str, Any], ...] | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": 1,
            "checkpoint_id": self.checkpoint_id,
            "role": self.role,
            "lineage_id": self.lineage_id,
            "league_opponents": list(self.league_opponents),
            "training_set_hashes": list(self.training_set_hashes),
            "team_set_manifest_hash": self.team_set_manifest_hash,
            "run_config_hash": self.run_config_hash,
            "learner_state_hash": self.learner_state_hash,
            "encoder_version": self.encoder_version,
            "data_class": self.data_class,
            "manifest": self.manifest,
        }
        if self.parent is not None:
            out["parent"] = self.parent
        if self.exploiter_target is not None:
            out["exploiter_target"] = self.exploiter_target
        if self.game_manifests is not None:
            out["game_manifests"] = list(self.game_manifests)
        return out

    @property
    def record_hash(self) -> str:
        """SHA-256 of the record's JCS bytes: how an evaluation plan names it."""
        return sha256_hex(jcs(self.to_json()))


def read_record(raw: object) -> CheckpointRecord:
    """Validate against the contract, then apply the reader rules."""
    validate("checkpoint-record", raw)
    assert isinstance(raw, dict)  # validate() accepted an object
    own = raw["checkpoint_id"]
    for field in ("league_opponents", "training_set_hashes"):
        if not _strictly_ascending(raw[field]):
            raise RecordError(f"checkpoint-record: {field} must be sorted and unique")
    if raw.get("parent") == own:
        raise RecordError("checkpoint-record: a checkpoint cannot be its own parent")
    if own in raw["league_opponents"]:
        raise RecordError("checkpoint-record: a checkpoint cannot train against itself")
    if raw.get("exploiter_target") == own:
        raise RecordError("checkpoint-record: a checkpoint cannot be its own exploiter target")
    game_manifests = raw.get("game_manifests")
    if game_manifests is not None:
        _check_game_manifests(raw["manifest"], game_manifests)
    return CheckpointRecord(
        checkpoint_id=own,
        role=raw["role"],
        lineage_id=raw["lineage_id"],
        parent=raw.get("parent"),
        league_opponents=tuple(raw["league_opponents"]),
        exploiter_target=raw.get("exploiter_target"),
        training_set_hashes=tuple(raw["training_set_hashes"]),
        team_set_manifest_hash=raw["team_set_manifest_hash"],
        run_config_hash=raw["run_config_hash"],
        learner_state_hash=raw["learner_state_hash"],
        encoder_version=raw["encoder_version"],
        data_class=raw["data_class"],
        manifest=raw["manifest"],
        game_manifests=None if game_manifests is None else tuple(game_manifests),
    )


def _check_game_manifests(manifest: dict[str, Any], found: list[dict[str, Any]]) -> None:
    """``game_manifests``: sorted by JCS bytes and unique, ``manifest`` among them, one engine
    commit and regulation (the sheet-variant check proper needs the engine: the Node side's)."""
    keys = [jcs(m) for m in found]
    if not all(a < b for a, b in itertools.pairwise(keys)):
        raise RecordError(
            "checkpoint-record: game_manifests must be sorted by JCS bytes, each once"
        )
    if manifest not in found:
        raise RecordError("checkpoint-record: manifest must be one of game_manifests")
    if len({(m["showdown_commit"], m["regulation"]) for m in found}) > 1:
        raise RecordError(
            "checkpoint-record: game_manifests must share one engine commit and regulation"
        )
