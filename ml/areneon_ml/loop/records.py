"""What a generation records: the policy identity, team-set manifest and CheckpointRecords."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig

# What the worker does with the pair when the loop plays it: sample at temperature 1 in float32
# (ADR-010 §2). They are part of the policy's identity, so a worker that changed them would be
# a different policy id.
MODE = "sample"
TEMPERATURE = 1
PRECISION = "float32"
FULL_TEMPERATURE_PPM = 1_000_000  # temperature 1 in millionths: the default, never written


def policy_id(
    preview: str,
    battle: str,
    mode: str = MODE,
    temperature_ppm: int | None = None,
    truncation: dict[str, Any] | None = None,
) -> str:
    """``policy:`` + SHA-256 of ``JCS({preview, battle, mode, temperature, precision})``.

    The loop plays ``sample`` (the default); the progress evaluation also plays ``argmax``,
    which is a different policy with a different id.

    Play settings (ADR-023 §1) enter the preimage only when set, so every earlier id is
    unchanged: ``temperature_ppm`` (1 to 999,999; the default 1,000,000 is never written)
    replaces ``temperature``, and ``truncation`` (``{"kind": "threshold", "min_ppm": n}``)
    is added. They apply to ``sample`` only.
    """
    body: dict[str, Any] = {
        "preview": preview,
        "battle": battle,
        "mode": mode,
        "temperature": TEMPERATURE,
        "precision": PRECISION,
    }
    if temperature_ppm is not None or truncation is not None:
        if mode != "sample":
            raise ValueError("a temperature or a truncation applies to sample mode only")
        if temperature_ppm is not None:
            if not 1 <= temperature_ppm < FULL_TEMPERATURE_PPM:
                raise ValueError("temperature_ppm must be 1 to 999999; the default is omitted")
            del body["temperature"]
            body["temperature_ppm"] = temperature_ppm
        if truncation is not None:
            if truncation.keys() != {"kind", "min_ppm"} or truncation["kind"] != "threshold":
                raise ValueError("an unknown truncation")
            if not 1 <= truncation["min_ppm"] < FULL_TEMPERATURE_PPM:
                raise ValueError("min_ppm must be 1 to 999999")
            body["truncation"] = dict(truncation)
    return "policy:" + sha256_hex(jcs(body))


def team_set_manifest(hashes: Iterable[str]) -> dict[str, Any]:
    """The content-addressed list of every team the run drew (protocol C7, T3)."""
    return {"schema_version": 1, "teams": sorted(set(hashes))}


def team_set_hash(manifest: dict[str, Any]) -> str:
    return sha256_hex(jcs(manifest))


@dataclass(frozen=True)
class RecordFacts:
    """What a CheckpointRecord says beyond the run config."""

    role: str
    checkpoint_id: str
    parent: str | None
    training_set_hashes: list[str]
    team_set_manifest_hash: str
    learner_state_hash: str
    encoder_version: str
    manifest: dict[str, Any]


def checkpoint_record(config: RunConfig, facts: RecordFacts) -> CheckpointRecord:
    """The record of one frozen checkpoint, validated by the contract and the reader rules.

    The loop's checkpoints are synthetic (the training loader refuses anything else) and
    play no league yet, so ``league_opponents`` is empty: a mirror match against the
    parent's pair is lineage, not a league opponent, and scripted baselines never count.
    """
    raw: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": facts.checkpoint_id,
        "role": facts.role,
        "lineage_id": config.lineage_id,
        "league_opponents": [],
        "training_set_hashes": sorted(facts.training_set_hashes),
        "team_set_manifest_hash": facts.team_set_manifest_hash,
        "run_config_hash": config.hash,
        "learner_state_hash": facts.learner_state_hash,
        "encoder_version": facts.encoder_version,
        "data_class": "synthetic",
        "manifest": facts.manifest,
    }
    if facts.parent is not None:
        raw["parent"] = facts.parent
    return read_record(raw)
