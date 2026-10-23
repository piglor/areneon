"""The CloneRunConfig reader (ADR-019 track 1, ADR-010 erratum A2).

The record is reviewed JSON, validated against the contract first and then
against the two rules the profile cannot say (they are the TS reader's too):
the attention heads divide the token width and a Team Preview order cannot
name more positions than a side holds (both checked by ``ModelConfig``).
Real-valued settings are integers in millionths (``_micro``) or billionths
(``_nano``); this module is the one place that turns them into floats, by one
exact division. The record is named only by ``hash``, the SHA-256 of its JCS
bytes, which a cloned checkpoint's record carries as ``run_config_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.learner.config import MICRO, NANO
from areneon_ml.model.network import ConfigError, ModelConfig


class CloneConfigError(ValueError):
    """A CloneRunConfig is not usable."""


@dataclass(frozen=True)
class CloneSettings:
    epochs: int
    minibatch_size: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    holdout_key: bytes
    holdout_ppm: int


@dataclass(frozen=True)
class CloneConfig:
    """A validated CloneRunConfig and its hash."""

    lineage_id: str
    format_id: str
    run_key: str
    init_preview: int
    init_battle: int
    learner_seed: int
    model: ModelConfig
    teacher_policy_id: str
    teacher_set_hash: str
    clone: CloneSettings
    device: str
    threads: int
    raw: dict[str, Any]
    hash: str


def read_clone_config(raw: object) -> CloneConfig:
    """Validate ``raw`` against the contract and the reader rules; copy nothing mutable out."""
    try:
        validate("clone-run-config", raw)
    except ContractError as exc:
        raise CloneConfigError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    try:
        model = ModelConfig.from_dict(raw["model"])
    except ConfigError as exc:
        raise CloneConfigError(f"clone-run-config: {exc}") from exc
    seeds, teacher, clone = raw["seeds"], raw["teacher"], raw["clone"]
    return CloneConfig(
        lineage_id=raw["lineage_id"],
        format_id=raw["format_id"],
        run_key=seeds["run_key"],
        init_preview=seeds["init_preview"],
        init_battle=seeds["init_battle"],
        learner_seed=seeds["learner"],
        model=model,
        teacher_policy_id=teacher["behaviour_policy_id"],
        teacher_set_hash=teacher["training_set_hash"],
        clone=CloneSettings(
            epochs=clone["epochs"],
            minibatch_size=clone["minibatch_size"],
            learning_rate=clone["learning_rate_micro"] / MICRO,
            adam_beta1=clone["adam_beta1_micro"] / MICRO,
            adam_beta2=clone["adam_beta2_micro"] / MICRO,
            adam_eps=clone["adam_eps_nano"] / NANO,
            max_grad_norm=clone["max_grad_norm_micro"] / MICRO,
            holdout_key=bytes.fromhex(clone["holdout"]["key"]),
            holdout_ppm=clone["holdout"]["ppm"],
        ),
        device=raw["learner"]["device"],
        threads=raw["learner"]["threads"],
        raw=raw,
        hash=sha256_hex(jcs(raw)),
    )
