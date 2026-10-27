"""The TrainingRunConfig reader (ADR-010 §5).

The record is reviewed JSON, validated against the contract first and then
against the five rules the profile cannot say (they are the TS reader's too):
the attention heads divide the token width, a Team Preview order cannot name
more positions than a side holds (both checked by ``ModelConfig``), and each
scripted baseline is listed once, a warm start names two different
checkpoints, and a value warm-up needs a warm start.
Real-valued settings are stored as integers in millionths (``_micro``) or
billionths (``_nano``); this module is the one place that turns them into
floats, by one exact division. The record is named only by ``run_config_hash``,
the SHA-256 of its JCS bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.model.network import ConfigError, ModelConfig

MICRO = 1_000_000
NANO = 1_000_000_000


class RunConfigError(ValueError):
    """A TrainingRunConfig is not usable."""


@dataclass(frozen=True)
class Baseline:
    actor: str
    policy_id: str
    games: int


@dataclass(frozen=True)
class GenerationSettings:
    games: int
    baselines: tuple[Baseline, ...]
    excluded_abilities: tuple[str, ...]
    max_turns: int
    n_parallel: int
    preview_ms: int
    battle_ms: int
    handshake_timeout_ms: int
    max_line_bytes: int


@dataclass(frozen=True)
class LinearSchedule:
    """``learning_rate_schedule`` of kind ``linear`` (ADR-010 erratum A1), in exact integers."""

    start_micro: int
    final_micro: int
    generations: int

    def rate_micro(self, update: int) -> int:
        """The rate of learner update ``update`` (from 0) in millionths.

        ``start + trunc((final - start) * min(t, generations) / generations)``: integer
        arithmetic, truncation toward zero, exactly ``final`` from update ``generations`` on.
        """
        if update < 0:
            raise RunConfigError("a learner update index counts from 0")
        delta = (self.final_micro - self.start_micro) * min(update, self.generations)
        step = abs(delta) // self.generations
        return self.start_micro + (step if delta >= 0 else -step)

    def rate(self, update: int) -> float:
        """``rate_micro`` as the optimiser's float: the one division by 10^6 for the schedule."""
        return self.rate_micro(update) / MICRO


@dataclass(frozen=True)
class PPOSettings:
    epochs: int
    minibatch_size: int
    learning_rate: float
    discount: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    normalize_advantages: bool
    target_kl: float | None
    # None for an absent or ``constant`` schedule: the optimiser keeps ``learning_rate``
    schedule: LinearSchedule | None
    # ``ppo.value_warmup.generations`` (ADR-019 section 4): the learner updates, from the first,
    # that train the value head only. None when absent (normal PPO throughout).
    value_warmup: int | None = None


@dataclass(frozen=True)
class RunConfig:
    """A validated TrainingRunConfig and its hash.

    ``team_dataset`` says the record names a generated team dataset (the Node side
    reads and verifies it; this reader only needs to know whether to start that script).
    """

    lineage_id: str
    format_id: str
    run_key: str
    init_preview: int
    init_battle: int
    learner_seed: int
    model: ModelConfig
    generation: GenerationSettings
    ppo: PPOSettings
    device: str
    threads: int
    team_dataset: bool
    #: ``warm_start`` (ADR-010 erratum A2): the cloned checkpoint id of each role, or ``None``
    #: for the seeded initialisation. The loop checks the files; this reader checks the ids.
    warm_start: Mapping[str, str] | None
    raw: dict[str, Any]
    hash: str


def _schedule(ppo: dict[str, Any]) -> LinearSchedule | None:
    found = ppo.get("learning_rate_schedule")
    if found is None or found["kind"] == "constant":
        return None
    return LinearSchedule(ppo["learning_rate_micro"], found["final_micro"], found["generations"])


def read_run_config(raw: object) -> RunConfig:
    """Validate ``raw`` against the contract and the reader rules; copy nothing mutable out."""
    try:
        validate("training-run-config", raw)
    except ContractError as exc:
        raise RunConfigError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    try:
        model = ModelConfig.from_dict(raw["model"])
    except ConfigError as exc:
        raise RunConfigError(f"training-run-config: {exc}") from exc
    actors = [b["actor"] for b in raw["generation"]["baseline_games"]]
    if len(set(actors)) != len(actors):
        raise RunConfigError("training-run-config: a scripted baseline may be listed once")
    warm = raw.get("warm_start")
    if warm is not None and warm["preview"] == warm["battle"]:
        raise RunConfigError(
            "training-run-config: warm_start names one checkpoint for both roles "
            "(a checkpoint holds one role's network)"
        )
    if "value_warmup" in raw["ppo"] and warm is None:
        raise RunConfigError(
            "training-run-config: ppo.value_warmup needs warm_start "
            "(only a cloned start has an untrained value head to warm up)"
        )
    gen, ppo, seeds = raw["generation"], raw["ppo"], raw["seeds"]
    return RunConfig(
        lineage_id=raw["lineage_id"],
        format_id=raw["format_id"],
        run_key=seeds["run_key"],
        init_preview=seeds["init_preview"],
        init_battle=seeds["init_battle"],
        learner_seed=seeds["learner"],
        model=model,
        generation=GenerationSettings(
            games=gen["games"],
            baselines=tuple(
                Baseline(b["actor"], b["policy_id"], b["games"]) for b in gen["baseline_games"]
            ),
            excluded_abilities=tuple(gen["excluded_abilities"]),
            max_turns=gen["max_turns"],
            n_parallel=gen["n_parallel"],
            preview_ms=gen["latency_budget_ms"]["preview_ms"],
            battle_ms=gen["latency_budget_ms"]["battle_ms"],
            handshake_timeout_ms=gen["handshake_timeout_ms"],
            max_line_bytes=gen["max_line_bytes"],
        ),
        ppo=PPOSettings(
            epochs=ppo["epochs"],
            minibatch_size=ppo["minibatch_size"],
            learning_rate=ppo["learning_rate_micro"] / MICRO,
            discount=ppo["discount_micro"] / MICRO,
            gae_lambda=ppo["gae_lambda_micro"] / MICRO,
            clip=ppo["clip_micro"] / MICRO,
            value_coef=ppo["value_coef_micro"] / MICRO,
            entropy_coef=ppo["entropy_coef_micro"] / MICRO,
            max_grad_norm=ppo["max_grad_norm_micro"] / MICRO,
            adam_beta1=ppo["adam_beta1_micro"] / MICRO,
            adam_beta2=ppo["adam_beta2_micro"] / MICRO,
            adam_eps=ppo["adam_eps_nano"] / NANO,
            normalize_advantages=ppo["normalize_advantages"],
            target_kl=ppo["target_kl_micro"] / MICRO if "target_kl_micro" in ppo else None,
            schedule=_schedule(ppo),
            value_warmup=ppo["value_warmup"]["generations"] if "value_warmup" in ppo else None,
        ),
        device=raw["learner"]["device"],
        threads=raw["learner"]["threads"],
        team_dataset="team_dataset" in raw,
        warm_start=None if warm is None else MappingProxyType(dict(warm)),
        raw=raw,
        hash=sha256_hex(jcs(raw)),
    )
