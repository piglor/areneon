"""The TrainingRunConfig reader (ADR-010 §5): the contract, then what the profile cannot say."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfigError, read_run_config

from tests.ml.model.support import SAMPLES

CONFIGS = SAMPLES / "training-run-config"


def raw(name: str = "valid/tiny.json") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((CONFIGS / name).read_text())
    return loaded


def test_the_tiny_sample_is_read_with_every_real_value_exact() -> None:
    config = read_run_config(raw())
    assert (config.lineage_id, config.run_key, config.device, config.threads) == (
        "tiny-main",
        "tiny-run",
        "cpu",
        1,
    )
    assert (config.init_preview, config.init_battle, config.learner_seed) == (11, 12, 13)
    ppo = config.ppo
    assert (ppo.epochs, ppo.minibatch_size) == (2, 16)
    assert ppo.learning_rate == 0.001
    assert (ppo.discount, ppo.gae_lambda, ppo.clip) == (1.0, 0.95, 0.2)
    assert (ppo.value_coef, ppo.entropy_coef, ppo.max_grad_norm) == (0.5, 0.01, 0.5)
    assert (ppo.adam_beta1, ppo.adam_beta2, ppo.adam_eps) == (0.9, 0.999, 1e-8)
    assert ppo.normalize_advantages is True
    gen = config.generation
    assert (gen.games, gen.max_turns, gen.n_parallel, gen.max_line_bytes) == (4, 6, 2, 4_000_000)
    assert (gen.preview_ms, gen.battle_ms, gen.handshake_timeout_ms) == (30000, 30000, 60000)
    assert gen.excluded_abilities == ("Illusion",)
    assert [(b.actor, b.policy_id, b.games) for b in gen.baselines] == [
        ("random-legal", "random-legal@tiny", 2)
    ]
    assert config.model.d_model == 16
    assert config.model.encoder.max_picks == 4
    assert config.format_id == "gen9championsvgc2026regmc"


def test_the_hash_is_the_sha256_of_the_jcs_bytes_and_ignores_key_order() -> None:
    config = read_run_config(raw())
    assert config.hash == sha256_hex(jcs(raw()))
    assert read_run_config(dict(reversed(raw().items()))).hash == config.hash
    other = raw()
    other["ppo"]["epochs"] += 1
    assert read_run_config(other).hash != config.hash


def test_no_baselines_is_a_valid_run() -> None:
    assert read_run_config(raw("valid/no-baselines.json")).generation.baselines == ()


READER_INVALID = sorted(p.name for p in (CONFIGS / "reader-invalid").iterdir())
INVALID = sorted(p.name for p in (CONFIGS / "invalid").iterdir())


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("heads-do-not-divide-width.json", "multiple of n_heads"),
        ("picks-exceed-team.json", "max_picks cannot exceed max_team"),
        ("duplicate-baseline.json", "may be listed once"),
    ],
)
def test_the_reader_rules_the_profile_cannot_say(name: str, message: str) -> None:
    assert name in READER_INVALID
    with pytest.raises(RunConfigError, match=message):
        read_run_config(raw(f"reader-invalid/{name}"))


def test_every_reader_invalid_sample_is_checked_above() -> None:
    assert len(READER_INVALID) == 3


@pytest.mark.parametrize("name", INVALID)
def test_a_record_the_schema_refuses_is_refused(name: str) -> None:
    with pytest.raises(RunConfigError, match="training-run-config"):
        read_run_config(raw(f"invalid/{name}"))
