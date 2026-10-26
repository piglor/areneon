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


def test_a_run_may_name_a_team_dataset_and_the_default_names_none() -> None:
    assert read_run_config(raw()).team_dataset is False
    named = read_run_config(raw("valid/team-dataset.json"))
    assert named.team_dataset is True
    assert named.raw["team_dataset"]["path"] == ".tmp/datasets/train-v1"
    assert named.hash != read_run_config(raw()).hash


def test_the_committed_run_configs_are_valid() -> None:
    files = sorted((SAMPLES.parents[2] / "config/training").glob("*.json"))
    assert files
    for file in files:
        config = read_run_config(json.loads(file.read_text()))
        assert config.team_dataset is True
        assert {b.actor for b in config.generation.baselines} == {"random-legal", "actor-rule"}


def test_damage_aware_is_a_scripted_baseline_beside_the_others() -> None:
    config = read_run_config(raw("valid/damage-aware.json"))
    assert [b.actor for b in config.generation.baselines] == [
        "random-legal",
        "actor-rule",
        "damage-aware",
    ]
    assert config.hash != read_run_config(raw()).hash


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


def test_the_kl_target_is_optional_and_divided_once() -> None:
    assert read_run_config(raw()).ppo.target_kl is None
    with_target = raw("valid/target-kl.json")
    config = read_run_config(with_target)
    assert with_target["ppo"]["target_kl_micro"] == 20000
    assert config.ppo.target_kl == 0.02
    assert config.hash != read_run_config(raw()).hash
    with pytest.raises(RunConfigError):
        read_run_config(raw("invalid/zero-target-kl.json"))


def test_model_options_are_optional() -> None:
    plain = read_run_config(raw())
    assert plain.model.options == ()
    assert plain.model.encoder.options == ()
    config = read_run_config(raw("valid/empty-options.json"))
    assert config.model.options == ()
    assert config.model.encoder.options == ()
    assert config.hash != plain.hash  # another record, another hash; the encoder is the same
    for name in ("unknown-model-option", "unknown-encoder-option"):
        with pytest.raises(RunConfigError):
            read_run_config(raw(f"invalid/{name}.json"))


def test_the_candidate_matchup_encoder_option_is_read_and_only_true_is_a_value() -> None:
    config = read_run_config(raw("valid/candidate-matchup.json"))
    assert config.model.encoder.options == (("candidate_matchup", True),)
    assert config.model.layout_changing
    assert config.hash != read_run_config(raw()).hash
    with pytest.raises(RunConfigError):
        read_run_config(raw("invalid/candidate-matchup-false.json"))
