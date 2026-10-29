"""``ppo.preview_bootstrap_micro`` (ADR-010 erratum A4 item 3, #1005): the Preview return mixes the
game score with the Battle network's FROZEN value of the Episode's first Battle decision."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from areneon_ml.learner import trainer
from areneon_ml.learner.config import MICRO, RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError, Trajectory
from areneon_ml.learner.diagnostics import explained_variance
from areneon_ml.learner.ppo import gae, normalise, terminal_rewards
from areneon_ml.learner.state import pack_state
from areneon_ml.learner.trainer import Learner

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs


def with_bootstrap(micro: int | None) -> Any:
    def edit(config: dict[str, Any]) -> None:
        if micro is not None:
            config["ppo"]["preview_bootstrap_micro"] = micro

    return edited_config(edit)


def captured_tables(student: Learner, config: Any) -> tuple[dict[str, Any], Any]:
    """Train one generation and keep each role's table as the epochs saw it."""
    seen: dict[str, Any] = {}
    original = Learner._epochs

    def spy(self: Learner, role: str, table: Any, warm: bool) -> Any:
        seen[role] = table
        return original(self, role, table, warm)

    Learner._epochs = spy  # type: ignore[method-assign]
    try:
        report = student.train(trajs(config))
    finally:
        Learner._epochs = original  # type: ignore[method-assign]
    return seen, report


def expected_preview(config: Any, micro: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Hand-computed from a separate learner that never trained: the frozen Battle values."""
    fresh = learner(config)
    tr = trajs(config)
    battle = [t for t in tr if t.role == "battle"]
    _, bvalues = fresh._score("battle", [s for t in battle for s in t.steps])
    first, start = {}, 0
    for t in battle:
        first[t.episode] = float(bvalues[start])
        start += len(t.steps)
    preview = [t for t in tr if t.role == "preview"]
    _, pvalues = fresh._score("preview", [s for t in preview for s in t.steps])
    w = micro / MICRO
    targets = [(1 - w) * t.score + w * first[t.episode] for t in preview]
    returns = torch.tensor(targets, dtype=torch.float32)
    advantages = returns - pvalues
    return returns, advantages, targets


def test_full_weight_returns_are_the_frozen_battle_values() -> None:
    config = with_bootstrap(MICRO)
    seen, report = captured_tables(learner(config), config)
    returns, advantages, targets = expected_preview(config, MICRO)
    table = seen["preview"]
    # gae returns (target - V) + V, one float32 rounding from the target
    assert torch.allclose(table.returns.cpu(), returns, rtol=0, atol=1e-6)
    assert torch.allclose(table.advantages.cpu(), normalise(advantages), rtol=0, atol=1e-5)
    # the returns are not the scores: the Battle values differ from 0, 1/2 and 1
    assert any(t not in (0.0, 0.5, 1.0) for t in targets)
    # the explained variance is of the returns the learner trains on (the bootstrapped ones)
    _, pvalues = learner(config)._score(
        "preview", [s for t in trajs(config) if t.role == "preview" for s in t.steps]
    )
    assert report.roles["preview"].explained_variance == pytest.approx(
        explained_variance(returns.double().tolist(), pvalues.double().tolist()), abs=1e-5
    )


def test_a_partial_weight_mixes_score_and_value_and_the_battle_role_is_unchanged() -> None:
    config = with_bootstrap(250_000)
    seen, _ = captured_tables(learner(config), config)
    returns, _, _ = expected_preview(config, 250_000)
    assert torch.allclose(seen["preview"].returns.cpu(), returns, rtol=0, atol=1e-6)
    plain_config = with_bootstrap(None)
    plain_seen, _ = captured_tables(learner(plain_config), plain_config)
    for name in ("returns", "advantages", "old_log_prob"):
        assert torch.equal(getattr(seen["battle"], name), getattr(plain_seen["battle"], name))


def test_the_battle_update_cannot_change_the_preview_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = with_bootstrap(MICRO)
    battle_first, _ = captured_tables(learner(config), config)
    monkeypatch.setattr(trainer, "ROLES", ("preview", "battle"))
    preview_first, _ = captured_tables(learner(config), config)
    for name in ("returns", "advantages"):
        assert torch.equal(
            getattr(battle_first["preview"], name), getattr(preview_first["preview"], name)
        )


def test_absent_and_zero_keep_todays_weights_and_the_report_key_is_only_set_with_the_field() -> (
    None
):
    absent = learner()
    absent_report = absent.train(trajs(tiny_config())).as_dict()
    zero_config = with_bootstrap(0)
    zero = learner(zero_config)
    zero_report = zero.train(trajs(zero_config)).as_dict()
    assert pack_state(absent).tensors == pack_state(zero).tensors
    assert all("preview_bootstrap_micro" not in r for r in absent_report.values())
    assert zero_report["preview"]["preview_bootstrap_micro"] == 0
    assert "preview_bootstrap_micro" not in zero_report["battle"]
    on_config = with_bootstrap(500_000)
    on = learner(on_config)
    on_report = on.train(trajs(on_config)).as_dict()
    assert on_report["preview"]["preview_bootstrap_micro"] == 500_000
    assert json.loads(pack_state(on).sidecar)["ppo"]["preview_bootstrap_micro"] == 500_000
    assert pack_state(on).tensors != pack_state(absent).tensors


def test_a_preview_step_without_a_battle_decision_is_refused() -> None:
    config = with_bootstrap(MICRO)
    tr = trajs(config)
    preview = [t for t in tr if t.role == "preview"]
    with pytest.raises(LearnerDataError, match="no Battle decision"):
        learner(config).train(preview)
    dropped = preview[0].episode
    partial = [t for t in tr if not (t.role == "battle" and t.episode == dropped)]
    with pytest.raises(LearnerDataError, match="no Battle decision"):
        learner(config).train(partial)
    # without the field the same trajectories train as before
    learner().train(partial)


def test_the_value_warm_up_and_a_battle_only_warm_start_still_bootstrap() -> None:
    def edit(c: dict[str, Any]) -> None:
        c["warm_start"] = {"battle": "b" * 64}
        c["ppo"]["value_warmup"] = {"generations": 1}
        c["ppo"]["preview_bootstrap_micro"] = MICRO

    config = edited_config(edit)
    seen, report = captured_tables(learner(config), config)
    returns, _, _ = expected_preview(config, MICRO)
    assert torch.allclose(
        seen["preview"].returns.cpu(), returns, rtol=0, atol=1e-6
    )  # unwarmed Preview
    assert report.roles["battle"].value_warmup is True
    assert report.roles["preview"].value_warmup is None


def test_the_reader_takes_the_field_and_the_range() -> None:
    assert (
        read_run_config(raw("valid/preview-bootstrap.json")).ppo.preview_bootstrap_micro == 500_000
    )
    assert (
        read_run_config(raw("valid/preview-bootstrap-zero.json")).ppo.preview_bootstrap_micro == 0
    )
    assert read_run_config(raw()).ppo.preview_bootstrap_micro is None
    for bad in (
        "invalid/preview-bootstrap-above-one.json",
        "invalid/preview-bootstrap-fractional.json",
    ):
        with pytest.raises(RunConfigError):
            read_run_config(raw(bad))


def test_the_single_step_rule_equals_gae_on_the_mixed_reward() -> None:
    value = torch.tensor([0.3])
    adv, ret = gae(terminal_rewards(1, 0.75), value, 0.99, 0.95)
    assert float(ret[0]) == 0.75
    assert float(adv[0]) == pytest.approx(0.45)


def test_trajectory_type_is_unchanged() -> None:
    assert Trajectory.__dataclass_fields__.keys() >= {"role", "steps", "score", "episode"}
