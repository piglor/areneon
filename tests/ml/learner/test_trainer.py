"""The learner: what a generation changes and leaves alone, and that it repeats (ADR-010 §1)."""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import random
from functools import cache
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import Trajectory, trajectories
from areneon_ml.learner.ppo import LossWeights, gae, policy_terms, ppo_loss, terminal_rewards
from areneon_ml.learner.trainer import Learner, LearnerError, configure_learner, to_device
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import (
    POLICY,
    edited_config,
    episode_lines,
    learner,
    make_stream,
    networks,
    stream_of,
    tiny_config,
)
from tests.ml.model.support import game


@cache
def _default_trajectories() -> tuple[Trajectory, ...]:
    config = tiny_config()
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    text, manifest = make_stream()
    return tuple(trajectories(text, manifest, encoders, POLICY))


def trajs(_config: RunConfig) -> list[Trajectory]:
    """The default stream's trajectories (every test config has the same encoder)."""
    return list(_default_trajectories())


def weights_id(student: Learner) -> dict[str, str]:
    return {role: checkpoint_id(checkpoint_bytes(student.networks[role], game())) for role in ROLES}


def test_a_generation_changes_both_networks_and_counts_itself() -> None:
    config = tiny_config()
    student = learner(config)
    before = weights_id(student)
    report = student.train(trajs(config))
    assert student.generation == 1
    after = weights_id(student)
    assert all(after[role] != before[role] for role in ROLES)
    assert set(report.roles) == set(ROLES)
    battle, preview = report.roles["battle"], report.roles["preview"]
    # 55 battle decisions and 3 preview decisions; 2 epochs of minibatches of 16
    assert (battle.steps, battle.optimiser_steps) == (55, 2 * math.ceil(55 / 16))
    assert (preview.steps, preview.optimiser_steps) == (3, 2)
    stored = report.as_dict()["battle"]
    assert all(math.isfinite(v) for k, v in stored.items() if k != "value_calibration")
    assert math.isfinite(stored["explained_variance"])
    # every decision falls in exactly one calibration bin
    assert sum(b["count"] for b in stored["value_calibration"]) == 55


def test_the_same_inputs_give_the_same_weights_and_another_seed_does_not() -> None:
    config = tiny_config()
    first, second = learner(config), learner(config)
    work = trajs(config)
    first.train(work)
    second.train(work)
    assert weights_id(first) == weights_id(second)
    assert first.generator.get_state().equal(second.generator.get_state())
    other = tiny_config()
    raw = copy.deepcopy(other.raw)
    raw["seeds"]["learner"] += 1
    third = learner(
        type(other)(**{**other.__dict__, "raw": raw, "learner_seed": raw["seeds"]["learner"]})
    )
    third.train(work)
    assert weights_id(third) != weights_id(first)


def test_a_role_with_no_trajectory_is_left_alone() -> None:
    config = tiny_config()
    student = learner(config)
    before = weights_id(student)
    state_before = [p.clone() for p in student.networks["battle"].parameters()]
    preview_only = [t for t in trajs(config) if t.role == "preview"]
    report = student.train(preview_only)
    assert set(report.roles) == {"preview"}
    after = weights_id(student)
    assert after["battle"] == before["battle"]
    assert after["preview"] != before["preview"]
    assert not student.optimisers["battle"].state
    assert all(
        torch.equal(a, b)
        for a, b in zip(state_before, student.networks["battle"].parameters(), strict=True)
    )


def test_old_probabilities_and_advantages_come_from_the_frozen_weights() -> None:
    config = tiny_config()
    student = learner(config)
    work = [t for t in trajs(config) if t.role == "battle"]
    table = student._table("battle", work)  # the learner's own view of the frozen policy
    steps = [s for t in work for s in t.steps]
    assert len(table.steps) == len(steps)
    network = student.networks["battle"]
    network.eval()
    with torch.no_grad():
        out = network(collate([s.query for s in steps]))
        terms = policy_terms(
            out.logits,
            collate([s.query for s in steps]).cand_mask,
            torch.tensor([s.chosen for s in steps]),
        )
    assert table.old_log_prob.tolist() == pytest.approx(terms.log_prob.tolist(), abs=1e-5)
    # the first trajectory's returns: GAE of its own frozen values and the game score at its end
    first = work[0]
    values = out.value[: len(first.steps)]
    _, returns = gae(
        terminal_rewards(len(first.steps), first.score),
        values,
        config.ppo.discount,
        config.ppo.gae_lambda,
    )
    assert table.returns[: len(first.steps)].tolist() == pytest.approx(returns.tolist(), abs=1e-5)
    # advantages were normalised: zero mean, unit deviation
    assert table.advantages.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert table.advantages.pow(2).mean().sqrt().item() == pytest.approx(1.0, abs=1e-4)


def test_advantage_normalisation_follows_the_config() -> None:
    plain = edited_config(lambda raw: raw["ppo"].update(normalize_advantages=False))
    student = learner(plain)
    table = student._table("battle", [t for t in trajs(plain) if t.role == "battle"])
    assert table.advantages.mean().item() != pytest.approx(0.0, abs=1e-3)


def test_a_chosen_candidate_that_won_becomes_likelier_and_one_that_lost_less_likely() -> None:
    config = tiny_config(learning_rate_micro=20000, epochs=8, entropy_coef_micro=0)
    examples = episode_lines(0, "p1", random.Random(3))
    preview = examples[0]
    assert preview["observation"]["kind"] == "preview"
    a, b = (c["choice"] for c in preview["candidates"][:2])
    won, lost = copy.deepcopy(preview), copy.deepcopy(preview)
    won.update(choice=a, outcome="win", episode_hash="1" * 64)
    lost.update(choice=b, outcome="loss", episode_hash="2" * 64)
    text, manifest = stream_of([won, lost])
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    work = trajectories(text, manifest, encoders, POLICY)
    student = learner(config)

    def probabilities() -> list[float]:
        network = student.networks["preview"]
        network.eval()
        with torch.no_grad():
            probs = torch.softmax(network(collate([work[0].steps[0].query])).logits[0], -1)
        return [probs[0].item(), probs[1].item()]

    before = probabilities()
    student.train(work)
    after = probabilities()
    assert after[0] > before[0]
    assert after[1] < before[1]


def test_the_gradient_norm_is_clipped_to_the_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    seen: list[float] = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(parameters: Any, max_norm: float, *args: Any, **kwargs: Any) -> torch.Tensor:
        seen.append(max_norm)
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    learner(config).train(trajs(config))
    assert seen
    assert set(seen) == {0.5}


def test_the_optimiser_is_built_from_the_config() -> None:
    config = tiny_config()
    student = learner(config)
    group = student.optimisers["battle"].param_groups[0]
    assert (group["lr"], group["betas"], group["eps"]) == (0.001, (0.9, 0.999), 1e-8)


def test_both_networks_are_needed() -> None:
    config = tiny_config()
    one = networks(config)
    del one["preview"]
    with pytest.raises(LearnerError, match="one network per role"):
        Learner(config, one, torch.device("cpu"))


def test_configuring_the_learner_sets_threads_and_deterministic_algorithms() -> None:
    config = tiny_config()
    assert configure_learner(config) == torch.device("cpu")
    assert torch.get_num_threads() == config.threads
    assert torch.are_deterministic_algorithms_enabled()


def test_a_cuda_config_is_refused_where_there_is_no_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = edited_config(lambda raw: raw["learner"].update(device="cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(LearnerError, match="no usable CUDA device"):
        configure_learner(cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "x")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    assert configure_learner(cuda) == torch.device("cuda")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # an operator's own setting is kept
    configure_learner(cuda)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    torch.use_deterministic_algorithms(True)


def test_a_batch_moves_whole() -> None:
    config = tiny_config()
    work = trajs(config)
    batch = collate([work[0].steps[0].query])
    moved = to_device(batch, torch.device("cpu"))
    assert moved.kind == batch.kind
    assert torch.equal(moved.cand_mask, batch.cand_mask)


def test_the_configured_loss_weights_reach_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    config = tiny_config(value_coef_micro=700_000, entropy_coef_micro=30_000, clip_micro=150_000)
    seen: list[LossWeights] = []
    real = ppo_loss

    def spy(terms: Any, value: Any, targets: Any, weights: LossWeights) -> Any:
        seen.append(weights)
        return real(terms, value, targets, weights)

    monkeypatch.setattr("areneon_ml.learner.trainer.ppo_loss", spy)
    learner(config).train(trajs(config))
    assert seen
    assert set(seen) == {LossWeights(clip=0.15, value_coef=0.7, entropy_coef=0.03)}


# --- the KL early stop (ADR-010 erratum A1, item 1) ---


def _scripted_kl(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    """Make the loss report these approximate KLs, one per minibatch, in order."""
    real = ppo_loss
    queue = iter(values)

    def loss(*args: Any, **kwargs: Any) -> Any:
        parts = real(*args, **kwargs)
        return dataclasses.replace(parts, approx_kl=torch.tensor(next(queue)))

    monkeypatch.setattr("areneon_ml.learner.trainer.ppo_loss", loss)


def _battle_only(config: RunConfig) -> list[Trajectory]:
    return [t for t in trajs(config) if t.role == "battle"]


def test_the_stop_fires_after_the_epoch_whose_mean_kl_is_above_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config(epochs=4, target_kl_micro=30000)
    assert config.ppo.target_kl == 0.03
    # 55 decisions in minibatches of 16: 4 per epoch. Means: 0.01, 0.04 (above 0.03), ...
    _scripted_kl(monkeypatch, [0.01] * 4 + [0.05, 0.07, 0.01, 0.03] + [0.9] * 8)
    report = learner(config).train(_battle_only(config)).roles["battle"]
    assert (report.epochs_run, report.kl_stopped) == (2, True)
    assert report.optimiser_steps == 8


def test_a_mean_equal_to_the_target_does_not_stop_and_the_last_epoch_still_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config(epochs=2, target_kl_micro=500000)
    _scripted_kl(monkeypatch, [0.25, 0.75, 0.5, 0.5] * 2)
    equal = learner(config).train(_battle_only(config)).roles["battle"]
    assert (equal.epochs_run, equal.kl_stopped) == (2, False)
    _scripted_kl(monkeypatch, [0.1] * 4 + [0.9] * 4)
    last = learner(config).train(_battle_only(config)).roles["battle"]
    assert (last.epochs_run, last.kl_stopped) == (2, True)


def test_the_stop_is_reproducible_and_skips_the_later_shuffles() -> None:
    config = tiny_config(epochs=4, target_kl_micro=1)
    work = _battle_only(config)
    first, second = learner(config), learner(config)
    one, two = first.train(work), second.train(work)
    assert one.as_dict() == two.as_dict()
    assert weights_id(first) == weights_id(second)
    report = one.roles["battle"]
    assert (report.epochs_run, report.kl_stopped) == (1, True)
    assert report.optimiser_steps == math.ceil(55 / 16)
    full = learner(tiny_config(epochs=4))
    full.train(work)
    assert not first.generator.get_state().equal(full.generator.get_state())


def test_a_target_never_reached_runs_every_epoch_and_reports_it() -> None:
    config = tiny_config(target_kl_micro=1_000_000_000)
    report = learner(config).train(trajs(config))
    assert report.as_dict()["battle"]["epochs_run"] == 2
    assert report.as_dict()["battle"]["kl_stopped"] is False
    assert report.as_dict()["preview"]["epochs_run"] == 2


def test_without_a_target_the_report_keys_are_omitted_and_the_weights_are_unchanged() -> None:
    config = tiny_config()
    assert config.ppo.target_kl is None
    plain = learner(config)
    report = plain.train(trajs(config))
    for role in ROLES:
        assert set(report.as_dict()[role]) == {
            "steps",
            "optimiser_steps",
            "policy_loss",
            "value_loss",
            "entropy",
            "clip_fraction",
            "approx_kl",
            "explained_variance",
            "value_calibration",
        }
    # a target that never fires trains the same weights as no target
    never = tiny_config(target_kl_micro=1_000_000_000)
    other = learner(never)
    other.train(trajs(never))
    assert weights_id(other) == weights_id(plain)
