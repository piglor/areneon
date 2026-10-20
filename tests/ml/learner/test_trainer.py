"""The learner: what a generation changes and leaves alone, and that it repeats (ADR-010 §1)."""

from __future__ import annotations

import copy
import math
import random
from functools import cache
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import Trajectory, trajectories
from areneon_ml.learner.ppo import gae, policy_terms, terminal_rewards
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
    assert all(math.isfinite(v) for v in report.as_dict()["battle"].values())


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
    assert configure_learner(cuda) == torch.device("cuda")
    torch.use_deterministic_algorithms(True)


def test_a_batch_moves_whole() -> None:
    config = tiny_config()
    work = trajs(config)
    batch = collate([work[0].steps[0].query])
    moved = to_device(batch, torch.device("cpu"))
    assert moved.kind == batch.kind
    assert torch.equal(moved.cand_mask, batch.cand_mask)
