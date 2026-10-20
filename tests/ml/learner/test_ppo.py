"""The PPO pieces against values worked out by hand (ADR-010 §1)."""

from __future__ import annotations

import math

import pytest
import torch
from areneon_ml.learner.ppo import (
    GAME_SCORE,
    LossWeights,
    PolicyTerms,
    Targets,
    gae,
    normalise,
    policy_terms,
    ppo_loss,
    terminal_rewards,
)

WEIGHTS = LossWeights(clip=0.2, value_coef=0.0, entropy_coef=0.0)


def t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def close(actual: torch.Tensor, *expected: float, tol: float = 1e-5) -> None:
    assert actual.reshape(-1).tolist() == pytest.approx(list(expected), abs=tol)


def test_the_game_score_is_win_one_tie_half_loss_zero() -> None:
    assert GAME_SCORE == {"win": 1.0, "tie": 0.5, "loss": 0.0}


def test_only_the_last_step_is_rewarded() -> None:
    close(terminal_rewards(4, 0.5), 0.0, 0.0, 0.0, 0.5)
    close(terminal_rewards(1, 1.0), 1.0)


def test_gae_by_hand() -> None:
    # rewards 0, 0, 1; values .5, .4, .3; gamma .9; lambda .8; the value after the end is 0.
    # delta = [0 + .9*.4 - .5, 0 + .9*.3 - .4, 1 + 0 - .3] = [-.14, -.13, .7]
    # A2 = .7; A1 = -.13 + .72*.7 = .374; A0 = -.14 + .72*.374 = .12928
    advantages, returns = gae(t(0, 0, 1), t(0.5, 0.4, 0.3), 0.9, 0.8)
    close(advantages, 0.12928, 0.374, 0.7)
    close(returns, 0.62928, 0.774, 1.0)


def test_gae_with_lambda_one_and_no_discount_is_the_monte_carlo_return() -> None:
    # A_t = G - V_t with G the game score
    advantages, returns = gae(terminal_rewards(3, 0.5), t(0.1, 0.2, 0.3), 1.0, 1.0)
    close(advantages, 0.4, 0.3, 0.2)
    close(returns, 0.5, 0.5, 0.5)


def test_gae_with_lambda_zero_is_the_one_step_error() -> None:
    advantages, _ = gae(t(0, 0, 1), t(0.5, 0.4, 0.3), 0.9, 0.0)
    close(advantages, -0.14, -0.13, 0.7)


def test_a_one_step_game_has_the_score_minus_the_value() -> None:
    advantages, returns = gae(t(1.0), t(0.25), 0.9, 0.8)
    close(advantages, 0.75)
    close(returns, 1.0)


def test_normalise_by_hand_and_a_constant_stays_zero() -> None:
    # mean 2, population deviation sqrt(2/3)
    close(normalise(t(1, 2, 3)), -math.sqrt(1.5), 0.0, math.sqrt(1.5), tol=1e-4)
    close(normalise(t(5, 5, 5)), 0.0, 0.0, 0.0)


def test_the_policy_is_a_softmax_over_the_real_candidates_only() -> None:
    logits = t(0.0, 0.0, 0.0, 99.0).reshape(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    terms = policy_terms(logits, mask, torch.tensor([1]))
    close(terms.log_prob, -math.log(3))
    close(terms.entropy, math.log(3))


def test_log_probability_and_entropy_of_a_skewed_policy() -> None:
    # p = [.25, .75]
    logits = t(math.log(1), math.log(3)).reshape(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    terms = policy_terms(logits, mask, torch.tensor([1]))
    close(terms.log_prob, math.log(0.75))
    close(terms.entropy, -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))


def test_the_loss_by_hand() -> None:
    old = t(math.log(0.5), math.log(0.5))
    new = t(math.log(0.75), math.log(0.25))  # ratios 1.5 and 0.5
    terms = PolicyTerms(new, t(0.6, 0.4))
    loss = ppo_loss(
        terms,
        t(0.5, 1.0),
        Targets(old, advantages=t(1.0, -1.0), returns=t(1.0, 0.0)),
        LossWeights(clip=0.2, value_coef=0.5, entropy_coef=0.1),
    )
    # surrogate: min(1.5*1, 1.2*1) = 1.2 and min(.5*-1, .8*-1) = -.8, so policy = -mean = -.2
    close(loss.policy, -0.2)
    # value: mean(.25, 1) = .625; entropy: mean(.6, .4) = .5
    close(loss.value, 0.625)
    close(loss.entropy, 0.5)
    close(loss.total, -0.2 + 0.5 * 0.625 - 0.1 * 0.5)
    close(loss.clip_fraction, 1.0)
    close(loss.approx_kl, ((0.5 - math.log(1.5)) + (-0.5 - math.log(0.5))) / 2)


def test_the_clip_stops_the_gradient_past_the_trust_region() -> None:
    # A positive advantage and a ratio above 1 + clip: the clipped branch wins, so no gradient.
    new = torch.tensor([math.log(1.5)], requires_grad=True)
    terms = PolicyTerms(new, t(0.0))
    loss = ppo_loss(terms, t(0.0), Targets(t(0.0), t(1.0), t(0.0)), WEIGHTS)
    torch.autograd.backward(loss.total)
    assert new.grad is not None
    assert new.grad.item() == 0.0
    # Inside the region the gradient is -ratio * A.
    inside = torch.tensor([math.log(1.1)], requires_grad=True)
    terms = PolicyTerms(inside, t(0.0))
    torch.autograd.backward(ppo_loss(terms, t(0.0), Targets(t(0.0), t(1.0), t(0.0)), WEIGHTS).total)
    assert inside.grad is not None
    assert inside.grad.item() == pytest.approx(-1.1)
