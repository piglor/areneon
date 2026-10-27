"""The PPO pieces (ADR-010 §1): GAE, the candidate-scoring policy terms and the loss.

All of it is plain tensor arithmetic on values the caller gives, so each piece
can be checked by hand. The policy is a softmax over the candidates of one
decision (padding is masked out); the reward is the game score at the end of
the game only, so every step before the last has zero immediate reward and a
refused pending choice is such a step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The game score of the evaluation protocol (S15 D1): win 1, tie 1/2, loss 0.
GAME_SCORE = {"win": 1.0, "tie": 0.5, "loss": 0.0}
_STD_FLOOR = 1e-8


def gae(
    rewards: torch.Tensor, values: torch.Tensor, discount: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantages and returns of one finished trajectory.

    ``rewards`` and ``values`` are [T]; the game has ended after step T-1, so the
    value after it is 0. ``delta_t = r_t + discount * V_{t+1} - V_t`` and
    ``A_t = delta_t + discount * lambda * A_{t+1}``; the return is ``A_t + V_t``.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(steps)):
        following = values[t + 1] if t + 1 < steps else torch.zeros((), dtype=values.dtype)
        delta = rewards[t] + discount * following - values[t]
        running = delta + discount * gae_lambda * running
        advantages[t] = running
    return advantages, advantages + values


def terminal_rewards(steps: int, score: float) -> torch.Tensor:
    """Zero for every step but the last, the game score for the last."""
    rewards = torch.zeros(steps)
    rewards[-1] = score
    return rewards


def normalise(advantages: torch.Tensor) -> torch.Tensor:
    """Zero mean and unit deviation (population deviation); a constant input stays zero."""
    centred = advantages - advantages.mean()
    return centred / (centred.pow(2).mean().sqrt() + _STD_FLOOR)


@dataclass(frozen=True)
class PolicyTerms:
    """Per decision: the chosen candidate's log-probability and the policy's entropy."""

    log_prob: torch.Tensor
    entropy: torch.Tensor


def policy_terms(logits: torch.Tensor, mask: torch.Tensor, chosen: torch.Tensor) -> PolicyTerms:
    """Log-probability of ``chosen`` [B] in a softmax over the real candidates of ``logits``."""
    floor = torch.finfo(logits.dtype).min
    log_probs = torch.log_softmax(logits.masked_fill(~mask, floor), dim=-1)
    probs = log_probs.exp()
    entropy = -torch.where(mask, probs * log_probs, torch.zeros_like(probs)).sum(-1)
    return PolicyTerms(log_probs.gather(1, chosen[:, None]).squeeze(1), entropy)


def anchor_kl(
    logits: torch.Tensor, mask: torch.Tensor, anchor_log_probs: torch.Tensor
) -> torch.Tensor:
    """``KL(pi_theta || pi_anchor)`` [B] over each decision's real candidates (ADR-019 section 4).

    ``pi_theta`` is the softmax of ``logits`` over the candidates ``mask`` keeps (padding is masked
    out) and ``anchor_log_probs`` [B, C] holds the frozen anchor's log-probabilities over the same
    candidates (any finite value at padded places): ``sum_c p_c * (log p_c - log q_c)``. The
    gradient flows through ``logits`` only.
    """
    floor = torch.finfo(logits.dtype).min
    log_probs = torch.log_softmax(logits.masked_fill(~mask, floor), dim=-1)
    probs = log_probs.exp()
    return torch.where(mask, probs * (log_probs - anchor_log_probs), torch.zeros_like(probs)).sum(
        -1
    )


@dataclass(frozen=True)
class Targets:
    """What the update is measured against, per decision, from the frozen weights."""

    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class LossWeights:
    clip: float
    value_coef: float
    entropy_coef: float


@dataclass(frozen=True)
class LossParts:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


def ppo_loss(
    terms: PolicyTerms, value: torch.Tensor, targets: Targets, weights: LossWeights
) -> LossParts:
    """The clipped surrogate, the value error and the entropy bonus, averaged over the batch.

    ``total = policy + value_coef * value - entropy_coef * entropy`` where
    ``policy = -mean(min(ratio * A, clip(ratio, 1 - clip, 1 + clip) * A))`` and
    ``value = mean((V - return)^2)``.
    """
    clip = weights.clip
    log_ratio = terms.log_prob - targets.old_log_prob
    ratio = log_ratio.exp()
    advantages = targets.advantages
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1.0 - clip, 1.0 + clip) * advantages)
    policy = -surrogate.mean()
    value_loss = (value - targets.returns).pow(2).mean()
    entropy = terms.entropy.mean()
    return LossParts(
        total=policy + weights.value_coef * value_loss - weights.entropy_coef * entropy,
        policy=policy,
        value=value_loss,
        entropy=entropy,
        clip_fraction=((ratio - 1.0).abs() > clip).float().mean(),
        approx_kl=((ratio - 1.0) - log_ratio).mean(),
    )
