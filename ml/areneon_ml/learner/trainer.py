"""The PPO learner (ADR-010 §1): one generation's trajectories in, updated networks out.

For each role the learner (1) recomputes the behaviour policy's log-probability
and the value of every recorded decision from the frozen weights, with no
gradient (no behaviour probability is stored in an Episode), (2) turns the
game score into per-step advantages and returns with GAE, and (3) runs the
configured PPO epochs over shuffled minibatches. The shuffling uses the
learner's own generator, whose state is part of the learner state.

Determinism: on CPU, with the thread count of the config and deterministic
algorithms, the same inputs on the same machine and software give the same
weights. Nothing is claimed across machines, torch builds or devices, and a
CUDA run is not bit-reproducible.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import Step, Trajectory
from areneon_ml.learner.ppo import (
    LossWeights,
    Targets,
    gae,
    normalise,
    policy_terms,
    ppo_loss,
    terminal_rewards,
)
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.network import ROLES, PolicyNetwork


class LearnerError(ValueError):
    """The learner cannot be built or run as asked."""


class Placement(Protocol):
    """What ``configure_learner`` reads of a TrainingRunConfig or a CloneRunConfig."""

    @property
    def device(self) -> str: ...

    @property
    def threads(self) -> int: ...


def configure_learner(config: Placement) -> torch.device:
    """Set this process up for the learner and return its device; call once at start-up."""
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise LearnerError("the config asks for cuda, but this torch has no usable CUDA device")
        # cuBLAS is deterministic only with a fixed workspace; set before the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=config.device == "cuda")
    torch.set_num_threads(config.threads)
    return torch.device(config.device)


def to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        kind=batch.kind,
        token_mask=batch.token_mask.to(device),
        own_mask=batch.own_mask.to(device),
        mon_cat=batch.mon_cat.to(device),
        mon_num=batch.mon_num.to(device),
        field_cat=batch.field_cat.to(device),
        field_num=batch.field_num.to(device),
        cand_mask=batch.cand_mask.to(device),
        cand_cat=batch.cand_cat.to(device),
        cand_tok=batch.cand_tok.to(device),
        team_tok=batch.team_tok.to(device),
        pending=batch.pending.to(device),
    )


@dataclass(frozen=True)
class RoleReport:
    """Mean loss terms over the role's minibatches, and how many optimiser steps ran."""

    steps: int
    optimiser_steps: int
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    approx_kl: float
    # Set only when the config has ``ppo.target_kl_micro`` (ADR-010 erratum A1): the epochs that
    # ran, and whether an epoch's mean approximate KL exceeded the target (the stop fired). Both
    # are left out of the stored report when absent, so a run without the field keeps its bytes.
    epochs_run: int | None = None
    kl_stopped: bool | None = None


@dataclass(frozen=True)
class TrainReport:
    roles: Mapping[str, RoleReport] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[str, float | int | bool]]:
        return {
            role: {k: v for k, v in dataclasses.asdict(report).items() if v is not None}
            for role, report in sorted(self.roles.items())
        }


@dataclass(frozen=True)
class _Table:
    """The decisions of one role, flattened, with their frozen-weight values."""

    steps: list[Step]
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class Learner:
    """Both networks, their optimisers and the generator of one run."""

    def __init__(
        self, config: RunConfig, networks: Mapping[str, PolicyNetwork], device: torch.device
    ) -> None:
        if set(networks) != set(ROLES):
            raise LearnerError(f"one network per role is needed: {sorted(ROLES)}")
        self.config = config
        self.device = device
        self.networks = {role: networks[role].to(device) for role in ROLES}
        ppo = config.ppo
        self.optimisers = {
            role: torch.optim.Adam(
                self.networks[role].parameters(),
                lr=ppo.learning_rate,
                betas=(ppo.adam_beta1, ppo.adam_beta2),
                eps=ppo.adam_eps,
            )
            for role in ROLES
        }
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.learner_seed)
        self.generation = 0
        # Millionths used by the newest update; set only under a linear schedule (it is stored in
        # the learner-state sidecar then, and nowhere otherwise).
        self.last_rate_micro: int | None = None

    # -- one generation ------------------------------------------------------

    def train(self, trajectories: Sequence[Trajectory]) -> TrainReport:
        """Run the configured PPO epochs on the generation's trajectories, then count it."""
        roles = {}
        schedule = self.config.ppo.schedule
        if schedule is not None:
            # ADR-010 erratum A1: the rate is a function of the config and the update index (the
            # generation counter, restored on a resume), set on both optimisers before any epoch.
            self.last_rate_micro = schedule.rate_micro(self.generation)
            for optimiser in self.optimisers.values():
                for group in optimiser.param_groups:
                    group["lr"] = schedule.rate(self.generation)
        for role in ROLES:
            mine = [t for t in trajectories if t.role == role]
            if mine:
                roles[role] = self._train_role(role, mine)
        self.generation += 1
        return TrainReport(roles)

    def _score(self, role: str, steps: Sequence[Step]) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of the chosen candidate and the value of each step, no gradient."""
        network = self.networks[role]
        size = self.config.ppo.minibatch_size
        log_probs, values = [], []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(steps), size):
                chunk = steps[start : start + size]
                batch = to_device(collate([s.query for s in chunk]), self.device)
                out = network(batch)
                chosen = torch.tensor([s.chosen for s in chunk], device=self.device)
                log_probs.append(policy_terms(out.logits, batch.cand_mask, chosen).log_prob)
                values.append(out.value)
        return torch.cat(log_probs), torch.cat(values)

    def _table(self, role: str, trajectories: Sequence[Trajectory]) -> _Table:
        ppo = self.config.ppo
        steps = [s for t in trajectories for s in t.steps]
        old_log_prob, values = self._score(role, steps)
        values = values.cpu()
        advantages, returns, start = [], [], 0
        for t in trajectories:
            end = start + len(t.steps)
            a, r = gae(
                terminal_rewards(len(t.steps), t.score),
                values[start:end],
                ppo.discount,
                ppo.gae_lambda,
            )
            advantages.append(a)
            returns.append(r)
            start = end
        adv = torch.cat(advantages)
        if ppo.normalize_advantages:
            adv = normalise(adv)
        return _Table(steps, old_log_prob, adv.to(self.device), torch.cat(returns).to(self.device))

    def _train_role(self, role: str, trajectories: Sequence[Trajectory]) -> RoleReport:
        ppo = self.config.ppo
        table = self._table(role, trajectories)
        network, optimiser = self.networks[role], self.optimisers[role]
        weights = LossWeights(ppo.clip, ppo.value_coef, ppo.entropy_coef)
        sums = torch.zeros(5)
        optimiser_steps = 0
        epochs_run = 0
        kl_stopped = False
        network.train()
        for _ in range(ppo.epochs):
            epoch_kl = 0.0
            minibatches = 0
            order = torch.randperm(len(table.steps), generator=self.generator).tolist()
            for start in range(0, len(order), ppo.minibatch_size):
                index = order[start : start + ppo.minibatch_size]
                batch = to_device(collate([table.steps[i].query for i in index]), self.device)
                out = network(batch)
                chosen = torch.tensor([table.steps[i].chosen for i in index], device=self.device)
                # `policy_terms` masks padding again although the network already pushes padded
                # logits to the dtype minimum: defence in depth, so a loss never depends on it.
                loss = ppo_loss(
                    policy_terms(out.logits, batch.cand_mask, chosen),
                    out.value,
                    Targets(
                        table.old_log_prob[index], table.advantages[index], table.returns[index]
                    ),
                    weights,
                )
                optimiser.zero_grad(set_to_none=True)
                torch.autograd.backward(loss.total)
                torch.nn.utils.clip_grad_norm_(network.parameters(), ppo.max_grad_norm)
                optimiser.step()
                optimiser_steps += 1
                minibatches += 1
                epoch_kl += float(loss.approx_kl.detach().cpu())
                sums += torch.stack(
                    [
                        loss.policy.detach(),
                        loss.value.detach(),
                        loss.entropy.detach(),
                        loss.clip_fraction,
                        loss.approx_kl.detach(),
                    ]
                ).cpu()
            epochs_run += 1
            # ADR-010 erratum A1: one check per complete epoch, on the mean of the epoch's
            # per-minibatch approximate KL in minibatch order; never inside an epoch.
            if ppo.target_kl is not None and epoch_kl / minibatches > ppo.target_kl:
                kl_stopped = True
                break
        policy, value, entropy, clip_fraction, approx_kl = (sums / optimiser_steps).tolist()
        stop = ppo.target_kl is not None
        return RoleReport(
            len(table.steps),
            optimiser_steps,
            policy,
            value,
            entropy,
            clip_fraction,
            approx_kl,
            epochs_run if stop else None,
            kl_stopped if stop else None,
        )
