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

from areneon_ml.learner.config import MICRO, RunConfig
from areneon_ml.learner.data import Step, Trajectory
from areneon_ml.learner.ppo import (
    LossWeights,
    Targets,
    anchor_kl,
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
        cand_num=None if batch.cand_num is None else batch.cand_num.to(device),
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
    # True only for a value warm-up update (``ppo.value_warmup``, ADR-019 section 4); left out of
    # the stored report otherwise, so a run without the field keeps its bytes.
    value_warmup: bool | None = None
    # Set only when the config has ``ppo.anchor`` (ADR-019 section 4): the mean KL(pi_theta ||
    # pi_anchor) over the role's minibatches and the beta (millionths) the update used. Both are
    # 0 on a value warm-up update (the policy is frozen and equals the anchor exactly). Left out
    # of the stored report when absent, so a run without the field keeps its bytes.
    anchor_kl: float | None = None
    anchor_beta_micro: int | None = None
    # Set only under ``ppo.entropy_schedule`` of kind ``linear`` (ADR-023 section 3): the entropy
    # weight (millionths) the update used. Left out of the stored report otherwise.
    entropy_coef_micro: int | None = None


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
    #: Per step, the anchor's log-probabilities over that decision's real candidates (in
    #: candidate order), from its frozen weights; ``None`` without an active anchor.
    anchor_log_probs: list[torch.Tensor] | None = None


VALUE_PREFIX = "value."


def value_warmup_active(config: RunConfig, generation: int) -> bool:
    """Whether the update that follows ``generation`` completed updates is a value warm-up one."""
    warmup = config.ppo.value_warmup
    return warmup is not None and generation < warmup


class Learner:
    """Both networks, their optimisers and the generator of one run."""

    def __init__(
        self,
        config: RunConfig,
        networks: Mapping[str, PolicyNetwork],
        device: torch.device,
        anchors: Mapping[str, PolicyNetwork] | None = None,
    ) -> None:
        if set(networks) != set(ROLES):
            raise LearnerError(f"one network per role is needed: {sorted(ROLES)}")
        if (anchors is None) != (config.ppo.anchor is None):
            raise LearnerError(
                "the anchor networks are given exactly when the config has ppo.anchor"
            )
        # Value warm-up and the anchor act only on the warm-started roles (#1165): the others
        # train as PPO from the first update.
        self.warm_roles = frozenset(config.warm_start or ())
        if anchors is not None and set(anchors) != self.warm_roles:
            raise LearnerError(
                f"one anchor network per warm-started role is needed: {sorted(self.warm_roles)}"
            )
        self.config = config
        self.device = device
        self.networks = {role: networks[role].to(device) for role in ROLES}
        # The frozen warm-start pair (ppo.anchor): evaluation mode, no gradient, never trained and
        # not part of the learner state (the loop loads it from the run's checkpoint store).
        self.anchors: dict[str, PolicyNetwork] = {}
        if anchors is not None:
            for role in sorted(anchors):
                anchor = anchors[role].to(device).eval()
                anchor.requires_grad_(False)
                self.anchors[role] = anchor
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
        # Beta (millionths) of the newest update; set only with ``ppo.anchor`` (stored in the
        # sidecar then, and nowhere otherwise).
        self.last_anchor_beta_micro: int | None = None
        # Entropy weight (millionths) of the newest update; set only under a linear
        # ``ppo.entropy_schedule`` (stored in the sidecar then, and nowhere otherwise).
        self.last_entropy_coef_micro: int | None = None

    def entropy_coef_micro(self, update: int) -> int | None:
        """Entropy weight of learner update ``update`` (from 0), or ``None`` without a linear
        ``ppo.entropy_schedule``.

        The schedule's index counts POLICY updates, as the anchor's does: with a value warm-up of N
        updates, updates 0 to N-1 report the starting weight (the policy is frozen and the entropy
        term is not in the value-only loss) and update ``u`` >= N uses the schedule at ``u - N``.
        The index is one value for the update, both roles alike (a role that a battle-only warm
        start leaves unwarmed trains plain PPO during the warm-up and uses the starting weight).
        """
        schedule = self.config.ppo.entropy_schedule
        if schedule is None:
            return None
        warmup = self.config.ppo.value_warmup or 0
        return schedule.rate_micro(max(0, update - warmup))

    def anchor_beta_micro(self, update: int) -> int | None:
        """Beta of learner update ``update`` (from 0), or ``None`` without ``ppo.anchor``.

        The anchor's schedule index counts from the first POLICY update: with a value warm-up of N
        updates, updates 0 to N-1 have beta 0 (the policy is frozen) and update ``u`` >= N uses the
        schedule's rate at ``u - N``, so the decay is not spent while nothing moves. Without a
        warm-up the index is the update index, as for the learning-rate schedule.
        """
        anchor = self.config.ppo.anchor
        if anchor is None:
            return None
        warmup = self.config.ppo.value_warmup or 0
        return 0 if update < warmup else anchor.beta_micro(update - warmup)

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
        warmup = value_warmup_active(self.config, self.generation)
        self.last_anchor_beta_micro = self.anchor_beta_micro(self.generation)
        self.last_entropy_coef_micro = self.entropy_coef_micro(self.generation)
        for role in ROLES:
            mine = [t for t in trajectories if t.role == role]
            if mine:
                roles[role] = self._train_role(role, mine, warmup and role in self.warm_roles)
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

    def _anchor_scores(self, role: str, steps: Sequence[Step]) -> list[torch.Tensor]:
        """The anchor's log-probabilities over each step's real candidates, no gradient."""
        network = self.anchors[role]
        size = self.config.ppo.minibatch_size
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(steps), size):
                batch = to_device(
                    collate([s.query for s in steps[start : start + size]]), self.device
                )
                floor = torch.finfo(torch.float32).min
                logits = network(batch).logits
                log_probs = torch.log_softmax(logits.masked_fill(~batch.cand_mask, floor), dim=-1)
                out.extend(log_probs[i][batch.cand_mask[i]] for i in range(log_probs.shape[0]))
        return out

    def _table(self, role: str, trajectories: Sequence[Trajectory], anchor: bool = False) -> _Table:
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
        return _Table(
            steps,
            old_log_prob,
            adv.to(self.device),
            torch.cat(returns).to(self.device),
            self._anchor_scores(role, steps) if anchor else None,
        )

    def _train_role(
        self, role: str, trajectories: Sequence[Trajectory], warm: bool = False
    ) -> RoleReport:
        """PPO epochs for one role; with ``warm`` only the value head learns (see below)."""
        # The anchor term is inactive while the policy is frozen (a value warm-up update).
        table = self._table(role, trajectories, anchor=role in self.anchors and not warm)
        network = self.networks[role]
        network.train()
        # Value warm-up: every parameter outside the value head is frozen, so the policy outputs
        # stay byte-identical, no gradient reaches them and Adam (which skips a parameter with no
        # gradient) leaves their moments absent. The loss is the value error alone (the PPO terms
        # carry no gradient to the value head and the coefficient only rescales it).
        frozen = [
            p for n, p in network.named_parameters() if warm and not n.startswith(VALUE_PREFIX)
        ]
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            return self._epochs(role, table, warm)
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)

    def _epochs(self, role: str, table: _Table, warm: bool) -> RoleReport:
        ppo = self.config.ppo
        network, optimiser = self.networks[role], self.optimisers[role]
        entropy_micro = self.last_entropy_coef_micro
        entropy_coef = ppo.entropy_coef if entropy_micro is None else entropy_micro / MICRO
        weights = LossWeights(ppo.clip, ppo.value_coef, entropy_coef)
        sums = torch.zeros(5)
        kl_sum = torch.zeros(())
        beta_micro = self.last_anchor_beta_micro if role in self.anchors else None
        beta = 0.0 if beta_micro is None else beta_micro / MICRO
        optimiser_steps = 0
        epochs_run = 0
        kl_stopped = False
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
                total = loss.total
                if table.anchor_log_probs is not None:
                    dense = torch.zeros(batch.cand_mask.shape, device=self.device)
                    for row, i in enumerate(index):
                        dense[row, batch.cand_mask[row]] = table.anchor_log_probs[i]
                    kl = anchor_kl(out.logits, batch.cand_mask, dense).mean()
                    total = total + beta * kl
                    kl_sum += kl.detach().cpu()
                optimiser.zero_grad(set_to_none=True)
                torch.autograd.backward(loss.value if warm else total)
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
            if not warm and ppo.target_kl is not None and epoch_kl / minibatches > ppo.target_kl:
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
            True if warm else None,
            (float(kl_sum) / optimiser_steps if table.anchor_log_probs is not None else 0.0)
            if beta_micro is not None
            else None,
            beta_micro,
            entropy_micro,
        )
