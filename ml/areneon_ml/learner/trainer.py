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
from typing import Any, Protocol

import torch

from areneon_ml.learner.config import MICRO, RunConfig
from areneon_ml.learner.data import LearnerDataError, Step, Trajectory
from areneon_ml.learner.diagnostics import CalibrationBin, calibration, explained_variance
from areneon_ml.learner.ppo import (
    LossWeights,
    Targets,
    anchor_kl,
    gae,
    normalise,
    policy_terms,
    ppo_loss,
    temper,
    terminal_rewards,
)
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.network import ROLES, PolicyNetwork, shared_trunk_of, tie_trunk


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
        **{
            name: value.to(device)
            for name in ("thr_idx", "thr_cat", "thr_num", "foe_idx", "foe_cat", "foe_num")
            if (value := getattr(batch, name)) is not None
        },
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
    # Set only when the config has ``generation.play`` (ADR-023 section 1): the temperature
    # (millionths) every policy term of the update is computed at. Left out of the stored report
    # otherwise.
    temperature_ppm: int | None = None
    # Value-head diagnostics (ADR-010 erratum A4 item 1): explained variance of the returns by the
    # frozen-weight values, and ten fixed bins of predicted value against the final game score.
    # Diagnostic only; always written for a role that trained, absent in older reports.
    explained_variance: float | None = None
    value_calibration: tuple[CalibrationBin, ...] | None = None
    # Set only on the Preview role under ``ppo.preview_bootstrap_micro`` (ADR-010 erratum A4 item
    # 3): the weight (millionths) of the frozen Battle value in the Preview return. Left out of
    # the stored report otherwise.
    preview_bootstrap_micro: int | None = None


@dataclass(frozen=True)
class TrainReport:
    roles: Mapping[str, RoleReport] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[str, Any]]:
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
    #: Value-head diagnostics from the frozen-weight values (see ``diagnostics``).
    explained_variance: float
    value_calibration: tuple[CalibrationBin, ...]
    #: Per step, the anchor's log-probabilities over that decision's real candidates (in
    #: candidate order), from its frozen weights; ``None`` without an active anchor.
    anchor_log_probs: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class _FrozenBattle:
    """The Battle role's pre-update scores: per step, and the first value of each Episode."""

    log_probs: torch.Tensor | None
    values: torch.Tensor | None
    first: Mapping[str, float]


VALUE_PREFIXES = ("value.", "value_net.")  # the head, or the separate value network


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
        # shared_trunk (ADR-010 erratum A4 item 5): the Preview network uses the Battle network's
        # trunk modules, the very same parameters; the rest of each network stays its own. (A
        # Battle-only warm start gives the Battle network's cloned trunk to both roles; a
        # two-role warm start is refused with the option by the config reader.)
        self.shared_trunk = shared_trunk_of(config.model)
        if self.shared_trunk:
            tie_trunk(networks["battle"], networks["preview"])
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

        def adam(parameters: Sequence[torch.nn.Parameter]) -> torch.optim.Adam:
            return torch.optim.Adam(
                parameters,
                lr=ppo.learning_rate,
                betas=(ppo.adam_beta1, ppo.adam_beta2),
                eps=ppo.adam_eps,
            )

        # Without shared_trunk: one Adam per role, as ever. With it: ONE Adam over the union of the
        # two networks' parameters (each shared parameter once), kept under both role keys.
        self.entries: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
            role: list(self.networks[role].named_parameters()) for role in ROLES
        }
        if self.shared_trunk:
            seen = {id(p) for _, p in self.entries["battle"]}
            self.entries["preview"] = [
                (n, p) for n, p in self.entries["preview"] if id(p) not in seen
            ]
            union = adam([p for role in ROLES for _, p in self.entries[role]])
            self.optimisers = {role: union for role in ROLES}
        else:
            self.optimisers = {role: adam(list(self.networks[role].parameters())) for role in ROLES}
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

    def optimiser_groups(
        self,
    ) -> list[tuple[torch.optim.Optimizer, list[tuple[str, str, torch.nn.Parameter]]]]:
        """Each distinct optimiser with its parameters in its own order, as (role, name, parameter).

        Two groups (one per role) normally; one under ``shared_trunk``, where the Preview entries
        are the Preview network's own parameters only (the trunk is stored under the Battle role).
        """
        flat = [(role, n, p) for role in ROLES for n, p in self.entries[role]]
        if self.shared_trunk:
            return [(self.optimisers["battle"], flat)]
        return [(self.optimisers[role], [e for e in flat if e[0] == role]) for role in ROLES]

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
        # Under ``ppo.preview_bootstrap_micro`` (> 0) both roles' targets are fixed from the
        # frozen (pre-update) weights BEFORE either role is updated: the Battle network is trained
        # first, so its values are read here, once, and the Preview return is built from them.
        frozen = self._frozen_battle(trajectories)
        # Every role's targets are read from the frozen (pre-update) weights BEFORE any role is
        # updated. Without shared_trunk the roles share nothing, so this equals reading each just
        # before its own epochs; with it, the Battle update moves the trunk the Preview scores with.
        tables = {}
        for role in ROLES:
            mine = [t for t in trajectories if t.role == role]
            if mine:
                warm = warmup and role in self.warm_roles
                tables[role] = (
                    self._table(role, mine, role in self.anchors and not warm, frozen),
                    warm,
                )
        if self.shared_trunk:
            roles.update(self._train_joint(tables, warmup))
        else:
            for role, (table, warm) in tables.items():
                roles[role] = self._train_role(role, table, warm)
        self.generation += 1
        return TrainReport(roles)

    def _score(self, role: str, steps: Sequence[Step]) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of the chosen candidate and the value of each step, no gradient."""
        network = self.networks[role]
        size = self.config.ppo.minibatch_size
        tau = self.config.generation.temperature_ppm
        log_probs, values = [], []
        network.eval()
        with torch.no_grad():
            for start in range(0, len(steps), size):
                chunk = steps[start : start + size]
                batch = to_device(collate([s.query for s in chunk]), self.device)
                out = network(batch)
                chosen = torch.tensor([s.chosen for s in chunk], device=self.device)
                log_probs.append(policy_terms(out.logits, batch.cand_mask, chosen, tau).log_prob)
                values.append(out.value)
        return torch.cat(log_probs), torch.cat(values)

    def _frozen_battle(self, trajectories: Sequence[Trajectory]) -> _FrozenBattle | None:
        """The Battle network's frozen log-probabilities and values, and its first value per
        Episode, or ``None`` unless ``ppo.preview_bootstrap_micro`` is above 0.

        Every Preview trajectory must have a Battle trajectory of the same Episode (its first
        decision is the position the Preview step led to); otherwise it is refused.
        """
        weight = self.config.ppo.preview_bootstrap_micro
        if not weight:
            return None
        battle = [t for t in trajectories if t.role == "battle"]
        steps = [s for t in battle for s in t.steps]
        log_probs, values = self._score("battle", steps) if steps else (None, None)
        first: dict[str, float] = {}
        start = 0
        for t in battle:
            assert values is not None  # a Battle trajectory holds steps
            first[t.episode] = float(values[start])
            start += len(t.steps)
        for t in trajectories:
            if t.role == "preview" and t.episode not in first:
                raise LearnerDataError(
                    f"Episode {t.episode!r} has a Preview step and no Battle decision: "
                    "ppo.preview_bootstrap_micro needs the Battle value of its first position"
                )
        return _FrozenBattle(log_probs, values, first)

    def _anchor_scores(self, role: str, steps: Sequence[Step]) -> list[torch.Tensor]:
        """The anchor's log-probabilities over each step's real candidates, no gradient."""
        network = self.anchors[role]
        size = self.config.ppo.minibatch_size
        tau = self.config.generation.temperature_ppm
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(steps), size):
                batch = to_device(
                    collate([s.query for s in steps[start : start + size]]), self.device
                )
                floor = torch.finfo(torch.float32).min
                logits = temper(network(batch).logits, tau)
                log_probs = torch.log_softmax(logits.masked_fill(~batch.cand_mask, floor), dim=-1)
                out.extend(log_probs[i][batch.cand_mask[i]] for i in range(log_probs.shape[0]))
        return out

    def _table(
        self,
        role: str,
        trajectories: Sequence[Trajectory],
        anchor: bool = False,
        frozen: _FrozenBattle | None = None,
    ) -> _Table:
        ppo = self.config.ppo
        steps = [s for t in trajectories for s in t.steps]
        if role == "battle" and frozen is not None and frozen.values is not None:
            # The same frozen pass the Preview returns were read from (no second forward pass).
            assert frozen.log_probs is not None
            old_log_prob, values = frozen.log_probs, frozen.values
        else:
            old_log_prob, values = self._score(role, steps)
        values = values.cpu()
        # Preview return under ppo.preview_bootstrap_micro: the terminal reward is
        # (1 - w) * score + w * V_battle(first Battle decision of the Episode), the Battle value
        # frozen before any update. The calibration below still compares values to the score.
        weight = (self.config.ppo.preview_bootstrap_micro or 0) / MICRO
        bootstrap = role == "preview" and frozen is not None
        advantages, returns, start = [], [], 0
        scores: list[float] = []
        for t in trajectories:
            scores.extend([t.score] * len(t.steps))
            end = start + len(t.steps)
            target = (
                (1 - weight) * t.score + weight * frozen.first[t.episode]
                if bootstrap and frozen is not None
                else t.score
            )
            a, r = gae(
                terminal_rewards(len(t.steps), target),
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
        all_returns = torch.cat(returns)
        # float32 tensors widened exactly to float64 for the diagnostics.
        value_list = values.double().tolist()
        return _Table(
            steps,
            old_log_prob,
            adv.to(self.device),
            all_returns.to(self.device),
            explained_variance(all_returns.double().tolist(), value_list),
            calibration(value_list, scores),
            self._anchor_scores(role, steps) if anchor else None,
        )

    def _train_role(self, role: str, table: _Table, warm: bool = False) -> RoleReport:
        """PPO epochs for one role; with ``warm`` only the value head learns (see below).

        ``table`` holds the role's targets, read before any update (the anchor term is inactive
        while the policy is frozen, a value warm-up update)."""
        network = self.networks[role]
        network.train()
        # Value warm-up: every parameter outside the value head is frozen, so the policy outputs
        # stay byte-identical, no gradient reaches them and Adam (which skips a parameter with no
        # gradient) leaves their moments absent. The loss is the value error alone (the PPO terms
        # carry no gradient to the value head and the coefficient only rescales it).
        frozen = [
            p for n, p in network.named_parameters() if warm and not n.startswith(VALUE_PREFIXES)
        ]
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            return self._epochs(role, table, warm)
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)

    def _weights(self) -> LossWeights:
        ppo = self.config.ppo
        entropy_micro = self.last_entropy_coef_micro
        entropy_coef = ppo.entropy_coef if entropy_micro is None else entropy_micro / MICRO
        return LossWeights(ppo.clip, ppo.value_coef, entropy_coef)

    def _minibatch_loss(
        self, role: str, table: _Table, warm: bool, index: list[int], weights: LossWeights
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        """One minibatch of one role: the tensor to backpropagate, the loss terms and, with an
        anchor, the (detached-by-the-caller) mean KL to the anchor."""
        tau = self.config.generation.temperature_ppm
        beta_micro = self.last_anchor_beta_micro if role in self.anchors else None
        beta = 0.0 if beta_micro is None else beta_micro / MICRO
        batch = to_device(collate([table.steps[i].query for i in index]), self.device)
        out = self.networks[role](batch)
        chosen = torch.tensor([table.steps[i].chosen for i in index], device=self.device)
        # `policy_terms` masks padding again although the network already pushes padded
        # logits to the dtype minimum: defence in depth, so a loss never depends on it.
        loss = ppo_loss(
            policy_terms(out.logits, batch.cand_mask, chosen, tau),
            out.value,
            Targets(table.old_log_prob[index], table.advantages[index], table.returns[index]),
            weights,
        )
        total = loss.total
        kl = None
        if table.anchor_log_probs is not None:
            dense = torch.zeros(batch.cand_mask.shape, device=self.device)
            for row, i in enumerate(index):
                dense[row, batch.cand_mask[row]] = table.anchor_log_probs[i]
            kl = anchor_kl(out.logits, batch.cand_mask, dense, tau).mean()
            total = total + beta * kl
        return (loss.value if warm else total), loss, kl

    @staticmethod
    def _tally(sums: torch.Tensor, loss: Any) -> None:
        sums += torch.stack(
            [
                loss.policy.detach(),
                loss.value.detach(),
                loss.entropy.detach(),
                loss.clip_fraction,
                loss.approx_kl.detach(),
            ]
        ).cpu()

    def _report(
        self,
        role: str,
        table: _Table,
        warm: bool,
        tally: tuple[torch.Tensor, torch.Tensor, int],
        epochs: tuple[int, bool],
    ) -> RoleReport:
        sums, kl_sum, optimiser_steps = tally
        epochs_run, kl_stopped = epochs
        policy, value, entropy, clip_fraction, approx_kl = (sums / optimiser_steps).tolist()
        stop = self.config.ppo.target_kl is not None
        beta_micro = self.last_anchor_beta_micro if role in self.anchors else None
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
            self.last_entropy_coef_micro,
            self.config.generation.temperature_ppm,
            table.explained_variance,
            table.value_calibration,
            self.config.ppo.preview_bootstrap_micro if role == "preview" else None,
        )

    def _epochs(self, role: str, table: _Table, warm: bool) -> RoleReport:
        ppo = self.config.ppo
        network, optimiser = self.networks[role], self.optimisers[role]
        weights = self._weights()
        sums = torch.zeros(5)
        kl_sum = torch.zeros(())
        optimiser_steps = 0
        epochs_run = 0
        kl_stopped = False
        for _ in range(ppo.epochs):
            epoch_kl = 0.0
            minibatches = 0
            order = torch.randperm(len(table.steps), generator=self.generator).tolist()
            for start in range(0, len(order), ppo.minibatch_size):
                index = order[start : start + ppo.minibatch_size]
                backward, loss, kl = self._minibatch_loss(role, table, warm, index, weights)
                if kl is not None:
                    kl_sum += kl.detach().cpu()
                optimiser.zero_grad(set_to_none=True)
                torch.autograd.backward(backward)
                torch.nn.utils.clip_grad_norm_(network.parameters(), ppo.max_grad_norm)
                optimiser.step()
                optimiser_steps += 1
                minibatches += 1
                epoch_kl += float(loss.approx_kl.detach().cpu())
                self._tally(sums, loss)
            epochs_run += 1
            # ADR-010 erratum A1: one check per complete epoch, on the mean of the epoch's
            # per-minibatch approximate KL in minibatch order; never inside an epoch.
            if not warm and ppo.target_kl is not None and epoch_kl / minibatches > ppo.target_kl:
                kl_stopped = True
                break
        return self._report(
            role, table, warm, (sums, kl_sum, optimiser_steps), (epochs_run, kl_stopped)
        )

    # -- the joint step of shared_trunk -----------------------------------------------------------

    @staticmethod
    def _spread(order: list[int], parts: int) -> list[list[int]]:
        """``order`` cut into ``parts`` consecutive chunks as evenly as possible: the first
        ``len % parts`` chunks hold one more step, and with fewer steps than ``parts`` only the
        first ones are non-empty."""
        base, extra = divmod(len(order), parts)
        out, position = [], 0
        for k in range(parts):
            size = base + (1 if k < extra else 0)
            out.append(order[position : position + size])
            position += size
        return out

    def _train_joint(
        self, tables: Mapping[str, tuple[_Table, bool]], warmup: bool
    ) -> dict[str, RoleReport]:
        """``shared_trunk``: every optimiser step backpropagates the SUM of the roles' minibatch
        losses (each a mean over its own decisions), clips over the union and steps the one Adam.

        Per epoch each role's steps are shuffled with the learner generator (Battle's draw, then
        Preview's) and cut into the same number M of minibatches: the leading role (Battle, else
        Preview) by ``ppo.minibatch_size``, the other spread as evenly as possible over the M (a
        role with fewer steps than M contributes to the first minibatches only, and a missing part
        contributes 0). During the value warm-up the shared trunk is frozen for both roles, and
        the warm role also freezes everything outside its value head. The ``target_kl`` check is
        per complete joint epoch and stops all roles when the epoch-mean ``approx_kl`` of any
        non-warm role exceeds it."""
        ppo = self.config.ppo
        networks = {role: self.networks[role] for role in tables}
        for network in networks.values():
            network.train()
        trunk = set(self.networks["battle"].trunk_keys())
        frozen = [
            p
            for role, (_, warm) in tables.items()
            for n, p in networks[role].named_parameters()
            if (warm and not n.startswith(VALUE_PREFIXES)) or (warmup and n in trunk)
        ]
        for parameter in frozen:
            parameter.requires_grad_(False)
        weights = self._weights()
        optimiser = self.optimisers["battle"]
        union = [p for r in ROLES for _, p in self.entries[r]]
        leader = next(iter(tables))
        sums = {role: torch.zeros(5) for role in tables}
        kl_sums = {role: torch.zeros(()) for role in tables}
        counts = dict.fromkeys(tables, 0)
        epochs_run, kl_stopped = 0, False
        try:
            for _ in range(ppo.epochs):
                orders = {
                    role: torch.randperm(len(table.steps), generator=self.generator).tolist()
                    for role, (table, _) in tables.items()
                }
                lead = orders[leader]
                size = ppo.minibatch_size
                chunks = {leader: [lead[s : s + size] for s in range(0, len(lead), size)]}
                for role in tables:
                    if role != leader:
                        chunks[role] = self._spread(orders[role], len(chunks[leader]))
                epoch_kl = dict.fromkeys(tables, 0.0)
                epoch_n = dict.fromkeys(tables, 0)
                for k in range(len(chunks[leader])):
                    parts = []
                    for role, (table, warm) in tables.items():
                        index = chunks[role][k]
                        if not index:
                            continue
                        backward, loss, kl = self._minibatch_loss(role, table, warm, index, weights)
                        if kl is not None:
                            kl_sums[role] += kl.detach().cpu()
                        parts.append(backward)
                        counts[role] += 1
                        epoch_n[role] += 1
                        epoch_kl[role] += float(loss.approx_kl.detach().cpu())
                        self._tally(sums[role], loss)
                    optimiser.zero_grad(set_to_none=True)
                    torch.autograd.backward(sum(parts[1:], parts[0]))
                    torch.nn.utils.clip_grad_norm_(union, ppo.max_grad_norm)
                    optimiser.step()
                epochs_run += 1
                if ppo.target_kl is not None and any(
                    epoch_kl[role] / epoch_n[role] > ppo.target_kl
                    for role, (_, warm) in tables.items()
                    if not warm
                ):
                    kl_stopped = True
                    break
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        return {
            role: self._report(
                role,
                table,
                warm,
                (sums[role], kl_sums[role], counts[role]),
                (epochs_run, kl_stopped),
            )
            for role, (table, warm) in tables.items()
        }
