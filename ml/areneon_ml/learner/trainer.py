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

from areneon_ml.learner.aux_moves import aux_terms, targets_tensors
from areneon_ml.learner.aux_opponent import HEADS, opponent_terms
from areneon_ml.learner.aux_opponent import targets_tensors as opponent_targets
from areneon_ml.learner.config import MICRO, RunConfig
from areneon_ml.learner.data import LearnerDataError, Step, Trajectory
from areneon_ml.learner.diagnostics import CalibrationBin, calibration, explained_variance
from areneon_ml.learner.imitation import DrawStream, TeacherSource
from areneon_ml.learner.labels import LabelStep
from areneon_ml.learner.ppo import (
    LossParts,
    LossWeights,
    Targets,
    anchor_kl,
    distill_kl,
    gae,
    normalise,
    policy_terms,
    ppo_loss,
    temper,
    terminal_rewards,
    uniform_floor_counts,
    uniform_kl,
)
from areneon_ml.learner.shaping import phi, shaped
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.network import ROLES, PolicyNetwork, PolicyOutput, shared_trunk_of, tie_trunk


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
    # Set only on the Battle role under ``ppo.search_distill`` (ADR-026 erratum E2): the labels the
    # update used, and the mean KL(pi' || pi_theta) over them before the update's epochs and after.
    # All three are left out of the stored report otherwise, so a run without the block keeps its
    # bytes. 0 labels (before the first label stream, or during the value warm-up) reports 0.0.
    search_labels: int | None = None
    search_kl_before: float | None = None
    search_kl_after: float | None = None
    # Set only on the Battle role under ``ppo.imitation`` (ADR-010 erratum A11.3): the teacher
    # steps the update drew, lambda_im (millionths) it used, the mean cross-entropy over them at
    # the frozen weights and after the epochs, and the agreement (ppm) with the teacher's choice
    # after the epochs. All five are left out of the stored report otherwise; a value warm-up
    # update (the term is off) reports 0 steps, coefficient 0 and 0.0 / 0.
    imitation_steps: int | None = None
    imitation_coef_micro: int | None = None
    imitation_ce_before: float | None = None
    imitation_ce_after: float | None = None
    imitation_top1_ppm: int | None = None
    # Set only on the Battle role under ``ppo.shaping`` (ADR-010 erratum A11.1): the potential's
    # weight (millionths) and the mean |gamma * Phi(o_{t+1}) - Phi(o_t)| over the role's steps
    # (millionths, truncated). Left out of the stored report otherwise.
    shaping_coef_micro: int | None = None
    shaping_mean_abs_micro: int | None = None
    # Set only on the Battle role under ``ppo.anchor.threatened`` (ADR-010 erratum A11.2): the
    # Threatened steps of the update's table, the mean KL(pi_theta || pi_anchor) over the
    # Threatened decisions of its minibatches (measured before each step, as ``anchor_kl``; 0.0
    # on a value warm-up update) and the Threatened beta (millionths) the update used. Left out
    # of the stored report otherwise.
    anchor_threatened_steps: int | None = None
    anchor_kl_threatened: float | None = None
    anchor_threatened_beta_micro: int | None = None
    # Set only on the Battle role under ``ppo.aux_moves`` (ADR-010 erratum A12.2): the unmasked
    # move-reveal targets of the update's table, the mean cross-entropy over the targets of its
    # minibatches (measured before each step, 0.0 with none) and the share whose highest-scored
    # unmasked move is in the label. Left out of the stored report otherwise.
    aux_moves_steps: int | None = None
    aux_moves_loss: float | None = None
    aux_moves_top1: float | None = None
    aux_moves_masked: int | None = None
    # Set only on the Battle role under ``ppo.uniform_kl`` (ADR-010 erratum A14.1): the mean
    # KL(u || pi_theta) over the minibatch decisions that have a term (measured before each step,
    # 0.0 with none), lambda (millionths) the update used, the table's decisions with a term and
    # the share (ppm) of their support candidates below 1/(100 n) at the frozen weights. A value
    # warm-up update (the term is off) reports 0.0, coefficient 0, 0 steps and 0. Left out of the
    # stored report otherwise.
    uniform_kl: float | None = None
    uniform_kl_coef_micro: int | None = None
    uniform_kl_steps: int | None = None
    uniform_floor_ppm: int | None = None
    # Set only on the Battle role under ``ppo.aux_opponent`` (ADR-010 erratum A14.2), for each head
    # with a coefficient: the unmasked targets of the update's table, the mean cross-entropy over
    # the targets of its minibatches (measured before each step, 0.0 with none), the share whose
    # highest-scored unmasked index is the label, and the units without a label. Left out of the
    # stored report otherwise.
    aux_action_steps: int | None = None
    aux_action_loss: float | None = None
    aux_action_top1: float | None = None
    aux_action_masked: int | None = None
    aux_item_steps: int | None = None
    aux_item_loss: float | None = None
    aux_item_top1: float | None = None
    aux_item_masked: int | None = None
    aux_ability_steps: int | None = None
    aux_ability_loss: float | None = None
    aux_ability_top1: float | None = None
    aux_ability_masked: int | None = None


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
    #: The labels the Battle update distils toward (``ppo.search_distill``); empty otherwise.
    labels: tuple[LabelStep, ...] = ()
    #: Mean distillation KL over ``labels`` at the frozen weights; ``None`` without the block.
    search_kl_before: float | None = None
    #: The teacher steps drawn for this update (``ppo.imitation``), their draw stream (shuffles
    #: per epoch) and the mean cross-entropy over them at the frozen weights; empty / ``None``
    #: otherwise.
    teacher: tuple[Step, ...] = ()
    teacher_stream: DrawStream | None = None
    imitation_ce_before: float | None = None
    #: Mean |gamma * Phi(o_{t+1}) - Phi(o_t)| in millionths (``ppo.shaping``, Battle only).
    shaping_mean_abs_micro: int | None = None
    #: Per step, whether it has a ``ppo.uniform_kl`` term (two or more non-pending candidates), and
    #: the share (ppm) of support candidates below 1/(100 n) at the frozen weights; ``None``
    #: without the block or on a value warm-up update.
    uniform_valid: list[bool] | None = None
    uniform_floor_ppm: int | None = None


@dataclass(frozen=True)
class _FrozenBattle:
    """The Battle role's pre-update scores: per step, and the first value of each Episode."""

    log_probs: torch.Tensor | None
    values: torch.Tensor | None
    first: Mapping[str, float]


def _targets(step: Step, head: str) -> tuple[tuple[int, int], ...]:
    """A step's unmasked (row, label) targets of an opponent head (none without labels)."""
    return step.opp.of(head) if step.opp is not None else ()


VALUE_PREFIXES = ("value.", "value_net.")  # the head, or the separate value network
#: What a value warm-up update still trains: the value head, the move-reveal head (A12.2) and the
#: opponent heads (A14.2).
WARMUP_PREFIXES = (
    *VALUE_PREFIXES,
    "aux_moves.",
    "aux_action.",
    "aux_item.",
    "aux_ability.",
)


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
        micro_batch: int | None = None,
    ) -> None:
        if micro_batch is not None and micro_batch < 1:
            raise LearnerError("the learner micro-batch must be at least 1")
        #: Operational (``--learner-micro-batch``, not a config field): a minibatch is run through
        #: the network in chunks of at most this many decisions, each backpropagated at once and
        #: scaled by its share (gradient accumulation), so the activations of one optimiser step
        #: never exceed one chunk's. ``None`` is the single pass of every earlier release.
        self.micro_batch = micro_batch
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
        # The Threatened beta (millionths) of the newest update; set only with
        # ``ppo.anchor.threatened`` (stored in the sidecar then, and nowhere otherwise), and the
        # running (KL sum, Threatened decisions) of the role's minibatches in the update.
        self.last_anchor_threatened_beta_micro: int | None = None
        self._threatened_kl: dict[str, list[float]] = {}
        # The running (CE sum, targets, top-1 hits) of the Battle minibatches in the update
        # (``ppo.aux_moves``).
        self._aux_tally = [0.0, 0.0, 0.0]
        # The running (KL sum, decisions with a term) of the Battle minibatches in the update
        # (``ppo.uniform_kl``), and lambda (millionths) of the newest update (set only with the
        # block; stored in the sidecar for a linear schedule only).
        self._uniform_tally = [0.0, 0.0]
        self.last_uniform_kl_coef_micro: int | None = None
        self._opp_tally = {head: [0.0, 0.0, 0.0] for head in HEADS}
        # Entropy weight (millionths) of the newest update; set only under a linear
        # ``ppo.entropy_schedule`` (stored in the sidecar then, and nowhere otherwise).
        self.last_entropy_coef_micro: int | None = None
        # lambda_im (millionths) of the newest update; set only with ``ppo.imitation`` (0 on a
        # value warm-up update). Stored in the sidecar for a linear schedule only.
        self.last_imitation_coef_micro: int | None = None
        # The member's effective PPO settings (ADR-021 section 8); set only by a population run,
        # which stores them in the learner-state sidecar. ``None`` keeps a single lineage's bytes.
        self.population_settings: dict[str, int] | None = None
        # True for the one update after a population copy (ADR-021 section 6): the slot trains on
        # games its donor did not play, so only the value heads learn and the policy is frozen. The
        # population loop sets it before the update and clears it after; it is not persisted (a
        # resume re-applies the copy from the committed ladder record).
        self.value_only = False

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

    def imitation_coef_micro(self, update: int) -> int | None:
        """lambda_im of learner update ``update`` (from 0), or ``None`` without ``ppo.imitation``.

        The term is off during the value warm-up (0: the policy is frozen) and the schedule's
        index counts from the first policy update, as the anchor's does.
        """
        settings = self.config.ppo.imitation
        if settings is None:
            return None
        warmup = self.config.ppo.value_warmup or 0
        return 0 if update < warmup else settings.coef_micro_at(update - warmup)

    def uniform_kl_coef_micro(self, update: int) -> int | None:
        """lambda of learner update ``update`` (from 0), or ``None`` without ``ppo.uniform_kl``.

        Off during the value warm-up (0: the policy is frozen); the schedule's index counts from
        the first policy update, as the anchor's does.
        """
        settings = self.config.ppo.uniform_kl
        if settings is None:
            return None
        warmup = self.config.ppo.value_warmup or 0
        return 0 if update < warmup else settings.coef_micro_at(update - warmup)

    def anchor_threatened_beta_micro(self, update: int) -> int | None:
        """Threatened beta of learner update ``update`` (from 0), or ``None`` without
        ``ppo.anchor.threatened``: the anchor's rule, with its own ``kl_micro`` and schedule and
        the same update index (counting from the first POLICY update, 0 while the policy is
        frozen by a value warm-up)."""
        anchor = self.config.ppo.anchor
        if anchor is None or anchor.threatened is None:
            return None
        warmup = self.config.ppo.value_warmup or 0
        return 0 if update < warmup else anchor.threatened.beta_micro(update - warmup)

    # -- one generation ------------------------------------------------------

    def train(
        self,
        trajectories: Sequence[Trajectory],
        labels: Sequence[LabelStep] | None = None,
        teacher: TeacherSource | None = None,
    ) -> TrainReport:
        """Run the configured PPO epochs on the generation's trajectories, then count it.

        ``labels`` are the usable search labels of the aggregated label streams
        (``ppo.search_distill``); they are given exactly when the config has the block (an empty
        sequence before the first label stream) and only the Battle update reads them.
        ``teacher`` is the teacher's Battle steps (``ppo.imitation``), given exactly when the
        config has the block; each update draws its share of them and encodes only those.
        """
        if (labels is None) != (self.config.ppo.search_distill is None):
            raise LearnerError("labels are given exactly when the config has ppo.search_distill")
        if (teacher is None) != (self.config.ppo.imitation is None):
            raise LearnerError("teacher steps are given exactly when the config has ppo.imitation")
        roles = {}
        schedule = self.config.ppo.schedule
        if schedule is not None:
            # ADR-010 erratum A1: the rate is a function of the config and the update index (the
            # generation counter, restored on a resume), set on both optimisers before any epoch.
            self.last_rate_micro = schedule.rate_micro(self.generation)
            for optimiser in self.optimisers.values():
                for group in optimiser.param_groups:
                    group["lr"] = schedule.rate(self.generation)
        warmup = value_warmup_active(self.config, self.generation) or self.value_only
        self.last_anchor_beta_micro = self.anchor_beta_micro(self.generation)
        self.last_anchor_threatened_beta_micro = self.anchor_threatened_beta_micro(self.generation)
        self._threatened_kl = {}
        self._aux_tally = [0.0, 0.0, 0.0]
        self._uniform_tally = [0.0, 0.0]
        self.last_uniform_kl_coef_micro = self.uniform_kl_coef_micro(self.generation)
        self._opp_tally = {head: [0.0, 0.0, 0.0] for head in HEADS}
        self.last_entropy_coef_micro = self.entropy_coef_micro(self.generation)
        self.last_imitation_coef_micro = self.imitation_coef_micro(self.generation)
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
                warm = warmup and (self.value_only or role in self.warm_roles)
                tables[role] = (
                    self._table(
                        role,
                        mine,
                        role in self.anchors and not warm,
                        frozen,
                        tuple(labels or ()) if role == "battle" and not warm else (),
                        teacher if role == "battle" and not warm else None,
                        warm,
                    ),
                    warm,
                )
        if self.shared_trunk:
            roles.update(self._train_joint(tables, warmup))
        else:
            for role, (table, warm) in tables.items():
                roles[role] = self._train_role(role, table, warm)
        self.generation += 1
        return TrainReport(roles)

    def _pass_size(self) -> int:
        """Decisions per forward pass of the no-gradient passes: the minibatch, or the
        micro-batch when that is smaller."""
        size = self.config.ppo.minibatch_size
        return size if self.micro_batch is None else min(size, self.micro_batch)

    def _score(self, role: str, steps: Sequence[Step]) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of the chosen candidate and the value of each step, no gradient."""
        network = self.networks[role]
        size = self._pass_size()
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
            # Under ppo.shaping the value head estimates score - Phi: the bootstrap reads V + Phi.
            first[t.episode] = float(values[start]) + self._phi(t.steps[0])
            start += len(t.steps)
        for t in trajectories:
            if t.role == "preview" and t.episode not in first:
                raise LearnerDataError(
                    f"Episode {t.episode!r} has a Preview step and no Battle decision: "
                    "ppo.preview_bootstrap_micro needs the Battle value of its first position"
                )
        return _FrozenBattle(log_probs, values, first)

    def _phi(self, step: Step) -> float:
        """``Phi`` of a Battle step's observation under ``ppo.shaping``; 0.0 without the block."""
        shaping = self.config.ppo.shaping
        if shaping is None:
            return 0.0
        if step.potential_d is None:
            raise LearnerError("ppo.shaping needs the Battle steps read with their potential")
        return phi(step.potential_d, shaping.coef_micro)

    def _anchor_scores(self, role: str, steps: Sequence[Step]) -> list[torch.Tensor]:
        """The anchor's log-probabilities over each step's real candidates, no gradient."""
        network = self.anchors[role]
        size = self._pass_size()
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

    def _check_labels(self, role: str, steps: Sequence[Step]) -> None:
        """A Battle update with a label-reading term needs the steps read with their labels."""
        ppo = self.config.ppo
        if role != "battle":
            return
        if ppo.aux_moves is not None and any(step.aux is None for step in steps):
            raise LearnerDataError(
                "ppo.aux_moves needs the Battle steps read with their move-reveal labels"
            )
        if ppo.aux_opponent is not None and any(step.opp is None for step in steps):
            raise LearnerDataError(
                "ppo.aux_opponent needs the Battle steps read with their opponent labels"
            )

    def _table(  # noqa: PLR0913, PLR0917
        self,
        role: str,
        trajectories: Sequence[Trajectory],
        anchor: bool = False,
        frozen: _FrozenBattle | None = None,
        labels: tuple[LabelStep, ...] = (),
        teacher: TeacherSource | None = None,
        warm: bool = False,
    ) -> _Table:
        ppo = self.config.ppo
        steps = [s for t in trajectories for s in t.steps]
        if role == "battle" and role in self.anchors and self._threatened_beta() is not None:
            for step in steps:
                if step.threatened is None:
                    raise LearnerDataError(
                        "ppo.anchor.threatened needs the threats, speeds and foe_priorities rows "
                        "of every Battle example: a decision has none"
                    )
        self._check_labels(role, steps)
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
        potentials: list[float] = []  # Phi per step (ppo.shaping, Battle only)
        shaped_abs = 0.0
        shaping = role == "battle" and ppo.shaping is not None
        for t in trajectories:
            scores.extend([t.score] * len(t.steps))
            end = start + len(t.steps)
            target = (
                (1 - weight) * t.score + weight * frozen.first[t.episode]
                if bootstrap and frozen is not None
                else t.score
            )
            rewards = terminal_rewards(len(t.steps), target)
            if shaping:
                # r'_t = r_t + gamma * Phi(o_{t+1}) - Phi(o_t), Phi after the last decision 0.
                phis = [self._phi(s) for s in t.steps]
                potentials.extend(phis)
                moved, magnitude = shaped(rewards.tolist(), phis, ppo.discount)
                shaped_abs += magnitude
                rewards = torch.tensor(moved)
            a, r = gae(rewards, values[start:end], ppo.discount, ppo.gae_lambda)
            advantages.append(a)
            returns.append(r)
            start = end
        adv = torch.cat(advantages)
        if ppo.normalize_advantages:
            adv = normalise(adv)
        all_returns = torch.cat(returns)
        # float32 tensors widened exactly to float64 for the diagnostics.
        value_list = values.double().tolist()
        drawn, stream, ce_before = self._draw_teacher(role, len(steps), teacher)
        # Under ppo.shaping the head estimates the shaped return (score - Phi at gamma 1), so the
        # calibration compares V + Phi with the final score; explained variance stays against the
        # returns the learner trains on.
        calibrated = (
            [v + p for v, p in zip(value_list, potentials, strict=True)] if shaping else value_list
        )
        return _Table(
            steps,
            old_log_prob,
            adv.to(self.device),
            all_returns.to(self.device),
            explained_variance(all_returns.double().tolist(), value_list),
            calibration(calibrated, scores),
            self._anchor_scores(role, steps) if anchor else None,
            labels,
            self._label_kl(labels)
            if role == "battle" and self.config.ppo.search_distill is not None
            else None,
            drawn,
            stream,
            ce_before,
            int(shaped_abs / len(steps) * MICRO) if shaping and steps else (0 if shaping else None),
            *self._uniform_frozen(role, steps, warm),
        )

    def _uniform_frozen(
        self, role: str, steps: Sequence[Step], warm: bool
    ) -> tuple[list[bool] | None, int | None]:
        """Which Battle steps have a ``ppo.uniform_kl`` term, and the share (ppm) of their support
        candidates below 1/(100 n) at the frozen weights, one no-gradient pass; ``(None, None)``
        without the block, for the Preview role and (the term being off) on a warm-up update."""
        if role != "battle" or self.config.ppo.uniform_kl is None or warm:
            return None, None
        network = self.networks[role]
        size = self._pass_size()
        tau = self.config.generation.temperature_ppm
        valid: list[bool] = []
        below = total = 0
        network.eval()
        with torch.no_grad():
            for start in range(0, len(steps), size):
                batch = to_device(
                    collate([s.query for s in steps[start : start + size]]), self.device
                )
                logits = network(batch).logits
                _, has = uniform_kl(logits, batch.cand_mask, batch.pending, tau)
                valid.extend(bool(v) for v in has.cpu().tolist())
                low, count = uniform_floor_counts(logits, batch.cand_mask, batch.pending, tau)
                below, total = below + low, total + count
        return valid, below * MICRO // total if total else 0

    def _draw_teacher(
        self, role: str, battle_steps: int, teacher: TeacherSource | None
    ) -> tuple[tuple[Step, ...], DrawStream | None, float | None]:
        """This update's teacher steps (``ppo.imitation``): K = floor(share * N_battle / 10^6)
        drawn without replacement by the update's own stream, and their cross-entropy at the
        frozen weights. Nothing is drawn, and the stream is not built, without the block."""
        settings = self.config.ppo.imitation
        if role != "battle" or settings is None or teacher is None:
            return (), None, None
        stream = DrawStream(self.config.learner_seed, self.generation)
        picked = stream.draw(len(teacher), settings.steps(battle_steps, len(teacher)))
        drawn = teacher.fetch(picked)  # encoded now, held for this update only
        return drawn, stream, self._imitation_eval(drawn)[0]

    def _imitation_eval(self, steps: Sequence[Step]) -> tuple[float, int]:
        """Mean cross-entropy (at the play temperature) of the Battle network over ``steps``, and
        how many of them it ranks the teacher's choice first (0.0 and 0 if none); no gradient."""
        if not steps:
            return 0.0, 0
        network = self.networks["battle"]
        size = self._pass_size()
        tau = self.config.generation.temperature_ppm
        total = torch.zeros(())
        hits = 0
        network.eval()
        with torch.no_grad():
            for start in range(0, len(steps), size):
                chunk = steps[start : start + size]
                batch = to_device(collate([s.query for s in chunk]), self.device)
                logits = network(batch).logits
                chosen = torch.tensor([s.chosen for s in chunk], device=self.device)
                total += (-policy_terms(logits, batch.cand_mask, chosen, tau).log_prob).sum().cpu()
                top = logits.masked_fill(~batch.cand_mask, torch.finfo(logits.dtype).min).argmax(-1)
                hits += int((top == chosen).sum())
        return float(total) / len(steps), hits

    def _label_dense(self, labels: Sequence[LabelStep], batch: Batch) -> torch.Tensor:
        """``ln pi'`` of ``labels`` as [B, C] (zero at padded places), on the batch's device."""
        dense = torch.zeros(batch.cand_mask.shape, device=self.device)
        for row, label in enumerate(labels):
            dense[row, : len(label.log_target)] = torch.tensor(
                label.log_target, dtype=torch.float32, device=self.device
            )
        return dense

    def _label_kl(self, labels: Sequence[LabelStep]) -> float:
        """Mean KL(pi' || pi_theta) of the Battle network over ``labels`` (0.0 if none)."""
        if not labels:
            return 0.0
        network = self.networks["battle"]
        size = self._pass_size()
        tau = self.config.generation.temperature_ppm
        total = torch.zeros(())
        network.eval()
        with torch.no_grad():
            for start in range(0, len(labels), size):
                chunk = labels[start : start + size]
                batch = to_device(collate([lab.query for lab in chunk]), self.device)
                kl = distill_kl(
                    network(batch).logits, batch.cand_mask, self._label_dense(chunk, batch), tau
                )
                total += kl.sum().cpu()
        return float(total) / len(labels)

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
            p for n, p in network.named_parameters() if warm and not n.startswith(WARMUP_PREFIXES)
        ]
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            return self._epochs(role, table, warm)
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)

    def _threatened_beta(self) -> int | None:
        """The Threatened beta of the update in progress, or ``None`` without the block."""
        return self.last_anchor_threatened_beta_micro

    def _weights(self) -> LossWeights:
        ppo = self.config.ppo
        entropy_micro = self.last_entropy_coef_micro
        entropy_coef = ppo.entropy_coef if entropy_micro is None else entropy_micro / MICRO
        return LossWeights(ppo.clip, ppo.value_coef, entropy_coef)

    def _minibatch_loss(  # noqa: PLR0913, PLR0915, PLR0917
        self,
        role: str,
        table: _Table,
        warm: bool,
        index: list[int],
        weights: LossWeights,
        label_index: list[int] | None = None,
        teacher_index: list[int] | None = None,
        shares: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        """One minibatch of one role: the tensor to backpropagate, the loss terms and, with an
        anchor, the (detached-by-the-caller) mean KL to the anchor.

        ``shares`` (micro-batching only) is this chunk's share of the minibatch's decisions, label
        steps, teacher steps, move-reveal targets, the decisions with a zero-avoiding term and the
        targets of the action, item and ability heads: each term is a mean over its own set, so it
        is scaled by the chunk's share of that set and the chunks' tensors add up to the
        minibatch's. At 1.0 (the default) nothing is multiplied."""
        main_share, label_share, teacher_share, aux_share, uniform_share, *head_shares = shares
        opp_shares = dict(zip(HEADS, head_shares, strict=True))

        def scaled(term: torch.Tensor, share: float) -> torch.Tensor:
            return term if share == 1.0 else term * share

        tau = self.config.generation.temperature_ppm
        beta_micro = self.last_anchor_beta_micro if role in self.anchors else None
        beta = 0.0 if beta_micro is None else beta_micro / MICRO
        batch = to_device(collate([table.steps[i].query for i in index]), self.device)
        aux = self.config.ppo.aux_moves if role == "battle" else None
        out, aux_ce, opp_ce = self._forward(role, table, index, batch)
        chosen = torch.tensor([table.steps[i].chosen for i in index], device=self.device)
        # `policy_terms` masks padding again although the network already pushes padded
        # logits to the dtype minimum: defence in depth, so a loss never depends on it.
        loss = ppo_loss(
            policy_terms(out.logits, batch.cand_mask, chosen, tau),
            out.value,
            Targets(table.old_log_prob[index], table.advantages[index], table.returns[index]),
            weights,
        )
        total = scaled(loss.total, main_share)
        kl = None
        if table.anchor_log_probs is not None:
            dense = torch.zeros(batch.cand_mask.shape, device=self.device)
            for row, i in enumerate(index):
                dense[row, batch.cand_mask[row]] = table.anchor_log_probs[i]
            kl_rows = anchor_kl(out.logits, batch.cand_mask, dense, tau)
            kl = kl_rows.mean()
            threatened = self._threatened_beta() if role == "battle" else None
            if threatened is None:
                total = total + scaled(beta * kl, main_share)
            else:
                # ADR-010 erratum A11.2: beta(d) per decision, not renormalised.
                flags = [bool(table.steps[i].threatened) for i in index]
                hot = torch.tensor(flags, device=self.device)
                betas = torch.where(hot, threatened / MICRO, beta)
                total = total + scaled((betas * kl_rows).mean(), main_share)
                tally = self._threatened_kl.setdefault(role, [0.0, 0.0])
                tally[0] += float(kl_rows.detach()[hot].sum().cpu())
                tally[1] += float(hot.sum().cpu())
        settings = self.config.ppo.search_distill
        if settings is not None and label_index and not warm:
            # ADR-026 erratum E2: lambda * mean KL(pi' || pi_theta) over this minibatch's labels,
            # a second forward pass of the Battle network on the labelled decisions. The gradient
            # reaches the shared trunk under shared_trunk, as the anchor's does.
            chosen_labels = [table.labels[i] for i in label_index]
            label_batch = to_device(collate([lab.query for lab in chosen_labels]), self.device)
            label_kl = distill_kl(
                self.networks[role](label_batch).logits,
                label_batch.cand_mask,
                self._label_dense(chosen_labels, label_batch),
                tau,
            ).mean()
            total = total + scaled(settings.coef_micro / MICRO * label_kl, label_share)
        imitation = self.config.ppo.imitation
        if imitation is not None and teacher_index and not warm:
            # ADR-010 erratum A11.3: lambda_im * the mean cross-entropy toward the teacher's choice
            # over this minibatch's teacher steps, at the play temperature over the real
            # candidates: a second forward pass of the Battle network on the teacher's fixed
            # states, with no clip and no value term. The gradient reaches the shared trunk under
            # shared_trunk, as the anchor's does.
            steps = [table.teacher[i] for i in teacher_index]
            teacher_batch = to_device(collate([s.query for s in steps]), self.device)
            picked = torch.tensor([s.chosen for s in steps], device=self.device)
            terms = policy_terms(
                self.networks[role](teacher_batch).logits, teacher_batch.cand_mask, picked, tau
            )
            imitation_term = (
                (self.last_imitation_coef_micro or 0) / MICRO * (-terms.log_prob).mean()
            )
            total = total + scaled(imitation_term, teacher_share)
        uniform_term: float | torch.Tensor = 0.0
        if self.config.ppo.uniform_kl is not None and role == "battle" and not warm:
            # ADR-010 erratum A14.1: lambda * the mean KL(u || pi_theta) over the minibatch's
            # decisions with two or more non-pending candidates, at the play temperature over the
            # real candidates (pending ones stay in the normaliser). No clip: its logit gradient
            # lambda (pi_k - 1/n) is bounded. The gradient reaches the shared trunk under
            # shared_trunk, as the anchor's does.
            rows, valid = uniform_kl(out.logits, batch.cand_mask, batch.pending, tau)
            count = int(valid.sum())
            if count:
                lam = (self.last_uniform_kl_coef_micro or 0) / MICRO
                uniform_term = scaled(lam * rows.sum() / count, uniform_share)
                self._uniform_tally[0] += float(rows.detach().sum().cpu())
                self._uniform_tally[1] += float(count)
        aux_term: float | torch.Tensor = 0.0
        if aux is not None and aux_ce is not None and aux_ce.numel():
            # ADR-010 erratum A12.2: coef * the mean cross-entropy over the minibatch's unmasked
            # move-reveal targets; it trains the head and, through the trunk, the representation
            # (frozen during a value warm-up, which then trains the head alone).
            aux_term = scaled(aux.coef_micro / MICRO * aux_ce.mean(), aux_share)
        aux_term = aux_term + self._opp_loss(opp_ce, opp_shares)
        warm_loss = scaled(loss.value, main_share)
        return (warm_loss + aux_term if warm else total + aux_term + uniform_term), loss, kl

    def _forward(
        self, role: str, table: _Table, index: Sequence[int], batch: Batch
    ) -> tuple[PolicyOutput, torch.Tensor | None, dict[str, torch.Tensor]]:
        """The policy output and, with the move-reveal head (A12.2) or the opponent heads (A14.2),
        the cross-entropies of their unmasked targets, from one trunk pass."""
        network = self.networks[role]
        aux = self.config.ppo.aux_moves if role == "battle" else None
        opp = self.config.ppo.aux_opponent if role == "battle" else None
        steps = [table.steps[i].aux or () for i in index]
        if opp is not None:
            out, moves, opp_logits = network.forward_opponent(
                batch, self._opp_picks(table, index, opp.heads), aux is not None
            )
            aux_ce = None if moves is None else self._aux_ce(moves, steps)
            return out, aux_ce, self._opp_ce(table, index, opp_logits)
        if aux is None:
            return network(batch), None, {}
        out, moves = network.forward_aux(batch)
        return out, self._aux_ce(moves, steps), {}

    def _opp_loss(
        self, opp_ce: Mapping[str, torch.Tensor], shares: Mapping[str, float]
    ) -> torch.Tensor | float:
        """ADR-010 erratum A14.2: per head, coef * the mean cross-entropy over the minibatch's
        unmasked targets, scaled by the chunk's share of the head's targets (1.0 unchunked)."""
        opp = self.config.ppo.aux_opponent
        total: torch.Tensor | float = 0.0
        for head, ce in opp_ce.items():
            assert opp is not None
            term = (opp.coef_micro(head) or 0) / MICRO * ce.mean()
            total = total + (term if shares[head] == 1.0 else term * shares[head])
        return total

    def _accumulate(  # noqa: PLR0913, PLR0917
        self,
        role: str,
        table: _Table,
        warm: bool,
        index: list[int],
        weights: LossWeights,
        label_index: list[int] | None,
        teacher_index: list[int] | None,
    ) -> tuple[LossParts, torch.Tensor | None]:
        """Micro-batched ``_minibatch_loss``: the minibatch in chunks of at most
        ``self.micro_batch`` decisions, each backpropagated at once (its graph freed) and scaled by
        its share, so the gradients accumulate to the minibatch's. The label and teacher steps are
        spread over the same chunks and each term is scaled by its own share. The returned loss
        terms and KL are the share-weighted sums, i.e. the minibatch means. The caller zeroes the
        gradients before and clips and steps after."""
        assert self.micro_batch is not None
        size = self.micro_batch
        chunks = [index[s : s + size] for s in range(0, len(index), size)]
        parts = len(chunks)
        labels = self._spread(label_index, parts) if label_index else [None] * parts
        teachers = self._spread(teacher_index, parts) if teacher_index else [None] * parts
        aux = self.config.ppo.aux_moves if role == "battle" else None
        counts = [sum(len(table.steps[i].aux or ()) for i in c) for c in chunks] if aux else None
        aux_total = sum(counts) if counts else 0
        uniform_valid = table.uniform_valid if role == "battle" else None
        uniform_counts = (
            [sum(1 for i in c if uniform_valid[i]) for c in chunks] if uniform_valid else None
        )
        uniform_total = sum(uniform_counts) if uniform_counts else 0
        opp = self.config.ppo.aux_opponent if role == "battle" else None
        head_counts: dict[str, list[int]] = {
            head: [self._opp_count(table, c, head) for c in chunks]
            for head in (opp.heads if opp else ())
        }
        head_totals = {head: sum(found) for head, found in head_counts.items()}
        fields = [torch.zeros(()) for _ in range(6)]
        kl_total: torch.Tensor | None = None
        for k, chunk in enumerate(chunks):
            main = len(chunk) / len(index)
            label = len(labels[k] or ()) / len(label_index) if label_index else 1.0
            teacher = len(teachers[k] or ()) / len(teacher_index) if teacher_index else 1.0
            aux_share = counts[k] / aux_total if counts and aux_total else 1.0
            uniform_share = (
                uniform_counts[k] / uniform_total if uniform_counts and uniform_total else 1.0
            )
            head_shares = tuple(
                head_counts[head][k] / head_totals[head] if head_totals.get(head) else 1.0
                for head in HEADS
            )
            backward, loss, kl = self._minibatch_loss(
                role,
                table,
                warm,
                chunk,
                weights,
                labels[k],
                teachers[k],
                (main, label, teacher, aux_share, uniform_share, *head_shares),
            )
            torch.autograd.backward(backward)
            terms = (
                loss.total,
                loss.policy,
                loss.value,
                loss.entropy,
                loss.clip_fraction,
                loss.approx_kl,
            )
            for j, term in enumerate(terms):
                fields[j] = fields[j] + term.detach().cpu() * main
            if kl is not None:
                weighted = kl.detach().cpu() * main
                kl_total = weighted if kl_total is None else kl_total + weighted
        return LossParts(*fields), kl_total

    def _aux_ce(
        self, logits: torch.Tensor, targets: Sequence[tuple[Any, ...]]
    ) -> torch.Tensor | None:
        """Cross-entropy of each unmasked target of the minibatch (empty if none), and the update's
        running tally of the sum, the count and the top-1 hits, measured before the step."""
        batch, rows, known, target = targets_tensors(targets, logits.shape[-1])
        if not batch.numel():
            return None
        selected = logits[batch.to(self.device), rows.to(self.device)]
        ce, hit = aux_terms(selected, known.to(self.device), target.to(self.device))
        tally = self._aux_tally
        tally[0] += float(ce.detach().sum().cpu())
        tally[1] += float(ce.numel())
        tally[2] += float(hit.sum().cpu())
        return ce

    @staticmethod
    def _opp_count(table: _Table, chunk: Sequence[int], head: str) -> int:
        """The unmasked targets of ``head`` in the chunk's steps."""
        return sum(len(_targets(table.steps[i], head)) for i in chunk)

    def _opp_picks(
        self, table: _Table, index: Sequence[int], heads: Sequence[str]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """(batch index, token row) of each head's unmasked targets in the minibatch."""
        picks = {}
        for head in heads:
            batch, rows, _ = opponent_targets([_targets(table.steps[i], head) for i in index])
            picks[head] = (batch.to(self.device), rows.to(self.device))
        return picks

    def _opp_ce(
        self, table: _Table, index: Sequence[int], logits: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Cross-entropy of each unmasked target of each head in the minibatch (heads with none
        are left out), and the update's running tallies, measured before the step."""
        found = {}
        for head, head_logits in logits.items():
            _, _, target = opponent_targets([_targets(table.steps[i], head) for i in index])
            if not target.numel():
                continue
            ce, hit = opponent_terms(head_logits, target.to(self.device))
            tally = self._opp_tally[head]
            tally[0] += float(ce.detach().sum().cpu())
            tally[1] += float(ce.numel())
            tally[2] += float(hit.sum().cpu())
            found[head] = ce
        return found

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
            *self._search_report(role, table),
            *self._imitation_report(role, table),
            *self._shaping_report(role, table),
            *self._threatened_report(role, table),
            *self._aux_report(role, table),
            *self._uniform_report(role, table, warm),
            *self._opp_report(role, table),
        )

    def _imitation_report(self, role: str, table: _Table) -> tuple[Any, Any, Any, Any, Any]:
        if role != "battle" or self.config.ppo.imitation is None:
            return None, None, None, None, None
        after, hits = self._imitation_eval(table.teacher)
        size = len(table.teacher)
        return (
            size,
            self.last_imitation_coef_micro,
            table.imitation_ce_before or 0.0,
            after,
            hits * MICRO // size if size else 0,
        )

    def _threatened_report(self, role: str, table: _Table) -> tuple[Any, Any, Any]:
        if role != "battle" or self.last_anchor_threatened_beta_micro is None:
            return None, None, None
        kl_sum, count = self._threatened_kl.get(role, [0.0, 0.0])
        return (
            sum(1 for step in table.steps if step.threatened),
            kl_sum / count if count else 0.0,
            self.last_anchor_threatened_beta_micro,
        )

    def _uniform_report(self, role: str, table: _Table, warm: bool) -> tuple[Any, Any, Any, Any]:
        if role != "battle" or self.config.ppo.uniform_kl is None:
            return None, None, None, None
        total, count = self._uniform_tally
        steps = 0 if warm or table.uniform_valid is None else sum(table.uniform_valid)
        return (
            total / count if count else 0.0,
            self.last_uniform_kl_coef_micro,
            steps,
            table.uniform_floor_ppm or 0,
        )

    def _aux_report(self, role: str, table: _Table) -> tuple[Any, Any, Any, Any]:
        if role != "battle" or self.config.ppo.aux_moves is None:
            return None, None, None, None
        total, count, hits = self._aux_tally
        # Denominators differ: ``steps`` counts the table's unmasked targets once; the loss and
        # top-1 average the minibatch targets of every epoch (``count``, steps times epochs).
        steps = sum(len(step.aux or ()) for step in table.steps)
        masked = sum(step.aux_masked or 0 for step in table.steps)
        return steps, total / count if count else 0.0, hits / count if count else 0.0, masked

    def _opp_report(self, role: str, table: _Table) -> tuple[Any, ...]:
        opp = self.config.ppo.aux_opponent
        if role != "battle" or opp is None:
            return (None,) * (4 * len(HEADS))
        out: list[Any] = []
        for head in HEADS:
            if head not in opp.heads:
                out.extend((None, None, None, None))
                continue
            total, count, hits = self._opp_tally[head]
            targets = [step.opp.of(head) for step in table.steps if step.opp]
            masked = sum(step.opp.masked[HEADS.index(head)] for step in table.steps if step.opp)
            out.extend(
                (
                    sum(len(t) for t in targets),
                    total / count if count else 0.0,
                    hits / count if count else 0.0,
                    masked,
                )
            )
        return tuple(out)

    def _shaping_report(self, role: str, table: _Table) -> tuple[int | None, int | None]:
        shaping = self.config.ppo.shaping
        if role != "battle" or shaping is None:
            return None, None
        return shaping.coef_micro, table.shaping_mean_abs_micro

    def _search_report(self, role: str, table: _Table) -> tuple[Any, Any, Any]:
        if role != "battle" or self.config.ppo.search_distill is None:
            return None, None, None
        return len(table.labels), table.search_kl_before, self._label_kl(table.labels)

    def _epochs(self, role: str, table: _Table, warm: bool) -> RoleReport:
        ppo = self.config.ppo
        network, optimiser = self.networks[role], self.optimisers[role]
        weights = self._weights()
        sums = torch.zeros(5)
        kl_sum = torch.zeros(())
        optimiser_steps = 0
        epochs_run = 0
        kl_stopped = False
        for epoch in range(ppo.epochs):
            epoch_kl = 0.0
            minibatches = 0
            order = torch.randperm(len(table.steps), generator=self.generator).tolist()
            parts = -(-len(order) // ppo.minibatch_size)
            label_chunks = self._label_chunks(table, parts, epoch)
            teacher_chunks = self._teacher_chunks(table, parts)
            for k, start in enumerate(range(0, len(order), ppo.minibatch_size)):
                index = order[start : start + ppo.minibatch_size]
                optimiser.zero_grad(set_to_none=True)
                if self.micro_batch is None:
                    backward, loss, kl = self._minibatch_loss(
                        role,
                        table,
                        warm,
                        index,
                        weights,
                        label_chunks[k] if label_chunks else None,
                        teacher_chunks[k] if teacher_chunks else None,
                    )
                    torch.autograd.backward(backward)
                else:
                    loss, kl = self._accumulate(
                        role,
                        table,
                        warm,
                        index,
                        weights,
                        label_chunks[k] if label_chunks else None,
                        teacher_chunks[k] if teacher_chunks else None,
                    )
                if kl is not None:
                    kl_sum += kl.detach().cpu()
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

    def _label_chunks(self, table: _Table, parts: int, epoch: int) -> list[list[int]] | None:
        """The table's labels shuffled and spread over ``parts`` minibatches (as the shared trunk
        spreads the second role); ``None`` without labels.

        The shuffle uses a generator of its own, seeded from the learner seed, the update index
        and the epoch, so it needs no stored state (a resume restores the update counter) and the
        learner's main generator is never touched: a run without the block, and the Preview
        role's shuffles in a run with it, keep their orders.
        """
        if not table.labels:
            return None
        seed = ((self.config.learner_seed * 1_000_003 + self.generation) * 1_000_003 + epoch) % (
            2**63
        )
        own = torch.Generator(device="cpu")
        own.manual_seed(seed)
        return self._spread(torch.randperm(len(table.labels), generator=own).tolist(), parts)

    def _teacher_chunks(self, table: _Table, parts: int) -> list[list[int]] | None:
        """The update's teacher steps shuffled (by the update's own draw stream, which also drew
        them) and spread over ``parts`` Battle minibatches, as the labels are; ``None`` without
        teacher steps. The learner's main generator is never touched."""
        if not table.teacher or table.teacher_stream is None:
            return None
        return self._spread(table.teacher_stream.shuffle(list(range(len(table.teacher)))), parts)

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

    def _train_joint(  # noqa: PLR0912, PLR0915
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
            if (warm and not n.startswith(WARMUP_PREFIXES)) or (warmup and n in trunk)
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
            for epoch in range(ppo.epochs):
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
                label_chunks = (
                    self._label_chunks(tables["battle"][0], len(chunks[leader]), epoch)
                    if "battle" in tables
                    else None
                )
                teacher_chunks = (
                    self._teacher_chunks(tables["battle"][0], len(chunks[leader]))
                    if "battle" in tables
                    else None
                )
                epoch_kl = dict.fromkeys(tables, 0.0)
                epoch_n = dict.fromkeys(tables, 0)
                for k in range(len(chunks[leader])):
                    parts = []
                    optimiser.zero_grad(set_to_none=True)
                    for role, (table, warm) in tables.items():
                        index = chunks[role][k]
                        if not index:
                            continue
                        label_index = label_chunks[k] if role == "battle" and label_chunks else None
                        teacher_index = (
                            teacher_chunks[k] if role == "battle" and teacher_chunks else None
                        )
                        if self.micro_batch is None:
                            backward, loss, kl = self._minibatch_loss(
                                role, table, warm, index, weights, label_index, teacher_index
                            )
                            parts.append(backward)
                        else:
                            loss, kl = self._accumulate(
                                role, table, warm, index, weights, label_index, teacher_index
                            )
                        if kl is not None:
                            kl_sums[role] += kl.detach().cpu()
                        counts[role] += 1
                        epoch_n[role] += 1
                        epoch_kl[role] += float(loss.approx_kl.detach().cpu())
                        self._tally(sums[role], loss)
                    if self.micro_batch is None:
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
