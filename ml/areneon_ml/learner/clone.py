"""The supervised cloning learner (ADR-019 track 1, ADR-010 erratum A2).

A learner-ready TrainingSet played by a scripted teacher is the label: for each
decision the teacher's choice is the target, and the loss is the cross-entropy
of a softmax over the real candidates of that decision (padding masked out,
exactly the policy of ``ppo.policy_terms``), so a candidate the encoder could
not back never takes probability. Preview and Battle are trained separately,
each from its own seeded initialisation, with the same ``ModelConfig`` and
encoder as PPO. The value head has no label here and is not trained; the PPO
phase that follows learns it.

A part of the Episodes is held back from training, chosen per Episode by a
keyed hash of the Episode hash (HMAC-SHA256 over its 64 hex characters, the
first 8 bytes big endian as ``n``, held back when ``n * 10^6 < ppm * 2^64``,
the rule shape of ADR-015); the report gives the top-1 agreement with the
teacher on both parts. Top-1 is the candidate with the largest logit (the
first on a tie); ``chance`` is the mean of ``1 / candidates`` over the same
decisions.

Determinism: on CPU, with the config's thread count and deterministic
algorithms (``configure_learner``), the same config and stream on the same
machine and software give bit-identical weights. Nothing is claimed across
machines or torch builds, and a CUDA run is not bit-reproducible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.training import stream_lines

from areneon_ml.learner.clone_config import CloneConfig, SituationWeights
from areneon_ml.learner.data import LearnerDataError, Step, Trajectory, trajectories
from areneon_ml.learner.ppo import policy_terms
from areneon_ml.learner.situation import GROUPS, Groups
from areneon_ml.learner.trainer import to_device
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, PolicyNetwork, init_network

_TWO_64 = 1 << 64
# A learned pair's behaviour policy id; a scripted baseline's is ``<name>@<commit>``.
_LEARNED_PREFIX = "policy:"


class CloneError(ValueError):
    """The cloning run cannot be built or run as asked."""


def held_back(key: bytes, episode_hash: str, ppm: int) -> bool:
    """Whether the Episode is in the held-back part: a keyed hash of its hash, in integers."""
    digest = hmac.new(key, episode_hash.encode("ascii"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") * 1_000_000 < ppm * _TWO_64


def cross_entropy(logits: torch.Tensor, mask: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ``-log softmax(logits over the real candidates)[chosen]``."""
    return -policy_terms(logits, mask, chosen).log_prob.mean()


def top1_hits(logits: torch.Tensor, mask: torch.Tensor, chosen: torch.Tensor) -> int:
    """How many decisions have ``chosen`` as their largest real-candidate logit (first on a tie)."""
    floor = torch.finfo(logits.dtype).min
    return int((logits.masked_fill(~mask, floor).argmax(dim=-1) == chosen).sum().item())


@dataclass(frozen=True)
class Agreement:
    """The teacher-agreement of a network on a set of decisions."""

    steps: int
    loss: float
    hits: int
    chance: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "loss": self.loss,
            "top1_hits": self.hits,
            "chance": self.chance,
        }


@dataclass(frozen=True)
class GroupAgreement:
    """The agreement of a network on the decisions of one situation group (ADR-010 erratum A13.4):
    ``loss`` is the UNWEIGHTED mean cross-entropy; on ``streak`` also how many teacher choices
    (``stall_teacher``) and clone top-1 candidates (``stall_clone``) use a stalling move in a
    streak slot."""

    agreement: Agreement
    stall_teacher: int | None = None
    stall_clone: int | None = None
    #: On ``streak``: request slots (over every grouped step) whose ident matches no roster entry,
    #: which the streak rule cannot judge; ``None`` on every other group.
    streak_unmatched: int | None = None

    def as_dict(self) -> dict[str, float | int]:
        out = self.agreement.as_dict()
        if self.streak_unmatched is not None:
            out["streak_unmatched"] = self.streak_unmatched
        if self.stall_teacher is not None and self.stall_clone is not None:
            out["stall_teacher"] = self.stall_teacher
            out["stall_clone"] = self.stall_clone
        return out


def _groups_dict(groups: dict[str, GroupAgreement]) -> dict[str, Any]:
    return {name: groups[name].as_dict() for name in GROUPS}


@dataclass(frozen=True)
class SituationReport:
    """What a Battle role reports under ``clone.situation_weights`` (ADR-010 erratum A13.4)."""

    weights: SituationWeights
    weight_mean_ppm: int
    train_groups: dict[str, GroupAgreement]
    holdout_groups: dict[str, GroupAgreement] | None

    def as_dict(self) -> dict[str, Any]:
        weights: dict[str, int] = {"streak_ppm": self.weights.streak_ppm}
        if self.weights.threatened_ppm is not None:
            weights["threatened_ppm"] = self.weights.threatened_ppm
        out: dict[str, Any] = {
            "situation_weights": weights,
            "weight_mean_ppm": self.weight_mean_ppm,
            "train_groups": _groups_dict(self.train_groups),
        }
        if self.holdout_groups is not None:
            out["holdout_groups"] = _groups_dict(self.holdout_groups)
        return out


@dataclass(frozen=True)
class RoleReport:
    """One role: the loss of each epoch and the agreement on the training and held-back parts."""

    optimiser_steps: int
    epoch_loss: tuple[float, ...]
    train: Agreement
    holdout: Agreement | None
    situation: SituationReport | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "optimiser_steps": self.optimiser_steps,
            "epoch_loss": list(self.epoch_loss),
            "train": self.train.as_dict(),
        }
        if self.holdout is not None:
            out["holdout"] = self.holdout.as_dict()
        if self.situation is not None:
            out.update(self.situation.as_dict())
        return out


@dataclass(frozen=True)
class Cloned:
    """Both trained networks, their optimisers and the generator, with the report."""

    networks: dict[str, PolicyNetwork]
    optimisers: dict[str, torch.optim.Adam]
    generator: torch.Generator
    roles: dict[str, RoleReport]
    team_set: tuple[str, ...]


def evaluate(
    network: PolicyNetwork, steps: Sequence[Step], size: int, device: torch.device
) -> Agreement:
    """Mean cross-entropy and top-1 hits over ``steps``, no gradient, in minibatches of ``size``."""
    network.eval()
    total, hits = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(steps), size):
            chunk = steps[start : start + size]
            batch = to_device(collate([s.query for s in chunk]), device)
            chosen = torch.tensor([s.chosen for s in chunk], device=device)
            logits = network(batch).logits
            total += float(cross_entropy(logits, batch.cand_mask, chosen).cpu()) * len(chunk)
            hits += top1_hits(logits, batch.cand_mask, chosen)
    chance = sum(1.0 / s.query.pending.shape[0] for s in steps) / len(steps) if steps else 0.0
    return Agreement(len(steps), total / len(steps) if steps else 0.0, hits, chance)


def evaluate_groups(
    network: PolicyNetwork, steps: Sequence[Step], size: int, device: torch.device
) -> dict[str, GroupAgreement]:
    """The agreement per situation group (``streak``, ``threatened``, ``other``), no gradient.

    A step with no group is not counted. An empty group reports 0 steps and 0.0 values.
    """
    network.eval()
    count = dict.fromkeys(GROUPS, 0)
    hit = dict.fromkeys(GROUPS, 0)
    loss_sum = dict.fromkeys(GROUPS, 0.0)
    chance_sum = dict.fromkeys(GROUPS, 0.0)
    stalls = [0, 0]  # teacher, clone: streak steps whose choice uses a stalling move
    grouped = [s for s in steps if s.group is not None]
    unmatched = sum(s.unmatched for s in grouped)
    with torch.no_grad():
        for start in range(0, len(grouped), size):
            chunk = grouped[start : start + size]
            batch = to_device(collate([s.query for s in chunk]), device)
            chosen = torch.tensor([s.chosen for s in chunk], device=device)
            logits = network(batch).logits
            floor = torch.finfo(logits.dtype).min
            picked = logits.masked_fill(~batch.cand_mask, floor).argmax(dim=-1).tolist()
            losses = (-policy_terms(logits, batch.cand_mask, chosen).log_prob).tolist()
            for step, loss, pick in zip(chunk, losses, picked, strict=True):
                name = str(step.group)
                count[name] += 1
                loss_sum[name] += loss
                hit[name] += int(pick == step.chosen)
                chance_sum[name] += 1.0 / step.query.pending.shape[0]
                if step.stall is not None:
                    stalls[0] += int(step.stall[step.chosen])
                    stalls[1] += int(step.stall[pick])
    out = {}
    for name in GROUPS:
        n = count[name]
        agreement = Agreement(
            n, loss_sum[name] / n if n else 0.0, hit[name], chance_sum[name] / n if n else 0.0
        )
        streak = name == "streak"
        out[name] = GroupAgreement(
            agreement,
            stalls[0] if streak else None,
            stalls[1] if streak else None,
            unmatched if streak else None,
        )
    return out


def situation_factors(steps: Sequence[Step], weights: SituationWeights) -> tuple[torch.Tensor, int]:
    """The loss factor of each training step and the floor of the mean weight in ppm.

    Step weights are integers (``streak_ppm``, ``threatened_ppm``, else 1,000,000); the factor is
    ``w_i / w_bar = w_i * |S| / sum(w)``, one exact integer ratio rounded once to float64 and then
    to float32 (ADR-010 erratum A13.3). The mean factor over ``steps`` is 1 up to that rounding.
    """
    table = {
        "streak": weights.streak_ppm,
        "threatened": weights.threatened_ppm,
        "other": 1_000_000,
    }
    found: list[int] = []
    for step in steps:
        weight = table[str(step.group)]
        found.append(1_000_000 if weight is None else weight)
    total = sum(found)
    factors = torch.tensor([w * len(found) / total for w in found], dtype=torch.float32)
    return factors, total // len(found)


def _train_role(  # noqa: PLR0913
    config: CloneConfig,
    network: PolicyNetwork,
    parts: tuple[Sequence[Step], Sequence[Step]],
    generator: torch.Generator,
    device: torch.device,
    *,
    weights: SituationWeights | None = None,
) -> tuple[torch.optim.Adam, RoleReport]:
    """Train ``network`` on ``parts[0]``; ``parts[1]`` is the held-back part, only measured.

    With ``weights`` (the Battle role under ``clone.situation_weights``) every decision's
    cross-entropy is scaled by its factor (``situation_factors``); the shuffle, minibatches and
    optimiser steps are the same as without them.
    """
    steps, held = parts
    settings = config.clone
    factors, weight_mean = situation_factors(steps, weights) if weights is not None else (None, 0)
    optimiser = torch.optim.Adam(
        network.parameters(),
        lr=settings.learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_eps,
    )
    epoch_loss: list[float] = []
    optimiser_steps = 0
    for _ in range(settings.epochs):
        network.train()
        total = 0.0
        order = torch.randperm(len(steps), generator=generator).tolist()
        for start in range(0, len(order), settings.minibatch_size):
            index = order[start : start + settings.minibatch_size]
            batch = to_device(collate([steps[i].query for i in index]), device)
            chosen = torch.tensor([steps[i].chosen for i in index], device=device)
            logits = network(batch).logits
            if factors is None:
                loss = cross_entropy(logits, batch.cand_mask, chosen)
            else:
                each = -policy_terms(logits, batch.cand_mask, chosen).log_prob
                loss = (factors[index].to(device) * each).mean()
            optimiser.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(network.parameters(), settings.max_grad_norm)
            optimiser.step()
            optimiser_steps += 1
            total += float(loss.detach().cpu()) * len(index)
        epoch_loss.append(total / len(steps))
    size = settings.minibatch_size
    situation = None
    if weights is not None:
        situation = SituationReport(
            weights,
            weight_mean,
            evaluate_groups(network, steps, size, device),
            evaluate_groups(network, held, size, device) if held else None,
        )
    report = RoleReport(
        optimiser_steps,
        tuple(epoch_loss),
        evaluate(network, steps, size, device),
        evaluate(network, held, size, device) if held else None,
        situation,
    )
    return optimiser, report


def own_team_set(stream: str | Path) -> tuple[str, ...]:
    """Sorted unique SHA-256 of the JCS bytes of each own team the examples' observations carry."""
    lines = stream_lines(stream) if isinstance(stream, Path) else iter(stream.split("\n")[:-1])
    return tuple(
        sorted({sha256_hex(jcs(json.loads(line)["observation"]["team"]["team"])) for line in lines})
    )


def check_teacher(config: CloneConfig, manifest: dict[str, Any]) -> None:
    """Refuse a TrainingSet that is not the one the config names, played by its teacher.

    Fail closed: the set must be synthetic and learner-ready, hash (JCS of the manifest) to
    ``teacher.training_set_hash`` and have been played by ``teacher.behaviour_policy_id``,
    which must be a scripted policy (``<name>@<commit>``), never a learned pair.
    """
    teacher = config.teacher_policy_id
    if teacher.startswith(_LEARNED_PREFIX):
        raise CloneError("a learned pair is not a teacher: only a scripted policy is cloned")
    if manifest.get("data_class") != "synthetic":
        raise CloneError("the TrainingSet is not synthetic: the cloning learner refuses it")
    if manifest.get("learner_ready") is not True:
        raise CloneError("the TrainingSet is not learner_ready: the cloning learner refuses it")
    if manifest.get("behaviour_policy_id") != teacher:
        raise CloneError(
            f"the TrainingSet was not played by the teacher {teacher!r}: "
            f"it names {manifest.get('behaviour_policy_id')!r}"
        )
    if sha256_hex(jcs(manifest)) != config.teacher_set_hash:
        raise CloneError(
            "the TrainingSet is not the one the config names (teacher.training_set_hash)"
        )


def situation_groups(config: CloneConfig) -> Groups | None:
    """How the loader reads situation groups for ``config``: with the block, ``threatened`` only
    when ``threatened_ppm`` is set (ADR-010 erratum A13.2); without it, no groups."""
    weights = config.clone.situation_weights
    return None if weights is None else Groups(weights.threatened_ppm is not None)


def split_parts(
    config: CloneConfig, found: Sequence[Trajectory]
) -> dict[str, tuple[list[Step], list[Step]]]:
    """Per role: the training steps, then the held-back ones (per Episode, never by group)."""
    key, ppm = config.clone.holdout_key, config.clone.holdout_ppm
    parts: dict[str, tuple[list[Step], list[Step]]] = {role: ([], []) for role in ROLES}
    for trajectory in found:
        side = 1 if held_back(key, trajectory.episode, ppm) else 0
        parts[trajectory.role][side].extend(trajectory.steps)
    return parts


def clone(
    config: CloneConfig,
    game: GameData,
    stream: str | Path,
    manifest: dict[str, Any],
    device: torch.device | None = None,
) -> Cloned:
    """Train a Preview and a Battle network to imitate the teacher of ``stream``.

    ``stream`` is the text or the path of the stream file; it is read only with its manifest.
    The device is the config's unless given (``configure_learner`` has set the process up).
    """
    check_teacher(config, manifest)
    device = device or torch.device(config.device)
    encoders = {role: Encoder(game, config.model.encoder) for role in ROLES}
    try:
        found = trajectories(
            stream, manifest, encoders, config.teacher_policy_id, groups=situation_groups(config)
        )
    except LearnerDataError as exc:
        raise CloneError(str(exc)) from exc
    parts = split_parts(config, found)
    for role in ROLES:
        if not parts[role][0]:
            raise CloneError(f"no {role} decision is left to train on after the held-back part")
    seeds = {"preview": config.init_preview, "battle": config.init_battle}
    networks = {
        role: init_network(role, game, config.model, seeds[role]).to(device) for role in ROLES
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.learner_seed)
    optimisers: dict[str, torch.optim.Adam] = {}
    roles: dict[str, RoleReport] = {}
    for role in ROLES:
        weights = config.clone.situation_weights if role == "battle" else None
        optimisers[role], roles[role] = _train_role(
            config, networks[role], parts[role], generator, device, weights=weights
        )
        networks[role].eval()
    return Cloned(networks, optimisers, generator, roles, own_team_set(stream))
