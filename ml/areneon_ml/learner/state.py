"""The learner state of one generation (ADR-010 §1): safetensors plus a JCS sidecar, never a pickle.

``learner-<g>.safetensors`` holds the weights of both networks, the Adam moment
tensors and the learner generator's state tensor; ``learner-<g>.json`` holds the
generation counter, the optimiser step counts, the run config's hash and the
hyperparameters in force. Every other source of randomness in a generation is
seeded from the plan, so the generator state is the only RNG state there is.
``learner_state_hash`` is the SHA-256 of ``JCS({tensors_sha256, sidecar})`` and
goes into the CheckpointRecord. A restore loads only through safetensors and a
JSON parser, recomputes both hashes and refuses on any mismatch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from areneon_contracts.contracts import CanonicalError, jcs, sha256_hex

from areneon_ml.checkpoint.files import CheckpointFileError, pack_safetensors, unpack_safetensors
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.network import ROLES, ModelConfig
from areneon_ml.model.options import options_dict

STATE_FORMAT = 1
_NAME = re.compile(r"[A-Za-z0-9_.]+")


class LearnerStateError(ValueError):
    """A learner state is missing, damaged, or not the one asked for."""


@dataclass(frozen=True)
class PackedState:
    tensors: bytes
    sidecar: bytes
    hash: str


def state_hash(tensors: bytes, sidecar: Mapping[str, Any]) -> str:
    """``learner_state_hash``: SHA-256 of ``JCS({tensors_sha256, sidecar})``."""
    return sha256_hex(jcs({"tensors_sha256": sha256_hex(tensors), "sidecar": dict(sidecar)}))


def _parameter_names(learner: Learner, role: str) -> list[str]:
    return [name for name, _ in learner.networks[role].named_parameters()]


def gather_tensors(
    networks: Mapping[str, torch.nn.Module],
    optimisers: Mapping[str, torch.optim.Optimizer],
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    """The named tensors of a learner state (weights, Adam moments, generator) and step counts."""
    tensors: dict[str, torch.Tensor] = {}
    steps: dict[str, dict[str, int]] = {}
    for role in ROLES:
        for name, tensor in networks[role].state_dict().items():
            tensors[f"net.{role}.{name}"] = tensor
        optimiser = optimisers[role]
        steps[role] = {}
        for name, parameter in networks[role].named_parameters():
            found = optimiser.state.get(parameter)
            if found:
                steps[role][name] = int(found["step"].item())
                tensors[f"opt.{role}.exp_avg.{name}"] = found["exp_avg"]
                tensors[f"opt.{role}.exp_avg_sq.{name}"] = found["exp_avg_sq"]
    tensors["rng.learner"] = generator.get_state()
    return tensors, steps


class LearnerOptionsError(LearnerStateError):
    """A learner state was made under other model options than the learner's (A4 item 2)."""


def model_options_of(model: ModelConfig) -> dict[str, Any]:
    """The options a sidecar records: ``{}`` without any, else ``network`` and/or ``encoder``."""
    out: dict[str, Any] = {}
    if model.options:
        out["network"] = options_dict(model.options)
    if model.encoder.options:
        out["encoder"] = options_dict(model.encoder.options)
    return out


def pack_state(learner: Learner) -> PackedState:
    """The two files' bytes for the learner as it is now."""
    tensors, steps = gather_tensors(learner.networks, learner.optimisers, learner.generator)
    sidecar = {
        "schema_version": STATE_FORMAT,
        "generation": learner.generation,
        "run_config_hash": learner.config.hash,
        "seeds": learner.config.raw["seeds"],
        "ppo": learner.config.raw["ppo"],
        "optimizer_steps": steps,
        "rng": {"learner": "rng.learner"},
    }
    model_options = model_options_of(learner.config.model)
    if model_options:  # the key is omitted without options, so such a state keeps its bytes
        sidecar["model_options"] = model_options
    if learner.last_rate_micro is not None:  # linear schedule only: the key is otherwise omitted
        sidecar["learning_rate_micro"] = learner.last_rate_micro
    if learner.last_anchor_beta_micro is not None:  # ppo.anchor only: otherwise the key is omitted
        sidecar["anchor_beta_micro"] = learner.last_anchor_beta_micro
    data = pack_safetensors(tensors)
    return PackedState(data, jcs(sidecar), state_hash(data, sidecar))


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise LearnerStateError("the sidecar repeats a key")
    return dict(pairs)


def _sidecar(raw: bytes) -> dict[str, Any]:
    try:
        found = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
        canonical = jcs(found)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, CanonicalError) as exc:
        raise LearnerStateError("the sidecar is not canonical JSON") from exc
    if not isinstance(found, dict) or canonical != raw:
        raise LearnerStateError("the sidecar is not an RFC 8785 (JCS) object")
    return found


def _steps(sidecar: Mapping[str, Any], learner: Learner) -> dict[str, dict[str, int]]:
    """The optimiser step counts the sidecar claims, checked against the networks' parameters."""
    claimed = sidecar.get("optimizer_steps")
    if not isinstance(claimed, dict) or set(claimed) != set(ROLES):
        raise LearnerStateError("the sidecar's optimizer_steps must name each role once")
    out: dict[str, dict[str, int]] = {}
    for role in ROLES:
        counts = claimed[role]
        names = set(_parameter_names(learner, role))
        if not isinstance(counts, dict) or not set(counts) <= names:
            raise LearnerStateError(
                f"the sidecar names an optimiser state the {role} network lacks"
            )
        for name, count in counts.items():
            if (
                not _NAME.fullmatch(name)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise LearnerStateError(
                    f"the sidecar's step count for {name} is not a positive integer"
                )
        out[role] = dict(counts)
    return out


def _check_sidecar(described: Mapping[str, Any], learner: Learner) -> int:
    """What the sidecar must say of itself; the generation it holds."""
    if described.get("schema_version") != STATE_FORMAT:
        raise LearnerStateError(
            f"unsupported learner state format {described.get('schema_version')!r}"
        )
    if described.get("model_options") != (model_options_of(learner.config.model) or None):
        raise LearnerOptionsError("the learner state was made under other model options")
    if described.get("run_config_hash") != learner.config.hash:
        raise LearnerStateError("the learner state was made under another TrainingRunConfig")
    generation = described.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise LearnerStateError("the sidecar's generation is not a counter")
    if described.get("rng") != {"learner": "rng.learner"}:
        raise LearnerStateError("the sidecar does not describe the learner generator")
    schedule = learner.config.ppo.schedule
    if schedule is None or generation == 0:
        expected_rate = None
    else:
        expected_rate = schedule.rate_micro(generation - 1)
    if described.get("learning_rate_micro") != expected_rate or (
        expected_rate is None and "learning_rate_micro" in described
    ):
        raise LearnerStateError("the sidecar's learning rate is not the schedule's rate")
    expected_beta = None if generation == 0 else learner.anchor_beta_micro(generation - 1)
    if described.get("anchor_beta_micro") != expected_beta or (
        expected_beta is None and "anchor_beta_micro" in described
    ):
        raise LearnerStateError("the sidecar's anchor weight is not the schedule's beta")
    return generation


def _apply_schedule(learner: Learner) -> None:
    """After a restore: the rate and beta of the update just done, as an uninterrupted run holds."""
    learner.last_anchor_beta_micro = (
        None if learner.generation == 0 else learner.anchor_beta_micro(learner.generation - 1)
    )
    schedule = learner.config.ppo.schedule
    if schedule is None or learner.generation == 0:
        return
    learner.last_rate_micro = schedule.rate_micro(learner.generation - 1)
    for optimiser in learner.optimisers.values():
        for group in optimiser.param_groups:
            group["lr"] = schedule.rate(learner.generation - 1)


def restore_state(learner: Learner, tensors: bytes, sidecar: bytes, expected_hash: str) -> None:
    """Load a learner state into ``learner``, which must be built from the same config.

    Refused: a pair whose hash is not ``expected_hash``, a sidecar of another run config
    or format, and tensors that are not exactly the networks', optimiser moments and generator
    state the sidecar describes. The learner is changed only once everything has been checked.
    """
    described = _sidecar(sidecar)
    if state_hash(tensors, described) != expected_hash:
        raise LearnerStateError(
            "the learner state does not hash to the expected learner_state_hash"
        )
    generation = _check_sidecar(described, learner)
    steps = _steps(described, learner)
    expected: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
        "rng.learner": (torch.uint8, tuple(learner.generator.get_state().shape))
    }
    for role in ROLES:
        network = learner.networks[role]
        for name, tensor in network.state_dict().items():
            expected[f"net.{role}.{name}"] = (torch.float32, tuple(tensor.shape))
        for name, parameter in network.named_parameters():
            if name in steps[role]:
                for moment in ("exp_avg", "exp_avg_sq"):
                    expected[f"opt.{role}.{moment}.{name}"] = (
                        torch.float32,
                        tuple(parameter.shape),
                    )
    try:
        loaded = unpack_safetensors(tensors, expected)
    except CheckpointFileError as exc:
        raise LearnerStateError(str(exc)) from exc
    for role in ROLES:
        network, optimiser = learner.networks[role], learner.optimisers[role]
        prefix = f"net.{role}."
        network.load_state_dict(
            {
                name[len(prefix) :]: tensor
                for name, tensor in loaded.items()
                if name.startswith(prefix)
            },
            strict=True,
        )
        state: dict[int, dict[str, torch.Tensor]] = {}
        for index, (name, parameter) in enumerate(network.named_parameters()):
            if name in steps[role]:
                state[index] = {
                    "step": torch.tensor(float(steps[role][name])),
                    "exp_avg": loaded[f"opt.{role}.exp_avg.{name}"].to(parameter.device),
                    "exp_avg_sq": loaded[f"opt.{role}.exp_avg_sq.{name}"].to(parameter.device),
                }
        optimiser.load_state_dict(
            {"state": state, "param_groups": optimiser.state_dict()["param_groups"]}
        )
    learner.generator.set_state(loaded["rng.learner"])
    learner.generation = generation
    _apply_schedule(learner)
