"""The learner state: safetensors plus a JCS sidecar, hashed, restored after checks (ADR-010 §1)."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from functools import cache
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.files import checkpoint_bytes, read_header
from areneon_ml.learner.state import (
    LearnerStateError,
    PackedState,
    pack_state,
    restore_state,
    state_hash,
)
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game


def weights(student: Learner) -> list[bytes]:
    return [checkpoint_bytes(student.networks[role], game()) for role in ROLES]


@cache
def trained() -> tuple[PackedState, list[bytes]]:
    student = learner()
    student.train(trajs(tiny_config()))
    return pack_state(student), weights(student)


def fresh() -> Learner:
    return learner()


def test_a_fresh_learner_packs_its_weights_and_generator_but_no_optimiser_state() -> None:
    packed = pack_state(learner())
    names = set(read_header(packed.tensors)) - {"__metadata__"}
    assert "rng.learner" in names
    assert not [n for n in names if n.startswith("opt.")]
    assert {n for n in names if n.startswith("net.")}
    sidecar = json.loads(packed.sidecar)
    assert sidecar["generation"] == 0
    assert sidecar["optimizer_steps"] == {"battle": {}, "preview": {}}


def test_a_trained_state_holds_the_moments_the_steps_and_the_generation() -> None:
    packed, _ = trained()
    sidecar = json.loads(packed.sidecar)
    assert sidecar["generation"] == 1
    assert sidecar["run_config_hash"] == tiny_config().hash
    assert sidecar["seeds"] == tiny_config().raw["seeds"]
    assert sidecar["ppo"] == tiny_config().raw["ppo"]
    battle = sidecar["optimizer_steps"]["battle"]
    assert battle
    assert set(battle.values()) == {2 * 4}  # 2 epochs of 4 minibatches (55 decisions, 16 each)
    names = set(read_header(packed.tensors))
    for name in battle:
        assert f"opt.battle.exp_avg.{name}" in names
        assert f"opt.battle.exp_avg_sq.{name}" in names


def test_the_hash_binds_the_tensor_bytes_and_the_sidecar() -> None:
    packed, _ = trained()
    assert packed.hash == sha256_hex(
        jcs({"tensors_sha256": sha256_hex(packed.tensors), "sidecar": json.loads(packed.sidecar)})
    )
    assert state_hash(packed.tensors, json.loads(packed.sidecar)) == packed.hash
    changed = json.loads(packed.sidecar)
    changed["generation"] += 1
    assert state_hash(packed.tensors, changed) != packed.hash
    assert state_hash(packed.tensors + b"\0", json.loads(packed.sidecar)) != packed.hash


def test_packing_is_repeatable_byte_for_byte() -> None:
    first, second = learner(), learner()
    first.train(trajs(tiny_config()))
    second.train(trajs(tiny_config()))
    assert pack_state(first) == pack_state(second)


def test_a_restored_learner_continues_exactly_like_one_that_never_stopped() -> None:
    packed, after_one = trained()
    resumed = fresh()
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    assert resumed.generation == 1
    assert weights(resumed) == after_one
    assert pack_state(resumed) == packed
    continuous = fresh()
    continuous.train(trajs(tiny_config()))
    continuous.train(trajs(tiny_config()))
    resumed.train(trajs(tiny_config()))
    assert weights(resumed) == weights(continuous)
    assert pack_state(resumed) == pack_state(continuous)
    assert resumed.generation == 2


def _refused(mutate: Callable[[PackedState], tuple[bytes, bytes, str]], message: str) -> None:
    packed, _ = trained()
    target = fresh()
    before = (weights(target), target.generator.get_state().clone(), target.generation)
    tensors, sidecar, expected = mutate(packed)
    with pytest.raises(LearnerStateError, match=message):
        restore_state(target, tensors, sidecar, expected)
    assert weights(target) == before[0]
    assert target.generator.get_state().equal(before[1])
    assert target.generation == before[2]
    assert not target.optimisers["battle"].state


def _resealed(
    packed: PackedState, edit: Callable[[dict[str, Any]], None]
) -> tuple[bytes, bytes, str]:
    sidecar = json.loads(packed.sidecar)
    edit(sidecar)
    return packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)


def test_a_flipped_tensor_byte_is_refused() -> None:
    def flip(p: PackedState) -> tuple[bytes, bytes, str]:
        data = bytearray(p.tensors)
        data[-1] ^= 1
        return bytes(data), p.sidecar, p.hash

    _refused(flip, "does not hash")


def test_an_edited_sidecar_is_refused() -> None:
    _refused(
        lambda p: (p.tensors, p.sidecar.replace(b'"generation":1', b'"generation":2'), p.hash),
        "does not hash",
    )


def test_the_expected_hash_must_match() -> None:
    _refused(lambda p: (p.tensors, p.sidecar, "0" * 64), "does not hash")


def test_a_sidecar_that_is_not_canonical_json_is_refused() -> None:
    _refused(lambda p: (p.tensors, b"{ }", p.hash), "not an RFC 8785")
    _refused(lambda p: (p.tensors, b"[]", p.hash), "not an RFC 8785")
    _refused(lambda p: (p.tensors, b"\xff", p.hash), "not canonical JSON")
    _refused(lambda p: (p.tensors, b'{"a":1,"a":1}', p.hash), "repeats a key")
    _refused(lambda p: (p.tensors, b'{"a":' + b"[" * 100000, p.hash), "not canonical JSON")


def test_a_state_of_another_run_config_is_refused() -> None:
    packed, _ = trained()
    other = learner(edited_config(lambda raw: raw["seeds"].update(learner=99)))
    with pytest.raises(LearnerStateError, match="another TrainingRunConfig"):
        restore_state(other, packed.tensors, packed.sidecar, packed.hash)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(schema_version=2), "unsupported learner state format"),
        (lambda s: s.update(generation=-1), "not a counter"),
        (lambda s: s.update(generation=True), "not a counter"),
        (lambda s: s.update(generation="1"), "not a counter"),
        (lambda s: s.update(rng={"learner": "elsewhere"}), "generator"),
        (lambda s: s.update(optimizer_steps={"battle": {}}), "name each role once"),
        (lambda s: s.update(optimizer_steps=[]), "name each role once"),
        (
            lambda s: s["optimizer_steps"]["battle"].update(nonsense=1),
            "optimiser state the battle network lacks",
        ),
        (lambda s: s["optimizer_steps"].update(battle=3), "optimiser state the battle network"),
    ],
)
def test_a_sidecar_that_contradicts_the_networks_is_refused(
    edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    _refused(lambda p: _resealed(p, edit), message)


def test_a_bad_step_count_is_refused() -> None:
    def bad(sidecar: dict[str, Any]) -> None:
        battle = sidecar["optimizer_steps"]["battle"]
        battle[next(iter(battle))] = 0

    _refused(lambda p: _resealed(p, bad), "not a positive integer")

    def boolean(sidecar: dict[str, Any]) -> None:
        battle = sidecar["optimizer_steps"]["battle"]
        battle[next(iter(battle))] = True

    _refused(lambda p: _resealed(p, boolean), "not a positive integer")


def test_a_sidecar_that_drops_optimiser_state_the_tensors_hold_is_refused() -> None:
    def drop(sidecar: dict[str, Any]) -> None:
        sidecar["optimizer_steps"]["battle"] = {}

    _refused(lambda p: _resealed(p, drop), "not the expected ones")


def test_tensors_of_the_wrong_shape_or_dtype_are_refused() -> None:
    packed, _ = trained()
    prefix = struct.unpack("<Q", packed.tensors[:8])[0]
    header = json.loads(packed.tensors[8 : 8 + prefix])
    name = next(n for n in header if n.startswith("net.battle."))
    header[name]["dtype"] = "U8"

    def rewrite(p: PackedState) -> tuple[bytes, bytes, str]:
        text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 8)
        data = struct.pack("<Q", len(text)) + text + p.tensors[8 + prefix :]
        return data, p.sidecar, state_hash(data, json.loads(p.sidecar))

    _refused(rewrite, "dtype or shape differs")


def test_the_networks_in_a_state_are_the_networks_of_its_checkpoints() -> None:
    packed, after_one = trained()
    resumed = fresh()
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    assert weights(resumed) == after_one


def test_the_generator_state_is_restored_so_the_next_shuffle_matches() -> None:
    packed, _ = trained()
    resumed = fresh()
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    original = fresh()
    original.train(trajs(tiny_config()))
    assert (
        torch.randperm(10, generator=resumed.generator).tolist()
        == torch.randperm(10, generator=original.generator).tolist()
    )
