"""The learner reads the ``probe-damage/2`` arrays from the stream (ADR-022 E-22.3, #1175)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.learner import parallel
from areneon_ml.learner.data import _Collector, trajectories
from areneon_ml.learner.trainer import to_device
from areneon_ml.loop.records import FEATURES_2_FAMILY, FEATURES_FAMILY, family_label
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, read_options

from tests.ml.learner.support import POLICY, edited_config, stream_of
from tests.ml.model.support import SAMPLES, TINY_ENCODER, TINY_MODEL, game

EXAMPLE: dict[str, Any] = json.loads(
    (SAMPLES / "training-example/valid/battle-features-2.json").read_text()
)
SET = {k: EXAMPLE["candidate_features"][k] for k in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
ARRAYS = ("thr_idx", "thr_cat", "thr_num", "foe_idx", "foe_cat", "foe_num")


def encoders(config: Any) -> dict[str, Encoder]:
    return {role: Encoder(game(), config) for role in ROLES}


def battle_steps() -> Any:
    text, manifest = stream_of([EXAMPLE])
    (battle,) = [
        t for t in trajectories(text, manifest, encoders(ON), POLICY) if t.role == "battle"
    ]
    return battle


def test_a_stream_with_the_arrays_becomes_steps_that_carry_them() -> None:
    (step,) = battle_steps().steps
    for name in ARRAYS:
        assert getattr(step.query, name) is not None, name
    assert step.query.thr_idx.shape[0] == len(EXAMPLE["candidate_features"]["threats"])
    query = {
        "observation": EXAMPLE["observation"],
        "candidates": EXAMPLE["candidates"],
        "candidate_features": EXAMPLE["candidate_features"],
    }
    direct = Encoder(game(), ON).encode(query)
    assert all(torch.equal(v, direct.tensors()[k]) for k, v in step.query.tensors().items())


def test_the_gradient_reaches_every_array_module() -> None:
    battle = battle_steps()
    net = init_network("battle", game(), replace(TINY_MODEL, encoder=ON), seed=1)
    out = net(collate([battle.steps[0].query]))
    out.logits[0, battle.steps[0].chosen].backward()
    for module in (
        net.threat_num, net.threat_row, net.threat_pool, net.foe_num, net.foe_row, net.foe_pool,
        net.cand_features,
    ):  # fmt: skip
        assert module is not None
        assert module.weight.grad is not None
        assert module.weight.grad.abs().sum() > 0


def test_the_batch_moves_to_a_device_with_its_arrays() -> None:
    battle = battle_steps()
    moved = to_device(collate([battle.steps[0].query]), torch.device("cpu"))
    for name in ARRAYS:
        assert getattr(moved, name) is not None
    plain = Encoder(game(), TINY_ENCODER).encode(
        {
            "observation": EXAMPLE["observation"],
            "candidates": EXAMPLE["candidates"],
        }
    )
    assert to_device(collate([plain]), torch.device("cpu")).thr_idx is None


def test_a_worker_sends_the_arrays_bit_for_bit() -> None:
    collector = _Collector(encoders(ON))
    collector.add(json.loads(json.dumps(EXAMPLE)))
    sent = parallel._Piece(collector.collected(), set(), set())
    back = parallel.unpack_piece(parallel.pack_piece(sent))
    before = next(v for k, v in sent.collected.steps.items() if k[1] == "battle")[0]
    after = next(v for k, v in back.collected.steps.items() if k[1] == "battle")[0]
    for name in ARRAYS:
        assert torch.equal(getattr(before.query, name), getattr(after.query, name)), name


def test_a_stream_of_the_first_set_is_refused_by_an_encoder_of_the_second() -> None:
    text, manifest = stream_of([EXAMPLE])
    first = replace(
        ON,
        options=read_options(
            "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/1"}}
        ),
    )
    with pytest.raises(Exception, match=r"another feature set|arrays"):
        trajectories(text, manifest, encoders(first), POLICY)


def test_the_family_label_names_the_turn_order_for_the_second_set() -> None:
    def config(name: str) -> Any:
        def edit(raw: dict[str, Any]) -> None:
            raw["model"]["encoder"]["options"] = {
                CANDIDATE_FEATURES: {
                    "feature_set": name,
                    "feature_set_hash": SET["feature_set_hash"],
                }
            }

        return edited_config(edit)

    assert family_label(config("probe-damage/1")) == FEATURES_FAMILY
    assert family_label(config("probe-damage/2")) == FEATURES_2_FAMILY
    assert "damage calculation and turn order on assumed opponent sets" in FEATURES_2_FAMILY
