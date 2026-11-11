"""The learner reads the ``probe-outcome/1`` rows from the stream (ADR-022 E-22.5.4, slice H5),
with and without the ``history_facts`` option."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.learner import parallel
from areneon_ml.learner.data import _Collector, trajectories
from areneon_ml.learner.trainer import to_device
from areneon_ml.loop.records import FEATURES_OUTCOME_FAMILY, family_label
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, HISTORY_FACTS, read_options

from tests.ml.learner.support import POLICY, edited_config, stream_of
from tests.ml.model.support import SAMPLES, TINY_ENCODER, TINY_MODEL, game

EXAMPLE: dict[str, Any] = json.loads(
    (SAMPLES / "training-example/valid/battle-features-outcome.json").read_text()
)
SET = {k: EXAMPLE["candidate_features"][k] for k in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
BOTH = replace(
    TINY_ENCODER,
    options=read_options("encoder", {CANDIDATE_FEATURES: SET, HISTORY_FACTS: True}),
)


def encoders(config: Any) -> dict[str, Encoder]:
    return {role: Encoder(game(), config) for role in ROLES}


def battle_steps(config: Any) -> Any:
    text, manifest = stream_of([EXAMPLE])
    (battle,) = [
        t for t in trajectories(text, manifest, encoders(config), POLICY) if t.role == "battle"
    ]
    return battle


@pytest.mark.parametrize("config", [ON, BOTH], ids=["outcome", "outcome-history"])
def test_a_stream_with_the_outcomes_becomes_steps_equal_to_a_direct_encode(config: Any) -> None:
    (step,) = battle_steps(config).steps
    direct = Encoder(game(), config).encode(
        {
            "observation": EXAMPLE["observation"],
            "candidates": EXAMPLE["candidates"],
            "candidate_features": EXAMPLE["candidate_features"],
        }
    )
    assert all(torch.equal(v, direct.tensors()[k]) for k, v in step.query.tensors().items())


def test_the_gradient_reaches_the_outcome_and_history_modules() -> None:
    battle = battle_steps(BOTH)
    net = init_network("battle", game(), replace(TINY_MODEL, encoder=BOTH), seed=1)
    out = net(collate([battle.steps[0].query]))
    out.logits[0, battle.steps[0].chosen].backward()
    assert net.cand_features is not None
    assert net.cand_features.weight.grad is not None
    assert net.cand_features.weight.grad.abs().sum() > 0
    for kind in ("outcome_kind", "outcome_pre", "outcome_branch_kind", "history_state"):
        grad = net.embeddings[kind].weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0, kind


def test_the_batch_moves_to_a_device_and_a_worker_sends_it_bit_for_bit() -> None:
    battle = battle_steps(ON)
    moved = to_device(collate([battle.steps[0].query]), torch.device("cpu"))
    assert moved.cand_num is not None
    collector = _Collector(encoders(ON))
    collector.add(json.loads(json.dumps(EXAMPLE)))
    sent = parallel._Piece(collector.collected(), set(), set())
    back = parallel.unpack_piece(parallel.pack_piece(sent))
    before = next(v for k, v in sent.collected.steps.items() if k[1] == "battle")[0]
    after = next(v for k, v in back.collected.steps.items() if k[1] == "battle")[0]
    for name, tensor in before.query.tensors().items():
        assert torch.equal(after.query.tensors()[name], tensor), name


def test_the_family_label_names_the_outcomes_and_the_history() -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["encoder"]["options"] = {CANDIDATE_FEATURES: SET}

    assert family_label(edited_config(edit)) == FEATURES_OUTCOME_FAMILY
    text = " ".join(FEATURES_OUTCOME_FAMILY.split())
    assert "move outcomes" in text
    assert "the player's own battle history" in text
