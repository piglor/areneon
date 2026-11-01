"""The learner reads the ``opponent_sheet`` block from the stream (ADR-027 slice 4, #1152)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.learner import parallel
from areneon_ml.learner.data import _Collector, trajectories
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.model.options import OPPONENT_SHEET, read_options

from tests.ml.learner.support import POLICY, stream_of
from tests.ml.model.support import SAMPLES, TINY_ENCODER, TINY_MODEL, game
from tests.ml.model.test_sheet import HASH, block

BASE: dict[str, Any] = json.loads((SAMPLES / "training-example/valid/battle.json").read_text())
EXAMPLE: dict[str, Any] = {**copy.deepcopy(BASE), "opponent_sheet": block(BASE)}
SET = {"feature_set": "opponent-sheet/1", "feature_set_hash": HASH}
ON = replace(TINY_ENCODER, options=read_options("encoder", {OPPONENT_SHEET: SET}))


def encoders(config: Any) -> dict[str, Encoder]:
    return {role: Encoder(game(), config) for role in ROLES}


def test_a_stream_with_blocks_becomes_steps_that_carry_them() -> None:
    text, manifest = stream_of([EXAMPLE])
    found = trajectories(text, manifest, encoders(ON), POLICY)
    (battle,) = [t for t in found if t.role == "battle"]
    (step,) = battle.steps
    query = {k: EXAMPLE[k] for k in ("observation", "candidates", "opponent_sheet")}
    direct = Encoder(game(), ON).encode(query)
    assert all(torch.equal(v, direct.tensors()[k]) for k, v in step.query.tensors().items())
    plain = Encoder(game(), TINY_ENCODER).encode(BASE)
    assert step.query.mon_num.shape[1] > plain.mon_num.shape[1]


def test_the_network_trains_on_a_stream_with_blocks() -> None:
    text, manifest = stream_of([EXAMPLE])
    (battle,) = [
        t for t in trajectories(text, manifest, encoders(ON), POLICY) if t.role == "battle"
    ]
    net = init_network("battle", game(), replace(TINY_MODEL, encoder=ON), seed=1)
    out = net(collate([battle.steps[0].query]))
    out.logits[0, battle.steps[0].chosen].backward()
    assert net.mon_num.weight.grad is not None
    assert net.mon_num.weight.grad.abs().sum() > 0


def test_a_stream_without_blocks_is_refused_by_an_encoder_that_needs_them() -> None:
    text, manifest = stream_of([BASE])
    with pytest.raises(EncodingError, match="no opponent_sheet"):
        trajectories(text, manifest, encoders(ON), POLICY)


def test_a_block_of_another_source_is_refused() -> None:
    other = replace(
        ON,
        options=read_options("encoder", {OPPONENT_SHEET: {**SET, "feature_set_hash": "4" * 64}}),
    )
    text, manifest = stream_of([EXAMPLE])
    with pytest.raises(EncodingError, match="another feature set"):
        trajectories(text, manifest, encoders(other), POLICY)


def test_a_worker_process_sends_the_block_columns_bit_for_bit() -> None:
    collector = _Collector(encoders(ON))
    collector.add(json.loads(json.dumps(EXAMPLE)))
    sent = parallel._Piece(collector.collected(), set(), set())
    back = parallel.unpack_piece(parallel.pack_piece(sent))
    (before,) = next(v for k, v in sent.collected.steps.items() if k[1] == "battle")[:1]
    (after,) = next(v for k, v in back.collected.steps.items() if k[1] == "battle")[:1]
    assert torch.equal(before.query.mon_num, after.query.mon_num)
    assert torch.equal(before.query.mon_cat, after.query.mon_cat)
