"""The learner reads ``candidate_features`` rows from the stream (ADR-022 §3, slice D3)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.learner import parallel
from areneon_ml.learner.data import LearnerDataError, _Collector, trajectories
from areneon_ml.learner.state import LearnerOptionsError, pack_state, restore_state
from areneon_ml.loop.records import FEATURES_FAMILY, family_label, features_hash, policy_id
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, read_options

from tests.ml.learner.support import POLICY, edited_config, learner, stream_of, tiny_config
from tests.ml.model.support import SAMPLES, TINY_ENCODER, TINY_MODEL, game

EXAMPLE: dict[str, Any] = json.loads(
    (SAMPLES / "training-example/valid/battle-features.json").read_text()
)
SET = {k: EXAMPLE["candidate_features"][k] for k in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))


def encoders(config: Any) -> dict[str, Encoder]:
    return {role: Encoder(game(), config) for role in ROLES}


def test_a_stream_with_rows_becomes_steps_that_carry_them() -> None:
    text, manifest = stream_of([EXAMPLE])
    found = trajectories(text, manifest, encoders(ON), POLICY)
    (battle,) = [t for t in found if t.role == "battle"]
    (step,) = battle.steps
    assert step.query.cand_num is not None
    assert step.query.cand_num.shape[0] == len(EXAMPLE["candidates"])
    assert step.query.cand_num.any()
    # The same decision from the A1 query form (the worker's) gives the same tensors.
    query = {
        "observation": EXAMPLE["observation"],
        "candidates": EXAMPLE["candidates"],
        "candidate_features": EXAMPLE["candidate_features"],
    }
    direct = Encoder(game(), ON).encode(query)
    assert all(torch.equal(v, direct.tensors()[k]) for k, v in step.query.tensors().items())


def test_the_network_trains_on_a_stream_with_rows() -> None:
    text, manifest = stream_of([EXAMPLE])
    (battle,) = [
        t for t in trajectories(text, manifest, encoders(ON), POLICY) if t.role == "battle"
    ]
    net = init_network("battle", game(), replace(TINY_MODEL, encoder=ON), seed=1)
    out = net(collate([battle.steps[0].query]))
    out.logits[0, battle.steps[0].chosen].backward()
    assert net.cand_features is not None
    assert net.cand_features.weight.grad is not None
    assert net.cand_features.weight.grad.abs().sum() > 0


def test_a_stream_whose_features_are_not_the_ones_the_encoder_reads_is_refused() -> None:
    text, manifest = stream_of([EXAMPLE])
    with pytest.raises((LearnerDataError, EncodingError), match="candidate_features"):
        trajectories(text, manifest, encoders(TINY_ENCODER), POLICY)
    other = replace(
        ON,
        options=read_options(
            "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set_hash": "4" * 64}}
        ),
    )
    with pytest.raises(EncodingError, match="another feature set"):
        trajectories(text, manifest, encoders(other), POLICY)


def test_a_stream_without_rows_is_refused_by_an_encoder_that_needs_them() -> None:
    plain = copy.deepcopy(EXAMPLE)
    del plain["candidate_features"]
    text, manifest = stream_of([plain])
    with pytest.raises(EncodingError, match="no candidate_features"):
        trajectories(text, manifest, encoders(ON), POLICY)


def test_the_incremental_load_equals_a_fresh_encode_with_rows(tmp_path: Path) -> None:
    text, manifest = stream_of([EXAMPLE])
    path = tmp_path / "stream.jsonl"
    path.write_text(text)
    from_text = trajectories(text, manifest, encoders(ON), POLICY)
    from_file = trajectories(path, manifest, encoders(ON), POLICY)
    for a, b in zip(from_text, from_file, strict=True):
        for x, y in zip(a.steps, b.steps, strict=True):
            assert all(torch.equal(v, y.query.tensors()[k]) for k, v in x.query.tensors().items())


def test_a_worker_sends_the_rows_numbers_bit_for_bit() -> None:
    collector = _Collector(encoders(ON))
    collector.add(json.loads(json.dumps(EXAMPLE)))
    sent = parallel._Piece(collector.collected(), set(), set())
    back = parallel.unpack_piece(parallel.pack_piece(sent))
    (before,) = next(v for k, v in sent.collected.steps.items() if k[1] == "battle")[:1]
    (after,) = next(v for k, v in back.collected.steps.items() if k[1] == "battle")[:1]
    assert before.query.cand_num is not None
    assert after.query.cand_num is not None
    assert torch.equal(before.query.cand_num, after.query.cand_num)
    # Without the option the field is absent and travels as such.
    plain = _Collector(encoders(TINY_ENCODER))
    plain_example = copy.deepcopy(EXAMPLE)
    del plain_example["candidate_features"]
    plain.add(plain_example)
    again = parallel.unpack_piece(
        parallel.pack_piece(parallel._Piece(plain.collected(), set(), set()))
    )
    step = next(v for k, v in again.collected.steps.items() if k[1] == "battle")[0]
    assert step.query.cand_num is None


def feature_config(hash_: str = SET["feature_set_hash"]) -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["encoder"]["options"] = {
            CANDIDATE_FEATURES: {"feature_set": "probe-damage/1", "feature_set_hash": hash_}
        }

    return edited_config(edit)


def test_the_learner_state_records_the_feature_set_and_restores_only_under_it() -> None:
    config = feature_config()
    packed = pack_state(learner(config))
    assert json.loads(packed.sidecar)["model_options"] == {"encoder": {CANDIDATE_FEATURES: SET}}
    target = learner(config)
    restore_state(target, packed.tensors, packed.sidecar, packed.hash)
    assert pack_state(target) == packed
    for other in (feature_config("5" * 64), None):
        target = learner(other) if other is not None else learner()
        with pytest.raises(LearnerOptionsError, match="other model options"):
            restore_state(target, packed.tensors, packed.sidecar, packed.hash)


def test_the_run_config_names_the_family_and_the_policy_id_carries_the_hash() -> None:
    plain, featured = tiny_config(), feature_config()
    assert features_hash(plain) is None
    assert family_label(plain) is None
    assert features_hash(featured) == SET["feature_set_hash"]
    assert family_label(featured) == FEATURES_FAMILY
    assert "damage calculation on assumed opponent sets at decision time" in FEATURES_FAMILY
    a, b = "a" * 64, "b" * 64
    assert policy_id(a, b, features=features_hash(featured)) != policy_id(a, b)
    assert featured.hash != plain.hash
