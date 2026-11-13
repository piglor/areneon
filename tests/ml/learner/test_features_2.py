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
from areneon_ml.loop.records import (
    FEATURES_2_FAMILY,
    FEATURES_3_FAMILY,
    FEATURES_BELIEF_FAMILY,
    FEATURES_FAMILY,
    family_label,
)
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.features import reads_arrays
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, FEATURE_SETS, read_options

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
    assert family_label(config("probe-damage/3")) == FEATURES_3_FAMILY
    assert "damage under the assumed foe set, move likelihood from the belief" in (
        " ".join(FEATURES_3_FAMILY.split())
    )
    assert "damage calculation and turn order on assumed opponent sets" in FEATURES_2_FAMILY


def test_the_belief_set_is_a_registered_set_read_like_the_first_and_labelled_as_such() -> None:
    name = "probe-damage-belief/1"
    assert name in FEATURE_SETS
    # its rows have probe-damage/1's shape: no arrays for the encoder to read
    assert not reads_arrays(name)

    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["encoder"]["options"] = {
            CANDIDATE_FEATURES: {"feature_set": name, "feature_set_hash": SET["feature_set_hash"]}
        }

    assert family_label(edited_config(edit)) == FEATURES_BELIEF_FAMILY
    assert "sampled opponent worlds" in FEATURES_BELIEF_FAMILY


# -- the Threatened flag through the real loader (ADR-010 erratum A11.2, #1366) -----------------


def with_rows(threat_ohko: int, relation: str, priority: int) -> dict[str, Any]:
    """The sample with own slot 0 reached by foe 1's thunderbolt (``threat_ohko`` rolls), the
    speed relation of that pair and the foe move's priority set as given."""
    example = json.loads(json.dumps(EXAMPLE))
    rows = example["candidate_features"]
    for row in rows["threats"]:
        if (row["own"], row["foe"], row["move"]) == (0, 1, "thunderbolt"):
            row["ohko_rolls"] = threat_ohko
    for row in rows["speeds"]:
        if (row["own"], row["foe"]) == (0, 1):
            row["relation"] = relation
    for row in rows["foe_priorities"]:
        if (row["foe"], row["move"]) == (1, "thunderbolt"):
            row["priority"] = priority
    return example


CASES = [
    (0, "foe_first", 0, False),  # no roll reaches the HP
    (5, "foe_first", 0, True),  # reaches, and the foe is faster at priority 0
    (5, "own_first", 0, False),  # reaches but acts after
    (5, "tie", 0, False),
    (5, "own_first", 1, True),  # priority above 0 acts first whatever the speed
    (5, "foe_first", -1, False),  # priority below 0 acts last
]


def test_the_loader_flags_a_step_exactly_as_classify_slot_says() -> None:
    from areneon_ml.learner.situation import classify_slot  # noqa: PLC0415

    for ohko, relation, priority, expected in CASES:
        example = with_rows(ohko, relation, priority)
        assert (classify_slot(example["candidate_features"], 0) == "threatened") is expected
        text, manifest = stream_of([example])
        trajs = trajectories(text, manifest, encoders(ON), POLICY)
        (battle,) = [t for t in trajs if t.role == "battle"]
        assert [s.threatened for s in battle.steps] == [expected], (ohko, relation, priority)
        # the worker transport keeps the flag
        collector = _Collector(encoders(ON))
        collector.add(json.loads(json.dumps(example)))
        sent = parallel._Piece(collector.collected(), set(), set())
        back = parallel.unpack_piece(parallel.pack_piece(sent))
        flags = [
            s.threatened for k, v in back.collected.steps.items() if k[1] == "battle" for s in v
        ]
        assert flags == [expected]


def test_a_stream_without_the_arrays_has_no_flag() -> None:
    plain = json.loads(json.dumps(EXAMPLE))
    del plain["candidate_features"]
    text, manifest = stream_of([plain])
    trajs = trajectories(text, manifest, encoders(TINY_ENCODER), POLICY)
    assert [s.threatened for t in trajs for s in t.steps] == [None] * sum(
        len(t.steps) for t in trajs
    )
