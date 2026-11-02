"""The ``probe-damage/3`` belief threats as encoder inputs (ADR-022, the opponent model's moves).

``belief_threats`` rows (a move the belief says a foe may hold, with its inclusion) fold into the
same threat rows the network already pools onto the own and foe tokens, with two more columns:
``belief`` (1 for such a row, 0 for a revealed move's) and ``inclusion`` (the belief's
probability, 1 for a revealed move). ``/2`` keeps its layout, version and tensors. The sample's
numbers are invented (``fixtures/contracts/v1/policy-query/valid/battle-features-3.json``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.model import features as rows_of
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError, Layout, encoder_version
from areneon_ml.model.gamedata import NONE, RESERVED
from areneon_ml.model.network import init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, read_options
from areneon_ml.worker.features import coverage_problem, damaging_moves, rows_problem

from .support import TINY_ENCODER, TINY_MODEL, game, sample
from .test_features_2 import SAMPLE as SAMPLE2
from .test_features_2 import token_of

SAMPLE = sample("policy-query/valid/battle-features-3.json")
SET = {key: SAMPLE["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
SET2 = {key: SAMPLE2["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
TWO = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET2}))
ENCODE2 = Encoder(game(), TWO)
THREATS = SAMPLE["candidate_features"]["threats"]
BELIEF = SAMPLE["candidate_features"]["belief_threats"]


def edited(edit: Any) -> dict[str, Any]:
    q = copy.deepcopy(SAMPLE)
    edit(q["candidate_features"])
    return q


def net(seed: int = 1) -> Any:
    return init_network("battle", game(), ON_MODEL, seed)


def test_the_option_reads_the_third_set_and_it_has_its_own_layout_and_version() -> None:
    assert read_options("encoder", {CANDIDATE_FEATURES: SET})
    layout = Layout.of(ON)
    assert layout.thr_num == rows_of.THREAT_NUMBERS_3
    assert layout.thr_num[-2:] == ("belief", "inclusion")
    two = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder",
            {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/2"}},
        ),
    )
    assert Layout.of(two).thr_num == rows_of.THREAT_NUMBERS  # /2 keeps its layout
    assert encoder_version(game(), ON) != encoder_version(game(), two)
    # the `/2` layout bytes carry no belief word
    assert "belief" not in str(Layout.of(two).as_dict())
    assert rows_of.reads_arrays("probe-damage/3")
    assert rows_of.reads_belief_threats("probe-damage/3")
    assert not rows_of.reads_belief_threats("probe-damage/2")
    assert rows_of.array_names("probe-damage/3") == (*rows_of.ARRAY_NAMES, "belief_threats")
    assert rows_of.array_names("probe-damage/1") == ()


def test_revealed_threats_keep_their_numbers_and_say_belief_zero_inclusion_one() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.thr_num is not None
    assert encoded.thr_idx is not None
    assert encoded.thr_num.shape[0] == len(THREATS) + len(BELIEF)
    # the /2 encoder on the /2 sample gives the same first six numbers per revealed row
    plain = ENCODE2.encode(SAMPLE2)
    assert plain.thr_num is not None
    for i in range(len(THREATS)):
        assert encoded.thr_num[i, :6].tolist() == pytest.approx(plain.thr_num[i, :6].tolist())
        assert encoded.thr_num[i, 6].item() == 0.0  # not the belief's
        assert encoded.thr_num[i, 7].item() == 1.0  # certain: the channel showed it


def test_a_belief_row_names_its_pokemon_foe_and_move_and_carries_its_inclusion() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.thr_idx is not None
    assert encoded.thr_cat is not None
    assert encoded.thr_num is not None
    for k, row in enumerate(BELIEF):
        i = len(THREATS) + k
        own, foe, mega = encoded.thr_idx[i].tolist()
        assert own == token_of(SAMPLE, row["own"], encoded)
        assert foe == encoded.n_own + row["foe"] - 1
        assert mega == int(row["mega"])
        assert encoded.thr_cat[i, 0].item() == game().moves.lookup(row["move"])
        assert encoded.thr_cat[i, 2].item() == NONE  # no foe priority row prices a belief move
        assert encoded.thr_num[i, 5].item() == 0.0
        assert encoded.thr_num[i, 6].item() == 1.0  # the belief's
        assert encoded.thr_num[i, 7].item() == pytest.approx(row["inclusion_ppm"] / 1e6)
        if row["kind"] == "computed":
            assert encoded.thr_cat[i, 1].item() == RESERVED
            assert encoded.thr_num[i, 0].item() == pytest.approx(row["mean_ppm"] / 1e6)
        else:
            assert encoded.thr_cat[i, 1].item() == RESERVED + 1 + rows_of.ALL_REASONS.index(
                row["reason"]
            )
            assert not encoded.thr_num[i, :5].any()
    assert any(r["kind"] == "unsupported" for r in BELIEF)


def test_the_arrays_are_required_and_a_belief_row_may_not_repeat_a_revealed_move() -> None:
    with pytest.raises(EncodingError, match="lacks the array belief_threats"):
        ENCODE.encode(edited(lambda f: f.pop("belief_threats")))
    first = THREATS[0]
    repeat = {
        **copy.deepcopy(BELIEF[0]),
        "foe": first["foe"],
        "move": first["move"],
        "own": first["own"],
        "mega": first["mega"],
    }
    with pytest.raises(EncodingError, match="repeats a move"):
        ENCODE.encode(edited(lambda f: f.update(belief_threats=[repeat, *f["belief_threats"]])))
    twice = edited(lambda f: f.update(belief_threats=[BELIEF[0], *f["belief_threats"]]))
    with pytest.raises(EncodingError, match="twice"):
        ENCODE.encode(twice)


def test_the_third_set_s_query_is_refused_by_a_second_set_checkpoint_and_the_reverse() -> None:
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE2.encode(SAMPLE)
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE.encode(SAMPLE2)
    # a /1 value carrying the array is refused
    one = Encoder(
        game(),
        replace(
            TINY_ENCODER,
            options=read_options(
                "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/1"}}
            ),
        ),
    )
    carrying = copy.deepcopy(SAMPLE)
    carrying["candidate_features"]["feature_set"] = "probe-damage/1"
    with pytest.raises(EncodingError):
        one.encode(carrying)


def test_the_network_reads_the_belief_rows() -> None:
    model = net()
    base = model(collate([ENCODE.encode(SAMPLE)])).logits
    assert base.isfinite().any()
    changed = edited(lambda f: [r.update(inclusion_ppm=1) for r in f["belief_threats"]])
    assert not torch.equal(base, model(collate([ENCODE.encode(changed)])).logits)
    dropped = edited(lambda f: f.update(belief_threats=[]))
    assert not torch.equal(base, model(collate([ENCODE.encode(dropped)])).logits)
    # alone is repeatable, and after another query of the same layout it is bit-identical
    target = ENCODE.encode(SAMPLE)
    other = ENCODE.encode(dropped)
    alone = model(collate([target])).logits
    assert torch.equal(alone, model(collate([target])).logits)
    after = model(collate([other])).logits
    assert after.shape[0] == 1
    assert torch.equal(alone, model(collate([target])).logits)


def test_the_worker_reader_holds_the_same_rules_as_the_stream_reader() -> None:
    features = SAMPLE["candidate_features"]
    assert rows_problem(features) is None
    assert coverage_problem(SAMPLE, features, damaging_moves(game())) is None
    problem = rows_problem({**features, "feature_set": "probe-damage/2"})
    assert problem is not None
    assert "present exactly" in problem
    no_belief = {k: v for k, v in features.items() if k != "belief_threats"}
    assert "present exactly" in (rows_problem(no_belief) or "")
    assert "sorted and unique" in (
        rows_problem({**features, "belief_threats": list(reversed(BELIEF))}) or ""
    )
    first = THREATS[0]
    repeat = {**BELIEF[0], "foe": first["foe"], "move": first["move"]}
    assert "repeats a move" in (rows_problem({**features, "belief_threats": [repeat]}) or "")
    foreign = {**features, "belief_threats": [*BELIEF, {**BELIEF[0], "own": 5}]}
    assert "belief_threats name an own Pokemon" in (
        coverage_problem(SAMPLE, foreign, damaging_moves(game())) or ""
    )


# -- a refused belief is stated, never a silent empty list --------------------------------------

REFUSED = sample("policy-query/valid/battle-features-3-refused.json")


def test_the_refusal_codes_are_the_contract_s_enum() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[3] / "packages/schema/contracts/v1/policy-query.schema.json"
        ).read_text()
    )
    enum = schema["$defs"]["CandidateFeatures"]["properties"]["belief_threats_refused"]["enum"]
    assert list(rows_of.BELIEF_REFUSALS) == enum


def test_a_refused_decision_reaches_the_field_token_as_its_code_category() -> None:
    encoded = ENCODE.encode(REFUSED)
    ordinary = ENCODE.encode(SAMPLE)
    column = len(Layout.of(ON).field_cat) - 1
    assert Layout.of(ON).field_cat[column] == ("belief_refusal", "belief_refusal")
    assert ordinary.field_cat[column].item() == NONE
    code = REFUSED["candidate_features"]["belief_threats_refused"]
    assert encoded.field_cat[column].item() == RESERVED + rows_of.BELIEF_REFUSALS.index(code)
    assert encoded.thr_idx is not None
    assert encoded.thr_idx.shape[0] == len(THREATS)  # no belief rows
    # the network tells "refused" from "no likely threats"
    model = net()
    quiet = copy.deepcopy(REFUSED)
    del quiet["candidate_features"]["belief_threats_refused"]
    assert not torch.equal(
        model(collate([encoded])).logits, model(collate([ENCODE.encode(quiet)])).logits
    )
    # each code is its own category
    other = copy.deepcopy(REFUSED)
    other["candidate_features"]["belief_threats_refused"] = "reader_flagged"
    assert not torch.equal(
        model(collate([encoded])).logits, model(collate([ENCODE.encode(other)])).logits
    )


def test_a_refusal_with_rows_or_outside_the_third_set_is_refused() -> None:
    with pytest.raises(EncodingError, match="no belief_threats"):
        ENCODE.encode(edited(lambda f: f.update(belief_threats_refused="reader_flagged")))
    for name in ("refused-with-rows", "refused-under-2"):
        bad = sample(f"policy-query/reader-invalid/features-3-{name}.json")
        assert "belief_threats_refused" in (rows_problem(bad["candidate_features"]) or "")
    # the other layouts have no refusal column: /2 keeps its field columns
    two = Layout.of(
        replace(
            TINY_ENCODER,
            options=read_options(
                "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/2"}}
            ),
        )
    )
    assert all(kind != "belief_refusal" for _, kind in two.field_cat)
