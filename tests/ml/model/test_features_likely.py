"""The ``probe-outcome-likely/1`` feature set as encoder input (ADR-022 erratum E-22.7, LT-F2).

The set holds ``probe-outcome/1``'s arrays and ``outcomes`` plus ``probe-damage/3``'s
``belief_threats`` rows (the moves a foe is likely to hold, from a frequency table). The encoder
reads it with both sets' folding: ``/3``'s belief rows and refusal code, ``probe-outcome/1``'s
widened kinds and outcome blocks. Nothing new is invented here, so every other set keeps its
layout, version and tensors. The sample's numbers are invented
(``fixtures/contracts/v1/policy-query/valid/battle-features-likely.json``).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import FORMAT_OPTIONS
from areneon_ml.loop.records import (
    FEATURES_LIKELY_FAMILY,
    FEATURES_OUTCOME_FAMILY,
)
from areneon_ml.model import features as rows_of
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError, Layout, encoder_version
from areneon_ml.model.gamedata import NONE, RESERVED
from areneon_ml.model.network import init_network
from areneon_ml.model.options import CANDIDATE_FEATURES, FEATURE_SETS, read_options
from areneon_ml.worker.features import coverage_problem, damaging_moves, rows_problem

from .support import TINY_ENCODER, TINY_MODEL, game, sample
from .test_features_2 import token_of

SAMPLE = sample("policy-query/valid/battle-features-likely.json")
REFUSED = sample("policy-query/valid/battle-features-likely-refused.json")
OUTCOME = sample("policy-query/valid/battle-features-outcome.json")
THREE = sample("policy-query/valid/battle-features-3.json")
SET = {key: SAMPLE["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}


def config_for(sample_: dict[str, Any]) -> Any:
    named = {key: sample_["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
    return replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: named}))


ON = config_for(SAMPLE)
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
ENCODE_OUTCOME = Encoder(game(), config_for(OUTCOME))
ENCODE_THREE = Encoder(game(), config_for(THREE))
THREATS = SAMPLE["candidate_features"]["threats"]
BELIEF = SAMPLE["candidate_features"]["belief_threats"]


def edited(edit: Any) -> dict[str, Any]:
    q = copy.deepcopy(SAMPLE)
    edit(q["candidate_features"])
    return q


def net(seed: int = 1) -> Any:
    return init_network("battle", game(), ON_MODEL, seed)


def test_the_option_reads_the_likely_set_and_it_has_its_own_layout_and_version() -> None:
    assert "probe-outcome-likely/1" in FEATURE_SETS
    layout = Layout.of(ON)
    assert layout.outcome  # the widened kinds and the outcome columns
    assert layout.thr_num == rows_of.THREAT_NUMBERS_3  # and /3's belief and inclusion columns
    for name in ("reads_arrays", "reads_outcomes", "reads_belief_threats"):
        assert getattr(rows_of, name)("probe-outcome-likely/1")
    assert rows_of.array_names("probe-outcome-likely/1") == (
        *rows_of.ARRAY_NAMES,
        "belief_threats",
        "outcomes",
    )
    # the other sets keep what they read
    assert not rows_of.reads_belief_threats("probe-outcome/1")
    assert not rows_of.reads_outcomes("probe-damage/3")
    assert rows_of.array_names("probe-outcome/1") == (*rows_of.ARRAY_NAMES, "outcomes")
    versions = {encoder_version(game(), config_for(s)) for s in (SAMPLE, OUTCOME, THREE)}
    assert len(versions) == 3
    assert FORMAT_OPTIONS  # a layout-changing option: checkpoint format "2" (the generic rule)


def test_the_shared_columns_are_the_outcome_sets_and_the_belief_rows_are_the_third_sets() -> None:
    encoded = ENCODE.encode(SAMPLE)
    outcome = ENCODE_OUTCOME.encode(OUTCOME)
    # the move parts' outcome columns equal the outcome set's for the same arrays and candidates
    assert encoded.cand_cat is not None
    assert outcome.cand_cat is not None
    assert encoded.cand_cat.tolist() == outcome.cand_cat.tolist()
    assert encoded.cand_num is not None
    assert outcome.cand_num is not None
    assert encoded.cand_num.tolist() == outcome.cand_num.tolist()
    assert encoded.thr_num is not None
    assert encoded.thr_idx is not None
    assert encoded.thr_cat is not None
    assert encoded.thr_num.shape[0] == len(THREATS) + len(BELIEF)
    for k, row in enumerate(BELIEF):
        i = len(THREATS) + k
        own, foe, mega = encoded.thr_idx[i].tolist()
        assert own == token_of(SAMPLE, row["own"], encoded)
        assert foe == encoded.n_own + row["foe"] - 1
        assert mega == int(row["mega"])
        assert encoded.thr_cat[i, 0].item() == game().moves.lookup(row["move"])
        assert encoded.thr_cat[i, 2].item() == NONE
        assert encoded.thr_num[i, 6].item() == 1.0
        assert encoded.thr_num[i, 7].item() == pytest.approx(row["inclusion_ppm"] / 1e6)
    for i in range(len(THREATS)):
        assert encoded.thr_num[i, 6].item() == 0.0
        assert encoded.thr_num[i, 7].item() == 1.0
    assert any(r["kind"] == "unsupported" for r in BELIEF)


def test_the_arrays_are_required_and_the_other_sets_refuse_this_sets_query() -> None:
    with pytest.raises(EncodingError, match="lacks the array belief_threats"):
        ENCODE.encode(edited(lambda f: f.pop("belief_threats")))
    with pytest.raises(EncodingError, match="lacks the array outcomes"):
        ENCODE.encode(edited(lambda f: f.pop("outcomes")))
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
    for other in (ENCODE_OUTCOME, ENCODE_THREE):
        with pytest.raises(EncodingError, match="another feature set"):
            other.encode(SAMPLE)
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE.encode(OUTCOME)


def test_a_refused_decision_reaches_the_field_token_as_its_code_category() -> None:
    encoded = ENCODE.encode(REFUSED)
    ordinary = ENCODE.encode(SAMPLE)
    column = len(Layout.of(ON).field_cat) - 1
    assert Layout.of(ON).field_cat[column] == ("belief_refusal", "belief_refusal")
    assert ordinary.field_cat[column].item() == NONE
    code = REFUSED["candidate_features"]["belief_threats_refused"]
    assert encoded.field_cat[column].item() == RESERVED + rows_of.BELIEF_REFUSALS.index(code)
    assert encoded.thr_idx is not None
    assert encoded.thr_idx.shape[0] == len(THREATS)  # no likely rows
    with pytest.raises(EncodingError, match="no belief_threats"):
        ENCODE.encode(edited(lambda f: f.update(belief_threats_refused="reader_flagged")))


def test_the_network_reads_the_likely_rows_and_the_outcomes() -> None:
    model = net()
    base = model(collate([ENCODE.encode(SAMPLE)])).logits
    assert base.isfinite().any()
    changed = edited(lambda f: [r.update(inclusion_ppm=1) for r in f["belief_threats"]])
    assert not torch.equal(base, model(collate([ENCODE.encode(changed)])).logits)
    dropped = edited(lambda f: f.update(belief_threats=[]))
    assert not torch.equal(base, model(collate([ENCODE.encode(dropped)])).logits)
    target = ENCODE.encode(SAMPLE)
    alone = model(collate([target])).logits
    model(collate([ENCODE.encode(dropped)]))
    assert torch.equal(
        alone, model(collate([target])).logits
    )  # alone and after another: bit-identical


def test_the_worker_reader_and_the_run_label() -> None:
    features = SAMPLE["candidate_features"]
    assert rows_problem(features) is None
    assert coverage_problem(SAMPLE, features, damaging_moves(game())) is None
    assert FEATURES_LIKELY_FAMILY != FEATURES_OUTCOME_FAMILY
    assert "likely to hold" in FEATURES_LIKELY_FAMILY
