"""The ``candidate_features`` encoder option (ADR-022 tier A, slice D3, #1028).

A battle query under a ``probe-damage/1`` checkpoint carries the rows of the engine's damage
calculation (``PolicyQuery.candidate_features``). The encoder maps each damaging move part of a
candidate to its rows (one group per foe location), an unsupported row to its own category with
no number, and refuses a query without the rows it needs. The sample rows are invented numbers
(``fixtures/contracts/v1/policy-query/valid/battle-features.json``).
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import (
    FORMAT_OPTIONS,
    CheckpointOptionsError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    read_header,
)
from areneon_ml.model import features as rows_of
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import (
    BASE_CAND_COLUMNS,
    Encoder,
    EncodingError,
    Layout,
    encoder_version,
)
from areneon_ml.model.gamedata import NONE, RESERVED
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    FeatureSetOption,
    OptionError,
    feature_set,
    options_dict,
    read_options,
)
from areneon_ml.worker.view import encoder_input, view_digest

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample
from .test_options import TODAY_ENCODER_VERSION

SAMPLE = sample("policy-query/valid/battle-features.json")
SET = {key: SAMPLE["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
WIDTH = len(rows_of.NUMBERS) * 2  # numbers of one move part: five per foe location
KIND = BASE_CAND_COLUMNS  # the first feature category column (no matchup option here)
COMPUTED, QUERY_KIND, RANDOM, SCENARIO = (RESERVED + k for k in range(4))
SOURCES = Path(__file__).resolve().parents[3] / "ml" / "areneon_ml" / "model"


def query(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(SAMPLE), **changes}


def candidate(q: dict[str, Any], choice: str) -> int:
    return [c["choice"] for c in q["candidates"]].index(choice)


def part(q: dict[str, Any], choice: str, slot: int) -> tuple[list[int], list[float]]:
    """The two feature categories and the ten numbers of one slot action."""
    encoded = ENCODE.encode(q)
    i = candidate(q, choice)
    assert encoded.cand_num is not None
    return (
        encoded.cand_cat[i, slot, KIND : KIND + 2].tolist(),
        encoded.cand_num[i, slot].tolist(),
    )


def rows_without(q: dict[str, Any], keep: Any) -> dict[str, Any]:
    out = copy.deepcopy(q)
    out["candidate_features"]["rows"] = [r for r in q["candidate_features"]["rows"] if keep(r)]
    return out


# -- the option --------------------------------------------------------------------------------


def test_the_option_reads_an_object_with_a_supported_set_and_a_hash() -> None:
    assert feature_set(ON.options) == FeatureSetOption(**SET)
    assert options_dict(ON.options) == {CANDIDATE_FEATURES: SET}
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL
    for bad in (
        True,
        1,
        {"feature_set": "probe-damage/1"},
        {**SET, "extra": 1},
        {**SET, "feature_set": "probe-damage/2"},
        {**SET, "feature_set_hash": "A" * 64},
        {**SET, "feature_set_hash": "0" * 63},
    ):
        with pytest.raises(OptionError):
            read_options("encoder", {CANDIDATE_FEATURES: bad})
    with pytest.raises(ConfigError, match="candidate_features"):
        ModelConfig.from_dict(
            {
                **TINY_MODEL.as_dict(),
                "encoder": {**TINY_ENCODER.as_dict(), "options": {CANDIDATE_FEATURES: 3}},
            }
        )


def test_without_the_option_the_layout_the_version_and_the_tensors_are_today_s() -> None:
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert "cand_num" not in Layout.of(TINY_ENCODER).as_dict()
    plain = encoder().encode(sample("policy-query/valid/battle.json"))
    assert plain.cand_num is None
    assert "cand_num" not in plain.tensors()


def test_the_hash_moves_the_encoder_version() -> None:
    other = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set_hash": "1" * 64}}
        ),
    )
    versions = {encoder_version(game(), c) for c in (TINY_ENCODER, ON, other)}
    assert len(versions) == 3


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    data = checkpoint_bytes(init_network("battle", game(), ON_MODEL, seed=1), game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)
    other = replace(
        ON_MODEL,
        encoder=replace(
            ON,
            options=read_options(
                "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set_hash": "2" * 64}}
            ),
        ),
    )
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=other)


# -- rows to columns -----------------------------------------------------------------------------


def test_a_single_target_move_fills_the_group_of_its_foe() -> None:
    q = query()
    # Slot 0, move 1, foe 1: computed, no OHKO count (null), so the known flag is 0.
    kinds, numbers = part(q, "move 1 1, move 1 1", 0)
    assert kinds == [COMPUTED, NONE]
    assert numbers == pytest.approx([0.15, 0.18, 0.12, 0.0, 0.0] + [0.0] * 5)
    # Slot 0, move 2, foe 2: the second group; OHKO count 0 of 16 is known.
    kinds, numbers = part(q, "move 2 2, move 1 1", 0)
    assert kinds == [NONE, COMPUTED]
    assert numbers == pytest.approx([0.0] * 5 + [0.1875, 0.2175, 0.1575, 1.0, 0.0])


def test_a_spread_move_fills_one_group_per_foe_it_hits() -> None:
    q = query()
    # Slot 1, move 4 (Heat Wave, all adjacent foes) has a row for each foe.
    kinds, numbers = part(q, "move 1 1, move 4", 1)
    assert kinds == [COMPUTED, COMPUTED]
    assert all(
        n > 0 for n in (numbers[0], numbers[1], numbers[2], numbers[5], numbers[6], numbers[7])
    )
    first = [r for r in q["candidate_features"]["rows"] if (r["slot"], r["move"]) == (1, 4)]
    assert numbers[0] == pytest.approx(first[0]["mean_ppm"] / 1_000_000)
    assert numbers[5] == pytest.approx(first[1]["mean_ppm"] / 1_000_000)


def test_an_unsupported_row_reaches_its_own_category_with_no_number() -> None:
    q = query()
    kinds, numbers = part(q, "move 1 2, move 1 1", 0)  # the sample's unsupported random_effect row
    assert kinds == [NONE, RANDOM]
    assert numbers == [0.0] * WIDTH
    for reason, expected in (("query_kind", QUERY_KIND), ("scenario_rule_table", SCENARIO)):
        changed = query()
        for row in changed["candidate_features"]["rows"]:
            if row["kind"] == "unsupported":
                row["reason"] = reason
        assert part(changed, "move 1 2, move 1 1", 0)[0] == [NONE, expected]
    assert len({COMPUTED, QUERY_KIND, RANDOM, SCENARIO}) == 4  # one learned category each


def test_a_status_move_a_switch_a_pass_and_an_ally_target_carry_no_row() -> None:
    q = query()
    none = ([NONE, NONE], [0.0] * WIDTH)
    assert part(q, "move 3, move 1 1", 0) == none  # Protect
    assert part(q, "move 4 1, move 1 1", 0) == none  # Sleep Powder, a status move
    assert part(q, "move 1 1, switch 3", 1) == none  # a switch
    assert part(q, "move 1 1, move 1 -1", 1) == none  # a damaging move at the ally


def test_a_preview_query_has_zero_numbers_and_no_rows() -> None:
    preview = sample("policy-query/valid/preview-features.json")
    encoded = ENCODE.encode(preview)
    assert encoded.cand_num is not None
    assert not encoded.cand_num.any()
    assert encoded.cand_cat[..., BASE_CAND_COLUMNS:].eq(NONE).all()


# -- fail closed -------------------------------------------------------------------------------


def test_a_missing_row_is_refused() -> None:
    # The foe at location 1 is alive, so the move aimed at it needs its row.
    gone = rows_without(SAMPLE, lambda r: (r["slot"], r["move"], r["target"]) != (0, 1, 1))
    with pytest.raises(EncodingError, match="no candidate_features row"):
        ENCODE.encode(gone)
    # A spread move with no row at all is refused too.
    no_spread = rows_without(SAMPLE, lambda r: (r["slot"], r["move"]) != (1, 4))
    with pytest.raises(EncodingError, match="no candidate_features row"):
        ENCODE.encode(no_spread)


def test_a_candidate_aimed_at_a_fainted_foe_needs_no_row() -> None:
    fainted = query()
    fainted["observation"]["log"].insert(-1, "|faint|p2b: Charizard")  # the foe at location 1
    fainted = rows_without(fainted, lambda r: r["target"] != 1)
    kinds, numbers = part(fainted, "move 1 1, move 1 2", 0)
    assert kinds == [NONE, NONE]
    assert numbers == [0.0] * WIDTH
    # The same query with a foe alive and the row gone is refused (the test above).


def test_the_rows_must_be_the_ones_the_encoder_reads() -> None:
    no_rows = {k: v for k, v in SAMPLE.items() if k != "candidate_features"}
    with pytest.raises(EncodingError, match="has no candidate_features"):
        ENCODE.encode(no_rows)
    wrong = query()
    wrong["candidate_features"]["feature_set_hash"] = "3" * 64
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE.encode(wrong)
    with pytest.raises(EncodingError, match="does not read"):
        encoder().encode(SAMPLE)  # an encoder without the option refuses rows
    twice = query()
    twice["candidate_features"]["rows"].append(twice["candidate_features"]["rows"][0])
    with pytest.raises(EncodingError, match="listed twice"):
        ENCODE.encode(twice)
    far = query()
    far["candidate_features"]["rows"][0]["target"] = 3
    with pytest.raises(EncodingError, match="foe location"):
        ENCODE.encode(far)


# -- the view, the session and the network ---------------------------------------------------------


def test_the_rows_are_part_of_the_view_only_when_the_query_has_them() -> None:
    assert encoder_input(SAMPLE)["candidate_features"] == SAMPLE["candidate_features"]
    plain = sample("policy-query/valid/battle.json")
    assert "candidate_features" not in encoder_input(plain)
    other = query()
    other["candidate_features"]["rows"][0]["mean_ppm"] += 1
    assert view_digest(other) != view_digest(SAMPLE)
    # decision_context and rqid still do not enter the view.
    a2 = query()
    a2["identity"] = {**a2["identity"], "rqid": 99}
    a2["observation"]["decision_context"] = {"deadline": None}
    assert view_digest(
        {**a2, "observation": {**a2["observation"], "identity": a2["identity"]}}
    ) == (view_digest(SAMPLE))


def test_an_incremental_session_gives_the_tensors_of_a_fresh_encode() -> None:
    session = ENCODE.session()
    a, b = session.encode(encoder_input(SAMPLE)), ENCODE.encode(encoder_input(SAMPLE))
    assert a.cand_num is not None
    assert b.cand_num is not None
    assert all(torch.equal(x, b.tensors()[k]) for k, x in a.tensors().items())


def net(seed: int = 1) -> Any:
    return init_network("battle", game(), ON_MODEL, seed)


def test_the_rows_are_an_input_of_the_score() -> None:
    model = net()
    base = model(collate([ENCODE.encode(SAMPLE)])).logits
    changed = query()
    for row in changed["candidate_features"]["rows"]:
        if row["kind"] == "computed":
            row["mean_ppm"] = min(1_000_000, row["mean_ppm"] * 3)
    assert not torch.allclose(base, model(collate([ENCODE.encode(changed)])).logits)
    # A candidate that uses no row scores the same as before only through the shared state; the
    # network reads the numbers per slot, so the move parts are what changed.


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    model = net()
    target = ENCODE.encode(SAMPLE)
    other = query()
    rows = other["candidate_features"]["rows"]
    other["candidate_features"]["rows"] = [*rows[:-1], {**rows[-1], "mean_ppm": 5}]
    alone = model(collate([target]))
    model(collate([ENCODE.encode(other)]))
    after = model(collate([target]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)
    assert torch.equal(alone.logits, net()(collate([target])).logits)


def test_the_a1_and_a2_forms_with_the_same_rows_get_identical_logits() -> None:
    a2 = sample("policy-query/valid/battle-a2.json")
    a2["candidate_features"] = copy.deepcopy(SAMPLE["candidate_features"])
    model = net()
    a1_logits = model(collate([ENCODE.encode(SAMPLE)])).logits
    assert torch.equal(a1_logits, model(collate([ENCODE.encode(a2)])).logits)


def test_a_padded_batch_matches_its_queries_scored_alone() -> None:
    model = net()
    short = query()
    short["candidates"] = short["candidates"][:10]
    a, b = ENCODE.encode(SAMPLE), ENCODE.encode(short)
    both = model(collate([a, b]))
    for row, single in enumerate((a, b)):
        alone = model(collate([single]))
        count = single.pending.shape[0]
        assert torch.allclose(both.logits[row, :count], alone.logits[0], atol=1e-5)


def test_collate_refuses_queries_of_two_feature_layouts() -> None:
    with_rows = ENCODE.encode(SAMPLE)
    without = encoder().encode(sample("policy-query/valid/battle.json"))
    with pytest.raises(ValueError, match="feature layout"):
        collate([with_rows, without])


def test_a_network_that_reads_features_refuses_a_batch_without_them() -> None:
    plain = collate([encoder().encode(sample("policy-query/valid/battle.json"))])
    with pytest.raises(ValueError, match="layout"):
        net()(plain)


def test_the_preview_network_ignores_the_rows() -> None:
    preview = init_network("preview", game(), ON_MODEL, seed=1)
    logits = preview(
        collate([ENCODE.encode(sample("policy-query/valid/preview-features.json"))])
    ).logits
    assert logits.shape[0] == 1


def test_no_game_content_is_written_in_the_row_columns() -> None:
    source = ast.parse((SOURCES / "features.py").read_text())
    strings = {
        n.value
        for n in ast.walk(source)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert not {"fire", "water", "tackle", "earthquake", "sunnyday"} & strings


def test_a_target_location_the_format_lacks_has_no_foe_and_needs_no_row() -> None:
    odd = query()
    odd["candidates"].append({"choice": "move 1 3, move 1 1", "pending": False})
    kinds, numbers = part(odd, "move 1 3, move 1 1", 0)
    assert kinds == [NONE, NONE]
    assert numbers == [0.0] * WIDTH


def test_a_feature_network_refuses_a_batch_whose_numbers_are_missing() -> None:
    batch = collate([ENCODE.encode(SAMPLE)])
    with pytest.raises(ValueError, match="reads candidate features"):
        net()(replace(batch, cand_num=None))


def test_a_location_beyond_the_foes_the_state_lists_holds_none() -> None:
    state = SimpleNamespace(opponent=SimpleNamespace(slots={"a": None}, roster=[]))
    assert not ENCODE._foe_alive(state, 2)  # type: ignore[arg-type]
    assert not ENCODE._foe_alive(state, 1)  # type: ignore[arg-type]  # the slot is empty
