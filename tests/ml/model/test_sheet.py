"""The ``opponent_sheet`` encoder option (ADR-027 slice 4, ADR-025 erratum E4, #1152).

A query under an ``opponent-sheet/1`` checkpoint carries the closed-regime block. Each opponent
token reads its slot: source tags and the refusal code as categories, entry ids as categories, ppm
as scaled numbers. The block is built on a real golden query's opponent roster; the numbers
are invented.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
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
from areneon_ml.model import sheet as sheet_of
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError, Layout, encoder_version
from areneon_ml.model.gamedata import NONE, RESERVED
from areneon_ml.model.network import ModelConfig, init_network
from areneon_ml.model.options import (
    OPPONENT_SHEET,
    FeatureSetOption,
    OptionError,
    options_dict,
    read_options,
    sheet_set,
)
from areneon_ml.reducer import reduce_channel

from .support import TINY_ENCODER, TINY_MODEL, game, sample
from .test_options import TODAY_ENCODER_VERSION

SCHEMA = (
    Path(__file__).resolve().parents[3] / "packages/schema/contracts/v1/opponent-sheet.schema.json"
)
HASH = "5" * 64
SET = {"feature_set": "opponent-sheet/1", "feature_set_hash": HASH}
ON = replace(TINY_ENCODER, options=read_options("encoder", {OPPONENT_SHEET: SET}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
QUERY = sample("policy-query/valid/battle.json")


def roster(query: dict[str, Any]) -> list[str]:
    obs = query["observation"]
    state = reduce_channel(obs["player"], obs["log"], obs["team"]["team"], game().vocabulary)
    return [m.base_species for m in state.opponent.roster if not m.phantom]


def block(query: dict[str, Any], refusal: str | None = None) -> dict[str, Any]:
    items = sorted(game().items.ids)[:3]
    slots = []
    for i, species in enumerate(roster(query)):
        if refusal is not None:
            slots.append(
                {
                    "species": species,
                    "brought_ppm": 0,
                    "brought_source": "unknown",
                    "item": {"source": "unknown", "entries": [], "other_ppm": 0},
                    "ability": {"source": "unknown", "entries": [], "other_ppm": 0},
                    "nature": {"source": "unknown", "entries": [], "other_ppm": 0},
                    "moves": {"source": "unknown", "entries": [], "slots_known": 0},
                }
            )
            continue
        slots.append(
            {
                "species": species,
                "brought_ppm": 666667,
                "brought_source": "predicted",
                "item": {
                    "source": "predicted",
                    "entries": [{"id": items[0], "ppm": 600000}, {"id": None, "ppm": 300000}],
                    "other_ppm": 100000,
                },
                "ability": {
                    "source": "revealed" if i == 0 else "predicted",
                    "entries": [{"id": sorted(game().abilities.ids)[i], "ppm": 1000000}],
                    "other_ppm": 0,
                },
                "nature": {
                    "source": "predicted",
                    "entries": [{"id": "adamant", "ppm": 250000}, {"id": "jolly", "ppm": 250000}],
                    "other_ppm": 500000,
                },
                "moves": {
                    "source": "predicted",
                    "entries": [{"id": sorted(game().moves.ids)[0], "inclusion_ppm": 700000}],
                    "slots_known": 0,
                },
            }
        )
    out: dict[str, Any] = {"source": {"kind": "belief", "hash": HASH}, "slots": slots}
    if refusal is not None:
        out["source"]["refusal"] = refusal
    return out


def query(**changes: Any) -> dict[str, Any]:
    q = copy.deepcopy(QUERY)
    q["opponent_sheet"] = block(q)
    q.update(changes)
    return q


def column(name: str) -> int:
    return [c for c, _ in Layout.of(ON, game()).mon_cat].index(name)


def number(name: str) -> int:
    return Layout.of(ON, game()).mon_num.index(name)


# -- the option --------------------------------------------------------------------------------


def test_the_option_reads_an_object_with_the_set_and_a_hash() -> None:
    assert sheet_set(ON.options) == FeatureSetOption(**SET)
    assert options_dict(ON.options) == {OPPONENT_SHEET: SET}
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL
    for bad in (
        True,
        {"feature_set": "opponent-sheet/1"},
        {**SET, "extra": 1},
        {**SET, "feature_set": "opponent-sheet/2"},
        {**SET, "feature_set_hash": "A" * 64},
    ):
        with pytest.raises(OptionError):
            read_options("encoder", {OPPONENT_SHEET: bad})


def test_without_the_option_nothing_changes_and_a_block_is_ignored() -> None:
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert not any(name.startswith("sheet_") for name, _ in Layout.of(TINY_ENCODER).mon_cat)
    plain = Encoder(game(), TINY_ENCODER)
    with_block = plain.encode(query())
    without = plain.encode(QUERY)
    assert torch.equal(with_block.mon_cat, without.mon_cat)
    assert torch.equal(with_block.mon_num, without.mon_num)


def test_the_option_moves_the_version_the_layout_and_the_checkpoint_format() -> None:
    assert encoder_version(game(), ON) != TODAY_ENCODER_VERSION
    other = replace(
        ON, options=read_options("encoder", {OPPONENT_SHEET: {**SET, "feature_set_hash": "6" * 64}})
    )
    assert encoder_version(game(), other) != encoder_version(game(), ON)
    layout = Layout.of(ON, game())
    assert [c for c, _ in layout.mon_cat if c.startswith("sheet_")] == [
        c for c, _ in sheet_of.CAT_COLUMNS
    ]
    assert [n for n in layout.mon_num if n.startswith("sheet_")] == list(sheet_of.NUM_COLUMNS)
    data = checkpoint_bytes(init_network("battle", game(), ON_MODEL, seed=1), game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    gd = game()
    loaded = load_checkpoint(data, gd, checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, gd, checkpoint_id(data), "battle", configured=TINY_MODEL)


def test_the_closed_words_are_the_contracts() -> None:
    schema = json.loads(SCHEMA.read_text())
    assert list(sheet_of.SOURCES) == schema["$defs"]["SheetFieldSource"]["enum"]
    assert (
        list(sheet_of.REFUSALS) == schema["$defs"]["SheetSource"]["properties"]["refusal"]["enum"]
    )


# -- the columns -------------------------------------------------------------------------------


def test_each_opponent_token_reads_its_slot_and_own_tokens_read_nothing() -> None:
    encoded = ENCODE.encode(query())
    n_own = encoded.n_own
    items = sorted(game().items.ids)
    for i in range(n_own):
        assert encoded.mon_cat[i, column("sheet_item_0")] == NONE
        assert encoded.mon_num[i, number("sheet_brought")] == 0
    first = n_own
    assert encoded.mon_cat[first, column("sheet_refusal")] == NONE
    assert encoded.mon_cat[first, column("sheet_item_source")] == RESERVED + sheet_of.SOURCES.index(
        "predicted"
    )
    assert encoded.mon_cat[
        first, column("sheet_ability_source")
    ] == RESERVED + sheet_of.SOURCES.index("revealed")
    assert encoded.mon_cat[first, column("sheet_item_0")] == game().items.lookup(items[0])
    assert encoded.mon_cat[first, column("sheet_item_1")] == NONE  # "holds no item" is NONE
    assert encoded.mon_cat[first, column("sheet_item_2")] == NONE  # padding
    assert encoded.mon_num[first, number("sheet_item_0_ppm")] == pytest.approx(0.6)
    assert encoded.mon_num[first, number("sheet_item_1_ppm")] == pytest.approx(0.3)
    assert encoded.mon_num[first, number("sheet_item_other")] == pytest.approx(0.1)
    assert encoded.mon_num[first, number("sheet_brought")] == pytest.approx(0.666667)
    assert encoded.mon_cat[first, column("sheet_nature_0")] == sheet_of.nature_category("adamant")
    assert encoded.mon_cat[first, column("sheet_nature_0")] >= RESERVED


def test_a_refused_block_is_its_code_on_every_opponent_token() -> None:
    q = query()
    q["opponent_sheet"] = block(q, "disguise_unsettled")
    encoded = ENCODE.encode(q)
    code = RESERVED + sheet_of.REFUSALS.index("disguise_unsettled")
    for i in range(encoded.n_own, encoded.mon_cat.shape[0]):
        assert encoded.mon_cat[i, column("sheet_refusal")] == code
        assert encoded.mon_cat[i, column("sheet_item_source")] == RESERVED + sheet_of.SOURCES.index(
            "unknown"
        )
        assert encoded.mon_num[i, number("sheet_brought")] == 0


def test_the_tensors_are_the_same_alone_and_after_others() -> None:
    q = query()
    alone = ENCODE.encode(q)
    ENCODE.encode(
        query(opponent_sheet={**q["opponent_sheet"], "slots": q["opponent_sheet"]["slots"]})
    )
    again = ENCODE.encode(q)
    assert torch.equal(alone.mon_cat, again.mon_cat)
    assert torch.equal(alone.mon_num, again.mon_num)
    batch = collate([alone, ENCODE.encode(q)])
    net = init_network("battle", game(), ON_MODEL, seed=1)
    out = net(batch)
    assert torch.isfinite(out.logits[~torch.isinf(out.logits)]).all()


# -- the refusals ------------------------------------------------------------------------------


def broken(change: Any) -> dict[str, Any]:
    q = query()
    change(q["opponent_sheet"])
    return q


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: b["slots"].pop(), "not one slot per previewed opponent"),
        (lambda b: b["source"].update(hash="7" * 64), "another feature set"),
        (lambda b: b["slots"][0].update(species="notaspecies"), "another species"),
        (
            lambda b: b["slots"][0]["item"]["entries"].__setitem__(
                0, {"id": "notanitem", "ppm": 1}
            ),
            "item the engine data lacks",
        ),
        (
            lambda b: b["slots"][0]["moves"]["entries"].append(
                {"id": "notamove", "inclusion_ppm": 1}
            ),
            "move the engine data lacks",
        ),
        (
            lambda b: b["slots"][0]["ability"]["entries"].append({"id": "notability", "ppm": 1}),
            "ability the engine data lacks",
        ),
        (
            lambda b: b["slots"][0]["nature"]["entries"].extend(
                {"id": f"n{i}", "ppm": 1} for i in range(4)
            ),
            "more entries than k",
        ),
    ],
)
def test_a_block_that_does_not_fit_is_refused(change: Any, message: str) -> None:
    with pytest.raises(EncodingError, match=message):
        ENCODE.encode(broken(change))


def test_a_config_with_the_option_needs_the_block() -> None:
    with pytest.raises(EncodingError, match="no opponent_sheet"):
        ENCODE.encode(QUERY)
