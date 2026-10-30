"""The ``exact_volatiles`` encoder option (ADR-010 erratum A4 item 7, #1019).

Volatiles and field and side conditions are exact vocabulary ids (one column per id of a version 2
engine-data table) instead of SHA-256 buckets, and a Pokémon token carries the types a
``typechange`` set. A version 1 table is refused; an id the vocabulary does not resolve fails
closed.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
from dataclasses import replace
from functools import cache
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
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import (
    EncodedQuery,
    Encoder,
    EncodingError,
    Layout,
    _typechange_num,
    bucket,
    encoder_version,
    exact_ids,
)
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    EXACT_VOLATILES,
    RECENT_HISTORY,
    read_options,
)
from areneon_ml.reducer import VocabularyError
from areneon_ml.reducer.state import PokemonState
from areneon_ml.reducer.vocab import PROTOCOL_VOLATILES

from .support import (
    FORMAT_DIR,
    RANDOM,
    TINY_ENCODER,
    TINY_MODEL,
    candidates_for,
    observation,
    request_points,
    sample,
)
from .test_encoder import PAIRS, kind_of, pair_query, same
from .test_options import TODAY_ENCODER_VERSION


@cache
def game2() -> GameData:
    return GameData.from_table(json.loads((FORMAT_DIR / "engine-data-2.json").read_text()))


@cache
def game1() -> GameData:
    return GameData.from_table(json.loads((FORMAT_DIR / "engine-data.json").read_text()))


ON = replace(TINY_ENCODER, options=read_options("encoder", {EXACT_VOLATILES: True}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
IDS = exact_ids(game2())
COLUMN = {name: i for i, name in enumerate(IDS)}
LAYOUT = Layout.of(ON, game2())
BASE = Layout.of(TINY_ENCODER, game2())
ENCODE = Encoder(game2(), ON)
PLAIN = Encoder(game2(), TINY_ENCODER)
FIRST = LAYOUT.mon_num.index("volatile_0")


def volatile_col(name: str) -> int:
    return FIRST + COLUMN[name]


def field_col(prefix: str, name: str) -> int:
    return LAYOUT.field_num.index(f"{prefix}_0") + COLUMN[name]


# -- the option is absent: today's bytes -----------------------------------------


def test_with_the_option_absent_everything_is_today_s() -> None:
    assert encoder_version(game1(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert Layout.of(TINY_ENCODER, game2()) == Layout.of(TINY_ENCODER)
    assert "options" not in TINY_ENCODER.as_dict()
    assert "exact_volatiles" not in json.dumps(Layout.of(TINY_ENCODER).as_dict())
    # A version 2 table without the option encodes as the version 1 table does.
    query = sample("policy-query/valid/battle.json")
    assert same(PLAIN.encode(query), Encoder(game1(), TINY_ENCODER).encode(query))


def test_the_option_changes_the_version_and_the_layout_only() -> None:
    assert ENCODE.version != PLAIN.version
    assert ON.as_dict()["options"] == {EXACT_VOLATILES: True}
    assert len(LAYOUT.mon_num) - len(BASE.mon_num) == len(IDS) - 4 + 2 + len(game2().types.ids) + 5
    assert len(LAYOUT.field_num) - len(BASE.field_num) == 3 * (len(IDS) - 4)
    assert LAYOUT.mon_num[:FIRST] == BASE.mon_num[:FIRST]
    assert LAYOUT.mon_cat == BASE.mon_cat
    assert LAYOUT.cand_cat == BASE.cand_cat


def test_the_ids_are_the_conditions_the_protocol_words_and_the_names_of_effects() -> None:
    assert list(IDS) == sorted(set(IDS))
    for name in ("confusion", "taunt", "stockpile", "firespin", "illusion", "spikes", "trickroom"):
        assert name in IDS, name
    assert set(IDS) >= PROTOCOL_VOLATILES
    table = game2()
    assert {*table.conditions.ids, *table.moves.ids, *table.items.ids} <= set(IDS)  # type: ignore[union-attr]


def test_only_true_is_a_value_and_the_model_config_round_trips() -> None:
    for bad in (False, 0, 1):
        with pytest.raises(ConfigError, match="invalid value"):
            ModelConfig.from_dict(
                {
                    **TINY_MODEL.as_dict(),
                    "encoder": {**TINY_ENCODER.as_dict(), "options": {EXACT_VOLATILES: bad}},
                }
            )
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL


def test_a_version_1_table_is_refused_with_a_clear_error() -> None:
    for call in (
        lambda: Encoder(game1(), ON),
        lambda: encoder_version(game1(), ON),
        lambda: Layout.of(ON, game1()),
        lambda: init_network("battle", game1(), ON_MODEL, seed=1),
    ):
        with pytest.raises(EncodingError, match="version 2 engine-data table"):
            call()
    with pytest.raises(EncodingError, match="needs the game data"):
        Layout.of(ON)


# -- hand-computed values ------------------------------------------------------------
# battle.json, then (turn 1) on the foe Venusaur (p2a) a confusion, two stockpiles' worth of the
# derived name, Fire Spin by name and a type change to Fire/Flying; on own Charizard (p1b) a
# typeless Burn Up; Spikes twice on the foe's side, one Tailwind on ours and a Trick Room.

TURN = [
    "|-start|p2a: Venusaur|confusion",
    "|-start|p2a: Venusaur|stockpile2",
    "|-start|p2a: Venusaur|move: Fire Spin",
    "|-start|p2a: Venusaur|typechange|Fire/Flying|[silent]",
    "|-start|p1b: Charizard|typechange|???|[from] move: Burn Up",
    "|-sidestart|p2: p2|move: Spikes",
    "|-sidestart|p2: p2|move: Spikes",
    "|-sidestart|p1: p1|move: Tailwind",
    "|-fieldstart|move: Trick Room",
    "|upkeep",
    "|turn|2",
]


def played(extra: list[str] | None = None) -> dict[str, Any]:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    log = query["observation"]["log"]
    at = max(i for i, line in enumerate(log) if line.startswith("|request|"))
    query["observation"]["log"] = [*log[:at], *TURN, *(extra or []), *log[at:]]
    return query


def test_the_volatile_columns_hold_one_per_exact_id() -> None:
    encoded = ENCODE.encode(played())
    foe = encoded.n_own  # the foe Venusaur in a
    row = encoded.mon_num[foe]
    block = row[FIRST : FIRST + len(IDS)]
    assert {IDS[i] for i in block.nonzero().flatten().tolist()} == {
        "confusion",
        "stockpile",  # stockpile2 resolves to its condition id
        "firespin",
        "typechange",
    }
    assert block[COLUMN["stockpile"]].item() == pytest.approx(1 + 2 / 8)  # layer 2
    assert block.sum().item() == pytest.approx(4.25)
    # Nothing shared with another id: with four buckets these would have collided.
    assert len({volatile_col(name) for name in IDS}) == len(IDS)
    assert encoded.mon_num[0, FIRST : FIRST + len(IDS)].sum().item() == 0.0


def foe_block(*lines: str) -> torch.Tensor:
    encoded = ENCODE.encode(played(list(lines)))
    return encoded.mon_num[encoded.n_own]


def test_the_derived_number_is_the_latest_one_the_engine_printed() -> None:
    """Stockpile counts up with no `-end` between layers, Perish Song down with none at all,
    and `-end|Stockpile` names the bare condition: the value follows the engine's sequence."""
    foe = "p2a: Venusaur"
    column = FIRST + COLUMN["stockpile"]
    one = foe_block(f"|-start|{foe}|stockpile1")[column]
    three = foe_block(
        f"|-start|{foe}|stockpile1", f"|-start|{foe}|stockpile2", f"|-start|{foe}|stockpile3"
    )[column]
    assert one.item() == pytest.approx(1 + 1 / 8)
    assert three.item() == pytest.approx(1 + 3 / 8)
    assert foe_block(f"|-start|{foe}|stockpile3", f"|-end|{foe}|Stockpile")[column].item() == 0.0
    perish = FIRST + COLUMN["perish"]
    three_left = foe_block(f"|-start|{foe}|perish3")[perish]
    none_left = foe_block(*(f"|-start|{foe}|perish{n}" for n in (3, 2, 1, 0)))[perish]
    assert three_left.item() == pytest.approx(1 + 3 / 8)
    assert none_left.item() == 1.0
    fallen = FIRST + COLUMN["fallen"]
    assert foe_block(f"|-start|{foe}|fallen4|[silent]")[fallen].item() == pytest.approx(1.5)
    assert (
        foe_block(f"|-start|{foe}|fallen4|[silent]", f"|-end|{foe}|fallen4|[silent]")[fallen].item()
        == 0.0
    )


def test_the_boosted_stat_of_protosynthesis_is_a_flag_per_stat() -> None:
    foe = "p2a: Venusaur"
    flags = LAYOUT.mon_num.index("boosted_stat_atk")
    atk = foe_block(f"|-start|{foe}|protosynthesisatk")
    spe = foe_block(f"|-start|{foe}|protosynthesisspe")
    assert atk[flags : flags + 5].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert spe[flags : flags + 5].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    column = FIRST + COLUMN["protosynthesis"]
    assert atk[column].item() == spe[column].item() == 1.0
    assert not torch.equal(atk, spe)
    ended = foe_block(f"|-start|{foe}|protosynthesisatk", f"|-end|{foe}|Protosynthesis")
    assert ended[flags : flags + 5].sum().item() == 0.0
    quark = foe_block(f"|-start|{foe}|quarkdrivespd")
    assert quark[flags + 3].item() == 1.0


def test_a_derived_suffix_that_is_neither_a_number_nor_a_stat_is_refused() -> None:
    mon = PokemonState(own=False, derived={"stockpile": "x"})
    with pytest.raises(EncodingError, match="neither a number nor a stat"):
        ENCODE._typechange(mon)


def test_a_typechange_sets_the_types_it_names() -> None:
    encoded = ENCODE.encode(played())
    start = LAYOUT.mon_num.index("types_known")
    names = game2().types.ids
    width = 2 + len(names)

    def read(token: int) -> list[float]:
        return encoded.mon_num[token, start : start + width].tolist()

    foe = read(encoded.n_own)
    assert foe[0] == 1.0
    assert foe[1] == 0.0
    assert {names[i] for i in range(len(names)) if foe[2 + i]} == {"fire", "flying"}
    charizard = read(1)  # own Charizard in b: Burn Up leaves it typeless
    assert charizard[:2] == [1.0, 1.0]
    assert sum(charizard[2:]) == 0.0
    assert read(0) == [0.0] * width  # its own types stay the species'


def test_the_guards_of_a_layout_without_game_data() -> None:
    with pytest.raises(EncodingError, match="needs the game data"):
        _typechange_num(ON, None)
    with pytest.raises(VocabularyError, match="without a condition list"):
        game1().vocabulary.resolve_volatile("confusion")


def test_the_copy_form_of_a_typechange_knows_only_that_it_changed() -> None:
    mon = PokemonState(own=False, types=[])
    row = ENCODE._typechange(mon)
    assert row[0] == 1.0
    assert sum(row[1:]) == 0.0
    assert ENCODE._typechange(PokemonState(own=False)) == [0.0] * len(row)


def test_the_field_and_side_columns_count_per_exact_id() -> None:
    encoded = ENCODE.encode(played())
    row = encoded.field_num
    assert row[field_col("field", "trickroom")].item() == 1.0
    assert row[field_col("own_side", "tailwind")].item() == 1.0
    assert row[field_col("opponent_side", "spikes")].item() == 2.0  # two layers
    assert row[field_col("opponent_side", "tailwind")].item() == 0.0
    first = LAYOUT.field_num.index("field_0")
    assert row[first:].sum().item() == 4.0


def test_an_id_the_vocabulary_does_not_resolve_fails_closed() -> None:
    with pytest.raises(EncodingError, match="not in the exact vocabulary"):
        ENCODE._volatile_column("notavolatile")
    with pytest.raises(EncodingError, match="not in the exact vocabulary"):
        ENCODE._condition_column("notacondition")
    mon = PokemonState(own=False, types=["plasma"])
    with pytest.raises(EncodingError, match="not in the engine data"):
        ENCODE._typechange(mon)
    # And through a state: a volatile no condition, move, ability or item names.
    state_mon = PokemonState(own=False, volatiles={"notavolatile"})
    with pytest.raises(EncodingError, match="exact vocabulary"):
        ENCODE._mon_num(state_mon, 1)


def test_the_bucketed_encoder_still_buckets() -> None:
    plain = PLAIN.encode(played())
    first = BASE.mon_num.index("volatile_0")
    block = plain.mon_num[plain.n_own, first : first + TINY_ENCODER.volatile_buckets]
    assert block.sum().item() == 4.0
    assert PLAIN._volatile_column("confusion") == bucket("confusion", 4)


# -- A1/A2 equality, the corpus and composition ------------------------------------------


def test_the_a1_and_a2_forms_encode_equally_on_the_committed_pairs() -> None:
    assert PAIRS["decisions"]
    for decision in PAIRS["decisions"]:
        assert same(
            ENCODE.encode(pair_query(PAIRS, decision, "a1")),
            ENCODE.encode(pair_query(PAIRS, decision, "a2")),
        ), decision["fixture"]


def test_every_request_point_of_a_random_corpus_slice_encodes_with_no_unknown_id() -> None:
    points = 0
    for path in RANDOM[:6]:
        battle = json.loads(path.read_text())
        for player in ("p1", "p2"):
            lines = battle["channels"][player]["lines"]
            for index, body in request_points(lines):
                obs = observation(
                    player, lines[: index + 1], battle["teams"][player], kind_of(body)
                )
                encoded = ENCODE.encode({"observation": obs, "candidates": candidates_for(body)})
                assert torch.isfinite(encoded.mon_num).all()
                assert encoded.mon_num.shape[1] == len(LAYOUT.mon_num)
                assert encoded.field_num.shape[0] == len(LAYOUT.field_num)
                points += 1
    assert points > 100


FEATURE_2 = sample("policy-query/valid/battle-features-2.json")


@pytest.mark.parametrize(
    ("others", "query"),
    [
        ({CANDIDATE_MATCHUP: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_TARGET: True}, sample("policy-query/valid/battle.json")),
        ({RECENT_HISTORY: True}, played()),
        (
            {
                CANDIDATE_FEATURES: {
                    key: FEATURE_2["candidate_features"][key]
                    for key in ("feature_set", "feature_set_hash")
                }
            },
            FEATURE_2,
        ),
        (
            {CANDIDATE_MATCHUP: True, CANDIDATE_TARGET: True, RECENT_HISTORY: True},
            sample("policy-query/valid/battle.json"),
        ),
    ],
    ids=["matchup", "target", "recent", "features-2", "all"],
)
def test_it_composes_with_each_other_encoder_option(
    others: dict[str, Any], query: dict[str, Any]
) -> None:
    without = replace(TINY_ENCODER, options=read_options("encoder", others))
    both = replace(TINY_ENCODER, options=read_options("encoder", {**others, EXACT_VOLATILES: True}))
    assert encoder_version(game2(), both) not in {
        encoder_version(game2(), without),
        ENCODE.version,
    }
    a, b = Encoder(game2(), without).encode(query), Encoder(game2(), both).encode(query)
    layout = Layout.of(both, game2())
    assert b.mon_num.shape[1] == len(layout.mon_num)
    assert b.field_num.shape[0] == len(layout.field_num)
    for name, tensor in a.tensors().items():
        if name not in ("mon_num", "field_num"):
            assert torch.equal(b.tensors()[name], tensor), name
    if RECENT_HISTORY in others:
        # The recent-history field columns are exact too, one per id.
        assert layout.field_num.index("field_turns_0") + len(IDS) == layout.field_num.index(
            "own_side_turns_0"
        )
    network: Any = init_network("battle", game2(), replace(TINY_MODEL, encoder=both), seed=3)
    out = network(collate([b]))
    assert torch.isfinite(out.logits[:, 0]).all()


# -- the network and the checkpoint -------------------------------------------------------


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    network: Any = init_network("battle", game2(), ON_MODEL, seed=3).eval()
    one = ENCODE.encode(played())
    other = ENCODE.encode(sample("policy-query/valid/battle-a2.json"))
    with torch.no_grad():
        alone = network(collate([one]))
        network(collate([other]))
        after = network(collate([one]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)


def test_a_training_step_reaches_the_exact_columns() -> None:
    network: Any = init_network("battle", game2(), ON_MODEL, seed=3)
    batch = collate([ENCODE.encode(played())] * 2)
    out = network(batch)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    grad = network.mon_num.weight.grad
    assert grad[:, volatile_col("confusion")].abs().sum() > 0
    assert grad[:, LAYOUT.mon_num.index("types_known")].abs().sum() > 0
    assert network.field_num.weight.grad[:, field_col("field", "trickroom")].abs().sum() > 0


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    network = init_network("battle", game2(), ON_MODEL, seed=3)
    data = checkpoint_bytes(network, game2())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game2(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game2(), checkpoint_id(data), "battle", configured=TINY_MODEL)


# -- golden values -----------------------------------------------------------------------
# Pinned like the encoder's: if one fails, the features changed. Review the diff, then re-pin.
GOLDEN_VERSION = "enc1:66db9179a27480ddf292c85ed7fa541a6ded2c49b61eef4a4b213ed6b74d635b"
GOLDEN_BATTLE = "be483c282c527158899575414a27852ab2114451b6c9b7be2af9cceff756153a"
GOLDEN_PLAYED = "a43bdd3b68f2b21f3f8dd48068e55c3979df39322890947f65cb769e721bde03"


def tensor_digest(encoded: EncodedQuery) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(encoded.tensors().items()):
        flat = tensor.contiguous()
        digest.update(f"{name}{tuple(flat.shape)}{flat.dtype}".encode())
        digest.update(ctypes.string_at(flat.data_ptr(), flat.numel() * flat.element_size()))
    return digest.hexdigest()


def test_the_version_and_the_tensors_of_fixed_queries_are_pinned() -> None:
    assert ENCODE.version == GOLDEN_VERSION
    assert tensor_digest(ENCODE.encode(sample("policy-query/valid/battle.json"))) == GOLDEN_BATTLE
    assert tensor_digest(ENCODE.encode(played())) == GOLDEN_PLAYED
