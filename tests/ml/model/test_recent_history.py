"""The ``recent_history`` encoder option (ADR-010 erratum A4 item 7, #1014).

The Pokémon and field features also carry the reducer's recent history (``Recent``, ``elapsed``,
``condition_turns``): the last move through the move vocabulary, ages and counts scaled by the
configured turn and stat scales and the fixed divisors the encoder version hashes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
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
    Encoder,
    EncodingError,
    Layout,
    bucket,
    encoder_version,
)
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    RECENT_HISTORY,
    read_options,
)
from areneon_ml.reducer.state import PokemonState, Recent

from .support import TINY_ENCODER, TINY_MODEL, game, sample
from .test_encoder import PAIRS, pair_query, same
from .test_options import TODAY_ENCODER_VERSION

ON = replace(TINY_ENCODER, options=read_options("encoder", {RECENT_HISTORY: True}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
PLAIN = Encoder(game(), TINY_ENCODER)
LAYOUT = Layout.of(ON)
BASE = Layout.of(TINY_ENCODER)


def mon_col(name: str) -> int:
    return LAYOUT.mon_num.index(name)


def field_col(name: str) -> int:
    return LAYOUT.field_num.index(name)


# -- the option is absent: today's bytes ---------------------------------------


def test_with_the_option_absent_the_layout_and_the_version_are_today_s() -> None:
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert "last_move" not in [name for name, _ in BASE.mon_cat]
    assert "active_turns" not in BASE.mon_num
    assert "weather_elapsed" not in BASE.field_num


def test_the_option_adds_columns_and_changes_nothing_else() -> None:
    assert ENCODE.version != PLAIN.version
    assert ON.as_dict()["options"] == {RECENT_HISTORY: True}
    # The new columns come last in the numbers and after the moves in the categories.
    assert [c for c in LAYOUT.mon_cat if c[0] != "last_move"] == list(BASE.mon_cat)
    assert LAYOUT.mon_num[: len(BASE.mon_num)] == BASE.mon_num
    assert LAYOUT.field_num[: len(BASE.field_num)] == BASE.field_num
    query = sample("policy-query/valid/battle.json")
    plain, with_option = PLAIN.encode(query), ENCODE.encode(query)
    keep = [i for i, (name, _) in enumerate(LAYOUT.mon_cat) if name != "last_move"]
    assert torch.equal(with_option.mon_cat[:, keep], plain.mon_cat)
    assert torch.equal(with_option.mon_num[:, : plain.mon_num.shape[1]], plain.mon_num)
    assert torch.equal(with_option.field_num[: plain.field_num.shape[0]], plain.field_num)
    assert torch.equal(with_option.field_cat, plain.field_cat)
    for name in ("cand_cat", "cand_tok", "team_tok", "pending"):
        assert torch.equal(getattr(with_option, name), getattr(plain, name)), name


def test_only_true_is_a_value_and_the_model_config_round_trips() -> None:
    for bad in (False, 0, 1):
        with pytest.raises(ConfigError, match="invalid value"):
            ModelConfig.from_dict(
                {
                    **TINY_MODEL.as_dict(),
                    "encoder": {**TINY_ENCODER.as_dict(), "options": {RECENT_HISTORY: bad}},
                }
            )
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL


def test_the_scales_the_columns_use_are_part_of_the_version() -> None:
    for changed in (replace(ON, turn_scale=50.0), replace(ON, stat_scale=250.0)):
        assert encoder_version(game(), changed) != ENCODE.version


# -- hand-computed values --------------------------------------------------------
# battle.json, then (turn 1) the foe Venusaur's Earthquake takes 59 of own Venusaur's 159, own
# Charizard Protects, a sun, an own Tailwind and a Trick Room start; (turn 2) own Venusaur's Giga
# Drain takes 30 points of 100 off the foe Charizard. Scales: turns / 100, own hit points / 500.

TURN_ONE = [
    "|move|p2a: Venusaur|Earthquake|p1a: Venusaur",
    "|-damage|p1a: Venusaur|100/159",
    "|move|p1b: Charizard|Protect|p1b: Charizard",
    "|-singleturn|p1b: Charizard|Protect",
    "|-weather|SunnyDay",
    "|-sidestart|p1: p1|move: Tailwind",
    "|-fieldstart|move: Trick Room",
    "|upkeep",
    "|turn|2",
]
TURN_TWO = ["|move|p1a: Venusaur|Giga Drain|p2b: Charizard", "|-damage|p2b: Charizard|70/100"]


def played() -> dict[str, Any]:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    log = query["observation"]["log"]
    at = max(i for i, line in enumerate(log) if line.startswith("|request|"))
    query["observation"]["log"] = [*log[:at], *TURN_ONE, *TURN_TWO, *log[at:]]
    return query


def values(row: torch.Tensor, names: dict[str, float]) -> None:
    for name, want in names.items():
        assert row[mon_col(name)].item() == pytest.approx(want, abs=1e-6), name


def test_the_pokemon_columns_equal_the_hand_computed_values() -> None:
    encoded = ENCODE.encode(played())
    no_target = {f"last_target_{s}_{c}": 0.0 for s in ("own", "opponent") for c in "abc"}
    earthquake, protect = game().moves.lookup("earthquake"), game().moves.lookup("protect")
    last = [name for name, _ in LAYOUT.mon_cat].index("last_move")
    # Own Venusaur: took 59 last turn (exact / 500), dealt 30 of 100 this turn to the foe in b.
    values(
        encoded.mon_num[0],
        {
            **no_target,
            "last_target_opponent_b": 1.0,
            "active_turns": 0.02,
            "last_move_age": 0.0,
            "last_action_known": 1.0,
            "last_action_age": 0.0,
            "last_action_order": 1 / 8,
            "hp_taken_this_turn": 0.0,
            "hp_taken_last_turn": 59 / 500,
            "hp_dealt_this_turn": 0.3,
            "hp_dealt_last_turn": 0.0,
            "protect_streak": 0.0,
        },
    )
    assert encoded.mon_cat[0, last].item() == game().moves.lookup("gigadrain")
    # Own Charizard: Protected on turn 1 (the second action), streak 1 of 4.
    values(
        encoded.mon_num[1],
        {
            **no_target,
            "last_target_own_b": 1.0,
            "active_turns": 0.02,
            "last_move_age": 0.01,
            "last_action_known": 1.0,
            "last_action_age": 0.01,
            "last_action_order": 2 / 8,
            "protect_streak": 0.25,
            "hp_taken_this_turn": 0.0,
        },
    )
    assert encoded.mon_cat[1, last].item() == protect
    # A bench Pokémon has nothing: zeros and the NONE move.
    values(encoded.mon_num[2], {**no_target, "active_turns": 0.0, "last_action_known": 0.0})
    assert encoded.mon_cat[2, last].item() == 0
    # Foe Venusaur: hit own slot a with Earthquake; dealt 59 own points, read as exact / 500.
    foe_a, foe_b = encoded.n_own, encoded.n_own + 1
    values(
        encoded.mon_num[foe_a],
        {
            **no_target,
            "last_target_own_a": 1.0,
            "last_move_age": 0.01,
            "last_action_order": 1 / 8,
            "hp_dealt_last_turn": 59 / 500,
            "hp_dealt_this_turn": 0.0,
            "hp_taken_this_turn": 0.0,
        },
    )
    assert encoded.mon_cat[foe_a, last].item() == earthquake
    # Foe Charizard: took 30 points of 100 this turn (an opponent's amount is read / 100).
    values(
        encoded.mon_num[foe_b], {**no_target, "hp_taken_this_turn": 0.3, "last_action_known": 0.0}
    )


def test_the_field_columns_equal_the_hand_computed_values() -> None:
    encoded = ENCODE.encode(played())
    row, buckets = encoded.field_num, ON.condition_buckets
    assert row[field_col("weather_elapsed")].item() == pytest.approx(0.01)
    assert row[field_col("terrain_elapsed")].item() == 0.0
    assert row[field_col(f"own_side_turns_{bucket('tailwind', buckets)}")].item() == pytest.approx(
        0.01
    )
    assert row[field_col(f"field_turns_{bucket('trickroom', buckets)}")].item() == pytest.approx(
        0.01
    )
    assert row[field_col(f"opponent_side_turns_{bucket('tailwind', buckets)}")].item() == 0.0
    # Nothing else is set: the counts of the base columns say which conditions exist.
    assert row[len(BASE.field_num) :].sum().item() == pytest.approx(0.03)


def test_a_condition_that_began_this_turn_has_zero_turns() -> None:
    query = played()
    query["observation"]["log"].insert(-1, "|-weather|RainDance")
    encoded = ENCODE.encode(query)
    assert encoded.field_num[field_col("weather_elapsed")].item() == 0.0
    assert encoded.field_cat[0].item() == game().weathers.lookup("raindance")


def test_a_target_the_encoder_cannot_place_is_refused() -> None:
    mon = PokemonState(own=True, recent=Recent(last_target="own:z"))
    with pytest.raises(EncodingError, match="last_target"):
        ENCODE._recent_num(mon, 1)
    mon.recent.last_target = "elsewhere:a"
    with pytest.raises(EncodingError, match="last_target"):
        ENCODE._recent_num(mon, 1)


# -- A1/A2 equality and composition ---------------------------------------------


def test_the_a1_and_a2_forms_encode_equally_on_the_committed_pairs() -> None:
    assert PAIRS["decisions"]
    for decision in PAIRS["decisions"]:
        assert same(
            ENCODE.encode(pair_query(PAIRS, decision, "a1")),
            ENCODE.encode(pair_query(PAIRS, decision, "a2")),
        ), decision["fixture"]


FEATURE_1 = sample("policy-query/valid/battle-features.json")
FEATURE_2 = sample("policy-query/valid/battle-features-2.json")


def feature_option(query: dict[str, Any]) -> dict[str, str]:
    return {key: query["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}


@pytest.mark.parametrize(
    ("others", "query"),
    [
        ({CANDIDATE_MATCHUP: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_TARGET: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_FEATURES: feature_option(FEATURE_1)}, FEATURE_1),
        ({CANDIDATE_FEATURES: feature_option(FEATURE_2)}, FEATURE_2),
        (
            {
                CANDIDATE_MATCHUP: True,
                CANDIDATE_TARGET: True,
                CANDIDATE_FEATURES: feature_option(FEATURE_2),
            },
            FEATURE_2,
        ),
    ],
    ids=["matchup", "target", "features-1", "features-2", "all"],
)
def test_it_composes_with_each_other_encoder_option(
    others: dict[str, Any], query: dict[str, Any]
) -> None:
    without = replace(TINY_ENCODER, options=read_options("encoder", others))
    both = replace(TINY_ENCODER, options=read_options("encoder", {**others, RECENT_HISTORY: True}))
    assert encoder_version(game(), both) not in {
        encoder_version(game(), without),
        ENCODE.version,
    }
    a, b = Encoder(game(), without).encode(query), Encoder(game(), both).encode(query)
    layout, base = Layout.of(both), Layout.of(without)
    keep = [i for i, (name, _) in enumerate(layout.mon_cat) if name != "last_move"]
    assert [layout.mon_cat[i] for i in keep] == list(base.mon_cat)
    assert torch.equal(b.mon_cat[:, keep], a.mon_cat)
    assert torch.equal(b.mon_num[:, : a.mon_num.shape[1]], a.mon_num)
    assert torch.equal(b.field_num[: a.field_num.shape[0]], a.field_num)
    for name, tensor in a.tensors().items():
        if name not in ("mon_cat", "mon_num", "field_num"):
            assert torch.equal(b.tensors()[name], tensor), name
    network: Any = init_network("battle", game(), replace(TINY_MODEL, encoder=both), seed=3)
    out = network(collate([b]))
    assert torch.isfinite(out.logits[:, 0]).all()
    # The new columns are an input of the score.
    changed = collate([b])
    changed.mon_num[..., len(a.mon_num[0]) : len(a.mon_num[0]) + 1] += 1.0
    assert not torch.equal(network(changed).logits, out.logits)


# -- the network and the checkpoint -------------------------------------------------


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3).eval()
    one, other = ENCODE.encode(played()), ENCODE.encode(sample("policy-query/valid/battle-a2.json"))
    with torch.no_grad():
        alone = network(collate([one]))
        network(collate([other]))
        after = network(collate([one]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)


def test_a_training_step_reaches_the_new_columns() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3)
    batch = collate([ENCODE.encode(played())] * 2)
    out = network(batch)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    first = len(BASE.mon_num)
    assert network.mon_num.weight.grad[:, first:].abs().sum() > 0
    assert network.field_num.weight.grad[:, len(BASE.field_num) :].abs().sum() > 0


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    network = init_network("battle", game(), ON_MODEL, seed=3)
    data = checkpoint_bytes(network, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)
