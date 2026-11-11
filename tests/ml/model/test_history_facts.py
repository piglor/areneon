"""The ``history_facts`` encoder option (ADR-022 erratum E-22.5.4, slice H5).

The Pokémon and field tokens also carry the history facts of the reduced state
(``BattleState.history``): counts over the configured turn scale and fixed divisors, flags 0/1,
the unknown of a fact the channel cannot settle as its own category or flag, and the field's last
move, faints and pending slot effects. Absent, every byte of today's encoder stays.
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
from areneon_ml.model import history as history_of
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder, EncodingError, Layout, encoder_version
from areneon_ml.model.gamedata import NONE, UNKNOWN, GameData
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    HISTORY_FACTS,
    RECENT_HISTORY,
    read_options,
)
from areneon_ml.reducer import reduce_channel
from areneon_ml.reducer.history import HistoryTracker
from areneon_ml.reducer.state import PokemonState

from .support import (
    FORMAT_DIR,
    RANDOM,
    TINY_ENCODER,
    TINY_MODEL,
    candidates_for,
    game,
    observation,
    request_points,
    sample,
)
from .test_encoder import PAIRS, pair_query, same
from .test_exact_volatiles import game2
from .test_options import TODAY_ENCODER_VERSION
from .test_recent_history import played

ON = replace(TINY_ENCODER, options=read_options("encoder", {HISTORY_FACTS: True}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
PLAIN = Encoder(game(), TINY_ENCODER)
LAYOUT = Layout.of(ON, game())
BASE = Layout.of(TINY_ENCODER)
STATE = next(i for i, (name, _) in enumerate(LAYOUT.mon_cat) if name == "history_state")
ROSTER = STATE + 1
LAST_ITEM = STATE + 2
KNOWN = history_of.KNOWN


def num(name: str) -> int:
    return LAYOUT.mon_num.index(name)


def values(row: torch.Tensor, expected: dict[str, float]) -> None:
    for name, want in expected.items():
        assert row[num(name)].item() == pytest.approx(want, abs=1e-6), name


def facts_of(query: dict[str, Any]) -> dict[str, Any]:
    o = query["observation"]
    state = reduce_channel(o["player"], o["log"], o["team"]["team"], game().vocabulary)
    assert state.history is not None
    facts: dict[str, Any] = state.history.facts()
    return facts


# -- absent: today's bytes ---------------------------------------------------------------------


def test_with_the_option_absent_the_layout_and_the_version_are_today_s() -> None:
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert not any(name.startswith("history") for name, _ in BASE.mon_cat + BASE.field_cat)
    assert not any(name.startswith("hist_") for name in BASE.mon_num + BASE.field_num)
    assert "options" not in TINY_ENCODER.as_dict()


def test_the_option_adds_columns_and_changes_nothing_else() -> None:
    assert ENCODE.version != PLAIN.version
    assert ON.as_dict()["options"] == {HISTORY_FACTS: True}
    mon_cat_added = list(history_of.mon_cat_columns())
    assert [c for c in LAYOUT.mon_cat if c not in mon_cat_added] == list(BASE.mon_cat)
    assert LAYOUT.mon_num[: len(BASE.mon_num)] == BASE.mon_num
    assert LAYOUT.field_num[: len(BASE.field_num)] == BASE.field_num
    query = sample("policy-query/valid/battle.json")
    plain, with_option = PLAIN.encode(query), ENCODE.encode(query)
    keep = [i for i, (name, _) in enumerate(LAYOUT.mon_cat) if name not in dict(mon_cat_added)]
    assert torch.equal(with_option.mon_cat[:, keep], plain.mon_cat)
    assert torch.equal(with_option.mon_num[:, : plain.mon_num.shape[1]], plain.mon_num)
    assert torch.equal(with_option.field_num[: plain.field_num.shape[0]], plain.field_num)
    assert torch.equal(with_option.field_cat[: plain.field_cat.shape[0]], plain.field_cat)
    for name in ("cand_cat", "cand_tok", "team_tok", "pending"):
        assert torch.equal(getattr(with_option, name), getattr(plain, name)), name


def test_only_true_is_a_value_and_the_model_config_round_trips() -> None:
    for bad in (False, 0, 1):
        with pytest.raises(ConfigError, match="invalid value"):
            ModelConfig.from_dict(
                {
                    **TINY_MODEL.as_dict(),
                    "encoder": {**TINY_ENCODER.as_dict(), "options": {HISTORY_FACTS: bad}},
                }
            )
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL


def test_the_scales_and_the_words_the_columns_use_are_part_of_the_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for changed in (replace(ON, turn_scale=50.0), replace(ON, max_moves=3)):
        assert encoder_version(game(), changed) != ENCODE.version
    for name, value in (("COUNT_SCALE", 4.0), ("LETTERS", 2)):
        with monkeypatch.context() as patch:
            patch.setattr(history_of, name, value)
            assert encoder_version(game(), ON) != ENCODE.version, name


def test_the_layout_needs_the_history_data_of_the_pin() -> None:
    bare = replace(game().vocabulary, history_data=None)
    no_data = replace(game(), vocabulary=bare)
    with pytest.raises(EncodingError, match="history facts data"):
        Layout.of(ON, no_data)
    with pytest.raises(EncodingError, match="game data"):
        Layout.of(ON, None)
    with pytest.raises(EncodingError, match="history facts data"):
        encoder_version(no_data, ON)
    with pytest.raises(history_of.HistoryError, match="history facts data"):
        history_of.stall_maximum(no_data)


# -- hand-computed values ----------------------------------------------------------------------
# `played()`: battle.json, then (turn 1) the foe Venusaur's Earthquake takes 59 of own Venusaur's
# 159, own Charizard Protects; (turn 2) own Venusaur's Giga Drain takes 30 of 100 off the foe
# Charizard. Scales: turns / 100, hits taken / 8, the stall counter over its engine-data maximum
# (729), a count of moves over turn_scale.


def test_the_pokemon_columns_equal_the_hand_computed_values() -> None:
    encoded = ENCODE.encode(played())
    foe_a, foe_b = encoded.n_own, encoded.n_own + 1
    # Own Venusaur in a: one action since its switch-in, hit once (the Earthquake).
    values(
        encoded.mon_num[0],
        {
            "hist_move_actions": 0.01,
            "hist_times_attacked": 1 / 8,
            "hist_hurt_this_turn": 0.0,
            "hist_stall_active": 0.0,
            "hist_last_move_failed_unknown": 0.0,
            "hist_attackers_unknown": 0.0,
            "hist_moves_used_known": 1.0,
        },
    )
    # Own Charizard in b: Protect started a stall chain (counter 3 of 729); whether that move
    # failed is not settled by the channel: its own flag.
    values(
        encoded.mon_num[1],
        {
            "hist_move_actions": 0.01,
            "hist_stall_active": 1.0,
            "hist_stall_counter": 3 / 729,
            "hist_last_move_failed": 0.0,
            "hist_last_move_failed_unknown": 1.0,
        },
    )
    # Foe Venusaur in a: its Earthquake is one action; it took nothing.
    values(
        encoded.mon_num[foe_a],
        {"hist_move_actions": 0.01, "hist_times_attacked": 0.0, "hist_moves_used_known": 0.0},
    )
    # Foe Charizard in b: hit once this turn by own slot a.
    values(
        encoded.mon_num[foe_b],
        {
            "hist_hurt_this_turn": 1.0,
            "hist_times_attacked": 1 / 8,
            "hist_attacked_by_own_a": 1.0,
            "hist_attacked_by_own_b": 0.0,
            "hist_attacked_by_opponent_a": 0.0,
            "hist_attackers_unknown": 0.0,
        },
    )
    for token in (0, 1, foe_a, foe_b):
        assert encoded.mon_cat[token, STATE].item() == KNOWN
        assert encoded.mon_cat[token, ROSTER].item() == KNOWN
        assert encoded.mon_cat[token, LAST_ITEM].item() == NONE
    # A bench Pokémon has no position: category NONE and zero position facts.
    assert encoded.mon_cat[2, STATE].item() == NONE
    assert not encoded.mon_num[2, num("hist_move_actions") : num("hist_ate_berry")].any()


def test_the_field_columns_equal_the_hand_computed_values() -> None:
    encoded = ENCODE.encode(played())
    cats = [name for name, _ in LAYOUT.field_cat]
    assert encoded.field_cat[cats.index("history_available")].item() == KNOWN
    assert encoded.field_cat[cats.index("history_last_move")].item() == game().moves.lookup(
        "gigadrain"
    )
    row = encoded.field_num
    assert row[LAYOUT.field_num.index("hist_total_fainted_own")].item() == 0.0
    assert row[LAYOUT.field_num.index("hist_total_fainted_opponent")].item() == 0.0
    pending = [n for n in LAYOUT.field_num if n.startswith("hist_pending_")]
    assert pending
    assert not any(row[LAYOUT.field_num.index(n)].item() for n in pending)


def wished() -> dict[str, Any]:
    """`played()` and then a Wish by own Venusaur and the foe Charizard fainting, into turn 3."""
    query = played()
    log = query["observation"]["log"]
    at = max(i for i, line in enumerate(log) if line.startswith("|request|"))
    extra = [
        "|move|p1a: Venusaur|Wish|p1a: Venusaur",
        "|faint|p2b: Charizard",
        "|upkeep",
        "|turn|3",
    ]
    query["observation"]["log"] = [*log[:at], *extra, *log[at:]]
    return query


def test_a_pending_slot_effect_and_a_faint_reach_the_field_token() -> None:
    query = wished()
    facts = facts_of(query)
    assert "wish" in facts["own"]["pending_slot"]["a"]
    remaining = facts["own"]["pending_slot"]["a"]["wish"]["remaining"]
    encoded = ENCODE.encode(query)
    name = "hist_pending_own_a_wish"
    assert encoded.field_num[LAYOUT.field_num.index(name)].item() == pytest.approx(
        remaining / ON.turn_scale
    )
    assert encoded.field_num[LAYOUT.field_num.index("hist_total_fainted_opponent")].item() == (
        pytest.approx(1 / ON.max_team)
    )


def test_the_ate_berry_flag_and_the_last_item_are_read_per_pokemon() -> None:
    query = played()
    log = query["observation"]["log"]
    at = max(i for i, line in enumerate(log) if line.startswith("|request|"))
    extra = ["|-enditem|p2a: Venusaur|Sitrus Berry|[eat]"]
    query["observation"]["log"] = [*log[:at], *extra, *log[at:]]
    encoded = ENCODE.encode(query)
    foe_a = encoded.n_own
    assert encoded.mon_num[foe_a, num("hist_ate_berry")].item() == 1.0
    assert encoded.mon_cat[foe_a, LAST_ITEM].item() == game().items.lookup("sitrusberry")
    assert encoded.mon_num[0, num("hist_ate_berry")].item() == 0.0


# -- unknown is its own value --------------------------------------------------------------------


def test_an_illusion_makes_the_position_unknown_and_nothing_else_is_filled_in() -> None:
    d = json.loads((FORMAT_DIR / "illusion" / "battle.json").read_text())
    wide = replace(ON, max_moves=8)  # the Illusion battle shows more moves than four
    encoder, layout = Encoder(game(), wide), Layout.of(wide, game())
    state_at = [name for name, _ in layout.mon_cat].index("history_state")
    seen = {NONE: 0, UNKNOWN: 0, KNOWN: 0}
    for player in ("p1", "p2"):
        log = d["channels"][player]["lines"]
        team = d["teams"][player]
        for i, body in request_points(log):
            kind = "preview" if body.get("teamPreview") else "battle"
            q = {
                "observation": observation(player, log[: i + 1], team, kind),
                "candidates": candidates_for(body)[:2],
            }
            encoded = encoder.encode(q)
            for token in range(encoded.mon_cat.shape[0]):
                state = int(encoded.mon_cat[token, state_at])
                seen[state] += 1
                if state == UNKNOWN:  # unknown: no number of the position is filled in
                    first = layout.mon_num.index("hist_move_actions")
                    last = layout.mon_num.index("hist_ate_berry")
                    assert not encoded.mon_num[token, first:last].any()
    assert seen[UNKNOWN] > 0
    assert seen[KNOWN] > 0
    assert seen[NONE] > 0


def test_a_tracker_that_refused_a_line_makes_every_fact_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(HistoryTracker, "facts", lambda self: {"refused": "malformed_line"})
    encoded = ENCODE.encode(played())
    assert (encoded.mon_cat[:, STATE] == UNKNOWN).all()
    assert (encoded.mon_cat[:, ROSTER] == UNKNOWN).all()
    assert (encoded.mon_cat[:, LAST_ITEM] == UNKNOWN).all()
    first, last = num("hist_move_actions"), num("hist_ate_berry") + 1
    assert not encoded.mon_num[:, first:last].any()
    cats = [name for name, _ in LAYOUT.field_cat]
    assert encoded.field_cat[cats.index("history_available")].item() == UNKNOWN
    assert encoded.field_cat[cats.index("history_last_move")].item() == NONE
    assert not encoded.field_num[len(BASE.field_num) :].any()


def test_a_state_without_the_history_facts_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("areneon_ml.model.encoder.reduce_channel", _reduce_without_history)
    with pytest.raises(EncodingError, match="history facts of the reduced state"):
        ENCODE.encode(played())


def _reduce_without_history(*args: Any, **kwargs: Any) -> Any:
    state = reduce_channel(*args, **kwargs)
    state.history = None
    return state


def test_a_fact_of_a_shape_the_encoder_does_not_know_is_refused() -> None:
    facts = facts_of(played())
    broken = copy.deepcopy(facts)
    broken["own"]["active"]["a"]["attackers_this_turn"] = ["elsewhere:a"]
    view = history_of.HistoryView(broken, game(), turn_scale=100.0, max_moves=4, max_team=6)
    state = reduce_channel(
        "p1",
        played()["observation"]["log"],
        played()["observation"]["team"]["team"],
        game().vocabulary,
    )
    with pytest.raises(history_of.HistoryError, match="attacker"):
        view.mon("own", state.own.slots, state.own.roster, state.own.roster[0])
    with pytest.raises(history_of.HistoryError, match="reduced state"):
        history_of.HistoryView(None, game(), turn_scale=1.0, max_moves=4, max_team=6)


def test_the_view_reads_every_unknown_of_a_synthetic_facts_object() -> None:
    facts = copy.deepcopy(facts_of(played()))
    own = facts["own"]
    own["active"]["a"].update(
        attackers_this_turn=None, last_move_failed=None, stall_counter=9, moves_used=["gigadrain"]
    )
    own["active"]["a"]["volatile_started"] = {"taunt": {"started_turn": 0, "after_residual": False}}
    own["active"]["b"] = {"name": "Charizard", "unknown": True}
    own["roster"]["Venusaur"] = {"ate_berry": None, "last_item": None}
    own["roster"]["Charizard"] = {"ate_berry": True, "last_item": history_of.UNSETTLED}
    facts["opponent"]["roster"]["Venusaur"]["last_item"] = "sitrusberry"
    view = history_of.HistoryView(facts, game(), turn_scale=100.0, max_moves=4, max_team=6)
    state = reduce_channel(
        "p1",
        played()["observation"]["log"],
        played()["observation"]["team"]["team"],
        game().vocabulary,
    )
    cat, nums = view.mon("own", state.own.slots, state.own.roster, state.own.roster[0])
    # The roster entry is unknown, the position's facts are known except two flagged ones.
    assert cat == [KNOWN, UNKNOWN, UNKNOWN]  # an unknown entry has no known last item
    named = dict(zip(history_of.mon_num_columns(4), nums, strict=True))
    assert named["hist_attackers_unknown"] == 1.0
    assert named["hist_last_move_failed_unknown"] == 1.0
    assert named["hist_stall_counter"] == pytest.approx(9 / 729)
    assert named["hist_volatile_count"] == pytest.approx(1 / history_of.COUNT_SCALE)
    assert named["hist_volatile_oldest_age"] == pytest.approx(2 / 100)
    assert named["hist_ate_berry"] == 0.0
    shown = state.own.roster[0].moves_own or state.own.roster[0].moves_revealed
    assert named[f"hist_used_{shown.index('gigadrain') + 1}"] == 1.0
    cat, nums = view.mon("own", state.own.slots, state.own.roster, state.own.roster[1])
    assert cat == [UNKNOWN, KNOWN, UNKNOWN]  # an unsettled last item is no item
    assert dict(zip(history_of.mon_num_columns(4), nums, strict=True))["hist_ate_berry"] == 1.0
    # A position whose fact names another Pokémon is unknown, not guessed.
    clash = PokemonState(own=True, name="Someone")
    state.own.roster.append(clash)
    state.own.slots["c"] = len(state.own.roster) - 1
    cat, _ = view.mon("own", state.own.slots, state.own.roster, clash)
    assert cat == [UNKNOWN, NONE, NONE]


# -- A1/A2, the whole corpus, composition ------------------------------------------------------


def test_the_a1_and_a2_forms_encode_equally_on_the_committed_pairs() -> None:
    assert PAIRS["decisions"]
    for decision in PAIRS["decisions"]:
        assert same(
            ENCODE.encode(pair_query(PAIRS, decision, "a1")),
            ENCODE.encode(pair_query(PAIRS, decision, "a2")),
        ), decision["fixture"]


def test_every_request_of_the_random_corpus_encodes_with_only_known_or_no_facts() -> None:
    seen = 0
    for path in RANDOM[:8]:
        d = json.loads(path.read_text())
        for player in ("p1", "p2"):
            log = d["channels"][player]["lines"]
            for i, body in request_points(log):
                kind = "preview" if body.get("teamPreview") else "battle"
                q = {
                    "observation": observation(player, log[: i + 1], d["teams"][player], kind),
                    "candidates": candidates_for(body)[:2],
                }
                encoded = ENCODE.encode(q)
                states = set(encoded.mon_cat[:, STATE].tolist())
                assert states <= {NONE, KNOWN}
                assert torch.isfinite(encoded.mon_num).all()
                seen += 1
    assert seen > 100


def test_an_incremental_session_gives_the_tensors_of_a_fresh_encode() -> None:
    d = json.loads(RANDOM[0].read_text())
    log = d["channels"]["p1"]["lines"]
    session = ENCODE.session()
    for i, body in request_points(log):
        kind = "preview" if body.get("teamPreview") else "battle"
        q = {
            "observation": observation("p1", log[: i + 1], d["teams"]["p1"], kind),
            "candidates": candidates_for(body)[:2],
        }
        assert same(session.encode(q), ENCODE.encode(q))


FEATURE_1 = sample("policy-query/valid/battle-features.json")
FEATURE_2 = sample("policy-query/valid/battle-features-2.json")
FEATURE_OUTCOME = sample("policy-query/valid/battle-features-outcome.json")


def feature_option(query: dict[str, Any]) -> dict[str, str]:
    return {key: query["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}


@pytest.mark.parametrize(
    ("others", "query"),
    [
        ({RECENT_HISTORY: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_MATCHUP: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_TARGET: True}, sample("policy-query/valid/battle.json")),
        ({CANDIDATE_FEATURES: feature_option(FEATURE_1)}, FEATURE_1),
        ({CANDIDATE_FEATURES: feature_option(FEATURE_2)}, FEATURE_2),
        ({CANDIDATE_FEATURES: feature_option(FEATURE_OUTCOME)}, FEATURE_OUTCOME),
        (
            {
                RECENT_HISTORY: True,
                CANDIDATE_MATCHUP: True,
                CANDIDATE_TARGET: True,
                CANDIDATE_FEATURES: feature_option(FEATURE_OUTCOME),
            },
            FEATURE_OUTCOME,
        ),
    ],
    ids=["recent", "matchup", "target", "features-1", "features-2", "outcome", "all"],
)
def test_it_composes_with_each_other_encoder_option(
    others: dict[str, Any], query: dict[str, Any]
) -> None:
    without = replace(TINY_ENCODER, options=read_options("encoder", others))
    both = replace(TINY_ENCODER, options=read_options("encoder", {**others, HISTORY_FACTS: True}))
    assert encoder_version(game(), both) not in {
        encoder_version(game(), without),
        ENCODE.version,
    }
    a, b = Encoder(game(), without).encode(query), Encoder(game(), both).encode(query)
    layout, base = Layout.of(both, game()), Layout.of(without, game())
    added = dict(history_of.mon_cat_columns())
    keep = [i for i, (name, _) in enumerate(layout.mon_cat) if name not in added]
    assert [layout.mon_cat[i] for i in keep] == list(base.mon_cat)
    assert torch.equal(b.mon_cat[:, keep], a.mon_cat)
    kept_num = [i for i, name in enumerate(layout.mon_num) if not name.startswith("hist_")]
    assert torch.equal(b.mon_num[:, kept_num], a.mon_num)
    kept_field = [i for i, name in enumerate(layout.field_num) if not name.startswith("hist_")]
    assert torch.equal(b.field_num[kept_field], a.field_num)
    for name, tensor in a.tensors().items():
        if name not in ("mon_cat", "mon_num", "field_num", "field_cat"):
            assert torch.equal(b.tensors()[name], tensor), name
    network: Any = init_network("battle", game(), replace(TINY_MODEL, encoder=both), seed=3)
    out = network(collate([b]))
    assert torch.isfinite(out.logits[:, 0]).all()
    # The new columns are an input of the score.
    changed = collate([b])
    changed.mon_num[..., layout.mon_num.index("hist_times_attacked")] += 1.0
    assert not torch.equal(network(changed).logits, out.logits)


def test_it_composes_with_exact_volatiles_on_a_version_two_table() -> None:
    v2 = game2()
    config = replace(
        TINY_ENCODER,
        options=read_options("encoder", {HISTORY_FACTS: True, "exact_volatiles": True}),
    )
    encoder = Encoder(v2, config)
    encoded = encoder.encode(played())
    layout = Layout.of(config, v2)
    assert encoded.mon_num.shape[1] == len(layout.mon_num)
    assert encoded.field_num.shape[0] == len(layout.field_num)
    assert (
        encoder.version
        != Encoder(
            v2, replace(config, options=read_options("encoder", {"exact_volatiles": True}))
        ).version
    )
    network: Any = init_network("battle", v2, replace(TINY_MODEL, encoder=config), seed=3)
    assert torch.isfinite(network(collate([encoded])).logits[:, 0]).all()


@pytest.mark.parametrize(
    "network_options",
    [{"shared_trunk": True}, {"separate_value": "network"}, {"separate_value": "detached"}],
)
def test_it_composes_with_the_network_options(network_options: dict[str, Any]) -> None:
    model = replace(ON_MODEL, options=read_options("network", network_options))
    for role in ("battle", "preview"):
        assert init_network(role, game(), model, seed=3) is not None
    battle: Any = init_network("battle", game(), model, seed=3)
    out = battle(collate([ENCODE.encode(played())]))
    assert torch.isfinite(out.logits[:, 0]).all()
    assert torch.isfinite(out.value).all()


# -- the network and the checkpoint --------------------------------------------------------------


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3).eval()
    one = ENCODE.encode(played())
    other = ENCODE.encode(sample("policy-query/valid/battle-a2.json"))
    with torch.no_grad():
        alone = network(collate([one]))
        network(collate([other]))
        after = network(collate([one]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)


def test_a_padded_batch_matches_its_queries_scored_alone() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3).eval()
    one = ENCODE.encode(played())
    two = ENCODE.encode(sample("policy-query/valid/battle-a2.json"))
    with torch.no_grad():
        both = network(collate([one, two]))
        alone_one, alone_two = network(collate([one])), network(collate([two]))
    n = two.pending.shape[0]
    assert torch.allclose(both.logits[0, : one.pending.shape[0]], alone_one.logits[0], atol=1e-5)
    assert torch.allclose(both.logits[1, :n], alone_two.logits[0, :n], atol=1e-5)


def test_a_training_step_reaches_the_new_columns() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3)
    batch = collate([ENCODE.encode(wished())] * 2)
    out = network(batch)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    assert network.mon_num.weight.grad[:, len(BASE.mon_num) :].abs().sum() > 0
    assert network.field_num.weight.grad[:, len(BASE.field_num) :].abs().sum() > 0
    assert network.embeddings["history_state"].weight.grad.abs().sum() > 0


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    network = init_network("battle", game(), ON_MODEL, seed=3)
    data = checkpoint_bytes(network, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.layout.mon_cat == LAYOUT.mon_cat
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)


def test_the_game_data_type_is_the_one_the_layout_reads() -> None:
    assert isinstance(game(), GameData)
