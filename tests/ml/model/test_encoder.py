"""The encoder (ADR-010 §2, §3): what it reads, what it ignores, and that it fails closed."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
from areneon_ml.model.candidates import UnsupportedCandidate
from areneon_ml.model.encoder import (
    EncodedQuery,
    Encoder,
    EncoderConfig,
    EncodingError,
    Layout,
    bucket,
    encoder_version,
)
from areneon_ml.model.gamedata import GameData
from areneon_ml.reducer import (
    OpponentSecretError,
    PokemonState,
    ReducerError,
    SideState,
    UnknownLineError,
)

from .support import (
    A2_DIR,
    RANDOM,
    TINY_ENCODER,
    candidates_for,
    encoder,
    game,
    observation,
    request_points,
    sample,
    table,
)

MANY = 40


def same(a: EncodedQuery, b: EncodedQuery) -> bool:
    return a.kind == b.kind and all(torch.equal(x, b.tensors()[k]) for k, x in a.tensors().items())


def strip_rqid(line: str) -> str:
    if not line.startswith("|request|"):
        return line
    body = json.loads(line[len("|request|") :])
    body.pop("rqid", None)
    return "|request|" + json.dumps(body, separators=(",", ":"))


def a2_fixture_points() -> list[
    tuple[str, str, int, list[str], list[dict[str, Any]], dict[str, Any]]
]:
    out = []
    for path in sorted(A2_DIR.glob("bot-vs-bot-*.json")):
        data = json.loads(path.read_text())
        for player in ("p1", "p2"):
            channels = data["channels"][player]
            for channel in channels if isinstance(channels, list) else [channels]:
                team = [{"gender": "", **mon} for mon in data["served_teams"][player]]
                for index, body in request_points(channel["lines"]):
                    out.append(
                        (path.name, player, index, channel["lines"][: index + 1], team, body)
                    )
    return out


def test_the_a1_and_a2_forms_of_every_adr005_fixture_request_encode_identically() -> None:
    points = a2_fixture_points()
    assert len(points) >= MANY
    context = sample("policy-query/valid/battle-a2.json")["observation"]["decision_context"]
    with_rqid = 0
    for name, player, index, lines, team, body in points:
        candidates = candidates_for(body)
        a2 = observation(player, lines, team, "battle") | {"decision_context": context}
        a2["identity"] = {"request_seq": 9, "request_hash": "a" * 64, "rqid": 7}
        a1 = observation(player, [strip_rqid(line) for line in lines], team, "battle")
        a1["identity"] = {"request_seq": 9, "request_hash": "a" * 64}
        with_rqid += lines != a1["log"]
        first = encoder().encode({"observation": a1, "candidates": candidates})
        second = encoder().encode({"observation": a2, "candidates": candidates})
        assert same(first, second), f"{name} {player} line {index}"
    assert with_rqid == len(points)  # the A2 forms really carry an rqid


def test_the_a1_and_a2_policy_query_samples_encode_identically() -> None:
    a1, a2 = sample("policy-query/valid/battle.json"), sample("policy-query/valid/battle-a2.json")
    assert "decision_context" in a2["observation"]
    assert a1["observation"]["identity"] != a2["observation"]["identity"]
    assert a1["observation"]["log"] != a2["observation"]["log"]
    assert same(encoder().encode(a1), encoder().encode(a2))


def test_nothing_but_the_observation_and_the_candidates_changes_the_tensors() -> None:
    query = sample("policy-query/valid/battle.json")
    base = encoder().encode(query)
    other = copy.deepcopy(query)
    other.update(query_id=77, policy_id="policy:other", sampling={"kind": "sample", "u": 5})
    other["identity"] = other["observation"]["identity"] = {
        "request_seq": 99,
        "request_hash": "f" * 64,
    }
    other["observation"]["manifest"]["regulation"] = "other"
    other["observation"]["data_class"] = "private"
    assert same(base, encoder().encode(other))
    verdicts = copy.deepcopy(query)
    for candidate in verdicts["candidates"]:
        candidate.update(legal=True, error="hidden verdict")
    assert same(base, encoder().encode(verdicts))


def test_the_pending_flag_is_an_input_and_so_are_the_log_and_the_candidates() -> None:
    query = sample("policy-query/valid/battle.json")
    base = encoder().encode(query)
    flagged = copy.deepcopy(query)
    flagged["candidates"][0]["pending"] = not flagged["candidates"][0]["pending"]
    changed = encoder().encode(flagged)
    assert not torch.equal(base.pending, changed.pending)
    assert torch.equal(base.cand_cat, changed.cand_cat)
    assert int(base.pending.sum()) != int(changed.pending.sum())
    shorter = copy.deepcopy(query)
    shorter["candidates"] = shorter["candidates"][:-1]
    assert encoder().encode(shorter).pending.shape[0] == base.pending.shape[0] - 1
    longer_log = copy.deepcopy(query)
    longer_log["observation"]["log"].insert(-1, "|-weather|SunnyDay")
    assert not torch.equal(base.field_cat, encoder().encode(longer_log).field_cat)


def test_the_opponent_tokens_hold_only_what_was_revealed() -> None:
    encoded = encoder().encode(sample("policy-query/valid/battle.json"))
    layout = Layout.of(TINY_ENCODER)
    num = {name: i for i, name in enumerate(layout.mon_num)}
    cat = {name: i for i, (name, _) in enumerate(layout.mon_cat)}
    opponents = slice(encoded.n_own, None)
    assert encoded.mon_num[: encoded.n_own, num["own"]].eq(1.0).all()
    assert encoded.mon_num[opponents, num["own"]].eq(0.0).all()
    for name in ("stats_known", "hp_stat", "stat_atk", "stat_spe", "pp_1", "pp_4"):
        assert encoded.mon_num[opponents, num[name]].eq(0.0).all(), name
    assert encoded.mon_cat[opponents, cat["ability"]].le(1).all()  # none or unknown
    assert encoded.mon_cat[: encoded.n_own, cat["move_1"]].gt(1).all()


def test_a_channel_with_a_split_marker_is_refused() -> None:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    query["observation"]["log"].insert(3, "|split|p2")
    with pytest.raises(OpponentSecretError):
        encoder().encode(query)


def test_every_request_point_of_the_random_corpus_encodes_into_range() -> None:
    sizes = {
        "species": len(game().species), "item": len(game().items), "ability": len(game().abilities),
        "status": len(game().statuses), "move": len(game().moves), "weather": len(game().weathers),
        "terrain": len(game().terrains), "gender": 4, "slot": 4, "request": 4,
    }  # fmt: skip
    layout = Layout.of(TINY_ENCODER)
    points = 0
    for path in RANDOM:
        battle = json.loads(path.read_text())
        for player in ("p1", "p2"):
            lines = battle["channels"][player]["lines"]
            for index, body in request_points(lines):
                obs = observation(player, lines[: index + 1], battle["teams"][player], "battle")
                try:
                    encoded = encoder().encode(
                        {"observation": obs, "candidates": candidates_for(body)}
                    )
                except UnknownLineError:
                    break  # a line type the reducer names as unsupported (Illusion)
                points += 1
                for column, (_, kind) in enumerate(layout.mon_cat):
                    assert encoded.mon_cat[:, column].lt(sizes[kind]).all(), kind
                    assert encoded.mon_cat[:, column].ge(0).all()
                for column, (_, kind) in enumerate(layout.field_cat):
                    assert int(encoded.field_cat[column]) < sizes[kind]
                for tensor in (encoded.mon_num, encoded.field_num, encoded.pending):
                    assert torch.isfinite(tensor).all()
                rows = encoded.mon_cat.shape[0]
                assert encoded.cand_tok.max() < rows
                assert encoded.team_tok.max() < rows
                assert encoded.n_own == len(battle["teams"][player])
    assert points > 500


def test_encoding_is_deterministic() -> None:
    query = sample("policy-query/valid/preview.json")
    assert same(encoder().encode(query), Encoder(game(), TINY_ENCODER).encode(query))


def test_the_preview_query_names_team_positions_and_the_battle_query_names_slots() -> None:
    preview = encoder().encode(sample("policy-query/valid/preview.json"))
    assert preview.kind == "preview"
    assert preview.team_tok.ge(0).all()
    assert preview.cand_cat.eq(0).all()
    assert preview.cand_tok.eq(-1).all()
    battle = encoder().encode(sample("policy-query/valid/battle.json"))
    assert battle.team_tok.eq(-1).all()
    assert set(battle.cand_cat[:, :, 0].flatten().tolist()) == {1, 2}  # moves and switches
    assert battle.cand_tok[:, :, 0].ge(0).all()


def test_a_switch_names_the_pokemon_at_that_position_of_the_request() -> None:
    query = sample("policy-query/valid/battle.json")
    lines = query["observation"]["log"]
    body = json.loads(lines[-1][len("|request|") :])
    roster = [p["ident"].split(": ", 1)[1] for p in body["side"]["pokemon"]]
    candidates = [{"choice": "switch 3, switch 4", "pending": False}]
    encoded = encoder().encode({"observation": query["observation"], "candidates": candidates})
    team = [m["name"] or m["species"] for m in query["observation"]["team"]["team"]]
    assert [team[int(i)] for i in encoded.cand_tok[0, :, 1]] == [roster[2], roster[3]]
    assert [team[int(i)] for i in encoded.cand_tok[0, :, 0]] == [roster[0], roster[1]]


def test_what_cannot_be_encoded_is_refused() -> None:
    query = sample("policy-query/valid/battle.json")
    observation_ = query["observation"]
    no_request = {
        **observation_,
        "log": [line for line in observation_["log"] if not line.startswith("|request|")],
    }
    with pytest.raises(EncodingError, match="no decision"):
        encoder().encode({"observation": no_request, "candidates": query["candidates"]})
    waiting = {
        **observation_,
        "log": [
            *observation_["log"][:-1],
            '|request|{"wait":true,"side":'
            + json.dumps(json.loads(observation_["log"][-1][9:])["side"])
            + "}",
        ],
    }
    with pytest.raises(EncodingError, match="no decision"):
        encoder().encode({"observation": waiting, "candidates": query["candidates"]})
    foreign = copy.deepcopy(observation_)
    request = json.loads(foreign["log"][-1][9:])
    request["side"]["pokemon"][0]["ident"] = "p1: Nobody"
    foreign["log"][-1] = "|request|" + json.dumps(request)
    with pytest.raises(ReducerError, match="not in the own team"):
        encoder().encode({"observation": foreign, "candidates": query["candidates"]})


@pytest.mark.parametrize(
    "choice",
    [
        "default",
        "move 5, move 1",  # the request lists four moves at most
        "move 9, move 1",
        "switch 7, move 1",
        "move 1, switch 6",
        "team 1234567",
    ],
)
def test_a_candidate_the_request_cannot_back_is_unsupported(choice: str) -> None:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    query["candidates"] = [{"choice": choice, "pending": False}]
    with pytest.raises(UnsupportedCandidate):
        encoder().encode(query)


def test_a_move_beyond_those_the_request_lists_is_unsupported() -> None:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    request = json.loads(query["observation"]["log"][-1][9:])
    request["active"][0]["moves"] = request["active"][0]["moves"][:1]
    query["observation"]["log"][-1] = "|request|" + json.dumps(request)
    query["candidates"] = [{"choice": "move 2 1, move 1 1", "pending": False}]
    with pytest.raises(UnsupportedCandidate, match="move the request lacks"):
        encoder().encode(query)
    query["candidates"] = [{"choice": "move 1 1, move 4 1", "pending": False}]
    assert encoder().encode(query).cand_cat[0, 0, 2] == 1


def test_a_preview_order_longer_than_max_picks_is_unsupported() -> None:
    query = sample("policy-query/valid/preview.json")
    query = {**query, "candidates": [{"choice": "team 12345", "pending": False}]}
    with pytest.raises(UnsupportedCandidate, match="max_picks"):
        encoder().encode(query)


def test_a_move_in_a_forced_switch_request_is_unsupported() -> None:
    query = copy.deepcopy(sample("policy-query/valid/battle.json"))
    request = json.loads(query["observation"]["log"][-1][9:])
    request.pop("active")
    request["forceSwitch"] = [True, False]
    query["observation"]["log"][-1] = "|request|" + json.dumps(request)
    query["candidates"] = [{"choice": "move 1, pass", "pending": False}]
    with pytest.raises(UnsupportedCandidate, match="move the request lacks"):
        encoder().encode(query)
    query["candidates"] = [{"choice": "switch 3, pass", "pending": False}]
    assert encoder().encode(query).cand_cat[0, 0, 0] == 2


def test_a_side_larger_than_max_team_is_refused() -> None:
    small = Encoder(
        game(), EncoderConfig(max_team=5, max_picks=4, volatile_buckets=4, condition_buckets=4)
    )
    with pytest.raises(EncodingError, match="max_team"):
        small.encode(sample("policy-query/valid/preview.json"))


def test_an_unsupported_gender_is_refused() -> None:
    mon = PokemonState(own=True, base_species="venusaur", species="venusaur", gender="X")
    with pytest.raises(EncodingError, match="gender"):
        encoder()._mon_cat(SideState(player="p1"), mon)


def test_the_encoder_version_carries_the_table_hash_and_every_feature_choice() -> None:
    version = encoder().version
    assert version == encoder_version(game(), TINY_ENCODER)
    assert version.startswith("enc1:")
    assert len(version) < 128
    assert encoder_version(game(), EncoderConfig()) != version
    assert encoder_version(game(), EncoderConfig(max_moves=3)) != encoder_version(
        game(), EncoderConfig()
    )
    changed = copy.deepcopy(table())
    changed["items"][0]["fling_power"] += 1
    assert encoder_version(GameData.from_table(changed), TINY_ENCODER) != version


@pytest.mark.parametrize(
    "change",
    [
        {"volatile_buckets": 0},
        {"condition_buckets": 0},
        {"max_moves": 0},
        {"max_team": 0},
        {"max_picks": 0},
        {"max_picks": 7},
        {"stat_scale": 0.0},
        {"turn_scale": -1.0},
    ],
)
def test_an_invalid_encoder_config_is_refused(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"at least 1|exceed|positive"):
        EncoderConfig(**change)


def test_buckets_are_stable_and_in_range() -> None:
    assert bucket("stealthrock", 32) == bucket("stealthrock", 32)
    assert all(0 <= bucket(name, 7) < 7 for name in ("a", "spikes", "tailwind", "reflect"))
    assert bucket("a", 1000) != bucket("b", 1000)
