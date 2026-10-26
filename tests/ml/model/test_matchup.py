"""The ``candidate_matchup`` encoder option (ADR-022 tier B, ADR-010 erratum A5 item 1).

Each move candidate carries the engine-data type-chart cells of the move's type against the shown
types of each opposing position and the ally, and a same-type flag. The cells are categories of
the export's own codes: no multiplier or same-type value exists in model code.
"""

from __future__ import annotations

import ast
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
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import (
    BASE_CAND_COLUMNS,
    MATCHUP_TYPES,
    SAME_TYPE_NOT,
    SAME_TYPE_YES,
    EncodedQuery,
    Encoder,
    Layout,
    encoder_version,
)
from areneon_ml.model.gamedata import NONE, RESERVED, UNKNOWN, GameDataError
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import CANDIDATE_MATCHUP, read_options

from .support import (
    RANDOM,
    TINY_ENCODER,
    TINY_MODEL,
    candidates_for,
    encoder,
    game,
    observation,
    request_points,
    sample,
    table,
)
from .test_encoder import PAIRS, a2_fixture_points, kind_of, pair_query, same
from .test_options import TODAY_ENCODER_VERSION

ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_MATCHUP: True}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
WIDTH = BASE_CAND_COLUMNS + 3 * MATCHUP_TYPES + 1
ENCODE = Encoder(game(), ON)
SOURCES = Path(__file__).resolve().parents[3] / "ml" / "areneon_ml" / "model"


def candidate(query: dict[str, Any], choice: str) -> int:
    return [c["choice"] for c in query["candidates"]].index(choice)


def columns(encoded: EncodedQuery, index: int, slot: int) -> list[int]:
    """The matchup columns of one slot action: foe 1, foe 2 and ally cells, then the flag."""
    return encoded.cand_cat[index, slot, BASE_CAND_COLUMNS:].tolist()


# -- the option is absent: today's bytes ---------------------------------------


def test_with_the_option_absent_the_layout_and_the_version_are_today_s() -> None:
    assert len(Layout.of(TINY_ENCODER).cand_cat) == BASE_CAND_COLUMNS
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    assert encoder().encode(sample("policy-query/valid/battle.json")).cand_cat.shape[2] == 5


def test_with_the_option_the_old_columns_and_every_other_tensor_are_unchanged() -> None:
    query = sample("policy-query/valid/battle.json")
    plain, with_option = encoder().encode(query), ENCODE.encode(query)
    assert ENCODE.version != encoder().version
    assert with_option.cand_cat.shape[2] == WIDTH
    assert torch.equal(with_option.cand_cat[:, :, :BASE_CAND_COLUMNS], plain.cand_cat)
    for name, tensor in plain.tensors().items():
        if name != "cand_cat":
            assert torch.equal(with_option.tensors()[name], tensor), name


def test_the_option_moves_the_version_and_only_true_is_a_value() -> None:
    assert encoder_version(game(), ON) != TODAY_ENCODER_VERSION
    assert ON.as_dict()["options"] == {CANDIDATE_MATCHUP: True}
    for bad in (False, 0, 1, 2):
        with pytest.raises(ConfigError, match="invalid value"):
            ModelConfig.from_dict(
                {
                    **TINY_MODEL.as_dict(),
                    "encoder": {**TINY_ENCODER.as_dict(), "options": {CANDIDATE_MATCHUP: bad}},
                }
            )
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL


# -- hand-checked cells --------------------------------------------------------
# Position order of a candidate's columns: foe 1, foe 2, ally (two type cells each), the flag.
# A cell is RESERVED (2) + the engine's code: 0 normal, 1 super effective, 2 resisted, 3 immune.
# The foes are Venusaur (grass, poison) and Charizard (fire, flying), the ally is Charizard.


def test_cells_equal_the_export_for_a_single_and_a_dual_typed_target() -> None:
    query = sample("policy-query/valid/battle.json")
    encoded = ENCODE.encode(query)
    # Slot 1, Venusaur, Giga Drain (grass): resisted by grass, poison, fire and flying.
    i = candidate(query, "move 1 1, move 1 1")
    assert columns(encoded, i, 0) == [4, 4, 4, 4, 4, 4, SAME_TYPE_YES]
    # Slot 2, Charizard, Flamethrower (fire) at the foe Venusaur: super effective on grass,
    # normal on poison; fire resists fire, flying is normal; the ally Venusaur the same.
    assert columns(encoded, i, 1) == [3, 2, 4, 2, 3, 2, SAME_TYPE_YES]
    # Venusaur's Sludge Bomb (poison): super effective on grass, resisted by poison.
    j = candidate(query, "move 2 1, move 3")
    assert columns(encoded, j, 0) == [3, 4, 2, 2, 2, 2, SAME_TYPE_YES]
    # A move of the user's second type counts: Charizard's Air Slash (flying).
    k = candidate(query, "move 1 1, move 2 1")
    assert columns(encoded, k, 1)[-1] == SAME_TYPE_YES


def test_an_immune_cell_a_dual_typed_target_and_a_move_outside_the_user_s_types() -> None:
    # Charizard's Air Slash becomes Earthquake (ground) in the request and the team: ground is
    # super effective on fire and immune on flying, so the foe Charizard shows [3, 5].
    query = json.loads(
        json.dumps(sample("policy-query/valid/battle.json")).replace("airslash", "earthquake")
    )
    encoded = ENCODE.encode(query)
    i = candidate(query, "move 1 1, move 2 1")
    assert columns(encoded, i, 1) == [4, 3, 3, 5, 4, 3, SAME_TYPE_NOT]
    assert RESERVED + 3 == 5  # the immune code, as the export holds it
    assert (
        table()["types"][game().types.lookup("flying") - RESERVED]["damage_taken"][
            game().types.lookup("ground") - RESERVED
        ]
        == 3
    )


def test_a_status_move_and_a_switch_and_a_pass_carry_none() -> None:
    query = sample("policy-query/valid/battle.json")
    encoded = ENCODE.encode(query)
    protect = candidate(query, "move 3, move 2 1")
    assert columns(encoded, protect, 0) == [NONE] * (WIDTH - BASE_CAND_COLUMNS)  # Protect
    sleep_powder = candidate(query, "move 4 1, move 1 -1")
    assert columns(encoded, sleep_powder, 0) == [NONE] * (WIDTH - BASE_CAND_COLUMNS)
    switch = candidate(query, "move 1 1, switch 4")
    assert columns(encoded, switch, 1) == [NONE] * (WIDTH - BASE_CAND_COLUMNS)
    assert encoded.cand_cat[switch, 1, :BASE_CAND_COLUMNS].tolist()[0] == 2


def test_an_empty_position_and_a_fainted_one_carry_none() -> None:
    query = sample("policy-query/valid/battle.json")
    fainted = copy.deepcopy(query)
    fainted["observation"]["log"].insert(-1, "|faint|p2b: Charizard")
    i = candidate(query, "move 1 1, move 2 1")
    cells = columns(ENCODE.encode(fainted), i, 1)
    assert cells[2:4] == [NONE, NONE]  # foe 2 has fainted
    assert cells[:2] == columns(ENCODE.encode(query), i, 1)[:2]
    # The matchup cells of GameData: a missing second type is none; a reserved move row too.
    data = game()
    mono = data.species.lookup("blastoise")
    assert data.matchup_cells(data.moves.lookup("thunderbolt"), mono, 2) == [RESERVED + 1, NONE]
    assert data.matchup_cells(NONE, mono, 2) == [NONE, NONE]
    assert data.matchup_cells(UNKNOWN, data.species.lookup("venusaur"), 2) == [NONE, NONE]
    with pytest.raises(GameDataError, match="more than 1 types"):
        data.matchup_cells(data.moves.lookup("thunderbolt"), data.species.lookup("venusaur"), 1)


# -- privacy: the player's view only --------------------------------------------


def matchup_part(encoded: EncodedQuery) -> torch.Tensor:
    return encoded.cand_cat[:, :, BASE_CAND_COLUMNS:]


def test_the_cells_read_only_the_species_a_position_shows() -> None:
    # Terastallisation is not in this format (no |-terastallize| line is read by the reducer, so
    # a channel with one is refused), and a type change by a volatile is not tracked: the cells
    # come from the shown species alone. Revealing a foe's item, ability or move changes none.
    query = sample("policy-query/valid/battle.json")
    base = matchup_part(ENCODE.encode(query))
    revealed = copy.deepcopy(query)
    revealed["observation"]["log"][-1:-1] = [
        "|-ability|p2a: Venusaur|Overgrow|[from] ability: Overgrow",
        "|-item|p2b: Charizard|Leftovers",
        "|move|p2a: Venusaur|Sludge Bomb|p1a: Venusaur",
    ]
    assert torch.equal(matchup_part(ENCODE.encode(revealed)), base)
    # Negative control: showing another species at a position changes the cells.
    switched = copy.deepcopy(query)
    log = switched["observation"]["log"]
    at = next(n for n, line in enumerate(log) if line.startswith("|switch|p2a:"))
    log[at] = "|switch|p2a: Blastoise|Blastoise, L50, M|100/100"
    assert not torch.equal(matchup_part(ENCODE.encode(switched)), base)


def test_the_unrevealed_opponent_team_is_not_an_input() -> None:
    # The encoder reads the observation and the candidates only, so there is no field for the
    # opponent's real team; two queries that differ only in `decision_context` and identity
    # encode equal tensors with the option, as without it.
    query = sample("policy-query/valid/battle-a2.json")
    other = copy.deepcopy(query)
    other["observation"]["identity"]["request_hash"] = "b" * 64
    other["observation"]["decision_context"] = None
    assert same(ENCODE.encode(query), ENCODE.encode(other))


# -- A1/A2 equality and the corpus -----------------------------------------------


def test_the_a1_and_a2_forms_encode_equally_with_the_option_on_the_committed_pairs() -> None:
    decisions = PAIRS["decisions"]
    moves = 0
    for decision in decisions:
        one, two = pair_query(PAIRS, decision, "a1"), pair_query(PAIRS, decision, "a2")
        first, second = ENCODE.encode(one), ENCODE.encode(two)
        assert same(first, second), decision["fixture"]
        moves += int((matchup_part(first) != 0).any())
    assert moves > 10  # the pairs really hold move candidates with cells


def test_the_a2_recordings_encode_equally_with_the_option_on() -> None:
    for name, player, index, lines, team, body in a2_fixture_points():
        a1 = observation(player, [line for line in lines], team, kind_of(body))
        a1["identity"] = {"request_seq": 1, "request_hash": "a" * 64}
        a2 = copy.deepcopy(a1)
        a2["identity"] = {"request_seq": 5, "request_hash": "b" * 64, "rqid": 3}
        candidates = candidates_for(body)
        assert same(
            ENCODE.encode({"observation": a1, "candidates": candidates}),
            ENCODE.encode({"observation": a2, "candidates": candidates}),
        ), f"{name} {player} {index}"


def test_every_request_point_of_the_random_corpus_encodes_into_range_with_the_option() -> None:
    layout = Layout.of(ON)
    limits = {"matchup": RESERVED + game().chart_codes, "same_type": RESERVED + 2}
    seen_cells: set[int] = set()
    points = 0
    for path in RANDOM:
        battle = json.loads(path.read_text())
        for player in ("p1", "p2"):
            lines = battle["channels"][player]["lines"]
            for index, body in request_points(lines):
                obs = observation(
                    player, lines[: index + 1], battle["teams"][player], kind_of(body)
                )
                encoded = ENCODE.encode({"observation": obs, "candidates": candidates_for(body)})
                points += 1
                for column, (_, kind) in enumerate(layout.cand_cat[BASE_CAND_COLUMNS:]):
                    values = encoded.cand_cat[:, :, BASE_CAND_COLUMNS + column]
                    assert values.ge(0).all()
                    assert values.lt(limits[kind]).all()
                    if kind == "matchup":
                        seen_cells |= set(values.flatten().tolist())
                # A battle candidate that is not a move holds none everywhere.
                moves = encoded.cand_cat[:, :, 0] != 1
                assert not matchup_part(encoded)[moves].any()
    assert points > 500
    assert {NONE, RESERVED + 0, RESERVED + 1, RESERVED + 2} <= seen_cells


def test_the_encoding_is_deterministic_and_a_preview_query_has_no_cells() -> None:
    query = sample("policy-query/valid/battle.json")
    assert same(ENCODE.encode(query), ENCODE.encode(copy.deepcopy(query)))
    preview = ENCODE.encode(sample("policy-query/valid/preview.json"))
    assert preview.cand_cat.eq(0).all()
    assert preview.cand_cat.shape[2] == WIDTH


# -- the network ----------------------------------------------------------------


def battle_batch(count: int = 2) -> Any:
    query = sample("policy-query/valid/battle.json")
    return collate([ENCODE.encode(query)] * count)


def test_the_network_consumes_the_cells_and_a_training_step_runs() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3)
    assert network.embeddings["matchup"].num_embeddings == RESERVED + game().chart_codes
    batch = battle_batch()
    out = network(batch)
    assert torch.isfinite(out.logits[batch.cand_mask]).all()
    # The output depends on the cells: another cell category changes the logits.
    changed = replace(batch, cand_cat=batch.cand_cat.clone())
    changed.cand_cat[:, :, 0, BASE_CAND_COLUMNS] = RESERVED + 3
    assert not torch.equal(network(changed).logits, out.logits)
    # One optimiser step reaches the new embeddings.
    optimiser = torch.optim.SGD(network.parameters(), lr=0.1)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    assert network.embeddings["matchup"].weight.grad is not None
    assert network.embeddings["matchup"].weight.grad.abs().sum() > 0
    assert network.embeddings["same_type"].weight.grad.abs().sum() > 0
    before = network.embeddings["matchup"].weight.detach().clone()
    optimiser.step()
    assert not torch.equal(network.embeddings["matchup"].weight, before)


def test_the_same_seed_gives_the_same_logits_and_a_preview_network_has_no_cell_embedding() -> None:
    one = init_network("battle", game(), ON_MODEL, seed=3)(battle_batch(1)).logits
    two = init_network("battle", game(), ON_MODEL, seed=3)(battle_batch(1)).logits
    assert torch.equal(one, two)
    preview = init_network("preview", game(), ON_MODEL, seed=3)
    assert "matchup" not in preview.embeddings
    out = preview(collate([ENCODE.encode(sample("policy-query/valid/preview.json"))]))
    assert torch.isfinite(out.logits[:, 0]).all()


def test_without_the_option_the_network_has_no_cell_embedding() -> None:
    plain = init_network("battle", game(), TINY_MODEL, seed=3)
    assert "matchup" not in plain.embeddings
    assert plain.slot_parts[0].in_features == 7 * TINY_MODEL.d_model


def test_the_checkpoint_of_the_option_is_format_two_and_loads_only_under_the_option() -> None:
    network = init_network("battle", game(), ON_MODEL, seed=3)
    data = checkpoint_bytes(network, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    assert checkpoint_bytes(loaded, game()) == data
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)
    plain = checkpoint_bytes(init_network("battle", game(), TINY_MODEL, seed=3), game())
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(plain, game(), checkpoint_id(plain), "battle", configured=ON_MODEL)


# -- no rule in code ----------------------------------------------------------------


def test_no_effectiveness_or_same_type_constant_is_written_in_model_code() -> None:
    names = {"_matchup", "_shown_species", "_foe_at", "_matchup_columns", "matchup_cells"}
    checked = 0
    for source in ("encoder.py", "gamedata.py"):
        tree = ast.parse((SOURCES / source).read_text())
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef) and function.name in names:
                checked += 1
                for node in ast.walk(function):
                    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                        # Only index arithmetic (0, 1, and a bool) is allowed: no multiplier.
                        assert not isinstance(node.value, float), (function.name, node.value)
                        assert node.value in (0, 1), (function.name, node.value)
    assert checked == len(names)
    text = "".join((SOURCES / name).read_text() for name in ("encoder.py", "gamedata.py"))
    for forbidden in ("1.5", "0.75", "effectiveness"):
        assert forbidden not in text.lower().replace("same-type", ""), forbidden
