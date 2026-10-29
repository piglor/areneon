"""The ``candidate_target`` encoder option (ADR-010 erratum A4 item 7, #1011).

Each move candidate carries the token of the Pokémon in the targeted active slot: the engine's
target location (``getAtLoc``: L > 0 is the foe in active slot L - 1, L < 0 the own active slot
-L - 1) read through the player's own request and the reduced slots. The Battle network gathers
that token's embedding into the slot part.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from dataclasses import replace
from functools import cache
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
from areneon_ml.model.candidates import parse_choice
from areneon_ml.model.encoder import (
    BASE_CAND_COLUMNS,
    Encoder,
    Layout,
    encoder_version,
)
from areneon_ml.model.network import ConfigError, ModelConfig, init_network
from areneon_ml.model.options import CANDIDATE_MATCHUP, CANDIDATE_TARGET, read_options
from areneon_ml.model.runtime import configure_cpu_inference

from .support import (
    RANDOM,
    TINY_ENCODER,
    TINY_MODEL,
    candidates_for,
    game,
    observation,
    request_points,
    sample,
)
from .test_encoder import PAIRS, a2_fixture_points, kind_of, pair_query, same
from .test_options import TODAY_ENCODER_VERSION

ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_TARGET: True}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
BOTH = replace(
    TINY_ENCODER,
    options=read_options("encoder", {CANDIDATE_MATCHUP: True, CANDIDATE_TARGET: True}),
)
ENCODE = Encoder(game(), ON)
ROOT = Path(__file__).resolve().parents[3]
CHOICES = sorted((ROOT / "fixtures" / "showdown").glob("**/choices/*.json"))
SPECIES = Layout.of(TINY_ENCODER).mon_cat.index(("species", "species"))


def candidate(query: dict[str, Any], choice: str) -> int:
    return [c["choice"] for c in query["candidates"]].index(choice)


def targets(encoded: Any, index: int) -> list[int]:
    result: list[int] = encoded.cand_tok[index, :, 2].tolist()
    return result


@cache
def channels() -> dict[str, Any]:
    """Each corpus case's player channel, replayed on the pinned engine by the Node tooling."""
    script = ROOT / "packages" / "showdown-bridge" / "scripts" / "choice-channels.ts"
    if shutil.which("node") is None or not (ROOT / "vendor/pokemon-showdown/dist/sim").exists():
        pytest.skip("needs node and the built pinned engine")
    done = subprocess.run(["node", str(script)], capture_output=True, check=True, text=True)
    loaded: dict[str, Any] = json.loads(done.stdout)
    return loaded


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# -- the option is absent: today's bytes ---------------------------------------


def test_with_the_option_absent_the_layout_and_the_version_are_today_s() -> None:
    assert Layout.of(TINY_ENCODER).cand_tok == ("actor", "switch_in")
    assert "cand_tok" not in Layout.of(TINY_ENCODER).as_dict()
    assert encoder_version(game(), TINY_ENCODER) == TODAY_ENCODER_VERSION
    query = sample("policy-query/valid/battle.json")
    assert Encoder(game(), TINY_ENCODER).encode(query).cand_tok.shape[2] == 2


def test_with_the_option_every_other_tensor_is_unchanged() -> None:
    query = sample("policy-query/valid/battle.json")
    plain, with_option = Encoder(game(), TINY_ENCODER).encode(query), ENCODE.encode(query)
    assert ENCODE.version != Encoder(game(), TINY_ENCODER).version
    assert with_option.cand_tok.shape[2] == 3
    assert torch.equal(with_option.cand_tok[:, :, :2], plain.cand_tok)
    for name, tensor in plain.tensors().items():
        if name != "cand_tok":
            assert torch.equal(with_option.tensors()[name], tensor), name


def test_only_true_is_a_value_and_the_model_config_round_trips() -> None:
    assert ON.as_dict()["options"] == {CANDIDATE_TARGET: True}
    for bad in (False, 0, 1):
        with pytest.raises(ConfigError, match="invalid value"):
            ModelConfig.from_dict(
                {
                    **TINY_MODEL.as_dict(),
                    "encoder": {**TINY_ENCODER.as_dict(), "options": {CANDIDATE_TARGET: bad}},
                }
            )
    assert ON_MODEL.layout_changing
    assert ModelConfig.from_dict(json.loads(json.dumps(ON_MODEL.as_dict()))) == ON_MODEL


# -- hand-checked tokens -------------------------------------------------------
# battle.json: own Venusaur (token 0, slot a) and Charizard (token 1, slot b); the foes are
# Venusaur in p2a and Charizard in p2b, whose tokens are n_own + their roster indices.


def test_foe_locations_map_to_the_engine_s_active_slots() -> None:
    query = sample("policy-query/valid/battle.json")
    encoded = ENCODE.encode(query)
    foe_a = encoded.n_own  # roster index 0 of the opponent
    foe_b = encoded.n_own + 1
    assert encoded.mon_cat[foe_a, SPECIES].item() == game().species.lookup("venusaur")
    assert encoded.mon_cat[foe_b, SPECIES].item() == game().species.lookup("charizard")
    assert targets(encoded, candidate(query, "move 1 1, move 1 2")) == [foe_a, foe_b]
    assert targets(encoded, candidate(query, "move 1 2, move 1 1")) == [foe_b, foe_a]


def test_ally_targets_map_to_the_ally_and_the_user_is_none() -> None:
    query = sample("policy-query/valid/battle.json")
    encoded = ENCODE.encode(query)
    # Slot b (token 1) at -1 is the own active slot 0: the ally, token 0. Slot a at -2 is slot b.
    ally = candidate(query, "move 4 1, move 1 -1")
    assert targets(encoded, ally)[1] == 0
    both = candidate(query, "move 4 -2, move 1 -1")
    assert targets(encoded, both) == [1, 0]
    # The user itself is none: slot a at -1.
    extra = copy.deepcopy(query)
    extra["candidates"].append({"choice": "move 4 -1, move 1 1", "pending": False})
    assert targets(ENCODE.encode(extra), len(extra["candidates"]) - 1) == [
        -1,
        ENCODE.encode(extra).n_own,
    ]


def test_untargeted_switch_and_pass_parts_have_no_target() -> None:
    query = sample("policy-query/valid/battle.json")
    encoded = ENCODE.encode(query)
    assert targets(encoded, candidate(query, "move 3, move 2 1"))[0] == -1  # no location
    switch = candidate(query, "move 1 1, switch 4")
    assert targets(encoded, switch)[1] == -1
    assert encoded.cand_tok[switch, 1, 1].item() >= 0  # the switch-in token is untouched


def test_a_fainted_foe_has_no_target() -> None:
    query = sample("policy-query/valid/battle.json")
    fainted = copy.deepcopy(query)
    fainted["observation"]["log"].insert(-1, "|faint|p2b: Charizard")
    i = candidate(query, "move 1 2, move 1 1")
    encoded = ENCODE.encode(fainted)
    assert targets(encoded, i)[0] == -1
    assert targets(encoded, i)[1] == ENCODE.encode(query).n_own


# -- the committed choice corpus: an independent mapping -------------------------


def expected_species(
    case: dict[str, Any], lines: list[str], body: dict[str, Any]
) -> dict[int, str | None]:
    """Species the target locations name, read from the log and request alone.

    Foe slot -> species of the last switch/drag line, None after its faint; own slot -> the
    request's ``side.pokemon`` entry that is active in that slot.
    """
    other = "p2" if case["player"] == "p1" else "p1"
    foes: dict[str, str | None] = {}
    for line in lines:
        parts = line.split("|")
        if (
            len(parts) > 3
            and parts[1] in ("switch", "drag", "detailschange", "replace")
            and (parts[2].startswith(other))
        ):
            foes[parts[2][2]] = to_id(parts[3].split(",")[0])
        elif len(parts) > 2 and parts[1] == "faint" and parts[2].startswith(other):
            foes[parts[2][2]] = None
    out: dict[int, str | None] = {}
    for n, letter in enumerate(sorted(foes), start=1):
        out[n] = foes[letter]
    mons = body["side"]["pokemon"]
    for n in range(1, 3):
        if n > len(mons) or mons[n - 1]["condition"].endswith(" fnt"):
            out[-n] = None
        else:
            out[-n] = to_id(mons[n - 1]["details"].split(",")[0])
    return out


def test_every_candidate_of_the_choice_corpus_maps_to_the_expected_token_or_none() -> None:
    assert CHOICES
    checked = mapped = nones = 0
    for path in CHOICES:
        case = json.loads(path.read_text())
        if case["phase"] not in ("move", "forceSwitch"):
            continue
        lines = channels()[path.name]["lines"]
        body = next(b for _, b in reversed(list(request_points(lines))))
        query = {
            "observation": observation(
                case["player"], lines, case["teams"][case["player"]], "battle"
            ),
            "candidates": [
                {"choice": c["choice"], "pending": c["pending"]} for c in case["candidates"]
            ],
        }
        encoded = ENCODE.encode(query)
        species = expected_species(case, lines, body)
        order = [to_id(m["ident"].split(": ", 1)[1]) for m in body["side"]["pokemon"]]
        for i, cand in enumerate(case["candidates"]):
            parsed = parse_choice(cand["choice"])
            for slot, action in enumerate(parsed.slots):
                got = int(encoded.cand_tok[i, slot, 2].item())
                checked += 1
                location = action.target if action.kind == "move" else None
                want = None
                if location is not None and (
                    location > 0 or (-location - 1 != slot and -location - 1 < len(order))
                ):
                    want = species.get(location)
                if want is None:
                    assert got == -1, (path.name, cand["choice"], slot)
                    nones += 1
                    continue
                assert location is not None
                assert got >= 0, (path.name, cand["choice"], slot)
                assert encoded.mon_cat[got, SPECIES].item() == game().species.lookup(want), (
                    path.name,
                    cand["choice"],
                    slot,
                )
                assert (got < encoded.n_own) == (location < 0), (path.name, cand["choice"])
                mapped += 1
    assert checked > 100
    assert mapped > 20
    assert nones > 20


# -- A1/A2 equality and the random corpus ----------------------------------------


def test_the_a1_and_a2_forms_encode_equally_on_the_committed_pairs() -> None:
    targeted = 0
    for decision in PAIRS["decisions"]:
        first = ENCODE.encode(pair_query(PAIRS, decision, "a1"))
        second = ENCODE.encode(pair_query(PAIRS, decision, "a2"))
        assert same(first, second), decision["fixture"]
        targeted += int((first.cand_tok[:, :, 2] >= 0).any())
    assert targeted > 10


def test_the_a2_recordings_encode_equally() -> None:
    for name, player, index, lines, team, body in a2_fixture_points():
        a1 = observation(player, list(lines), team, kind_of(body))
        a1["identity"] = {"request_seq": 1, "request_hash": "a" * 64}
        a2 = copy.deepcopy(a1)
        a2["identity"] = {"request_seq": 5, "request_hash": "b" * 64, "rqid": 3}
        candidates = candidates_for(body)
        assert same(
            ENCODE.encode({"observation": a1, "candidates": candidates}),
            ENCODE.encode({"observation": a2, "candidates": candidates}),
        ), f"{name} {player} {index}"


def test_every_request_point_of_the_random_corpus_encodes_into_range() -> None:
    points = 0
    seen = set()
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
                column = encoded.cand_tok[:, :, 2]
                assert column.ge(-1).all()
                assert column.lt(encoded.mon_cat.shape[0]).all()
                # Only a move part holds a target, and never the acting Pokémon itself.
                assert not column[encoded.cand_cat[:, :, 0] != 1].ge(0).any()
                assert not (column == encoded.cand_tok[:, :, 0])[column >= 0].any()
                seen |= {int(v >= encoded.n_own) for v in column[column >= 0]}
    assert points > 500
    assert seen == {0, 1}  # foes and allies both occur


# -- composition, the network, the checkpoint --------------------------------------


def test_it_composes_with_matchup_and_with_features() -> None:
    both = Layout.of(BOTH)
    assert both.cand_tok == ("actor", "switch_in", "target")
    assert len(both.cand_cat) == BASE_CAND_COLUMNS + 7
    assert encoder_version(game(), BOTH) not in {
        encoder_version(game(), ON),
        encoder_version(
            game(),
            replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_MATCHUP: True})),
        ),
    }
    query = sample("policy-query/valid/battle.json")
    combined = Encoder(game(), BOTH).encode(query)
    alone = ENCODE.encode(query)
    assert torch.equal(combined.cand_tok, alone.cand_tok)
    network: Any = init_network("battle", game(), replace(TINY_MODEL, encoder=BOTH), seed=3)
    out = network(collate([combined]))
    assert torch.isfinite(out.logits[:, 0]).all()
    feature_set = {"feature_set": "probe-damage/1", "feature_set_hash": "c" * 64}
    full = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder",
            {CANDIDATE_MATCHUP: True, CANDIDATE_TARGET: True, "candidate_features": feature_set},
        ),
    )
    layout = Layout.of(full)
    assert layout.cand_tok == ("actor", "switch_in", "target")
    assert layout.cand_num
    net: Any = init_network("battle", game(), replace(TINY_MODEL, encoder=full), seed=3)
    d = TINY_MODEL.d_model
    assert net.slot_parts[0].in_features == (7 + 7 + 2 + 1 + 1) * d


def battle_batch(count: int = 2) -> Any:
    return collate([ENCODE.encode(sample("policy-query/valid/battle.json"))] * count)


def test_the_network_gathers_the_target_token_and_a_training_step_reaches_it() -> None:
    network: Any = init_network("battle", game(), ON_MODEL, seed=3)
    assert network.slot_parts[0].in_features == 8 * TINY_MODEL.d_model
    batch = battle_batch()
    out = network(batch)
    assert torch.isfinite(out.logits[batch.cand_mask]).all()
    changed = replace(batch, cand_tok=batch.cand_tok.clone())
    changed.cand_tok[:, :, 0, 2] = 1 - 0 if changed.cand_tok[0, 0, 0, 2] != 1 else 0
    assert not torch.equal(network(changed).logits, out.logits)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    assert network.slot_parts[0].weight.grad[:, 7 * TINY_MODEL.d_model :].abs().sum() > 0


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    configure_cpu_inference()
    network: Any = init_network("battle", game(), ON_MODEL, seed=3).eval()
    one = ENCODE.encode(sample("policy-query/valid/battle.json"))
    other = ENCODE.encode(sample("policy-query/valid/battle-a2.json"))
    with torch.no_grad():
        alone = network(collate([one]))
        network(collate([other]))
        after = network(collate([one]))
        a2_form = network(collate([ENCODE.encode(sample("policy-query/valid/battle-a2.json"))]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)
    assert torch.isfinite(a2_form.logits[:, 0]).all()
    # A padded batch matches each query scored alone (to float tolerance).
    with torch.no_grad():
        both = network(collate([one, other]))
    count = one.pending.shape[0]
    assert torch.allclose(both.logits[0, :count], alone.logits[0], atol=1e-5)


def test_the_preview_network_ignores_the_column_and_a_batch_of_another_layout_is_refused() -> None:
    preview = init_network("preview", game(), ON_MODEL, seed=3)
    out = preview(collate([ENCODE.encode(sample("policy-query/valid/preview.json"))]))
    assert torch.isfinite(out.logits[:, 0]).all()
    plain = Encoder(game(), TINY_ENCODER).encode(sample("policy-query/valid/battle.json"))
    with pytest.raises(ValueError, match="candidate token layout"):
        collate([plain, ENCODE.encode(sample("policy-query/valid/battle.json"))])
    network: Any = init_network("battle", game(), ON_MODEL, seed=3)
    with pytest.raises(ValueError, match="candidate tokens"):
        network(collate([plain]))


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    network = init_network("battle", game(), ON_MODEL, seed=3)
    data = checkpoint_bytes(network, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)
