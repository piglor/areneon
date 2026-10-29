"""The ``factored_preview`` network option (ADR-010 erratum A4 item 4).

Under the option the Preview network scores an order as the sum over places of
``u(place, Pokemon token)`` plus ``v(lead pair)`` from the tokens at places 1 and 2. The
cross-attention and interaction scorer are not built for Preview; the Battle network ignores it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from areneon_ml.checkpoint.files import (
    FORMAT,
    FORMAT_OPTIONS,
    CheckpointOptionsError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    read_header,
)
from areneon_ml.model.batch import collate
from areneon_ml.model.network import ConfigError, ModelConfig, PolicyNetwork, init_network
from areneon_ml.model.options import FACTORED_PREVIEW, OptionError, read_options
from areneon_ml.model.runtime import configure_cpu_inference

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample
from .test_encoder import PAIRS, pair_query

ON_MODEL = replace(TINY_MODEL, options=read_options("network", {FACTORED_PREVIEW: True}))


@pytest.fixture
def cpu_inference() -> Iterator[None]:
    deterministic, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    configure_cpu_inference()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


def preview(model: ModelConfig = ON_MODEL, seed: int = 1) -> PolicyNetwork:
    return init_network("preview", game(), model, seed)


def preview_query() -> Any:
    return encoder().encode(sample("policy-query/valid/preview.json"))


def test_only_true_is_a_value_and_the_option_is_a_network_option() -> None:
    assert read_options("network", {FACTORED_PREVIEW: True})
    bad_values: list[Any] = [False, 0, 1, {}]
    for bad in bad_values:
        with pytest.raises(OptionError, match="invalid value"):
            read_options("network", {FACTORED_PREVIEW: bad})
    with pytest.raises(OptionError, match="unknown encoder option"):
        read_options("encoder", {FACTORED_PREVIEW: True})
    assert ModelConfig.from_dict(ON_MODEL.as_dict()) == ON_MODEL
    assert ON_MODEL.layout_changing


def test_absent_option_keeps_the_preview_tensor_names() -> None:
    names = set(init_network("preview", game(), TINY_MODEL, 1).state_dict())
    assert {"candidate.weight", "cross.query.weight", "score.0.weight", "pending.weight"} <= names
    assert not any(n.startswith(("place_score", "lead_score")) for n in names)


def test_present_option_swaps_the_scorer_for_the_two_terms() -> None:
    names = set(preview().state_dict())
    assert {"place.weight", "place_score.0.weight", "lead_score.2.bias", "value.0.weight"} <= names
    assert not any(
        n.startswith(("candidate", "cross", "score", "pending", "norm_c")) for n in names
    )
    battle = set(init_network("battle", game(), ON_MODEL, 1).state_dict())
    assert battle == set(init_network("battle", game(), TINY_MODEL, 1).state_dict())


def test_battle_network_logits_ignore_the_option() -> None:
    batch = collate([encoder().encode(sample("policy-query/valid/battle.json"))])
    plain = init_network("battle", game(), TINY_MODEL, 3)(batch)
    option = init_network("battle", game(), ON_MODEL, 3)(batch)
    assert torch.equal(plain.logits, option.logits)
    assert torch.equal(plain.value, option.value)


def test_the_score_is_the_hand_computed_sum() -> None:
    net = preview()
    captured: dict[str, torch.Tensor] = {}
    net.final_norm.register_forward_hook(lambda _m, _i, out: captured.update(x=out))
    encoded = preview_query()
    batch = collate([encoded])
    out = net(batch)
    mons = captured["x"][0, 2:]  # the tokens after the aggregation and field tokens
    picks = encoded.team_tok  # [N, P], -1 for none
    assert picks.shape[1] >= 2

    def token(i: int) -> torch.Tensor:
        return torch.zeros(mons.shape[-1]) if i < 0 else mons[i]

    with torch.no_grad():
        for n in range(picks.shape[0]):
            total = torch.zeros(())
            for place in range(picks.shape[1]):
                pair = torch.cat([net.place.weight[place], token(int(picks[n, place]))])
                total = total + net.place_score(pair)[0]
            lead = torch.cat([token(int(picks[n, 0])), token(int(picks[n, 1]))])
            total = total + net.lead_score(lead)[0]
            assert torch.allclose(out.logits[0, n], total, atol=1e-5), n


def test_zeroed_terms_leave_exactly_the_other_one() -> None:
    net = preview()
    with torch.no_grad():
        for p in net.lead_score.parameters():
            p.zero_()
        for module in net.place_score:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
        last = cast(torch.nn.Linear, net.place_score[2])
        last.bias.fill_(0.5)
    encoded = preview_query()
    logits = net(collate([encoded])).logits[0]
    assert torch.allclose(logits, torch.full_like(logits, 0.5 * TINY_ENCODER.max_picks))


def test_logits_are_the_same_alone_and_after_other_queries(cpu_inference: None) -> None:
    net = preview()
    target = preview_query()
    raw = sample("policy-query/valid/preview.json")
    others = [
        encoder().encode({**raw, "candidates": raw["candidates"][:5]}),
        encoder().encode({**raw, "candidates": raw["candidates"][::-1]}),
    ]
    alone = net(collate([target]))
    for other in others:
        net(collate([other]))
    after = net(collate([target]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)
    assert torch.equal(alone.logits, preview()(collate([target])).logits)


def test_the_a1_and_a2_forms_score_equally(cpu_inference: None) -> None:
    net = preview()
    checked = 0
    for decision in PAIRS["decisions"]:
        if decision["kind"] != "preview":
            continue
        one = encoder().encode(pair_query(PAIRS, decision, "a1"))
        two = encoder().encode(pair_query(PAIRS, decision, "a2"))
        assert torch.equal(net(collate([one])).logits, net(collate([two])).logits)
        checked += 1
    assert checked


def test_padding_candidates_get_the_minimum_and_stay_out_of_the_softmax() -> None:
    long_, short = (
        preview_query(),
        encoder().encode(
            {
                **sample("policy-query/valid/preview.json"),
                "candidates": sample("policy-query/valid/preview.json")["candidates"][:3],
            }
        ),
    )
    out = preview()(collate([long_, short]))
    assert torch.all(out.logits[1, 3:] == torch.finfo(out.logits.dtype).min)
    assert torch.isfinite(out.logits[1, :3]).all()


def test_a_lead_pair_needs_two_picks() -> None:
    one_pick = replace(TINY_ENCODER, max_picks=1)
    with pytest.raises(ConfigError, match="lead pair"):
        init_network("preview", game(), replace(ON_MODEL, encoder=one_pick), 1)


def test_the_checkpoint_is_format_two_and_loads_only_under_the_option() -> None:
    data = checkpoint_bytes(preview(), game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "preview", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    assert checkpoint_bytes(loaded, game()) == data
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game(), checkpoint_id(data), "preview", configured=TINY_MODEL)
    plain = checkpoint_bytes(init_network("preview", game(), TINY_MODEL, 1), game())
    assert read_header(plain)["__metadata__"]["areneon.format"] == FORMAT
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(plain, game(), checkpoint_id(plain), "preview", configured=ON_MODEL)
