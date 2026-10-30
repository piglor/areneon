"""The ``separate_value`` and ``policy_output_scale_ppm`` network options (ADR-010 erratum A4
item 7): the value stops pulling the policy trunk, and the initial policy is near uniform."""

from __future__ import annotations

import math
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
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ModelConfig, PolicyNetwork, init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    FACTORED_PREVIEW,
    POLICY_OUTPUT_SCALE_PPM,
    RECENT_HISTORY,
    SEPARATE_VALUE,
    OptionError,
    read_options,
)
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker.sampler import to_ppm
from areneon_ml.worker.view import encoder_input

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample


def model(**options: Any) -> ModelConfig:
    return replace(TINY_MODEL, options=read_options("network", options))


def batch(role: str) -> Batch:
    return collate([encoder().encode(sample(f"policy-query/valid/{role}.json"))])


def build(role: str, config: ModelConfig, seed: int = 1) -> PolicyNetwork:
    return init_network(role, game(), config, seed)


def policy_names(names: list[str]) -> list[str]:
    return [n for n in names if not n.startswith(("value.", "value_net."))]


def test_values_are_a_closed_set() -> None:
    for good in ("network", "detached"):
        assert read_options("network", {SEPARATE_VALUE: good})
    bad_modes: list[Any] = ["shared", "", True, 1, {}]
    for bad in bad_modes:
        with pytest.raises(OptionError, match="invalid value"):
            read_options("network", {SEPARATE_VALUE: bad})
    assert read_options("network", {POLICY_OUTPUT_SCALE_PPM: 10000})
    bad_scales: list[Any] = [0, -1, 1_000_000, 2_000_000, True, 0.5, "10000"]
    for bad in bad_scales:
        with pytest.raises(OptionError, match="invalid value"):
            read_options("network", {POLICY_OUTPUT_SCALE_PPM: bad})
    with pytest.raises(OptionError, match="unknown encoder option"):
        read_options("encoder", {SEPARATE_VALUE: "network"})
    for options in ({SEPARATE_VALUE: "detached"}, {POLICY_OUTPUT_SCALE_PPM: 10000}):
        config = model(**options)
        assert config.layout_changing
        assert ModelConfig.from_dict(config.as_dict()) == config


def test_absent_options_keep_every_tensor_name() -> None:
    names = list(build("battle", TINY_MODEL).state_dict())
    assert not any(n.startswith("value_net") for n in names)
    assert {"value.0.weight", "value.2.bias", "score.2.weight"} <= set(names)


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_a_separate_network_shares_no_layer_and_keeps_the_policy_untouched(role: str) -> None:
    plain = build(role, TINY_MODEL)
    sep = build(role, model(separate_value="network"))
    sd = sep.state_dict()
    assert not any(n.startswith("value.") for n in sd)  # the head moved into value_net
    own = {n for n in sd if n.startswith("value_net.")}
    assert {"value_net.value.0.weight", "value_net.blocks.0.attention.query.weight"} <= own
    assert {n for n in sd if n not in own} == set(plain.state_dict()) - {
        n for n in plain.state_dict() if n.startswith("value.")
    }
    # no storage is shared between the two sub-networks
    policy_ptrs = {p.data_ptr() for n, p in sep.named_parameters() if n in set(sd) - own}
    assert not policy_ptrs & {p.data_ptr() for n, p in sep.named_parameters() if n in own}
    # the policy is the plain network's (same seed, same draw order); the value is its own
    out, ref = sep(batch(role)), plain(batch(role))
    assert torch.equal(out.logits, ref.logits)
    assert not torch.equal(out.value, ref.value)


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_the_detached_head_keeps_the_layout_and_the_outputs(role: str) -> None:
    plain = build(role, TINY_MODEL)
    det = build(role, model(separate_value="detached"))
    assert list(det.state_dict()) == list(plain.state_dict())
    a, b = det(batch(role)), plain(batch(role))
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.value, b.value)


def grads(net: PolicyNetwork) -> set[str]:
    return {n for n, p in net.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}


@pytest.mark.parametrize("role", ["battle", "preview"])
@pytest.mark.parametrize("mode", ["network", "detached"])
def test_no_gradient_crosses_between_policy_and_value(role: str, mode: str) -> None:
    config = model(separate_value=mode)
    net = build(role, config)
    out = net(batch(role))
    live = out.logits[batch(role).cand_mask]
    # the policy loss: reaches the policy parameters, never the value network or head
    live.log_softmax(-1)[0].backward()
    after_policy = grads(net)
    assert after_policy
    assert not any(n.startswith(("value.", "value_net.")) for n in after_policy)
    # the value loss: reaches the value parameters, never the policy trunk or scorer
    net = build(role, config)
    net(batch(role)).value.sum().backward()
    after_value = grads(net)
    assert after_value
    assert all(n.startswith(("value.", "value_net.")) for n in after_value)
    if mode == "network":
        assert any(n.startswith("value_net.blocks.") for n in after_value)
    else:  # detached: only the head learns from the value
        assert not any(n.startswith("blocks.") for n in after_value)
        assert any(n.startswith("value.") for n in after_value)
    # and without the option the value error does reach the trunk (the control)
    plain = build(role, TINY_MODEL)
    plain(batch(role)).value.sum().backward()
    assert any(n.startswith("blocks.") for n in grads(plain))


def test_the_value_does_not_depend_on_the_policy_trunk() -> None:
    net = build("battle", model(separate_value="network"))
    before = net(batch("battle")).value.clone()
    with torch.no_grad():
        for p in net.blocks.parameters():
            p.add_(1.0)
        net.aggregate.add_(1.0)
    assert torch.equal(net(batch("battle")).value, before)


def spread(net: PolicyNetwork, role: str) -> float:
    b = batch(role)
    with torch.no_grad():
        return float(net(b).logits[b.cand_mask].std())


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_the_initial_logits_spread_is_about_one_hundredth(role: str) -> None:
    plain = spread(build(role, TINY_MODEL, seed=7), role)
    scaled_net = build(role, model(policy_output_scale_ppm=10000), seed=7)
    assert plain > 0
    ratio = plain / spread(scaled_net, role)
    assert 99.0 < ratio < 101.0
    # a seeded second draw agrees; the value head is not scaled
    other = build(role, TINY_MODEL, seed=7)
    assert torch.equal(
        scaled_net.state_dict()["value.2.weight"], other.state_dict()["value.2.weight"]
    )
    # the init is the same draw: only the last scorer layer's weights differ
    differing = {
        n for n, t in scaled_net.state_dict().items() if not torch.equal(t, other.state_dict()[n])
    }
    assert differing == {"score.2.weight"}


def test_the_scale_composes_with_the_factored_preview() -> None:
    both = model(**{FACTORED_PREVIEW: True, POLICY_OUTPUT_SCALE_PPM: 10000})
    only = model(**{FACTORED_PREVIEW: True})
    ratio = spread(build("preview", only, 7), "preview") / spread(
        build("preview", both, 7), "preview"
    )
    assert 99.0 < ratio < 101.0
    differing = {
        n
        for n, t in build("preview", both, 7).state_dict().items()
        if not torch.equal(t, build("preview", only, 7).state_dict()[n])
    }
    assert differing == {"place_score.2.weight", "lead_score.2.weight"}


def test_the_value_options_compose_with_the_factored_preview_and_the_scale() -> None:
    config = model(
        **{FACTORED_PREVIEW: True, SEPARATE_VALUE: "network", POLICY_OUTPUT_SCALE_PPM: 10000}
    )
    net = build("preview", config)
    names = set(net.state_dict())
    assert {"place_score.0.weight", "value_net.value.0.weight"} <= names
    assert not any(n.startswith("value.") for n in names)
    out = net(batch("preview"))
    assert torch.isfinite(out.logits[batch("preview").cand_mask]).all()
    net(batch("preview")).value.sum().backward()
    assert not any(not n.startswith("value_net.") for n in grads(net))


@pytest.mark.parametrize(
    "options",
    [{SEPARATE_VALUE: "network"}, {SEPARATE_VALUE: "detached"}, {POLICY_OUTPUT_SCALE_PPM: 10000}],
)
def test_the_checkpoint_is_format_two_and_loads_only_under_the_same_options(
    options: dict[str, Any],
) -> None:
    config = model(**options)
    net = build("battle", config)
    data = checkpoint_bytes(net, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=config)
    assert loaded.config == config
    assert checkpoint_bytes(loaded, game()) == data
    others = [
        TINY_MODEL,
        model(separate_value="network"),
        model(separate_value="detached"),
        model(policy_output_scale_ppm=20000),
    ]
    for other in others:
        if other == config:
            continue
        with pytest.raises(CheckpointOptionsError, match="other model options"):
            load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=other)


@pytest.mark.parametrize("mode", ["network", "detached"])
def test_a_separate_value_reads_the_probe_rows_and_option_columns_of_a_rich_encoder(
    mode: str,
) -> None:
    """probe-damage/2 rows, recent history and candidate matchup: the value network builds the
    probe layers and only the embeddings its trunk reads, and the gradients stay apart."""
    raw = sample("policy-query/valid/battle-features-2.json")
    feature_set = {
        key: raw["candidate_features"][key] for key in ("feature_set", "feature_set_hash")
    }
    rich = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder",
            {
                CANDIDATE_FEATURES: feature_set,
                CANDIDATE_MATCHUP: True,
                RECENT_HISTORY: True,
            },
        ),
    )
    config = replace(
        TINY_MODEL, encoder=rich, options=read_options("network", {SEPARATE_VALUE: mode})
    )
    rich_batch = collate([Encoder(game(), rich).encode(raw)])
    net = init_network("battle", game(), config, 3)
    assert net.probing
    out = net(rich_batch)
    assert torch.isfinite(out.value).all()
    names = set(net.state_dict())
    if mode == "network":
        value_net = net.value_net
        assert value_net is not None
        assert value_net.probing
        assert {"value_net.threat_row.weight", "value_net.foe_pool.weight"} <= names
        assert not any(n.startswith(("value_net.slot_parts", "value_net.score")) for n in names)
        # only the kinds its trunk reads are embedded
        kinds = {n.split(".")[2] for n in names if n.startswith("value_net.embeddings.")}
        assert {"move", "probe_kind"} <= kinds
        assert "target" not in kinds
    live = out.logits[rich_batch.cand_mask]
    live.log_softmax(-1)[0].backward(retain_graph=True)
    assert not any(n.startswith(("value.", "value_net.")) for n in grads(net))
    net.zero_grad(set_to_none=True)
    out.value.sum().backward()
    value_grads = grads(net)
    assert value_grads
    assert all(n.startswith(("value.", "value_net.")) for n in value_grads)
    if mode == "network":
        assert any(n.startswith("value_net.threat_row") for n in value_grads)


def test_the_scorer_evaluates_a_finite_value_under_a_separate_value_network() -> None:
    config = model(separate_value="network", policy_output_scale_ppm=10000)
    nets = {role: build(role, config, seed=2) for role in ("battle", "preview")}
    scorer = NetworkScorer(game(), nets)
    for name in ("battle", "preview"):
        view = encoder_input(sample(f"policy-query/valid/{name}.json"))
        probabilities, value = scorer.evaluate(view)
        assert math.isfinite(value)
        assert isinstance(to_ppm(value), int)
        assert abs(sum(probabilities) - 1) < 1e-4
        assert max(probabilities) - min(probabilities) < 0.05  # near uniform at 1/100 scale
