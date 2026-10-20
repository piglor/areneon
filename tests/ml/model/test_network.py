"""The Battle and Preview networks (ADR-010 §2, §3): shape, determinism, batch invariance."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.learner.trainer import to_device
from areneon_ml.model.batch import collate
from areneon_ml.model.network import (
    ConfigError,
    ModelConfig,
    PolicyNetwork,
    encoder_config_from_dict,
    init_network,
)
from areneon_ml.model.runtime import configure_cpu_inference
from areneon_ml.model.scorer import NetworkScorer, ScorerError
from areneon_ml.worker.view import encoder_input

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample


@pytest.fixture
def cpu_inference() -> Iterator[None]:
    """The evaluation setting, restored afterwards (it is process-global)."""
    deterministic, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    configure_cpu_inference()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


def network(role: str, seed: int = 1) -> PolicyNetwork:
    return init_network(role, game(), TINY_MODEL, seed)


def query(name: str) -> Any:
    return encoder().encode(sample(f"policy-query/valid/{name}.json"))


def test_initialisation_is_a_function_of_the_seed_and_leaves_the_global_rng_alone() -> None:
    state = torch.get_rng_state()
    first, again = network("battle", 1), network("battle", 1)
    other = network("battle", 2)
    assert torch.equal(torch.get_rng_state(), state)
    a, b, c = (net.state_dict() for net in (first, again, other))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


@pytest.mark.parametrize("seed", [-1, 2**63, 2**64])
def test_a_seed_outside_the_range_is_refused(seed: int) -> None:
    with pytest.raises(ConfigError, match="seed must be between"):
        init_network("battle", game(), TINY_MODEL, seed)


def test_sizes_are_configuration_and_the_vocabularies_come_from_the_game_data() -> None:
    small = init_network("battle", game(), TINY_MODEL, 1)
    wide = init_network(
        "battle",
        game(),
        ModelConfig(encoder=TINY_MODEL.encoder, d_model=32, n_layers=2, n_heads=4, d_ff=64),
        1,
    )
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in small.parameters())
    assert small.embeddings["species"].num_embeddings == len(game().species)
    assert small.embeddings["move"].num_embeddings == len(game().moves)
    assert small.embeddings["item"].num_embeddings == len(game().items)
    assert small.embeddings["ability"].num_embeddings == len(game().abilities)
    # The attribute tables are derived from the engine data, not parameters of the checkpoint.
    assert not any("attributes_" in name for name in small.state_dict())


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_a_network_scores_every_candidate_and_values_the_state(role: str) -> None:
    encoded = query(role)
    out = network(role)(collate([encoded]))
    assert out.logits.shape == (1, encoded.pending.shape[0])
    assert out.value.shape == (1,)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.value).all()


def test_a_network_refuses_the_other_roles_batch() -> None:
    with pytest.raises(ValueError, match="cannot score a preview batch"):
        network("battle")(collate([query("preview")]))
    with pytest.raises(ConfigError, match="role"):
        init_network("value", game(), TINY_MODEL, 1)


def test_the_battle_and_preview_networks_are_separate_networks() -> None:
    battle, preview = network("battle").state_dict(), network("preview").state_dict()
    assert set(battle) != set(preview)
    assert any(k.startswith("slot_parts") for k in battle)
    assert any(k.startswith("place") for k in preview)


def different_queries(role: str) -> list[Any]:
    """Queries unlike the target: a shortened candidate list, the other form, another request."""
    raw = sample(f"policy-query/valid/{role}.json")
    short = {**raw, "candidates": raw["candidates"][:5]}
    reversed_ = {**raw, "candidates": raw["candidates"][::-1]}
    queries = [short, reversed_]
    if role == "battle":
        queries.append(sample("policy-query/valid/battle-a2.json"))
    return [encoder().encode(q) for q in queries]


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_a_query_scores_the_same_alone_and_after_other_queries(
    role: str, cpu_inference: None
) -> None:
    # Batch invariance (ADR-010 §2): one query per forward pass, deterministic CPU; the queries
    # scored before differ from the target in size, order and form.
    net = network(role)
    target = query(role)
    others = different_queries(role)
    assert {o.pending.shape[0] for o in others} != {target.pending.shape[0]}
    alone = net(collate([target]))
    for other in others:
        net(collate([other]))
    after = net(collate([target]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)
    fresh = network(role)(collate([target]))  # and in another instance of the same weights
    assert torch.equal(alone.logits, fresh.logits)


def test_a_padded_batch_matches_its_queries_scored_alone() -> None:
    net = network("battle")
    long_ = query("battle")
    short = encoder().encode(
        {
            **sample("policy-query/valid/battle.json"),
            "candidates": sample("policy-query/valid/battle.json")["candidates"][:7],
        }
    )
    both = net(collate([long_, short]))
    for row, single in enumerate([long_, short]):
        alone = net(collate([single]))
        count = single.pending.shape[0]
        assert torch.allclose(both.logits[row, :count], alone.logits[0], atol=1e-5)
        assert torch.allclose(both.value[row], alone.value[0], atol=1e-5)
    assert both.logits[1, 7:].eq(torch.finfo(torch.float32).min).all()


def test_candidate_scores_do_not_depend_on_the_other_candidates_or_their_order() -> None:
    net = network("battle")
    raw = sample("policy-query/valid/battle.json")
    shuffled = copy.deepcopy(raw)
    order = list(range(len(raw["candidates"])))[::-1]
    shuffled["candidates"] = [raw["candidates"][i] for i in order]
    base = net(collate([encoder().encode(raw)])).logits[0]
    moved = net(collate([encoder().encode(shuffled)])).logits[0]
    assert torch.allclose(base[order], moved, atol=1e-5)


def test_the_pending_flag_is_an_input_of_the_score() -> None:
    net = network("battle")
    raw = sample("policy-query/valid/battle.json")
    flipped = copy.deepcopy(raw)
    for candidate in flipped["candidates"]:
        candidate["pending"] = not candidate["pending"]
    base = net(collate([encoder().encode(raw)])).logits
    changed = net(collate([encoder().encode(flipped)])).logits
    assert not torch.allclose(base, changed)


def test_the_a1_and_a2_forms_get_identical_logits(cpu_inference: None) -> None:
    net = network("battle")
    a1 = encoder().encode(sample("policy-query/valid/battle.json"))
    a2 = encoder().encode(sample("policy-query/valid/battle-a2.json"))
    assert torch.equal(net(collate([a1])).logits, net(collate([a2])).logits)


def test_the_network_trains(cpu_inference: None) -> None:
    net = network("battle")
    out = net(collate([query("battle")]))
    (torch.log_softmax(out.logits, -1)[0, 3] + out.value.sum()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 20
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_collate_refuses_what_it_cannot_stack() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collate([])
    with pytest.raises(ValueError, match="one kind"):
        collate([query("battle"), query("preview")])
    assert collate([query("battle")]).size == 1


def test_the_scorer_gives_one_probability_per_candidate(cpu_inference: None) -> None:
    scorer = NetworkScorer(game(), {"battle": network("battle"), "preview": network("preview")})
    raw = sample("policy-query/valid/battle.json")
    view = encoder_input(raw)
    probabilities = scorer.probabilities(view)
    assert len(probabilities) == len(raw["candidates"])
    assert abs(sum(probabilities) - 1.0) < 1e-5
    assert scorer.probabilities(view) == probabilities
    assert scorer.encoder_version == encoder().version
    preview = scorer.probabilities(encoder_input(sample("policy-query/valid/preview.json")))
    assert len(preview) == 360


def test_the_scorer_refuses_networks_that_cannot_serve_one_session() -> None:
    battle, preview = network("battle"), network("preview")
    with pytest.raises(ScorerError, match="one network per role"):
        NetworkScorer(game(), {"battle": battle})
    with pytest.raises(ScorerError, match="holds a preview network"):
        NetworkScorer(game(), {"battle": preview, "preview": preview})
    other = init_network(
        "preview",
        game(),
        ModelConfig(
            encoder=replace(TINY_ENCODER, volatile_buckets=5),
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
        ),
        1,
    )
    with pytest.raises(ScorerError, match="different encoders"):
        NetworkScorer(game(), {"battle": battle, "preview": other})


def test_deterministic_cpu_inference_is_switched_on(cpu_inference: None) -> None:
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1


def test_model_config_round_trips_and_is_strict() -> None:
    config = ModelConfig(
        encoder=replace(TINY_ENCODER, max_moves=3, stat_scale=400.0),
        d_model=24,
        n_layers=1,
        n_heads=3,
        d_ff=48,
    )
    assert ModelConfig.from_dict(json.loads(json.dumps(config.as_dict()))) == config
    with pytest.raises(ConfigError, match="keys must be exactly"):
        ModelConfig.from_dict({**config.as_dict(), "extra": 1})
    with pytest.raises(ConfigError, match="encoder must be an object"):
        ModelConfig.from_dict({**config.as_dict(), "encoder": 3})
    with pytest.raises(ConfigError, match="d_model must be an integer"):
        ModelConfig.from_dict({**config.as_dict(), "d_model": True})
    with pytest.raises(ConfigError, match="encoder config keys"):
        encoder_config_from_dict({"max_moves": 4})
    with pytest.raises(ConfigError, match="stat_scale must be a number"):
        encoder_config_from_dict({**config.encoder.as_dict(), "stat_scale": "x"})
    with pytest.raises(ConfigError, match="at least 1"):
        encoder_config_from_dict({**config.encoder.as_dict(), "max_moves": 0})
    with pytest.raises(ConfigError, match="multiple of n_heads"):
        replace(config, d_model=10, n_heads=4)
    with pytest.raises(ConfigError, match="n_layers must be at least 1"):
        replace(config, n_layers=0)
    with pytest.raises(ConfigError, match="d_model must be at least 1 and at most"):
        replace(config, d_model=1 << 40)


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_a_forward_pass_builds_every_tensor_on_the_batch_device(role: str) -> None:
    """A GPU learner moves the network and batch; a CPU-only constant would fail on `meta` here."""
    meta = torch.device("meta")
    net = init_network(role, game(), TINY_MODEL, 1).to(meta)
    out = net(to_device(collate([query(role)]), meta))
    assert out.logits.device == meta
    assert out.value.device == meta
