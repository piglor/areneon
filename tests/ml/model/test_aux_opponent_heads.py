"""The ``aux_opponent_heads`` network option (ADR-010 erratum A14.2, #1434): three training-only
opponent heads on the Battle network, built last, never on the play path."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import areneon_ml
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
from areneon_ml.model.options import AUX_OPPONENT_HEADS, OptionError, read_options
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker.view import encoder_input

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample


def model(**options: Any) -> ModelConfig:
    return replace(TINY_MODEL, options=read_options("network", options))


WITH = model(aux_opponent_heads=True)
BOTH = model(aux_opponent_heads=True, aux_move_head=True)


def batch(role: str) -> Batch:
    return collate([encoder().encode(sample(f"policy-query/valid/{role}.json"))])


def build(role: str, config: ModelConfig, seed: int = 1) -> PolicyNetwork:
    return init_network(role, game(), config, seed)


def picks(b: Batch) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    last = b.token_mask.shape[1] - 1
    index = torch.tensor([0, 0])
    rows = torch.tensor([last, last - 1])
    return {"action": (index, rows), "item": (index[:1], rows[:1]), "ability": (index, rows)}


def test_only_true_is_a_value_and_the_option_is_network_scope_and_layout_changing() -> None:
    assert read_options("network", {AUX_OPPONENT_HEADS: True})
    bad_values: list[Any] = [False, 0, 1, "yes", None, {}]
    for bad in bad_values:
        with pytest.raises(OptionError):
            read_options("network", {AUX_OPPONENT_HEADS: bad})
    with pytest.raises(OptionError, match="unknown encoder option"):
        read_options("encoder", {AUX_OPPONENT_HEADS: True})
    assert WITH.layout_changing
    assert ModelConfig.from_dict(WITH.as_dict()) == WITH


def test_the_battle_network_gains_the_three_heads_and_the_preview_network_nothing() -> None:
    names = set(build("battle", WITH).state_dict())
    plain = set(build("battle", TINY_MODEL).state_dict())
    expected = {
        f"{head}.{i}.{kind}"
        for head in ("aux_action", "aux_item", "aux_ability")
        for i in (0, 2)
        for kind in ("weight", "bias")
    }
    assert names - plain == expected
    assert plain <= names
    assert list(build("preview", WITH).state_dict()) == list(
        build("preview", TINY_MODEL).state_dict()
    )
    battle = build("battle", WITH)
    assert battle.aux_action is not None
    assert battle.aux_item is not None
    assert battle.aux_ability is not None
    sizes = (len(game().moves) + 1, len(game().items), len(game().abilities))
    for head, size in zip(
        (battle.aux_action, battle.aux_item, battle.aux_ability), sizes, strict=True
    ):
        assert head[0].in_features == head[0].out_features == WITH.d_model
        assert head[2].out_features == size
    assert build("preview", WITH).aux_action is None


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_the_other_weights_logits_and_value_are_the_plain_networks_at_one_seed(role: str) -> None:
    plain, heads = build(role, TINY_MODEL, 5), build(role, WITH, 5)
    plain_state, heads_state = plain.state_dict(), heads.state_dict()
    for name, tensor in plain_state.items():
        assert torch.equal(tensor, heads_state[name]), name
    a, b = heads(batch(role)), plain(batch(role))
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.value, b.value)


def test_the_heads_are_built_after_the_move_head_and_the_separate_value_network() -> None:
    options = {
        "aux_opponent_heads": True,
        "aux_move_head": True,
        "separate_value": "network",
        "policy_output_scale_ppm": 10000,
    }
    both = model(**options)
    without = model(aux_move_head=True, separate_value="network", policy_output_scale_ppm=10000)
    a, b = build("battle", both, 3), build("battle", without, 3)
    for name, tensor in b.state_dict().items():
        assert torch.equal(tensor, a.state_dict()[name]), name
    assert torch.equal(a(batch("battle")).logits, b(batch("battle")).logits)
    assert not any(n.startswith("value_net.aux_") for n in a.state_dict())


def test_forward_opponent_gives_the_forward_output_bit_for_bit_and_the_rows_asked_for() -> None:
    net = build("battle", WITH, 2)
    b = batch("battle")
    out, moves, heads = net.forward_opponent(b, picks(b))
    ref = net(b)
    assert torch.equal(out.logits, ref.logits)
    assert torch.equal(out.value, ref.value)
    assert moves is None
    sizes = {"action": len(game().moves) + 1, "item": len(game().items)}
    assert heads["action"].shape == (2, sizes["action"])
    assert heads["item"].shape == (1, sizes["item"])
    assert heads["ability"].shape == (2, len(game().abilities))
    assert net.forward_opponent(b, {})[2] == {}
    with pytest.raises(ValueError, match="without aux_opponent_heads"):
        build("battle", TINY_MODEL).forward_opponent(b, picks(b))


def test_forward_opponent_also_gives_the_move_head_logits_exactly_as_forward_aux_does() -> None:
    net = build("battle", BOTH, 2)
    b = batch("battle")
    _, moves, _ = net.forward_opponent(b, picks(b), moves=True)
    assert moves is not None
    assert torch.equal(moves, net.forward_aux(b)[1])


def test_forward_does_not_compute_the_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    net = build("battle", WITH)

    def boom(*_: Any) -> Any:
        raise AssertionError("a head ran on the play path")

    for head in (net.aux_action, net.aux_item, net.aux_ability):
        assert head is not None
        monkeypatch.setattr(head, "forward", boom)
    net(batch("battle"))
    net.value_output(batch("battle"))


def test_the_scorer_and_the_worker_never_call_the_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    nets = {
        role: build(role, WITH if role == "battle" else TINY_MODEL, 2)
        for role in ("battle", "preview")
    }

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("the play path called the opponent heads")

    monkeypatch.setattr(PolicyNetwork, "forward_opponent", boom)
    monkeypatch.setattr(PolicyNetwork, "_opponent_head", boom)
    scorer = NetworkScorer(game(), nets)
    for name in ("battle", "preview"):
        view = encoder_input(sample(f"policy-query/valid/{name}.json"))
        probabilities, value = scorer.evaluate(view)
        assert abs(sum(probabilities) - 1) < 1e-4
        assert torch.isfinite(torch.tensor(value))


def test_no_source_on_the_play_path_names_the_heads() -> None:
    root = Path(areneon_ml.__file__).parent
    play = [root / "model" / "scorer.py", *sorted((root / "worker").glob("*.py"))]
    for path in play:
        tree = ast.parse(path.read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        assert not {
            n for n in names if re.search(r"forward_opponent|aux_action|aux_item|aux_ability", n)
        }, path


def test_the_encoder_version_and_tensors_are_the_plain_ones() -> None:
    assert Encoder(game(), TINY_ENCODER).version == encoder().version
    assert WITH.encoder == TINY_MODEL.encoder
    plain = encoder().encode(sample("policy-query/valid/battle.json"))
    again = Encoder(game(), WITH.encoder).encode(sample("policy-query/valid/battle.json"))
    assert all(torch.equal(v, again.tensors()[k]) for k, v in plain.tensors().items())


def test_the_checkpoint_is_format_two_and_loads_only_under_the_same_options() -> None:
    net = build("battle", WITH)
    data = checkpoint_bytes(net, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=WITH)
    assert loaded.config == WITH
    assert checkpoint_bytes(loaded, game()) == data
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)
    plain = checkpoint_bytes(build("battle", TINY_MODEL), game())
    with pytest.raises(CheckpointOptionsError, match="other model options"):
        load_checkpoint(plain, game(), checkpoint_id(plain), "battle", configured=WITH)
