"""The ``aux_move_head`` network option (ADR-010 erratum A12.2, #1384): a training-only move-reveal
head on the Battle network, built last, never on the play path."""

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
from areneon_ml.model.options import AUX_MOVE_HEAD, OptionError, read_options
from areneon_ml.model.scorer import NetworkScorer
from areneon_ml.worker.view import encoder_input

from .support import TINY_ENCODER, TINY_MODEL, encoder, game, sample


def model(**options: Any) -> ModelConfig:
    return replace(TINY_MODEL, options=read_options("network", options))


WITH = model(aux_move_head=True)


def batch(role: str) -> Batch:
    return collate([encoder().encode(sample(f"policy-query/valid/{role}.json"))])


def build(role: str, config: ModelConfig, seed: int = 1) -> PolicyNetwork:
    return init_network(role, game(), config, seed)


def test_only_true_is_a_value_and_the_option_is_network_scope_and_layout_changing() -> None:
    assert read_options("network", {AUX_MOVE_HEAD: True})
    bad_values: list[Any] = [False, 0, 1, "yes", None, {}]
    for bad in bad_values:
        with pytest.raises(OptionError):
            read_options("network", {AUX_MOVE_HEAD: bad})
    with pytest.raises(OptionError, match="unknown encoder option"):
        read_options("encoder", {AUX_MOVE_HEAD: True})
    assert WITH.layout_changing
    assert ModelConfig.from_dict(WITH.as_dict()) == WITH


def test_the_battle_network_gains_the_head_and_the_preview_network_nothing() -> None:
    names = set(build("battle", WITH).state_dict())
    plain = set(build("battle", TINY_MODEL).state_dict())
    assert names - plain == {
        "aux_moves.0.weight",
        "aux_moves.0.bias",
        "aux_moves.2.weight",
        "aux_moves.2.bias",
    }
    assert plain <= names
    assert list(build("preview", WITH).state_dict()) == list(
        build("preview", TINY_MODEL).state_dict()
    )
    assert build("preview", WITH).aux_moves is None
    head = build("battle", WITH).aux_moves
    assert head is not None
    assert head[0].in_features == head[0].out_features == WITH.d_model
    assert head[2].out_features == len(game().moves)


@pytest.mark.parametrize("role", ["battle", "preview"])
def test_the_other_weights_logits_and_value_are_the_plain_networks_at_one_seed(role: str) -> None:
    plain, aux = build(role, TINY_MODEL, 5), build(role, WITH, 5)
    plain_state, aux_state = plain.state_dict(), aux.state_dict()
    for name, tensor in plain_state.items():
        assert torch.equal(tensor, aux_state[name]), name
    a, b = aux(batch(role)), plain(batch(role))
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.value, b.value)


def test_the_head_is_built_after_the_separate_value_network_and_the_scale() -> None:
    options = {"aux_move_head": True, "separate_value": "network", "policy_output_scale_ppm": 10000}
    both, without = model(**options), model(separate_value="network", policy_output_scale_ppm=10000)
    a, b = build("battle", both, 3), build("battle", without, 3)
    for name, tensor in b.state_dict().items():
        assert torch.equal(tensor, a.state_dict()[name]), name
    assert not any(n.startswith("value_net.aux_moves") for n in a.state_dict())
    assert torch.equal(a(batch("battle")).logits, b(batch("battle")).logits)


def test_forward_aux_gives_the_forward_output_bit_for_bit_and_the_head_logits() -> None:
    net = build("battle", WITH, 2)
    b = batch("battle")
    out, aux = net.forward_aux(b)
    ref = net(b)
    assert torch.equal(out.logits, ref.logits)
    assert torch.equal(out.value, ref.value)
    rows = b.token_mask.shape[1]
    assert aux.shape == (1, rows, len(game().moves))
    assert torch.equal(aux, net.aux_logits(b))
    with pytest.raises(ValueError, match="without aux_move_head"):
        build("battle", TINY_MODEL).aux_logits(b)


def test_forward_does_not_compute_the_head(monkeypatch: pytest.MonkeyPatch) -> None:
    net = build("battle", WITH)

    def boom(*_: Any) -> Any:
        raise AssertionError("the head ran on the play path")

    assert net.aux_moves is not None
    monkeypatch.setattr(net.aux_moves, "forward", boom)
    net(batch("battle"))  # raises if the head were called
    net.value_output(batch("battle"))


def test_the_scorer_and_the_worker_never_call_the_head(monkeypatch: pytest.MonkeyPatch) -> None:
    nets = {
        role: build(role, WITH if role == "battle" else TINY_MODEL, 2)
        for role in ("battle", "preview")
    }

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("the play path called the move-reveal head")

    monkeypatch.setattr(PolicyNetwork, "aux_logits", boom)
    monkeypatch.setattr(PolicyNetwork, "forward_aux", boom)
    monkeypatch.setattr(PolicyNetwork, "_aux", boom)
    scorer = NetworkScorer(game(), nets)
    for name in ("battle", "preview"):
        view = encoder_input(sample(f"policy-query/valid/{name}.json"))
        probabilities, value = scorer.evaluate(view)
        assert abs(sum(probabilities) - 1) < 1e-4
        assert torch.isfinite(torch.tensor(value))


def test_no_source_on_the_play_path_names_the_head() -> None:
    root = Path(areneon_ml.__file__).parent
    play = [root / "model" / "scorer.py", *sorted((root / "worker").glob("*.py"))]
    for path in play:
        tree = ast.parse(path.read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        assert not {n for n in names if re.search(r"aux_logits|forward_aux|aux_moves", n)}, path


def test_the_encoder_version_and_tensors_are_the_plain_ones() -> None:
    # a network option is not part of the encoder config: nothing the worker encodes changes
    assert Encoder(game(), TINY_ENCODER).version == encoder().version
    assert WITH.encoder == TINY_MODEL.encoder
    plain = encoder().encode(sample("policy-query/valid/battle.json"))
    again = Encoder(game(), WITH.encoder).encode(sample("policy-query/valid/battle.json"))
    assert all(torch.equal(v, again.tensors()[k]) for k, v in plain.tensors().items())


def test_the_gradient_of_the_head_reaches_the_trunk_and_never_the_policy_scorer() -> None:
    net = build("battle", WITH)
    b = batch("battle")
    _, aux = net.forward_aux(b)
    torch.autograd.backward(aux[:, b.token_mask.shape[1] - 1].sum())
    live = {n for n, p in net.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("aux_moves.") for n in live)
    assert any(n.startswith("blocks.") for n in live)
    assert not any(n.startswith(("value.", "score.", "slot_parts.", "cross.")) for n in live)


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
