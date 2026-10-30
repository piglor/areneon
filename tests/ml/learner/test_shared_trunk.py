"""The ``shared_trunk`` network option (ADR-010 erratum A4 item 5, #1009): one trunk for both
roles, one Adam over the union, complete per-role checkpoints with equal trunk tensors, and a
pair or a state that disagrees is refused."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml.checkpoint.files import (
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    pack_safetensors,
    read_header,
    unpack_safetensors,
)
from areneon_ml.learner.config import RunConfig, RunConfigError
from areneon_ml.learner.state import (
    LearnerOptionsError,
    LearnerStateError,
    pack_state,
    restore_state,
    state_hash,
)
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.network import (
    ROLES,
    ModelConfig,
    TrunkError,
    check_shared_trunk,
    init_network,
    shared_trunk_of,
    tie_trunk,
)
from areneon_ml.model.scorer import NetworkScorer, ScorerError

from tests.ml.learner.support import edited_config, learner, networks, tiny_config
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game


def shared_config(**ppo: Any) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["options"] = {"shared_trunk": True}
        raw["ppo"].update(ppo)

    return edited_config(edit)


def trunk_state(student: Any, role: str) -> dict[str, torch.Tensor]:
    network = student.networks[role]
    state = network.state_dict()
    return {key: state[key].clone() for key in network.trunk_keys()}


def test_the_option_is_registered_and_changes_the_checkpoint_format() -> None:
    config = shared_config().model
    assert shared_trunk_of(config)
    assert not shared_trunk_of(tiny_config().model)
    assert config.layout_changing
    assert config.as_dict()["options"] == {"shared_trunk": True}


def test_the_trunk_keys_are_the_shared_layers_only() -> None:
    config = shared_config().model
    battle = init_network("battle", game(), config, 1)
    preview = init_network("preview", game(), config, 2)
    keys = battle.trunk_keys()
    assert keys == preview.trunk_keys()
    assert "aggregate" in keys
    assert any(k.startswith("blocks.0.") for k in keys)
    assert not any(
        k.startswith(("score.", "cross.", "candidate.", "value.", "slot_parts.")) for k in keys
    )


def test_the_tied_networks_share_parameters_and_one_optimiser() -> None:
    student = learner(shared_config())
    battle, preview = student.networks["battle"], student.networks["preview"]
    assert battle.aggregate is preview.aggregate
    assert battle.blocks is preview.blocks
    assert student.optimisers["battle"] is student.optimisers["preview"]
    union = student.optimisers["battle"].param_groups[0]["params"]
    assert len({id(p) for p in union}) == len(union)
    owned = {role: {n for n, _ in student.entries[role]} for role in ROLES}
    assert set(battle.trunk_keys()) <= owned["battle"]
    assert not owned["preview"] & set(preview.trunk_keys())
    plain = learner(tiny_config())
    assert plain.networks["battle"].aggregate is not plain.networks["preview"].aggregate
    assert plain.optimisers["battle"] is not plain.optimisers["preview"]


def test_tying_networks_of_different_configs_is_refused() -> None:
    a = init_network("battle", game(), shared_config().model, 1)
    b = init_network("preview", game(), tiny_config().model, 2)
    with pytest.raises(TrunkError, match="same model config"):
        tie_trunk(a, b)


def test_an_update_leaves_equal_trunk_tensors_in_both_complete_checkpoints() -> None:
    config = shared_config()
    student = learner(config)
    before = trunk_state(student, "battle")
    head_before = {
        n: t.clone()
        for n, t in student.networks["preview"].state_dict().items()
        if n.startswith("score.")
    }
    student.train(trajs(config))
    after = trunk_state(student, "battle")
    assert any(not torch.equal(before[k], after[k]) for k in before)  # the trunk learned
    preview_state = student.networks["preview"].state_dict()
    assert any(not torch.equal(head_before[k], preview_state[k]) for k in head_before)
    loaded = {}
    for role in ROLES:
        data = checkpoint_bytes(student.networks[role], game())
        loaded[role] = load_checkpoint(
            data, game(), checkpoint_id(data), role, configured=config.model
        )
        assert set(loaded[role].state_dict()) == set(student.networks[role].state_dict())
    check_shared_trunk(loaded)
    for key in loaded["battle"].trunk_keys():
        assert torch.equal(loaded["battle"].state_dict()[key], loaded["preview"].state_dict()[key])


def test_a_pair_with_differing_trunks_is_refused_by_the_check_and_the_scorer() -> None:
    config = shared_config().model
    pair = {role: init_network(role, game(), config, 1 + i) for i, role in enumerate(ROLES)}
    with pytest.raises(TrunkError, match="differ"):
        check_shared_trunk(pair)
    with pytest.raises(ScorerError, match="differ"):
        NetworkScorer(game(), pair)
    tie_trunk(pair["battle"], pair["preview"])
    check_shared_trunk(pair)
    NetworkScorer(game(), pair)


def test_a_pair_that_disagrees_on_the_option_is_refused() -> None:
    pair = {
        "battle": init_network("battle", game(), shared_config().model, 1),
        "preview": init_network("preview", game(), tiny_config().model, 2),
    }
    with pytest.raises(TrunkError, match="only one"):
        check_shared_trunk(pair)


def test_a_pair_without_the_option_is_not_looked_at() -> None:
    check_shared_trunk(networks(tiny_config()))


def test_the_state_round_trips_and_a_resume_equals_an_uninterrupted_run() -> None:
    config = shared_config()
    straight = learner(config)
    straight.train(trajs(config))
    straight.train(trajs(config))
    first = learner(config)
    first.train(trajs(config))
    packed = pack_state(first)
    resumed = learner(config)
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    assert pack_state(resumed) == packed
    resumed.train(trajs(config))
    assert pack_state(resumed) == pack_state(straight)


def test_the_adam_state_is_stored_once_for_the_trunk() -> None:
    config = shared_config()
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)
    names = unpack_names(packed.tensors)
    assert "opt.battle.exp_avg.aggregate" in names
    assert "opt.preview.exp_avg.aggregate" not in names
    assert "net.preview.aggregate" in names  # the weights stay complete per role


def unpack_names(data: bytes) -> set[str]:
    return {n for n in read_header(data) if n != "__metadata__"}


@pytest.mark.parametrize("direction", ["to_plain", "from_plain"])
def test_a_state_across_the_two_layouts_is_refused(direction: str) -> None:
    shared, plain = shared_config(), tiny_config()
    made, target_config = (shared, plain) if direction == "to_plain" else (plain, shared)
    packed = pack_state(learner(made))
    target = learner(target_config)
    before = pack_state(target)
    with pytest.raises(LearnerOptionsError, match="other model options"):
        restore_state(target, packed.tensors, packed.sidecar, packed.hash)
    assert pack_state(target) == before


def test_a_state_whose_trunk_tensors_differ_is_refused() -> None:
    config = shared_config()
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)
    # Rebuild the file with a perturbed Preview copy of one trunk tensor, hash and all.
    expected = {}
    header = read_header(packed.tensors)
    dtypes = {"F32": torch.float32, "U8": torch.uint8}
    for name, entry in header.items():
        if name != "__metadata__":
            expected[name] = (dtypes[entry["dtype"]], tuple(entry["shape"]))
    tensors = unpack_safetensors(packed.tensors, expected)
    tensors["net.preview.aggregate"] = tensors["net.preview.aggregate"] + 1
    data = pack_safetensors(tensors)
    sidecar = json.loads(packed.sidecar)
    target = learner(config)
    before = pack_state(target)
    with pytest.raises(LearnerStateError, match="shared trunk"):
        restore_state(target, data, jcs(sidecar), state_hash(data, sidecar))
    assert pack_state(target) == before


def test_the_gradient_is_clipped_over_the_union_of_parameters() -> None:
    config = shared_config(max_grad_norm_micro=1)  # a tiny norm: every step is clipped hard
    clipped, free = learner(config), learner(shared_config())
    clipped.train(trajs(config))
    free.train(trajs(config))
    a, b = pack_state(clipped), pack_state(free)
    assert a.hash != b.hash


def test_the_option_does_not_change_a_run_without_it() -> None:
    plain = tiny_config()
    student = learner(plain)
    student.train(trajs(plain))
    assert student.optimisers["battle"] is not student.optimisers["preview"]
    assert "model_options" not in pack_state(student).sidecar.decode()


def test_the_reader_refuses_a_two_role_warm_start_but_takes_a_battle_only_one() -> None:
    def edit(warm: dict[str, str]) -> Any:
        def inner(raw: dict[str, Any]) -> None:
            raw["model"]["options"] = {"shared_trunk": True}
            raw["warm_start"] = warm

        return inner

    with pytest.raises(RunConfigError, match="shared_trunk cannot be combined"):
        edited_config(edit({"preview": "a" * 64, "battle": "b" * 64}))
    assert edited_config(edit({"battle": "b" * 64})).warm_start == {"battle": "b" * 64}


def test_the_model_config_round_trips_with_the_option() -> None:
    config = shared_config().model
    assert ModelConfig.from_dict(config.as_dict()) == config
    assert dataclasses.replace(config, options=()).as_dict() == tiny_config().model.as_dict()


def test_the_joint_step_is_one_adam_step_on_the_sum_of_the_roles_losses() -> None:
    """One epoch, one minibatch each: the update equals a hand-made step on the summed losses
    (targets from the pre-update weights, generator draws in Battle, Preview order)."""
    config = shared_config(epochs=1, minibatch_size=1000)
    run, hand = learner(config), learner(config)
    work = trajs(config)
    tables = {r: hand._table(r, [t for t in work if t.role == r], False, None) for r in ROLES}
    generator = torch.Generator()
    generator.manual_seed(config.learner_seed)
    weights = hand._weights()
    parts = {}
    for role in ROLES:
        order = torch.randperm(len(tables[role].steps), generator=generator).tolist()
        parts[role], _, _ = hand._minibatch_loss(role, tables[role], False, order, weights)
    trunk = hand.networks["battle"].aggregate
    grad_b = torch.autograd.grad(parts["battle"], trunk, retain_graph=True)[0]
    grad_p = torch.autograd.grad(parts["preview"], trunk, retain_graph=True)[0]
    assert grad_b.abs().sum() > 0
    assert grad_p.abs().sum() > 0  # both roles' gradients reach the shared trunk
    optimiser = hand.optimisers["battle"]
    optimiser.zero_grad(set_to_none=True)
    torch.autograd.backward(parts["battle"] + parts["preview"])
    assert trunk.grad is not None
    assert torch.allclose(trunk.grad, grad_b + grad_p, atol=1e-7)
    union = [p for r in ROLES for _, p in hand.entries[r]]
    torch.nn.utils.clip_grad_norm_(union, config.ppo.max_grad_norm)
    optimiser.step()
    report = run.train(work)
    for role in ROLES:
        expected = hand.networks[role].state_dict()
        for key, value in run.networks[role].state_dict().items():
            assert torch.allclose(value, expected[key], atol=1e-7, rtol=0), (role, key)
    assert {r: v.optimiser_steps for r, v in report.roles.items()} == {"battle": 1, "preview": 1}


def test_a_role_with_fewer_steps_than_minibatches_fills_the_first_ones_only() -> None:
    assert Learner._spread(list(range(5)), 3) == [[0, 1], [2, 3], [4]]
    assert Learner._spread([0, 1], 4) == [[0], [1], [], []]
    assert Learner._spread([], 2) == [[], []]


def test_the_target_kl_stop_is_joint() -> None:
    config = shared_config(epochs=3, minibatch_size=4, target_kl_micro=1)
    report = learner(config).train(trajs(config))
    runs = {r.epochs_run for r in report.roles.values()}
    assert runs == {1}
    assert all(r.kl_stopped for r in report.roles.values())
