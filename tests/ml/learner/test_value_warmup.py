"""The optional value warm-up (ADR-019 section 4, #1111): the reader rule, the frozen policy,
the value-only update, the report and sidecar keys, and a resume in the middle of the warm-up."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.learner.trainer import to_device, value_warmup_active
from areneon_ml.model.batch import collate
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs


def warm(generations: int, **ppo: Any) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["ppo"]["value_warmup"] = {"generations": generations}
        config["ppo"].update(ppo)

    return edited_config(edit)


def snapshot(student: Any, role: str) -> dict[str, torch.Tensor]:
    return {n: t.detach().clone() for n, t in student.networks[role].state_dict().items()}


def changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
    return {n for n in before if not torch.equal(before[n], after[n])}


def test_the_reader_needs_a_warm_start_and_a_positive_count() -> None:
    config = warm(2)
    assert config.ppo.value_warmup == 2
    assert tiny_config().ppo.value_warmup is None
    with pytest.raises(RunConfigError, match="needs warm_start"):
        edited_config(lambda c: c["ppo"].update(value_warmup={"generations": 2}))
    for bad in ({"generations": 0}, {"generations": 1, "epochs": 1}, {}, {"generations": 1.5}):
        data = copy.deepcopy(raw())
        data["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        data["ppo"]["value_warmup"] = bad
        with pytest.raises(RunConfigError):
            read_run_config(data)


def test_a_run_without_the_field_keeps_its_hash_and_weights() -> None:
    plain = tiny_config()
    assert "value_warmup" not in plain.raw["ppo"]
    assert warm(1).hash != plain.hash
    assert not value_warmup_active(plain, 0)
    assert value_warmup_active(warm(2), 0)
    assert value_warmup_active(warm(2), 1)
    assert not value_warmup_active(warm(2), 2)


def test_only_the_value_head_moves_through_the_warm_up_then_ppo_runs() -> None:
    config = warm(2)
    student = learner(config)
    before = {role: snapshot(student, role) for role in ROLES}
    for _ in range(2):
        report = student.train(trajs(config))
        assert all(r.value_warmup is True for r in report.roles.values())
        assert all(r["value_warmup"] is True for r in report.as_dict().values())
    for role in ROLES:
        moved = changed(before[role], snapshot(student, role))
        assert moved
        assert all(name.startswith("value.") for name in moved)
        # Adam holds state for the value head only: the frozen parameters have no moments
        network = student.networks[role]
        with_state = {
            n for n, p in network.named_parameters() if student.optimisers[role].state.get(p)
        }
        assert with_state == {n for n, _ in network.named_parameters() if n.startswith("value.")}
        assert all(p.requires_grad for p in network.parameters())
    after_warmup = {role: snapshot(student, role) for role in ROLES}
    third = student.train(trajs(config))
    assert all(r.value_warmup is None for r in third.roles.values())
    assert "value_warmup" not in third.as_dict()["battle"]
    for role in ROLES:
        assert any(
            not n.startswith("value.") for n in changed(after_warmup[role], snapshot(student, role))
        )


def test_the_policy_outputs_are_byte_identical_through_the_warm_up() -> None:
    config = warm(1)
    student = learner(config)
    work = trajs(config)
    queries = {
        role: [s.query for t in work if t.role == role for s in t.steps][:5] for role in ROLES
    }

    def logits() -> dict[str, torch.Tensor]:
        out = {}
        for role in ROLES:
            net = student.networks[role].eval()
            with torch.no_grad():
                out[role] = net(to_device(collate(queries[role]), student.device)).logits.clone()
        return out

    before = logits()
    student.train(work)
    after = logits()
    for role in ROLES:
        assert torch.equal(before[role], after[role])


def test_the_target_kl_does_not_stop_a_warm_up_update() -> None:
    config = warm(1, target_kl_micro=1)
    report = learner(config).train(trajs(config)).roles["battle"]
    assert (report.epochs_run, report.kl_stopped) == (config.ppo.epochs, False)


def test_the_sidecar_carries_the_warm_up_only_when_set() -> None:
    config = warm(3)
    student = learner(config)
    student.train(trajs(config))
    assert json.loads(pack_state(student).sidecar)["ppo"]["value_warmup"] == {"generations": 3}
    plain = learner()
    plain.train(trajs(tiny_config()))
    assert "value_warmup" not in json.loads(pack_state(plain).sidecar)["ppo"]


@pytest.mark.parametrize("stop_after", [1, 2])
def test_a_resume_mid_warm_up_or_at_its_end_reaches_the_same_state_hash(stop_after: int) -> None:
    config = warm(2)
    continuous = learner(config)
    for _ in range(stop_after):
        continuous.train(trajs(config))
    halfway = pack_state(continuous)
    resumed = learner(config)
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert pack_state(resumed) == halfway
    for _ in range(3 - stop_after):
        continuous.train(trajs(config))
        resumed.train(trajs(config))
    assert pack_state(resumed) == pack_state(continuous)
