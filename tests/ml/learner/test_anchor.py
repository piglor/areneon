"""The optional KL anchor (ADR-019 section 4, ADR-010 erratum A2 item 3, #968): the reader rule,
the integer beta rule, the KL term against a hand computation, the composition with the value
warm-up, the report and sidecar keys, and a resume."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml.learner.config import AnchorSettings, RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.ppo import anchor_kl
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state, state_hash
from areneon_ml.learner.trainer import Learner, LearnerError, to_device
from areneon_ml.model.batch import collate
from areneon_ml.model.network import ROLES, init_network

from tests.ml.learner.support import edited_config, learner, networks, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game

KL = 100_000


def anchored(
    schedule: dict[str, Any] | None = None, *, kl: int = KL, warmup: int | None = None, **ppo: Any
) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["ppo"]["anchor"] = {"kl_micro": kl, "schedule": schedule or {"kind": "constant"}}
        if warmup is not None:
            config["ppo"]["value_warmup"] = {"generations": warmup}
        config["ppo"].update(ppo)

    return edited_config(edit)


def other_anchors() -> dict[str, Any]:
    """Anchor networks that differ from the learner's (other initialisation seeds)."""
    config = tiny_config()
    return {role: init_network(role, game(), config.model, 900 + i) for i, role in enumerate(ROLES)}


# -- the reader ---------------------------------------------------------------


def test_the_reader_needs_a_warm_start_and_reads_both_schedules() -> None:
    constant = anchored()
    assert constant.ppo.anchor == AnchorSettings(KL, None)
    linear = anchored({"kind": "linear", "final_micro": 0, "generations": 4})
    assert linear.ppo.anchor is not None
    assert linear.ppo.anchor.schedule is not None
    assert tiny_config().ppo.anchor is None
    assert constant.hash != linear.hash != tiny_config().hash
    with pytest.raises(RunConfigError, match=r"ppo\.anchor needs warm_start"):
        edited_config(
            lambda c: c["ppo"].update(anchor={"kl_micro": KL, "schedule": {"kind": "constant"}})
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"kl_micro": 0, "schedule": {"kind": "constant"}},
        {"kl_micro": 100_000_001, "schedule": {"kind": "constant"}},
        {"kl_micro": KL},
        {"schedule": {"kind": "constant"}},
        {"kl_micro": KL, "schedule": {"kind": "constant"}, "role": "battle"},
        {"kl_micro": KL, "schedule": {"kind": "linear", "final_micro": 0, "generations": 0}},
        {"kl_micro": KL, "schedule": {"kind": "linear", "final_micro": -1, "generations": 2}},
        {"kl_micro": KL, "schedule": {"kind": "linear", "generations": 2}},
        {"kl_micro": KL, "schedule": {"kind": "cosine"}},
        {"kl_micro": 1.5, "schedule": {"kind": "constant"}},
    ],
)
def test_the_schema_refuses_a_malformed_anchor(bad: dict[str, Any]) -> None:
    data = copy.deepcopy(raw())
    data["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
    data["ppo"]["anchor"] = bad
    with pytest.raises(RunConfigError):
        read_run_config(data)


# -- beta: the integer rule -----------------------------------------------------


def test_beta_follows_the_integer_linear_rule() -> None:
    down = anchored({"kind": "linear", "final_micro": 0, "generations": 3}).ppo.anchor
    assert down is not None
    # 100000 + trunc(-100000 * t / 3): truncation toward zero, exactly final from t = generations
    assert [down.beta_micro(t) for t in (0, 1, 2, 3, 4, 100)] == [
        100_000,
        66_667,
        33_334,
        0,
        0,
        0,
    ]
    up = anchored({"kind": "linear", "final_micro": 200_000, "generations": 3}).ppo.anchor
    assert up is not None
    assert [up.beta_micro(t) for t in (0, 1, 2, 3, 9)] == [
        100_000,
        133_333,
        166_666,
        200_000,
        200_000,
    ]
    constant = anchored().ppo.anchor
    assert constant is not None
    assert [constant.beta_micro(t) for t in (0, 5, 1000)] == [KL, KL, KL]
    with pytest.raises(RunConfigError, match="from 0"):
        constant.beta_micro(-1)
    with pytest.raises(RunConfigError, match="from 0"):
        up.beta_micro(-1)


def test_the_schedule_counts_from_the_first_policy_update() -> None:
    plain = learner(anchored({"kind": "linear", "final_micro": 0, "generations": 4}))
    assert [plain.anchor_beta_micro(u) for u in range(6)] == [100_000, 75_000, 50_000, 25_000, 0, 0]
    warm = learner(anchored({"kind": "linear", "final_micro": 0, "generations": 4}, warmup=2))
    # updates 0 and 1 are the value warm-up (policy frozen): beta 0; the decay starts at update 2
    assert [warm.anchor_beta_micro(u) for u in range(8)] == [
        0,
        0,
        100_000,
        75_000,
        50_000,
        25_000,
        0,
        0,
    ]
    assert learner(tiny_config()).anchor_beta_micro(3) is None


# -- the KL term ----------------------------------------------------------------


def test_the_anchor_term_equals_a_hand_computed_kl_on_three_candidates() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.5, 9.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, False]])
    q = [0.2, 0.5, 0.3]
    padded = torch.tensor([[math.log(x) for x in q] + [0.0]], dtype=torch.float32)
    z = sum(math.exp(x) for x in (1.0, 2.0, 0.5))
    p = [math.exp(x) / z for x in (1.0, 2.0, 0.5)]
    hand = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q, strict=True))
    got = anchor_kl(logits, mask, padded)
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(hand, rel=1e-5)
    assert hand == pytest.approx(0.0707, abs=1e-4)  # by hand: 0.03357 + 0.14380 - 0.10668
    # identical distributions have no divergence; the padded logit never matters
    same = torch.log_softmax(logits.masked_fill(~mask, -1e30), dim=-1)
    assert float(anchor_kl(logits, mask, same)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(
        anchor_kl(logits.clone().index_fill_(1, torch.tensor([3]), -9.0), mask, padded)[0]
    ) == pytest.approx(hand, rel=1e-5)


def test_the_anchor_term_has_a_gradient_to_the_logits_only() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    anchor = torch.log_softmax(torch.tensor([[0.0, 0.0, 0.0]]), dim=-1).requires_grad_(False)
    torch.autograd.backward(anchor_kl(logits, mask, anchor).sum())
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_the_trainer_scores_the_anchor_once_per_generation_without_gradient() -> None:
    config = anchored()
    student = learner(config, other_anchors())
    calls = {role: 0 for role in ROLES}
    for role in ROLES:
        net = student.anchors[role]
        original = net.forward

        def counting(batch: Any, _r: str = role, _o: Any = original) -> Any:
            calls[_r] += 1
            assert not torch.is_grad_enabled()
            return _o(batch)

        net.forward = counting  # type: ignore[method-assign]
    work = trajs(config)
    student.train(work)
    size = config.ppo.minibatch_size
    for role in ROLES:
        steps = sum(len(t.steps) for t in work if t.role == role)
        assert calls[role] == -(-steps // size)  # one pass over the generation, not per epoch
        assert not any(p.requires_grad for p in student.anchors[role].parameters())
        assert not student.anchors[role].training


# -- training --------------------------------------------------------------------


def test_the_report_has_the_anchor_keys_only_when_set() -> None:
    plain = learner(tiny_config()).train(trajs(tiny_config())).as_dict()
    assert all("anchor_kl" not in r and "anchor_beta_micro" not in r for r in plain.values())
    config = anchored()
    out = learner(config, other_anchors()).train(trajs(config)).as_dict()
    for role_report in out.values():
        assert role_report["anchor_beta_micro"] == KL
        assert role_report["anchor_kl"] > 0


def test_an_equal_anchor_starts_near_zero_and_a_stronger_beta_holds_the_policy_closer() -> None:
    config = anchored(epochs=1)
    first = learner(config).train(trajs(config)).roles["battle"]
    # the first minibatch has KL 0 exactly (policy equals anchor); later ones only what one
    # optimiser step moved
    assert first.anchor_kl is not None
    assert 0.0 <= first.anchor_kl < 0.01

    def drift(kl: int) -> float:
        c = anchored(kl=kl, epochs=4)
        report = learner(c).train(trajs(c)).roles["battle"]
        assert report.anchor_kl is not None
        return report.anchor_kl

    assert drift(100_000_000) < drift(1)


def test_a_zero_final_beta_still_reports_the_kl() -> None:
    config = anchored({"kind": "linear", "final_micro": 0, "generations": 1})
    student = learner(config, other_anchors())
    student.train(trajs(config))
    report = student.train(trajs(config)).roles["preview"]
    assert report.anchor_beta_micro == 0
    assert report.anchor_kl is not None
    assert report.anchor_kl > 0


def test_a_run_with_beta_zero_trains_as_one_without_an_anchor_term() -> None:
    zero = anchored({"kind": "linear", "final_micro": 0, "generations": 1}, kl=1)
    with_term = learner(zero, other_anchors())
    with_term.train(trajs(zero))
    with_term.train(trajs(zero))  # beta is 0 from the second update on
    no_term = learner(tiny_config())
    no_term.train(trajs(tiny_config()))
    no_term.train(trajs(tiny_config()))
    # the first update differs (beta 1e-6), the weights are not the same; the second update's
    # gradient has no anchor part, so only the first update's tiny effect remains
    for role in ROLES:
        for (n, a), (_, b) in zip(
            with_term.networks[role].named_parameters(),
            no_term.networks[role].named_parameters(),
            strict=True,
        ):
            assert torch.allclose(a, b, atol=1e-3), n


# -- the value warm-up composition ------------------------------------------------------


def test_the_anchor_is_inactive_while_the_policy_is_frozen_then_acts() -> None:
    config = anchored({"kind": "linear", "final_micro": 0, "generations": 4}, warmup=2)
    student = learner(config, other_anchors())
    before = {r: {n: t.clone() for n, t in student.networks[r].state_dict().items()} for r in ROLES}
    for _ in range(2):
        reports = student.train(trajs(config))
        for role_report in reports.as_dict().values():
            assert role_report["value_warmup"] is True
            assert role_report["anchor_kl"] == 0.0
            assert role_report["anchor_beta_micro"] == 0
    for role in ROLES:
        after = student.networks[role].state_dict()
        moved = {n for n in before[role] if not torch.equal(before[role][n], after[n])}
        assert moved
        assert all(n.startswith("value.") for n in moved)
    third = student.train(trajs(config)).as_dict()
    fourth = student.train(trajs(config)).as_dict()
    for role in ROLES:
        assert "value_warmup" not in third[role]
        assert third[role]["anchor_beta_micro"] == 100_000  # anchor update 0
        assert third[role]["anchor_kl"] > 0
        assert fourth[role]["anchor_beta_micro"] == 75_000  # anchor update 1
    assert json.loads(pack_state(student).sidecar)["anchor_beta_micro"] == 75_000


# -- the learner state ----------------------------------------------------------------


def test_the_sidecar_records_beta_only_when_set() -> None:
    plain = learner(tiny_config())
    plain.train(trajs(tiny_config()))
    assert "anchor_beta_micro" not in json.loads(pack_state(plain).sidecar)
    config = anchored({"kind": "linear", "final_micro": 0, "generations": 4})
    student = learner(config)
    assert "anchor_beta_micro" not in json.loads(pack_state(student).sidecar)  # nothing run yet
    student.train(trajs(config))
    sidecar = json.loads(pack_state(student).sidecar)
    assert sidecar["anchor_beta_micro"] == 100_000
    assert sidecar["ppo"]["anchor"]["kl_micro"] == KL


@pytest.mark.parametrize("stop_after", [1, 2, 3])
def test_a_resume_reaches_the_same_state_hash(stop_after: int) -> None:
    config = anchored({"kind": "linear", "final_micro": 0, "generations": 3}, warmup=1)
    continuous = learner(config, other_anchors())
    for _ in range(stop_after):
        continuous.train(trajs(config))
    halfway = pack_state(continuous)
    resumed = learner(config, other_anchors())
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.last_anchor_beta_micro == continuous.last_anchor_beta_micro
    assert pack_state(resumed) == halfway
    for _ in range(4 - stop_after):
        continuous.train(trajs(config))
        resumed.train(trajs(config))
    assert pack_state(resumed) == pack_state(continuous)


def test_a_sidecar_with_another_beta_is_refused() -> None:
    config = anchored({"kind": "linear", "final_micro": 0, "generations": 4})
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)
    for change in ("set", "drop"):
        sidecar = json.loads(packed.sidecar)
        if change == "set":
            sidecar["anchor_beta_micro"] = 1
        else:
            del sidecar["anchor_beta_micro"]
        with pytest.raises(LearnerStateError, match="anchor weight"):
            restore_state(
                learner(config), packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)
            )
    plain = learner(tiny_config())
    plain.train(trajs(tiny_config()))
    stored = pack_state(plain)
    sidecar = json.loads(stored.sidecar)
    sidecar["anchor_beta_micro"] = 5
    with pytest.raises(LearnerStateError, match="anchor weight"):
        restore_state(
            learner(tiny_config()),
            stored.tensors,
            jcs(sidecar),
            state_hash(stored.tensors, sidecar),
        )


def test_the_learner_wants_anchor_networks_exactly_when_the_config_has_an_anchor() -> None:
    config = anchored()
    device = torch.device("cpu")
    with pytest.raises(LearnerError, match="exactly when"):
        Learner(config, networks(config), device)
    with pytest.raises(LearnerError, match="exactly when"):
        Learner(tiny_config(), networks(tiny_config()), device, networks(tiny_config()))
    with pytest.raises(LearnerError, match="one anchor network per role"):
        Learner(config, networks(config), device, {"battle": networks(config)["battle"]})


def test_the_anchor_networks_are_a_separate_frozen_pair() -> None:
    config = anchored()
    student = learner(config)
    for role in ROLES:
        assert student.anchors[role] is not student.networks[role]
    work = trajs(config)
    queries = [s.query for t in work if t.role == "battle" for s in t.steps][:4]
    batch = to_device(collate(queries), student.device)
    anchor_logits = student.anchors["battle"](batch).logits.clone()
    student.train(work)
    assert torch.equal(student.anchors["battle"](batch).logits, anchor_logits)
