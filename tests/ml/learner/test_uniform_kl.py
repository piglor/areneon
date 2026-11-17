"""``ppo.uniform_kl`` (ADR-010 erratum A14.1, #1433): the term and its logit gradient by hand, the
non-pending support, the temperature, the schedule index, the value warm-up, the report and sidecar
keys, composition with the anchor and the entropy bonus, and a resume."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml.learner import ppo, trainer
from areneon_ml.learner.config import (
    RunConfig,
    RunConfigError,
    UniformKlSettings,
    read_run_config,
)
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state, state_hash
from areneon_ml.model.batch import collate

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_anchor import anchored, other_anchors
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs

COEF = 100_000


def uniform(
    coef: int = COEF, schedule: dict[str, Any] | None = None, **ppo_fields: Any
) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["ppo"]["uniform_kl"] = {
            "coef_micro": coef,
            "schedule": schedule or {"kind": "constant"},
        }
        config["ppo"].update(ppo_fields)

    return edited_config(edit)


def warmed(generations: int = 1, schedule: dict[str, Any] | None = None) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["ppo"]["value_warmup"] = {"generations": generations}
        config["ppo"]["uniform_kl"] = {
            "coef_micro": COEF,
            "schedule": schedule or {"kind": "constant"},
        }

    return edited_config(edit)


# -- the term, by hand ---------------------------------------------------------------------------


def hand(logits: list[float], pending: list[bool]) -> tuple[float, list[float]]:
    """KL(u || pi) over the non-pending candidates, pi over all of them, and its logit gradient."""
    total = sum(math.exp(z) for z in logits)
    pi = [math.exp(z) / total for z in logits]
    support = [i for i, p in enumerate(pending) if not p]
    n = len(support)
    kl = -math.log(n) - sum(math.log(pi[i]) for i in support) / n
    return kl, [p - (1.0 / n if i in support else 0.0) for i, p in enumerate(pi)]


def test_the_term_and_its_logit_gradient_on_three_candidates() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True]])
    pending = torch.tensor([[0.0, 0.0, 1.0]])
    rows, valid = ppo.uniform_kl(logits, mask, pending)
    want, grad = hand([0.0, 1.0, 2.0], [False, False, True])
    assert valid.tolist() == [True]
    assert rows.item() == pytest.approx(want, rel=1e-6)
    (gradient,) = torch.autograd.grad(rows.sum(), logits)
    # pi_k - 1/n on the support, pi_k (a mild push down) on the pending candidate
    assert gradient[0].tolist() == pytest.approx(grad, rel=1e-5)
    assert grad[2] > 0
    assert grad[0] < 0  # the least likely support candidate is lifted


def test_a_uniform_policy_has_no_term_and_no_pending_is_the_plain_kl() -> None:
    logits = torch.zeros(1, 4, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    rows, _ = ppo.uniform_kl(logits, mask, torch.zeros(1, 4))
    assert rows.item() == pytest.approx(0.0, abs=1e-6)
    skewed = torch.tensor([[3.0, 0.0, 0.0, -2.0]])
    want, _ = hand([3.0, 0.0, 0.0, -2.0], [False] * 4)
    assert ppo.uniform_kl(skewed, mask, torch.zeros(1, 4))[0].item() == pytest.approx(want)


def test_pending_candidates_are_not_in_the_support_but_stay_in_the_normaliser() -> None:
    logits = torch.tensor([[0.5, -0.5, 4.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    pending = torch.tensor([[0.0, 0.0, 1.0]])
    with_pending, _ = ppo.uniform_kl(logits, mask, pending)
    without, _ = ppo.uniform_kl(logits[:, :2], mask[:, :2], pending[:, :2])
    # the big pending logit takes probability mass: the support is far below uniform
    assert with_pending.item() > without.item() + 1.0
    # the support itself ignores the pending flag's value but not the padding
    assert ppo.uniform_support(mask, pending).tolist() == [[True, True, False]]


@pytest.mark.parametrize(
    "pending", [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], ids=["one-non-pending", "all-pending"]
)
def test_fewer_than_two_non_pending_candidates_give_no_term(pending: list[float]) -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    rows, valid = ppo.uniform_kl(
        logits, torch.ones(1, 3, dtype=torch.bool), torch.tensor([pending])
    )
    assert valid.tolist() == [False]
    assert rows.item() == 0.0
    (gradient,) = torch.autograd.grad(rows.sum(), logits)
    assert not gradient.any()


def test_padding_is_masked_and_the_temperature_scales_the_logits() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 50.0]])
    mask = torch.tensor([[True, True, True, False]])
    pending = torch.zeros(1, 4)
    padded, _ = ppo.uniform_kl(logits, mask, pending)
    plain, _ = ppo.uniform_kl(logits[:, :3], mask[:, :3], pending[:, :3])
    assert padded.item() == pytest.approx(plain.item(), rel=1e-6)
    tempered, _ = ppo.uniform_kl(logits, mask, pending, 500_000)
    doubled, _ = ppo.uniform_kl(logits * 2.0, mask, pending)
    assert tempered.item() == pytest.approx(doubled.item(), rel=1e-6)
    assert tempered.item() != pytest.approx(plain.item(), rel=1e-3)


def test_the_floor_counts_the_support_candidates_below_one_in_a_hundred_n() -> None:
    # n = 2 support candidates: the threshold is 1/200 = 0.005; pi = softmax(0, 8, 8): ~0.0003
    logits = torch.tensor([[0.0, 8.0, 8.0], [0.0, 0.0, 0.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    pending = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # row 0: the support {1, 2} is far above the threshold (the pending 0 is below but not counted);
    # row 1: support {0, 1}, uniform 1/3 each, above 1/200
    assert ppo.uniform_floor_counts(logits, mask, pending) == (0, 4)
    low = torch.tensor([[8.0, 0.0, 8.0]])
    assert ppo.uniform_floor_counts(low, mask[:1], torch.tensor([[0.0, 0.0, 1.0]])) == (1, 2)
    one = torch.tensor([[0.0, 0.0]])
    assert ppo.uniform_floor_counts(
        one, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 1.0]])
    ) == (0, 0)


# -- the reader and the schedule -----------------------------------------------------------------


def test_the_reader_reads_both_schedules_and_absent_means_none() -> None:
    assert read_run_config(raw()).ppo.uniform_kl is None
    assert uniform().ppo.uniform_kl == UniformKlSettings(COEF, None)
    linear = uniform(schedule={"kind": "linear", "final_micro": 0, "generations": 4}).ppo.uniform_kl
    assert linear is not None
    assert linear.schedule is not None
    assert [linear.coef_micro_at(t) for t in (0, 1, 2, 4, 9)] == [COEF, 75_000, 50_000, 0, 0]
    assert UniformKlSettings(COEF, None).coef_micro_at(7) == COEF
    with pytest.raises(RunConfigError):
        UniformKlSettings(COEF, None).coef_micro_at(-1)
    assert uniform().hash != tiny_config().hash


def test_the_corpus_samples_read_or_are_refused() -> None:
    for good in ("valid/uniform-kl-constant.json", "valid/uniform-kl-linear.json"):
        assert read_run_config(raw(good)).ppo.uniform_kl is not None
    for bad in (
        "invalid/uniform-kl-zero.json",
        "invalid/uniform-kl-above-cap.json",
        "invalid/uniform-kl-float.json",
        "invalid/uniform-kl-no-schedule.json",
        "invalid/uniform-kl-extra-field.json",
    ):
        with pytest.raises(RunConfigError):
            read_run_config(raw(bad))


def test_the_value_warm_up_turns_the_term_off_and_offsets_the_index() -> None:
    config = warmed(2, {"kind": "linear", "final_micro": 0, "generations": 2})
    student = learner(config)
    # updates 0 and 1 are warm-up (off), then t = 0, 1, 2, ...
    assert [student.uniform_kl_coef_micro(u) for u in range(6)] == [0, 0, COEF, COEF // 2, 0, 0]
    assert learner().uniform_kl_coef_micro(3) is None


# -- the trainer ---------------------------------------------------------------------------------


def frozen_kl(student: Any, config: RunConfig) -> tuple[float, int, int]:
    """The independent reading: the mean term over the table's decisions with a term, how many
    there are, and the floor share (ppm), from a fresh forward pass of the untouched network."""
    steps = [s for t in trajs(config) if t.role == "battle" for s in t.steps]
    batch = collate([s.query for s in steps])
    student.networks["battle"].eval()
    with torch.no_grad():
        logits = student.networks["battle"](batch).logits
    rows, valid = ppo.uniform_kl(logits, batch.cand_mask, batch.pending)
    below, total = ppo.uniform_floor_counts(logits, batch.cand_mask, batch.pending)
    return (
        float(rows[valid].mean()),
        int(valid.sum()),
        below * 1_000_000 // total if total else 0,
    )


def one_pass(**fields: Any) -> RunConfig:
    # one epoch over one minibatch: the reported KL is the frozen weights' mean over the decisions
    return uniform(epochs=1, minibatch_size=4096, **fields)


def test_the_report_carries_the_mean_the_count_and_the_floor_of_the_frozen_weights() -> None:
    config = one_pass()
    student = learner(config)
    mean, count, floor = frozen_kl(student, config)
    report = student.train(trajs(config)).as_dict()
    battle = report["battle"]
    assert battle["uniform_kl"] == pytest.approx(mean, rel=1e-5)
    assert battle["uniform_kl_steps"] == count > 0
    assert battle["uniform_floor_ppm"] == floor
    assert battle["uniform_kl_coef_micro"] == COEF
    # a decision with no pending-free pair would not count: some decisions do have a term, and the
    # tiny stream's every third candidate is pending, so the count is below the step count
    assert count <= battle["steps"]
    assert not {
        "uniform_kl",
        "uniform_kl_coef_micro",
        "uniform_kl_steps",
        "uniform_floor_ppm",
    } & set(report["preview"])


def test_the_term_moves_the_battle_weights_and_leaves_the_preview_alone() -> None:
    plain = learner()
    plain.train(trajs(tiny_config()))
    config = uniform(coef=1_000_000)
    student = learner(config)
    student.train(trajs(config))
    a, b = pack_state(plain), pack_state(student)
    assert a.tensors != b.tensors
    for (name, x), (_, y) in zip(
        plain.networks["preview"].named_parameters(),
        student.networks["preview"].named_parameters(),
        strict=True,
    ):
        assert torch.equal(x, y), name
    assert any(
        not torch.equal(x, y)
        for (_, x), (_, y) in zip(
            plain.networks["battle"].named_parameters(),
            student.networks["battle"].named_parameters(),
            strict=True,
        )
    )


def test_absent_keeps_every_key_and_never_enters_the_term(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the zero-avoiding term ran")

    monkeypatch.setattr(trainer, "uniform_kl", refuse)
    monkeypatch.setattr(trainer, "uniform_floor_counts", refuse)
    student = learner()
    report = student.train(trajs(tiny_config())).as_dict()
    assert not any(key.startswith("uniform") for role in report.values() for key in role)
    assert not any(key.startswith("uniform") for key in json.loads(pack_state(student).sidecar))
    assert student.last_uniform_kl_coef_micro is None


def test_the_term_composes_with_the_anchor_and_the_entropy_bonus() -> None:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["ppo"]["anchor"] = {"kl_micro": 50_000, "schedule": {"kind": "constant"}}
        config["ppo"]["uniform_kl"] = {"coef_micro": COEF, "schedule": {"kind": "constant"}}
        config["ppo"]["entropy_schedule"] = {"kind": "linear", "final_micro": 0, "generations": 2}

    config = edited_config(edit)
    student = learner(config, other_anchors())
    battle = student.train(trajs(config)).as_dict()["battle"]
    assert {"anchor_kl", "anchor_beta_micro", "uniform_kl", "entropy_coef_micro"} <= set(battle)
    assert battle["uniform_kl"] > 0
    plain = anchored()
    only_anchor = learner(plain, other_anchors())
    only_anchor.train(trajs(plain))
    assert pack_state(student).tensors != pack_state(only_anchor).tensors


def test_a_value_warm_up_update_reports_the_term_off_and_trains_the_weights_of_none() -> None:
    config = warmed(1)
    student = learner(config)
    report = student.train(trajs(config)).as_dict()["battle"]
    assert (report["uniform_kl"], report["uniform_kl_coef_micro"]) == (0.0, 0)
    assert (report["uniform_kl_steps"], report["uniform_floor_ppm"]) == (0, 0)

    def edit(c: dict[str, Any]) -> None:
        c["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        c["ppo"]["value_warmup"] = {"generations": 1}

    bare_config = edited_config(edit)
    bare = learner(bare_config)
    bare.train(trajs(bare_config))
    assert pack_state(student).tensors == pack_state(bare).tensors
    # the next update is a policy update: the term is on
    second = student.train(trajs(config)).as_dict()["battle"]
    assert second["uniform_kl_coef_micro"] == COEF
    assert second["uniform_kl_steps"] > 0


# -- the sidecar and a resume --------------------------------------------------------------------


def test_the_sidecar_records_the_weight_for_a_linear_schedule_only() -> None:
    linear = uniform(schedule={"kind": "linear", "final_micro": 0, "generations": 2})
    student = learner(linear)
    used = []
    for _ in range(3):
        report = student.train(trajs(linear)).as_dict()["battle"]
        used.append(report["uniform_kl_coef_micro"])
        assert json.loads(pack_state(student).sidecar)["uniform_kl_coef_micro"] == used[-1]
    assert used == [COEF, COEF // 2, 0]
    constant = learner(uniform())
    constant.train(trajs(uniform()))
    assert "uniform_kl_coef_micro" not in json.loads(pack_state(constant).sidecar)


def test_a_resume_mid_schedule_reproduces_the_weight_and_the_state_hash() -> None:
    config = uniform(schedule={"kind": "linear", "final_micro": 0, "generations": 4})
    continuous = learner(config)
    continuous.train(trajs(config))
    continuous.train(trajs(config))
    halfway = pack_state(continuous)
    resumed = learner(config)
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.last_uniform_kl_coef_micro == continuous.last_uniform_kl_coef_micro
    assert pack_state(resumed) == halfway
    continuous.train(trajs(config))
    resumed.train(trajs(config))
    assert pack_state(resumed) == pack_state(continuous)


@pytest.mark.parametrize("linear_run", [True, False])
def test_a_sidecar_whose_weight_key_disagrees_with_the_schedule_is_refused(
    linear_run: bool,
) -> None:
    config = (
        uniform(schedule={"kind": "linear", "final_micro": 0, "generations": 4})
        if linear_run
        else uniform()
    )
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)

    def resealed(edit: Callable[[dict[str, Any]], None]) -> tuple[bytes, bytes, str]:
        sidecar = json.loads(packed.sidecar)
        edit(sidecar)
        return packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)

    def wrong(sidecar: dict[str, Any]) -> None:
        sidecar["uniform_kl_coef_micro"] = 999_999

    def dropped(sidecar: dict[str, Any]) -> None:
        sidecar.pop("uniform_kl_coef_micro", None)

    for edit in (wrong, dropped):
        if not linear_run and edit is dropped:
            continue  # unchanged state: nothing to refuse
        tensors, sidecar, digest = resealed(edit)
        with pytest.raises(LearnerStateError, match="uniform-KL weight"):
            restore_state(learner(config), tensors, sidecar, digest)
