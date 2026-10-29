"""``generation.play.temperature_ppm`` in the learner (ADR-023 section 1, #1046): every policy
term is computed on the tempered policy ``softmax(z / tau)`` over the real candidates, against
values worked out by hand (R4), and an untempered config keeps its values and report keys."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.ppo import (
    LossWeights,
    Targets,
    anchor_kl,
    policy_terms,
    ppo_loss,
    temper,
)
from areneon_ml.model.batch import collate

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_anchor import other_anchors
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs

TAU = 500_000  # temperature 0.5: the logits are doubled
WEIGHTS = LossWeights(clip=0.2, value_coef=0.0, entropy_coef=0.0)


def tempered_config(tau: int = TAU, **ppo: Any) -> Any:
    def edit(config: dict[str, Any]) -> None:
        config["generation"]["play"] = {"temperature_ppm": tau}
        config["ppo"].update(ppo)

    return edited_config(edit)


def softmax(values: list[float]) -> list[float]:
    z = sum(math.exp(v) for v in values)
    return [math.exp(v) / z for v in values]


# -- the config ---------------------------------------------------------------


def test_the_reader_reads_the_temperature_and_absent_is_none() -> None:
    assert tiny_config().generation.temperature_ppm is None
    assert tempered_config().generation.temperature_ppm == TAU
    assert tempered_config().hash != tiny_config().hash


def test_the_reader_refuses_the_explicit_default_and_the_schema_the_rest() -> None:
    for bad in (1_000_000, 1_000_001, 0, -5):
        document = raw()
        document["generation"]["play"] = {"temperature_ppm": bad}
        with pytest.raises(RunConfigError):
            read_run_config(document)
    for bad_shape in ({}, {"temperature_ppm": 5, "truncation": 1}, {"temperature_ppm": 0.5}):
        document = raw()
        document["generation"]["play"] = bad_shape
        with pytest.raises(RunConfigError):
            read_run_config(document)


# -- R4: hand-built three-candidate example -------------------------------------


def test_the_tempered_terms_equal_hand_values_and_differ_from_temperature_one() -> None:
    # Old policy logits (0, 1, 2); new (0, 2, 1); chosen candidate 1; tau 0.5 doubles the logits.
    old_logits, new_logits = [0.0, 1.0, 2.0], [0.0, 2.0, 1.0]
    chosen = torch.tensor([1])
    mask = torch.ones(1, 3, dtype=torch.bool)
    old = torch.tensor([old_logits])
    new = torch.tensor([new_logits])

    def hand(scale: float) -> tuple[float, float, float, float]:
        p_old = softmax([scale * x for x in old_logits])
        p_new = softmax([scale * x for x in new_logits])
        log_old, log_new = math.log(p_old[1]), math.log(p_new[1])
        entropy = -sum(p * math.log(p) for p in p_new)
        return log_old, log_new, entropy, (math.exp(log_new - log_old) - 1) - (log_new - log_old)

    def terms(tau: int | None) -> tuple[float, float, float, float]:
        before = policy_terms(old, mask, chosen, tau)
        after = policy_terms(new, mask, chosen, tau)
        loss = ppo_loss(
            after,
            torch.zeros(1),
            Targets(before.log_prob, torch.ones(1), torch.zeros(1)),
            WEIGHTS,
        )
        return (
            float(before.log_prob),
            float(after.log_prob),
            float(after.entropy),
            float(loss.approx_kl),
        )

    tempered, plain = terms(TAU), terms(None)
    assert tempered == pytest.approx(hand(2.0), abs=1e-5)
    assert plain == pytest.approx(hand(1.0), abs=1e-5)
    # negative control: the tempered values are not the temperature-1 ones
    for t_value, p_value in zip(tempered, plain, strict=True):
        assert abs(t_value - p_value) > 1e-3
    # worked numbers at tau 0.5: old p = softmax(0, 2, 4), new p = softmax(0, 4, 2)
    assert tempered[0] == pytest.approx(math.log(math.exp(2) / (1 + math.exp(2) + math.exp(4))))
    assert tempered[1] == pytest.approx(math.log(math.exp(4) / (1 + math.exp(4) + math.exp(2))))


def test_a_padded_candidate_never_enters_the_tempered_policy() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 99.0]])
    mask = torch.tensor([[True, True, True, False]])
    got = policy_terms(logits, mask, torch.tensor([2]), TAU)
    p = softmax([0.0, 2.0, 4.0])
    assert float(got.log_prob) == pytest.approx(math.log(p[2]), abs=1e-5)
    assert float(got.entropy) == pytest.approx(-sum(x * math.log(x) for x in p), abs=1e-5)
    # a padded logit at the dtype minimum survives the scale (it overflows to -inf, then is masked)
    floor = torch.finfo(torch.float32).min
    padded = torch.tensor([[0.0, 1.0, 2.0, floor]], requires_grad=True)
    out = policy_terms(padded, mask, torch.tensor([2]), 250_000)
    torch.autograd.backward(out.log_prob.sum())
    assert padded.grad is not None
    assert torch.isfinite(padded.grad).all()


def test_the_scale_is_one_float64_division_and_none_is_the_identity() -> None:
    logits = torch.tensor([[0.25, -1.5, 3.0]])
    assert temper(logits, None) is logits
    assert torch.equal(temper(logits, TAU), logits * 2.0)
    assert torch.equal(temper(logits, 250_000), logits * 4.0)


def test_the_anchor_kl_uses_the_temperature_on_both_policies() -> None:
    # pi_theta = softmax(2 z); the anchor's log-probabilities are given at the same tau.
    z, anchor_z = [1.0, 2.0, 0.5], [0.0, 1.0, 1.0]
    mask = torch.ones(1, 3, dtype=torch.bool)
    q = softmax([2 * x for x in anchor_z])
    p = softmax([2 * x for x in z])
    hand = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q, strict=True))
    anchor_log = torch.tensor([[math.log(x) for x in q]])
    got = anchor_kl(torch.tensor([z]), mask, anchor_log, TAU)
    assert float(got[0]) == pytest.approx(hand, rel=1e-5)
    untempered = anchor_kl(torch.tensor([z]), mask, anchor_log)
    assert abs(float(untempered[0]) - hand) > 1e-3  # negative control


# -- the trainer ---------------------------------------------------------------


def test_the_trainer_scores_the_old_log_probabilities_at_the_temperature() -> None:
    config = tempered_config()
    work = trajs(config)
    steps = [s for t in work if t.role == "battle" for s in t.steps][:6]
    student, plain = learner(config), learner(tiny_config())
    got, _ = student._score("battle", steps)
    base, _ = plain._score("battle", steps)
    net = student.networks["battle"]
    net.eval()
    with torch.no_grad():
        batch = collate([s.query for s in steps])
        logits = net(batch).logits * 2.0
    expected = torch.log_softmax(logits.masked_fill(~batch.cand_mask, -1e30), dim=-1)
    chosen = torch.tensor([s.chosen for s in steps])
    assert torch.allclose(got, expected.gather(1, chosen[:, None]).squeeze(1), atol=1e-5)
    assert not torch.allclose(got, base, atol=1e-4)  # same weights, another policy


def test_the_anchor_is_scored_at_the_temperature() -> None:
    config = edited_config(_anchored_and_tempered)
    anchors = other_anchors()
    student = learner(config, anchors)
    steps = [s for t in trajs(config) if t.role == "battle" for s in t.steps][:5]
    got = student._anchor_scores("battle", steps)
    net = anchors["battle"]
    net.eval()
    with torch.no_grad():
        batch = collate([s.query for s in steps])
        logits = net(batch).logits * 2.0
    for row, scores in enumerate(got):
        mask = batch.cand_mask[row]
        expected = torch.log_softmax(logits[row][mask], dim=-1)
        assert torch.allclose(scores, expected, atol=1e-5)


def _anchored_and_tempered(config: dict[str, Any]) -> None:
    config["generation"]["play"] = {"temperature_ppm": TAU}
    config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
    config["ppo"]["anchor"] = {"kl_micro": 100_000, "schedule": {"kind": "constant"}}


def test_the_report_states_the_temperature_per_role_only_when_set() -> None:
    plain = learner(tiny_config()).train(trajs(tiny_config())).as_dict()
    assert all("temperature_ppm" not in v for v in plain.values())
    config = tempered_config()
    report = learner(config).train(trajs(config)).as_dict()
    assert {role: v["temperature_ppm"] for role, v in report.items()} == {
        "battle": TAU,
        "preview": TAU,
    }
    # the same data and weights give another approximate KL on the tempered policy
    assert report["battle"]["approx_kl"] != plain["battle"]["approx_kl"]


def test_a_tempered_anchored_update_trains_and_reports_the_kl() -> None:
    config = edited_config(_anchored_and_tempered)
    report = learner(config, other_anchors()).train(trajs(config)).as_dict()
    assert report["battle"]["temperature_ppm"] == TAU
    assert report["battle"]["anchor_kl"] > 0
