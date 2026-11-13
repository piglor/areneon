"""``ppo.anchor.threatened`` (ADR-010 erratum A11.2, #1366): the reader rules, the two integer
betas, the per-decision anchor term against a hand computation, the report and sidecar keys, the
refusals and a resume. The Threatened flag itself is ``learner/situation.py``
(``test_situation.py``); here a step's flag is set by hand."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml.learner import parallel
from areneon_ml.learner.config import (
    AnchorBeta,
    AnchorSettings,
    LinearSchedule,
    RunConfig,
    RunConfigError,
)
from areneon_ml.learner.config import read_run_config as read_config
from areneon_ml.learner.data import Collected, LearnerDataError, Step, Trajectory
from areneon_ml.learner.ppo import anchor_kl
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state, state_hash
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.batch import collate

from tests.ml.learner import test_parallel
from tests.ml.learner.support import learner, tiny_config
from tests.ml.learner.test_anchor import KL, anchored, other_anchors
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs

HOT = 1_000_000
CONSTANT: dict[str, Any] = {"kind": "constant"}


def threatening(
    *,
    other: int = KL,
    other_schedule: dict[str, Any] | None = None,
    hot: int = HOT,
    hot_schedule: dict[str, Any] | None = None,
    warmup: int | None = None,
) -> RunConfig:
    """A reader-valid anchored config whose Learner settings also carry ``threatened``.

    The reader refuses ``threatened`` without a ``probe-damage/2`` feature set, and a network with
    that option needs rows the learner tests' streams lack, so the settings are set on the config
    after it was read (the reader rules have their own tests below)."""
    config = anchored(other_schedule, kl=max(other, 1), warmup=warmup)
    anchor = config.ppo.anchor
    assert anchor is not None
    linear = hot_schedule or CONSTANT
    schedule = (
        None
        if linear["kind"] == "constant"
        else LinearSchedule(hot, linear["final_micro"], linear["generations"])
    )
    settings = AnchorSettings(other, anchor.schedule, AnchorBeta(hot, schedule))
    return dataclasses.replace(config, ppo=dataclasses.replace(config.ppo, anchor=settings))


def flagged(work: list[Trajectory], every: int = 3) -> list[Trajectory]:
    """The Battle steps of ``work`` with every ``every``-th one Threatened and the rest not."""
    out = []
    count = 0
    for t in work:
        steps = []
        for step in t.steps:
            hot = t.role == "battle" and count % every == 0
            count += t.role == "battle"
            steps.append(dataclasses.replace(step, threatened=hot if t.role == "battle" else None))
        out.append(dataclasses.replace(t, steps=tuple(steps)))
    return out


# -- the reader -------------------------------------------------------------------------------


def features_config(feature_set: str | None, anchor: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(raw())
    data["warm_start"] = {"battle": "b" * 64}
    data["ppo"]["anchor"] = anchor
    if feature_set is not None:
        encoder = data["model"].setdefault("encoder", {})
        encoder["options"] = {
            "candidate_features": {"feature_set": feature_set, "feature_set_hash": "c" * 64}
        }
    return data


THREATENED = {"kl_micro": HOT, "schedule": CONSTANT}


def test_the_reader_takes_threatened_with_a_feature_set_that_carries_the_rows() -> None:
    for feature_set in ("probe-damage/2", "probe-outcome/1"):
        both = read_config(
            features_config(
                feature_set, {"kl_micro": KL, "schedule": CONSTANT, "threatened": THREATENED}
            )
        )
        assert both.ppo.anchor == AnchorSettings(KL, None, AnchorBeta(HOT, None))
    only = read_config(
        features_config(
            "probe-damage/2",
            {
                "kl_micro": 0,
                "schedule": CONSTANT,
                "threatened": {
                    "kl_micro": HOT,
                    "schedule": {"kind": "linear", "final_micro": 0, "generations": 4},
                },
            },
        )
    )
    assert only.ppo.anchor is not None
    assert only.ppo.anchor.kl_micro == 0
    assert only.ppo.anchor.threatened is not None
    assert only.ppo.anchor.threatened.schedule is not None
    assert only.ppo.anchor.threatened.beta_micro(2) == 500_000
    assert anchored().ppo.anchor == AnchorSettings(KL, None)  # no block: today's settings
    assert anchored().hash != only.hash


@pytest.mark.parametrize(
    ("feature_set", "anchor", "message"),
    [
        (None, {"kl_micro": KL, "schedule": CONSTANT, "threatened": THREATENED}, "probe-damage/2"),
        (
            "probe-damage/1",
            {"kl_micro": KL, "schedule": CONSTANT, "threatened": THREATENED},
            "probe-damage/2",
        ),
        (
            "probe-damage/3",
            {"kl_micro": KL, "schedule": CONSTANT, "threatened": THREATENED},
            "probe-damage/2",
        ),
        (
            "probe-damage/2",
            {"kl_micro": 0, "schedule": CONSTANT},
            r"kl_micro 0 means no anchor outside Threatened",
        ),
        (
            "probe-damage/2",
            {
                "kl_micro": KL,
                "schedule": CONSTANT,
                "threatened": {"kl_micro": 0, "schedule": CONSTANT},
            },
            None,
        ),
        (
            "probe-damage/2",
            {"kl_micro": KL, "schedule": CONSTANT, "threatened": {"kl_micro": 5}},
            None,
        ),
        (
            "probe-damage/2",
            {
                "kl_micro": KL,
                "schedule": CONSTANT,
                "threatened": {**THREATENED, "role": "battle"},
            },
            None,
        ),
    ],
)
def test_the_reader_refuses_what_the_profile_and_its_rules_say(
    feature_set: str | None, anchor: dict[str, Any], message: str | None
) -> None:
    with pytest.raises(RunConfigError, match=message):
        read_config(features_config(feature_set, anchor))


# -- the two betas ------------------------------------------------------------------------------


def test_each_beta_follows_its_own_integer_schedule_and_the_policy_update_index() -> None:
    config = threatening(
        other=0,
        hot_schedule={"kind": "linear", "final_micro": 0, "generations": 4},
        warmup=2,
    )
    student = learner(config)
    assert [student.anchor_beta_micro(u) for u in range(5)] == [0, 0, 0, 0, 0]
    # updates 0 and 1 are the value warm-up: 0; then 1000000, 750000, 500000, ... from update 2
    assert [student.anchor_threatened_beta_micro(u) for u in range(7)] == [
        0,
        0,
        1_000_000,
        750_000,
        500_000,
        250_000,
        0,
    ]
    both = learner(
        threatening(
            other_schedule={"kind": "linear", "final_micro": 0, "generations": 2},
            hot=300_000,
        )
    )
    assert [both.anchor_beta_micro(u) for u in range(3)] == [KL, KL // 2, 0]
    assert [both.anchor_threatened_beta_micro(u) for u in range(3)] == [300_000] * 3
    assert learner(anchored()).anchor_threatened_beta_micro(0) is None
    assert learner(tiny_config()).anchor_threatened_beta_micro(0) is None


# -- the per-decision anchor term -----------------------------------------------------------------


def minibatch(
    student: Learner, flags: list[bool]
) -> tuple[torch.Tensor, Any, torch.Tensor | None, torch.Tensor]:
    """One Battle minibatch through ``_minibatch_loss`` and the pieces of its hand computation."""
    work = [t for t in trajs(student.config) if t.role == "battle"]
    steps = [
        dataclasses.replace(s, threatened=f) for s, f in zip(work[0].steps, flags, strict=False)
    ]
    table = student._table("battle", [dataclasses.replace(work[0], steps=tuple(steps))], True)
    index = list(range(len(steps)))
    student.last_anchor_beta_micro = student.anchor_beta_micro(student.generation)
    student.last_anchor_threatened_beta_micro = student.anchor_threatened_beta_micro(
        student.generation
    )
    weights = student._weights()
    backward, loss, kl = student._minibatch_loss("battle", table, False, index, weights)
    batch = collate([table.steps[i].query for i in index])
    logits = student.networks["battle"](batch).logits
    dense = torch.zeros(batch.cand_mask.shape)
    for row in range(len(index)):
        dense[row, batch.cand_mask[row]] = table.anchor_log_probs[row]  # type: ignore[index]
    rows = anchor_kl(logits, batch.cand_mask, dense, None)
    return backward, loss, kl, rows


def test_the_anchor_term_weights_each_decision_by_its_own_beta_and_is_not_renormalised() -> None:
    config = threatening(other=200_000, hot=900_000)
    student = learner(config, other_anchors())
    flags = [True, False, False, True, False, False]
    backward, loss, kl, rows = minibatch(student, flags)
    betas = torch.tensor([0.9 if f else 0.2 for f in flags])
    expected = loss.total + (betas * rows).mean()
    assert torch.allclose(backward, expected, rtol=1e-5)
    assert kl is not None
    # anchor_kl stays the mean over all decisions
    assert float(kl.detach()) == pytest.approx(float(rows.mean()), rel=1e-5)
    # not a renormalised mean over the Threatened decisions
    assert not torch.allclose(backward, loss.total + 0.9 * rows[torch.tensor(flags)].mean())
    # no decision Threatened: the term is the plain anchor's, beta_other times the mean KL
    _, loss2, _, rows2 = minibatch(student, [False] * 6)
    plain = learner(anchored(kl=200_000), other_anchors())
    plain_backward, _, _, _ = minibatch(plain, [False] * 6)
    assert torch.allclose(plain_backward, loss2.total + 0.2 * rows2.mean(), rtol=1e-5)


def test_a_zero_other_beta_leaves_only_the_threatened_decisions_anchored() -> None:
    student = learner(threatening(other=0, hot=500_000), other_anchors())
    flags = [False, True, False, False, True, False]
    backward, loss, _, rows = minibatch(student, flags)
    hot = torch.tensor(flags)
    assert torch.allclose(backward, loss.total + (0.5 * rows * hot).mean(), rtol=1e-5)
    none, loss_none, _, _ = minibatch(student, [False] * 6)
    assert torch.allclose(none, loss_none.total, rtol=1e-6)  # no pull outside Threatened


# -- training: report, refusals, sidecar ----------------------------------------------------------


def test_the_report_counts_the_threatened_steps_and_their_kl_only_for_the_battle_role() -> None:
    config = threatening()
    student = learner(config, other_anchors())
    work = flagged(trajs(config))
    out = student.train(work).as_dict()
    expected = sum(1 for t in work if t.role == "battle" for s in t.steps if s.threatened)
    assert expected > 0
    assert out["battle"]["anchor_threatened_steps"] == expected
    assert out["battle"]["anchor_threatened_beta_micro"] == HOT
    assert out["battle"]["anchor_kl_threatened"] > 0
    assert out["battle"]["anchor_beta_micro"] == KL
    assert not any(k.startswith("anchor_threatened") for k in out["preview"])
    assert "anchor_kl_threatened" not in out["preview"]
    plain = learner(anchored(), other_anchors()).train(trajs(config)).as_dict()
    assert all("anchor_kl_threatened" not in r for r in plain.values())
    assert all("anchor_threatened_steps" not in r for r in plain.values())


def test_no_threatened_step_reports_a_zero_kl() -> None:
    config = threatening()
    student = learner(config, other_anchors())
    work = [
        dataclasses.replace(
            t,
            steps=tuple(
                dataclasses.replace(s, threatened=False if t.role == "battle" else None)
                for s in t.steps
            ),
        )
        for t in trajs(config)
    ]
    report = student.train(work).as_dict()["battle"]
    assert report["anchor_threatened_steps"] == 0
    assert report["anchor_kl_threatened"] == 0.0


def test_the_value_warmup_reports_beta_zero_and_no_threatened_kl() -> None:
    config = threatening(warmup=1)
    student = learner(config, other_anchors())
    report = student.train(flagged(trajs(config))).as_dict()["battle"]
    assert report["value_warmup"] is True
    assert report["anchor_threatened_beta_micro"] == 0
    assert report["anchor_kl_threatened"] == 0.0
    second = student.train(flagged(trajs(config))).as_dict()["battle"]
    assert second["anchor_threatened_beta_micro"] == HOT
    assert second["anchor_kl_threatened"] > 0


def test_a_battle_decision_without_the_rows_is_refused() -> None:
    config = threatening()
    student = learner(config, other_anchors())
    with pytest.raises(LearnerDataError, match="threats, speeds and foe_priorities"):
        student.train(trajs(config))  # the default stream carries no candidate_features
    # a Preview step with no flag is fine: it is never Threatened
    student.train(flagged(trajs(config)))


def test_a_threatened_beta_pulls_the_policy_toward_the_anchor_where_it_applies() -> None:
    """Held anchor on Threatened decisions only: the policy on them stays closer to the anchor
    than with no anchor at all."""

    def drift(hot: int) -> float:
        config = threatening(other=0, hot=hot)
        student = learner(config, other_anchors())
        report = student.train(flagged(trajs(config), every=1)).as_dict()["battle"]
        return float(report["anchor_kl_threatened"])

    assert drift(100_000_000) < drift(1)


def test_the_sidecar_records_the_threatened_beta_only_when_set() -> None:
    plain = learner(anchored())
    plain.train(trajs(anchored()))
    assert "anchor_threatened_beta_micro" not in json.loads(pack_state(plain).sidecar)
    config = threatening(hot_schedule={"kind": "linear", "final_micro": 0, "generations": 4})
    student = learner(config)
    assert "anchor_threatened_beta_micro" not in json.loads(pack_state(student).sidecar)
    student.train(flagged(trajs(config)))
    sidecar = json.loads(pack_state(student).sidecar)
    assert sidecar["anchor_threatened_beta_micro"] == HOT
    student.train(flagged(trajs(config)))
    assert json.loads(pack_state(student).sidecar)["anchor_threatened_beta_micro"] == 750_000


def test_a_sidecar_with_another_threatened_beta_is_refused() -> None:
    config = threatening(hot_schedule={"kind": "linear", "final_micro": 0, "generations": 4})
    student = learner(config)
    student.train(flagged(trajs(config)))
    packed = pack_state(student)
    for change in ("set", "drop"):
        sidecar = json.loads(packed.sidecar)
        if change == "set":
            sidecar["anchor_threatened_beta_micro"] = 1
        else:
            del sidecar["anchor_threatened_beta_micro"]
        with pytest.raises(LearnerStateError, match="Threatened anchor weight"):
            restore_state(
                learner(config), packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)
            )
    plain = learner(anchored())
    plain.train(trajs(anchored()))
    stored = pack_state(plain)
    sidecar = json.loads(stored.sidecar)
    sidecar["anchor_threatened_beta_micro"] = 5
    with pytest.raises(LearnerStateError, match="Threatened anchor weight"):
        restore_state(
            learner(anchored()), stored.tensors, jcs(sidecar), state_hash(stored.tensors, sidecar)
        )


@pytest.mark.parametrize("stop_after", [1, 2, 3])
def test_a_resume_reaches_the_same_state_hash(stop_after: int) -> None:
    config = threatening(
        other=0, hot_schedule={"kind": "linear", "final_micro": 0, "generations": 3}, warmup=1
    )
    continuous = learner(config, other_anchors())
    for _ in range(stop_after):
        continuous.train(flagged(trajs(config)))
    halfway = pack_state(continuous)
    resumed = learner(config, other_anchors())
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.last_anchor_threatened_beta_micro == continuous.last_anchor_threatened_beta_micro
    assert pack_state(resumed) == halfway
    for _ in range(4 - stop_after):
        continuous.train(flagged(trajs(config)))
        resumed.train(flagged(trajs(config)))
    assert pack_state(resumed) == pack_state(continuous)


def test_the_flag_travels_between_loader_processes_and_none_stays_none() -> None:
    piece = test_parallel._one_piece()
    key, items = next(iter(piece.collected.steps.items()))
    shaped = [
        dataclasses.replace(items[0], threatened=True),
        dataclasses.replace(items[0], threatened=False),
        dataclasses.replace(items[0], threatened=None),
    ]
    sent = parallel._Piece(
        Collected(list(piece.collected.order), dict(piece.collected.scores), {key: shaped}),
        set(),
        set(),
    )
    got = parallel.unpack_piece(parallel.pack_piece(sent))
    assert [s.threatened for s in got.collected.steps[key]] == [True, False, None]
    assert isinstance(got.collected.steps[key][0], Step)
