"""The operational learner micro-batch (``--learner-micro-batch``): a minibatch is run in chunks,
each backpropagated at once and scaled by its share. The default (None) is the untouched single
pass; with N the gradients and the report means equal the single pass's up to float rounding, for
every learner option; a chunk larger than the batch is the single pass exactly."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import Trajectory
from areneon_ml.learner.trainer import Learner, LearnerError
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import edited_config, networks, tiny_config
from tests.ml.learner.test_anchor import anchored, other_anchors
from tests.ml.learner.test_anchor_threatened import flagged, threatening
from tests.ml.learner.test_aux_moves import aux_config, aux_trajectories
from tests.ml.learner.test_aux_opponent import opp_config, opp_trajectories
from tests.ml.learner.test_imitation import imitating, teacher_steps
from tests.ml.learner.test_search_distill import battle_labels, distilled
from tests.ml.learner.test_shared_trunk import shared_config
from tests.ml.learner.test_trainer import trajs
from tests.ml.learner.test_uniform_kl import uniform

CPU = torch.device("cpu")
MICRO = 5  # does not divide the minibatch of 16, nor the 55 Battle steps


@dataclasses.dataclass
class Case:
    config: RunConfig
    anchors: dict[str, Any] | None = None
    work: list[Trajectory] | None = None
    labels: Any = None
    teacher: Any = None
    generation: int = 0


def plain() -> Case:
    return Case(tiny_config())


def anchor() -> Case:
    return Case(anchored(), other_anchors())


def anchor_warmup() -> Case:
    return Case(anchored(warmup=1), other_anchors())


def shared() -> Case:
    return Case(shared_config())


def aux() -> Case:
    config = aux_config()
    return Case(config, work=aux_trajectories(config))


def aux_opponent() -> Case:
    config = opp_config()
    return Case(config, work=opp_trajectories(config))


def threatened() -> Case:
    config = threatening(other=200_000, hot=900_000)
    return Case(config, other_anchors(), work=flagged(trajs(config)))


def imitation() -> Case:
    return Case(imitating(coef=1_000_000, share=1_000_000))


def distill() -> Case:
    return Case(distilled(coef_micro=1_000_000), labels=battle_labels(), generation=1)


def shared_imitation() -> Case:
    return Case(imitating(coef=1_000_000, share=1_000_000, shared=True))


def uniform_term() -> Case:
    return Case(uniform(coef=1_000_000))


def uniform_anchor() -> Case:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["ppo"]["anchor"] = {"kl_micro": 50_000, "schedule": {"kind": "constant"}}
        config["ppo"]["uniform_kl"] = {"coef_micro": 1_000_000, "schedule": {"kind": "constant"}}

    return Case(edited_config(edit), other_anchors())


def uniform_shared() -> Case:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"shared_trunk": True}
        config["ppo"]["uniform_kl"] = {"coef_micro": 1_000_000, "schedule": {"kind": "constant"}}

    return Case(edited_config(edit))


def uniform_aux_opponent() -> Case:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"aux_opponent_heads": True}
        config["ppo"]["aux_opponent"] = {"action_micro": 50_000, "item_micro": 25_000}
        config["ppo"]["uniform_kl"] = {"coef_micro": 1_000_000, "schedule": {"kind": "constant"}}

    config = edited_config(edit)
    return Case(config, work=opp_trajectories(config))


CASES: dict[str, Callable[[], Case]] = {
    "plain": plain,
    "anchor": anchor,
    "anchor-warmup": anchor_warmup,
    "shared-trunk": shared,
    "aux-moves": aux,
    "aux-opponent": aux_opponent,
    "anchor-threatened": threatened,
    "imitation": imitation,
    "search-distill": distill,
    "shared-trunk-imitation": shared_imitation,
    "uniform-kl": uniform_term,
    "uniform-kl-anchor": uniform_anchor,
    "uniform-kl-shared-trunk": uniform_shared,
    "uniform-kl-aux-opponent": uniform_aux_opponent,
}


def run(
    case: Case, micro: int | None, tmp_path: Path
) -> tuple[Any, dict[str, Any], list[list[torch.Tensor]]]:
    """One update; the report and the gradients each optimiser step saw (before clipping)."""
    student = Learner(
        case.config,
        networks(case.config),
        CPU,
        None if case.anchors is None else dict(case.anchors),
        micro_batch=micro,
    )
    student.generation = case.generation
    teacher = (
        teacher_steps(case.config, tmp_path) if case.config.ppo.imitation is not None else None
    )
    grads: list[list[torch.Tensor]] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters: Any, max_norm: float, *args: Any, **kwargs: Any) -> Any:
        params = list(parameters)
        grads.append([p.grad.detach().clone() for p in params if p.grad is not None])
        return original(params, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = spy
    try:
        work = case.work if case.work is not None else trajs(case.config)
        report = student.train(work, case.labels, teacher).as_dict()
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    return student, report, grads


def close(a: Any, b: Any, where: str = "") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for key in a:
            close(a[key], b[key], f"{where}/{key}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            close(x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=5e-2, abs_tol=2e-3), (where, a, b)
    else:
        assert a == b, where


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_chunked_gradients_and_report_equal_the_single_pass(name: str, tmp_path: Path) -> None:
    case = CASES[name]()
    _, single, g_single = run(case, None, tmp_path)
    _, chunked, g_chunked = run(case, MICRO, tmp_path)
    # the first optimiser step sees the same parameters and the same minibatch in both runs
    assert len(g_single) == len(g_chunked)
    for one, many in zip(g_single[0], g_chunked[0], strict=True):
        assert torch.allclose(one, many, rtol=1e-3, atol=1e-5)
    # same keys, integer values equal, float means equal within tolerance
    close(single, chunked)


def test_a_chunk_larger_than_the_batch_is_the_single_pass_exactly(tmp_path: Path) -> None:
    case = anchor()
    one, report_one, _ = run(case, None, tmp_path)
    big, report_big, _ = run(case, 10_000, tmp_path)
    assert report_one == report_big
    for role in ROLES:
        for (n, a), (_, b) in zip(
            one.networks[role].named_parameters(),
            big.networks[role].named_parameters(),
            strict=True,
        ):
            assert torch.equal(a, b), n


def test_the_default_never_enters_the_chunked_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the chunked path ran")

    monkeypatch.setattr(Learner, "_accumulate", refuse)
    for factory in (plain, shared):
        run(factory(), None, tmp_path)
    with pytest.raises(AssertionError, match="chunked path"):
        run(plain(), MICRO, tmp_path)


def test_the_default_is_deterministic_and_equals_an_explicit_none(tmp_path: Path) -> None:
    case = plain()
    a, report_a, _ = run(case, None, tmp_path)
    plain_learner = Learner(case.config, networks(case.config), CPU)
    assert plain_learner.micro_batch is None
    report_b = plain_learner.train(trajs(case.config)).as_dict()
    assert report_a == report_b
    for role in ROLES:
        for (n, x), (_, y) in zip(
            a.networks[role].named_parameters(),
            plain_learner.networks[role].named_parameters(),
            strict=True,
        ):
            assert torch.equal(x, y), n


def test_the_micro_batch_must_be_positive() -> None:
    with pytest.raises(LearnerError, match="micro-batch"):
        Learner(tiny_config(), networks(tiny_config()), CPU, micro_batch=0)


def test_the_no_gradient_passes_are_chunked_too() -> None:
    student = Learner(tiny_config(), networks(tiny_config()), CPU, micro_batch=4)
    assert student._pass_size() == 4
    assert Learner(tiny_config(), networks(tiny_config()), CPU)._pass_size() == 16
    assert Learner(tiny_config(), networks(tiny_config()), CPU, micro_batch=99)._pass_size() == 16
    steps = [s for t in trajs(student.config) if t.role == "battle" for s in t.steps][:10]
    chunked = student._score("battle", steps)
    whole = Learner(tiny_config(), networks(tiny_config()), CPU)._score("battle", steps)
    assert torch.allclose(chunked[0], whole[0], atol=1e-5)
    assert torch.allclose(chunked[1], whole[1], atol=1e-5)


def test_a_value_warmup_update_is_chunked(tmp_path: Path) -> None:
    case = Case(edited_config(_warm))
    _, single, g1 = run(case, None, tmp_path)
    _, chunked, g2 = run(case, MICRO, tmp_path)
    for one, many in zip(g1[0], g2[0], strict=True):
        assert torch.allclose(one, many, rtol=1e-3, atol=1e-5)
    close(single, chunked)


def _warm(config: dict[str, Any]) -> None:
    config["warm_start"] = {"battle": "b" * 64}
    config["ppo"]["value_warmup"] = {"generations": 1}
