"""The optional learning-rate schedule (ADR-010 erratum A1, item 2): the integer rule, the
optimiser's rate, the sidecar key and a resume in the middle of a schedule."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.learner.config import LinearSchedule, RunConfigError, read_run_config
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state, state_hash

from tests.ml.learner.support import edited_config, learner, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs


def linear(final: int, generations: int) -> Callable[[dict[str, Any]], None]:
    def edit(config: dict[str, Any]) -> None:
        config["ppo"]["learning_rate_schedule"] = {
            "kind": "linear",
            "final_micro": final,
            "generations": generations,
        }

    return edit


# (final_micro, generations, {t: rate}) from learning_rate_micro 1000
TABLE = [
    # decreasing: 1000 -> 100 over 4 (exact quarters of -900)
    (100, 4, {0: 1000, 1: 775, 3: 325, 4: 100, 9: 100}),
    # decreasing, truncation toward zero: -899 / 4 = -224.75 -> -224 (a floor would give -225)
    (101, 4, {0: 1000, 1: 776, 3: 326, 4: 101, 9: 101}),
    # increasing, truncated: +100 / 3 = 33.33 and 66.67
    (1100, 3, {0: 1000, 1: 1033, 2: 1066, 3: 1100, 8: 1100}),
    # one generation: the final rate from update 1
    (500, 1, {0: 1000, 1: 500, 2: 500}),
    # no change
    (1000, 5, {0: 1000, 2: 1000, 5: 1000}),
]


@pytest.mark.parametrize(("final", "generations", "expected"), TABLE)
def test_the_integer_rule_at_the_edges(
    final: int, generations: int, expected: dict[int, int]
) -> None:
    config = edited_config(linear(final, generations))
    schedule = config.ppo.schedule
    assert schedule is not None
    assert {t: schedule.rate_micro(t) for t in expected} == expected
    # the first, the one before the end, the end and five past it
    assert schedule.rate_micro(generations) == final
    assert schedule.rate_micro(generations + 5) == final
    assert schedule.rate(0) == 0.001


def test_a_negative_update_is_refused_and_the_float_is_one_division() -> None:
    schedule = LinearSchedule(1000, 100, 4)
    with pytest.raises(RunConfigError):
        schedule.rate_micro(-1)
    assert schedule.rate(1) == 775 / 1_000_000


def test_absent_and_constant_mean_no_schedule_and_the_hash_still_differs() -> None:
    assert read_run_config(raw()).ppo.schedule is None
    constant = raw("valid/constant-schedule.json")
    assert read_run_config(constant).ppo.schedule is None
    assert read_run_config(constant).hash != read_run_config(raw()).hash
    config = read_run_config(raw("valid/linear-schedule.json"))
    assert config.ppo.schedule == LinearSchedule(1000, 100, 4)
    for bad in ("invalid/schedule-zero-generations.json", "invalid/schedule-unknown-kind.json"):
        with pytest.raises(RunConfigError):
            read_run_config(raw(bad))


def test_each_update_sets_both_optimisers_to_its_rate() -> None:
    config = edited_config(linear(100, 4))
    student = learner(config)
    rates = []
    for _ in range(3):
        student.train(trajs(config))
        rates.append({g["lr"] for opt in student.optimisers.values() for g in opt.param_groups})
    assert rates == [{0.001}, {775 / 1e6}, {550 / 1e6}]
    assert student.last_rate_micro == 550


def test_without_a_schedule_the_rate_is_untouched_and_the_sidecar_has_no_rate_key() -> None:
    for config in (
        tiny_config(),
        edited_config(lambda c: c["ppo"].update(learning_rate_schedule={"kind": "constant"})),
    ):
        student = learner(config)
        student.train(trajs(config))
        assert {g["lr"] for g in student.optimisers["battle"].param_groups} == {0.001}
        assert "learning_rate_micro" not in json.loads(pack_state(student).sidecar)


def test_a_constant_schedule_trains_the_weights_of_no_schedule() -> None:
    plain = learner()
    plain.train(trajs(tiny_config()))
    constant_config = edited_config(
        lambda c: c["ppo"].update(learning_rate_schedule={"kind": "constant"})
    )
    constant = learner(constant_config)
    constant.train(trajs(constant_config))
    assert pack_state(plain).tensors == pack_state(constant).tensors


def test_the_sidecar_records_the_rate_of_the_update_just_done_only_for_linear() -> None:
    config = edited_config(linear(100, 4))
    student = learner(config)
    assert "learning_rate_micro" not in json.loads(pack_state(student).sidecar)
    student.train(trajs(config))
    assert json.loads(pack_state(student).sidecar)["learning_rate_micro"] == 1000
    student.train(trajs(config))
    assert json.loads(pack_state(student).sidecar)["learning_rate_micro"] == 775


def test_a_resume_mid_schedule_reproduces_the_rate_and_the_state_hash() -> None:
    config = edited_config(linear(100, 4))
    continuous = learner(config)
    continuous.train(trajs(config))
    continuous.train(trajs(config))
    halfway = pack_state(continuous)
    resumed = learner(config)
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.generation == 2
    assert resumed.last_rate_micro == 775
    assert pack_state(resumed) == halfway
    continuous.train(trajs(config))
    resumed.train(trajs(config))
    assert {g["lr"] for g in resumed.optimisers["preview"].param_groups} == {550 / 1e6}
    assert pack_state(resumed) == pack_state(continuous)
    assert json.loads(pack_state(resumed).sidecar)["learning_rate_micro"] == 550


def _resealed(packed: Any, edit: Callable[[dict[str, Any]], None]) -> tuple[bytes, bytes, str]:
    sidecar = json.loads(packed.sidecar)
    edit(sidecar)
    return packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)


@pytest.mark.parametrize("linear_run", [True, False])
def test_a_sidecar_whose_rate_key_disagrees_with_the_schedule_is_refused(linear_run: bool) -> None:
    config = edited_config(linear(100, 4)) if linear_run else tiny_config()
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)

    def wrong(sidecar: dict[str, Any]) -> None:
        sidecar["learning_rate_micro"] = 999

    def dropped(sidecar: dict[str, Any]) -> None:
        sidecar.pop("learning_rate_micro", None)

    for edit in (wrong, dropped):
        tensors, sidecar, digest = _resealed(packed, edit)
        if not linear_run and edit is dropped:
            continue  # unchanged state: nothing to refuse
        with pytest.raises(LearnerStateError, match="learning rate"):
            restore_state(learner(config), tensors, sidecar, digest)
