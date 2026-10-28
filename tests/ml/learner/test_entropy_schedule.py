"""The optional entropy-coefficient schedule (ADR-023 section 3, #1048): the integer rule, the
weight the loss uses, the sidecar and report keys, the value warm-up offset and a resume."""

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
        config["ppo"]["entropy_schedule"] = {
            "kind": "linear",
            "final_micro": final,
            "generations": generations,
        }

    return edit


def start_micro() -> int:
    return int(tiny_config().raw["ppo"]["entropy_coef_micro"])


def test_the_rule_at_the_edges_falling_rising_and_to_zero() -> None:
    e0 = start_micro()
    falling = edited_config(linear(0, 4)).ppo.entropy_schedule
    assert falling == LinearSchedule(e0, 0, 4)
    assert [falling.rate_micro(t) for t in (0, 2, 4, 9)] == [e0, e0 - (e0 * 2) // 4, 0, 0]
    rising = edited_config(linear(e0 + 100, 3)).ppo.entropy_schedule
    assert rising is not None
    # +100 / 3: 33 and 66 (truncated toward zero), exactly the final from update 3 on
    assert [rising.rate_micro(t) for t in (0, 1, 2, 3, 8)] == [
        e0,
        e0 + 33,
        e0 + 66,
        e0 + 100,
        e0 + 100,
    ]
    odd = edited_config(linear(e0 - 1 - 3, 4)).ppo.entropy_schedule
    assert odd is not None
    assert odd.rate_micro(1) == e0 - 1  # -4 / 4 exactly; -3 would truncate toward zero
    with pytest.raises(RunConfigError):
        rising.rate_micro(-1)


def test_absent_and_constant_mean_no_schedule_and_the_hash_still_differs() -> None:
    assert read_run_config(raw()).ppo.entropy_schedule is None
    constant = raw("valid/entropy-constant.json")
    assert read_run_config(constant).ppo.entropy_schedule is None
    assert read_run_config(constant).hash != read_run_config(raw()).hash
    assert read_run_config(raw("valid/entropy-linear.json")).ppo.entropy_schedule is not None
    for bad in ("invalid/entropy-zero-generations.json", "invalid/entropy-unknown-kind.json"):
        with pytest.raises(RunConfigError):
            read_run_config(raw(bad))


def test_each_update_uses_its_weight_and_the_report_and_sidecar_record_it() -> None:
    e0 = start_micro()
    config = edited_config(linear(0, 2))
    student = learner(config)
    used = []
    for _ in range(3):
        report = student.train(trajs(config)).as_dict()
        assert {r["entropy_coef_micro"] for r in report.values()} == {
            student.last_entropy_coef_micro
        }
        used.append(student.last_entropy_coef_micro)
        assert json.loads(pack_state(student).sidecar)["entropy_coef_micro"] == used[-1]
    assert used == [e0, e0 - e0 // 2, 0]


def test_without_a_linear_schedule_nothing_is_recorded_and_the_weights_are_todays() -> None:
    plain = learner()
    plain.train(trajs(tiny_config()))
    constant_config = edited_config(
        lambda c: c["ppo"].update(entropy_schedule={"kind": "constant"})
    )
    constant = learner(constant_config)
    report = constant.train(trajs(constant_config)).as_dict()
    assert all("entropy_coef_micro" not in r for r in report.values())
    assert "entropy_coef_micro" not in json.loads(pack_state(constant).sidecar)
    assert pack_state(plain).tensors == pack_state(constant).tensors
    assert "entropy_coef_micro" not in json.loads(pack_state(plain).sidecar)


def test_a_schedule_that_stays_at_the_start_trains_the_weights_of_no_schedule() -> None:
    plain = learner()
    plain.train(trajs(tiny_config()))
    flat_config = edited_config(linear(start_micro(), 3))
    flat = learner(flat_config)
    flat.train(trajs(flat_config))
    assert pack_state(plain).tensors == pack_state(flat).tensors


def test_the_value_warm_up_offsets_the_index() -> None:
    e0 = start_micro()

    def edit(c: dict[str, Any]) -> None:
        linear(0, 2)(c)
        c["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        c["ppo"]["value_warmup"] = {"generations": 2}

    config = edited_config(edit)
    student = learner(config)
    # updates 0 and 1 are warm-up (index 0), update 2 is the first policy update (index 0)
    assert [student.entropy_coef_micro(u) for u in range(6)] == [e0, e0, e0, e0 - e0 // 2, 0, 0]


def test_a_resume_mid_schedule_reproduces_the_weight_and_the_state_hash() -> None:
    config = edited_config(linear(0, 4))
    continuous = learner(config)
    continuous.train(trajs(config))
    continuous.train(trajs(config))
    halfway = pack_state(continuous)
    resumed = learner(config)
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.last_entropy_coef_micro == continuous.last_entropy_coef_micro
    assert pack_state(resumed) == halfway
    continuous.train(trajs(config))
    resumed.train(trajs(config))
    assert pack_state(resumed) == pack_state(continuous)


@pytest.mark.parametrize("linear_run", [True, False])
def test_a_sidecar_whose_weight_key_disagrees_with_the_schedule_is_refused(
    linear_run: bool,
) -> None:
    config = edited_config(linear(0, 4)) if linear_run else tiny_config()
    student = learner(config)
    student.train(trajs(config))
    packed = pack_state(student)

    def resealed(edit: Callable[[dict[str, Any]], None]) -> tuple[bytes, bytes, str]:
        sidecar = json.loads(packed.sidecar)
        edit(sidecar)
        return packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)

    def wrong(sidecar: dict[str, Any]) -> None:
        sidecar["entropy_coef_micro"] = 999_999

    def dropped(sidecar: dict[str, Any]) -> None:
        sidecar.pop("entropy_coef_micro", None)

    for edit in (wrong, dropped):
        if not linear_run and edit is dropped:
            continue  # unchanged state: nothing to refuse
        tensors, sidecar, digest = resealed(edit)
        with pytest.raises(LearnerStateError, match="entropy weight"):
            restore_state(learner(config), tensors, sidecar, digest)
