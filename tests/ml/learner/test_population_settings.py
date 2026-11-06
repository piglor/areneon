"""The population run's effective PPO settings in the learner-state sidecar (ADR-021 section 8)."""

from __future__ import annotations

import json

import pytest
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state

from tests.ml.learner.support import learner

SETTINGS = {"learning_rate_micro": 1000, "entropy_coef_micro": 10_000, "epochs": 2}


def test_a_single_lineage_sidecar_has_no_settings_key() -> None:
    packed = pack_state(learner())
    assert "population_settings" not in json.loads(packed.sidecar)


def test_a_population_sidecar_names_the_settings_and_restores_only_under_them() -> None:
    member = learner()
    member.population_settings = dict(SETTINGS)
    packed = pack_state(member)
    assert json.loads(packed.sidecar)["population_settings"] == SETTINGS
    # The settings enter the state hash: the same learner without them hashes differently.
    member.population_settings = None
    assert pack_state(member).hash != packed.hash
    same = learner()
    same.population_settings = dict(SETTINGS)
    restore_state(same, packed.tensors, packed.sidecar, packed.hash)
    assert same.generation == 0


@pytest.mark.parametrize(
    "held", [None, {**SETTINGS, "epochs": 3}, {"learning_rate_micro": 1000}], ids=str
)
def test_a_sidecar_whose_settings_differ_from_the_learners_is_refused(
    held: dict[str, int] | None,
) -> None:
    member = learner()
    member.population_settings = dict(SETTINGS)
    packed = pack_state(member)
    other = learner()
    other.population_settings = held
    with pytest.raises(LearnerStateError, match="PPO settings"):
        restore_state(other, packed.tensors, packed.sidecar, packed.hash)


def test_a_single_lineage_sidecar_is_refused_by_a_population_learner() -> None:
    packed = pack_state(learner())
    member = learner()
    member.population_settings = dict(SETTINGS)
    with pytest.raises(LearnerStateError, match="PPO settings"):
        restore_state(member, packed.tensors, packed.sidecar, packed.hash)
