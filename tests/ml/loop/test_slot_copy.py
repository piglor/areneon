"""The copy of one learner onto another (ADR-021 section 6, slice 8)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from areneon_ml.loop.slot_copy import SlotCopyError, copy_slot

from tests.ml.learner.support import learner, tiny_config


def test_a_slot_cannot_be_copied_onto_itself() -> None:
    one = learner()
    with pytest.raises(SlotCopyError, match="itself"):
        copy_slot(one, one, ppo=one.config.ppo, seed=1, settings={})


def test_a_donor_without_moments_clears_the_losers() -> None:
    donor, loser = learner(), learner(dataclasses.replace(tiny_config(), init_battle=7))
    optimiser, entries = loser.optimiser_groups()[0]
    parameter = entries[0][2]
    optimiser.state[parameter] = {"step": torch.tensor(3.0)}
    copy_slot(donor, loser, ppo=donor.config.ppo, seed=5, settings={"epochs": 2})
    assert not optimiser.state.get(parameter)
    assert loser.population_settings == {"epochs": 2}
    assert loser.config.learner_seed == 5
    assert loser.generator.initial_seed() == 5


def test_learners_with_other_optimisers_or_parameters_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor, loser = learner(), learner()
    groups = loser.optimiser_groups()
    monkeypatch.setattr(loser, "optimiser_groups", lambda: groups[:1])
    with pytest.raises(SlotCopyError, match="different optimisers"):
        copy_slot(donor, loser, ppo=donor.config.ppo, seed=1, settings={})
    renamed = [(opt, [("preview", "x." + n, p) for _, n, p in entries]) for opt, entries in groups]
    monkeypatch.setattr(loser, "optimiser_groups", lambda: renamed)
    with pytest.raises(SlotCopyError, match="different parameters"):
        copy_slot(donor, loser, ppo=donor.config.ppo, seed=1, settings={})
