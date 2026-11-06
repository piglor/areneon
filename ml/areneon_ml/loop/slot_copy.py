"""The copy itself (ADR-021 section 6): a loser slot takes a donor's frozen state.

The loser's networks take the donor's weights and its Adam moments (with their step counts, so
the cumulative learner steps follow the donor); its learner generator is re-seeded from the
``member-learner`` draw the ladder record names; its PPO settings become the explored ones. Both
learners must be built from one run config (same shapes) and sit on the same device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from areneon_ml.learner.config import PPOSettings
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.network import ROLES


class SlotCopyError(ValueError):
    """Two learners that cannot be copied one onto the other."""


def copy_slot(
    donor: Learner,
    loser: Learner,
    *,
    ppo: PPOSettings,
    seed: int,
    settings: Mapping[str, Any],
) -> None:
    """Make ``loser`` the donor's weights and Adam state under new settings and a new generator.

    ``settings`` is the PPO half of the explored settings, kept on the learner for its next
    sidecar. The donor is not changed.
    """
    if donor is loser:
        raise SlotCopyError("a slot cannot be copied onto itself")
    donor_groups = donor.optimiser_groups()
    loser_groups = loser.optimiser_groups()
    if len(donor_groups) != len(loser_groups):
        raise SlotCopyError("the two learners have different optimisers")
    for role in ROLES:
        loser.networks[role].load_state_dict(donor.networks[role].state_dict(), strict=True)
    for (donor_opt, donor_entries), (loser_opt, loser_entries) in zip(
        donor_groups, loser_groups, strict=True
    ):
        if [(r, n) for r, n, _ in donor_entries] != [(r, n) for r, n, _ in loser_entries]:
            raise SlotCopyError("the two learners have different parameters")
        for (_, _, mine), (_, _, theirs) in zip(donor_entries, loser_entries, strict=True):
            found = donor_opt.state.get(mine)
            if found:
                loser_opt.state[theirs] = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in found.items()
                }
            else:
                loser_opt.state.pop(theirs, None)
    loser.generation = donor.generation
    loser.config = dataclasses.replace(loser.config, ppo=ppo, learner_seed=seed)
    for optimiser, _ in loser_groups:
        for group in optimiser.param_groups:
            group["lr"] = ppo.learning_rate
    loser.generator.manual_seed(seed)
    loser.population_settings = dict(settings)
