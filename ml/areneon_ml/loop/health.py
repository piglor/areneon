"""Health stops and quarantine of a diverging member (ADR-021 sections 6 and 12, slice 9).

Pure functions plus one small guard over a learner; the population loop calls them while it trains
a generation. A member is *stopped* by the first of these, judged from its own train report and
the generation's game rows (``population.health`` of the run config):

1. ``non_finite_loss``: any value of the train report is not a finite number;
2. ``clip_fraction``: the Battle role's clip fraction is above ``clip_ppm`` in this and the two
   generations before it (three consecutive generations);
3. ``approx_kl``: the same for the Battle role's approximate KL against ``kl_ppm`` (millionths);
4. ``forfeits``: more than ``forfeit_ppm`` of the member's games of the generation were forfeits.

A generation that does not count towards a streak: one of a slot's copy or replacement and the
ones before it (the slot is new), a value-only update (its policy did not move), a report that is
missing or holds a non-finite value (that is stop 1 already). The comparison is on integers where
the report holds a float (``value * 10^6 > limit``) and strict.

A stopped member is *quarantined* in the generation's ``ladder.json`` (selection outcome
``health``). Its update of that generation is discarded (:class:`Guard` puts the pre-update
weights, Adam moments and generator back), so its frozen pair is the one it already had and no
non-finite weight is ever frozen; it trains no more. In the next generation its slot takes a copy
of the top live member (:func:`areneon_ml.loop.selection.replacements`, the copy mechanics of slice
8, value-only first update included). If half or more of the members are quarantined in one
generation the run stops with a :class:`HealthIncident` and nothing of that generation is
committed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.trainer import Learner
from areneon_ml.loop.driver import LoopError

#: The generations in a row, the current one included, that make a clip or KL stop.
STREAK = 3
MICRO = 1_000_000


class HealthIncident(LoopError):
    """Half or more of the members stopped in one generation: the run stops, nothing is kept."""


@dataclass(frozen=True)
class Limits:
    clip_ppm: int
    kl_ppm: int
    forfeit_ppm: int


def limits(config: RunConfig) -> Limits:
    raw = config.raw["population"]["health"]
    return Limits(raw["clip_ppm"], raw["kl_ppm"], raw["forfeit_ppm"])


# -- the report ----------------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def non_finite(report: Mapping[str, Any]) -> bool:
    """True if any number of a train report is not finite."""
    return not _finite(report)


def sanitize(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``None`` (the canonical JSON has no NaN).

    A report with only finite values is returned as it is, so a healthy run keeps its bytes.
    """
    if _finite(value):
        return value
    if isinstance(value, float):
        return None
    if isinstance(value, Mapping):
        return {k: sanitize(v) for k, v in value.items()}
    return [sanitize(v) for v in value]


def _over(report: Mapping[str, Any] | None, key: str, limit_ppm: int) -> bool:
    """The Battle role's ``key`` is above ``limit_ppm`` millionths in a (stored) report."""
    if not report or report.get("battle", {}).get("value_warmup"):
        return False
    value = report.get("battle", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return bool(math.isfinite(value) and value * MICRO > limit_ppm)


def forfeit_over(rows: Sequence[Mapping[str, Any]], member: str, found: Limits) -> bool:
    """More than ``forfeit_ppm`` of the member's games of the generation were forfeits."""
    mine = [r for r in rows if r.get("member") == member]
    lost = sum(
        1
        for r in mine
        if r.get("outcome") == "forfeit" and r["seats"].get(r["forfeit"]["loser"]) == member
    )
    return lost * MICRO > found.forfeit_ppm * len(mine)


def stop_cause(
    report: Mapping[str, Any],
    earlier: Sequence[Mapping[str, Any] | None],
    rows: Sequence[Mapping[str, Any]],
    member: str,
    found: Limits,
) -> str | None:
    """The first health stop of a member's generation, or ``None``.

    ``earlier`` are the stored reports of the generations before this one, newest first and as
    many as ``STREAK - 1``; an entry that does not count (see the module words) is ``None``.
    """
    if non_finite(report):
        return "non_finite_loss"
    window = [report, *earlier]
    if len(window) >= STREAK:
        for cause, key, limit in (
            ("clip_fraction", "clip_fraction", found.clip_ppm),
            ("approx_kl", "approx_kl", found.kl_ppm),
        ):
            if all(_over(r, key, limit) for r in window[:STREAK]):
                return cause
    if forfeit_over(rows, member, found):
        return "forfeits"
    return None


def check_incident(stopped: Mapping[str, str], members: int) -> None:
    """Half or more of the members stopped: refuse, with fixed words (no member, no value)."""
    if 2 * len(stopped) >= members:
        raise HealthIncident(
            "health: half or more of the members are quarantined in one generation; "
            "nothing of the generation is committed"
        )


# -- the discarded update ------------------------------------------------------------------------


class Guard:
    """A learner's weights, Adam moments and generator, kept to undo one update."""

    def __init__(self, learner: Learner) -> None:
        self._networks = {
            role: {k: v.detach().clone() for k, v in network.state_dict().items()}
            for role, network in learner.networks.items()
        }
        self._optimisers = [
            copy.deepcopy(optimiser.state_dict()) for optimiser, _ in learner.optimiser_groups()
        ]
        self._generator = learner.generator.get_state().clone()

    def restore(self, learner: Learner) -> None:
        """Put the state back; the generation counter and the update's schedule keys stay."""
        for role, state in self._networks.items():
            learner.networks[role].load_state_dict(state, strict=True)
        for (optimiser, _), state in zip(learner.optimiser_groups(), self._optimisers, strict=True):
            optimiser.load_state_dict(copy.deepcopy(state))
        learner.generator.set_state(self._generator)
