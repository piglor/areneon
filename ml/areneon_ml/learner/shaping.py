"""Potential-based reward shaping on the Battle role (ADR-010 erratum A11.1, #1365).

``Phi(o) = w * d(o) / 10^6`` where ``d(o)`` is the centred HP and knock-out differential of
ADR-026's non-learned leaf (``hpKoDifferential`` minus 500,000), read from the player's own view
of the decision ``o`` and nothing else: the own exact hit points and standing from the last
``|request|``, the opponent's shown share from the reduced state, and every brought opponent
Pokemon not yet seen counted full and standing (never guessed). ``d`` is an exact integer in
-500,000 .. 500,000; the one division is ``phi``. Phi of the state after the last decision is 0.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from areneon_ml.reducer.state import BattleState

PPM = 1_000_000
_REQUEST = "|request|"


class ShapingError(ValueError):
    """The potential cannot be read from an observation (the learner reports it as data error)."""


def _own_share(request: str) -> tuple[int, int, int]:
    """(n, own HP share in ppm, own standing share in ppm) from a ``|request|`` line."""
    if not request.startswith(_REQUEST):
        raise ShapingError("the observation holds no request to read the own side from")
    side = json.loads(request[len(_REQUEST) :]).get("side")
    team = None if not isinstance(side, dict) else side.get("pokemon")
    if not isinstance(team, list) or not team:
        raise ShapingError("the last request lists no own Pokemon")
    total = 0
    alive = 0
    for entry in team:
        first = str(entry["condition"]).split(" ")[0]
        if first == "0":
            continue
        hp, _, maxhp = first.partition("/")
        if not hp.isdigit() or not maxhp.isdigit() or int(maxhp) <= 0:
            raise ShapingError("an own condition is not a hit point count")
        total += int(hp) * PPM // int(maxhp)
        alive += 1
    n = len(team)
    return n, total // n, alive * PPM // n


def _opponent_share(state: BattleState, n: int) -> tuple[int, int]:
    """(opponent HP share, opponent standing share) in ppm; unseen brought Pokemon count full."""
    seen = 0
    fainted = 0
    total = 0
    for mon in state.opponent.roster:
        if mon.phantom:
            continue
        if mon.fainted:
            seen += 1
            fainted += 1
        elif mon.hp_num is not None and mon.hp_den:
            seen += 1
            total += mon.hp_num * PPM // mon.hp_den
    if seen > n:
        raise ShapingError(f"{seen} opponent Pokemon were seen but a side brings {n}")
    total += (n - seen) * PPM
    return total // n, (n - fainted) * PPM // n


def differential(state: BattleState, last_request: str | None) -> int:
    """``d(o)``: ``floor((own HP - opp HP + own standing - opp standing) / 4)``, in integers."""
    if last_request is None:
        raise ShapingError("the observation holds no request to read the own side from")
    n, own_hp, own_standing = _own_share(last_request)
    opp_hp, opp_standing = _opponent_share(state, n)
    return (own_hp - opp_hp + own_standing - opp_standing) // 4


def phi(d: int, coef_micro: int) -> float:
    """``Phi = w * d / 10^6`` with ``w = coef_micro / 10^6``, the one division."""
    return coef_micro * d / (PPM * PPM)


def shaped(
    rewards: Sequence[float], potentials: Sequence[float], discount: float
) -> tuple[list[float], float]:
    """``r'_t = r_t + gamma * Phi(o_{t+1}) - Phi(o_t)`` with ``Phi`` of the state after the last
    decision 0, and the summed ``|gamma * Phi(o_{t+1}) - Phi(o_t)|`` of the trajectory."""
    if len(rewards) != len(potentials):
        raise ShapingError("a reward and a potential for every step")
    out: list[float] = []
    magnitude = 0.0
    for t, reward in enumerate(rewards):
        following: Any = potentials[t + 1] if t + 1 < len(potentials) else 0.0
        delta = discount * following - potentials[t]
        magnitude += abs(delta)
        out.append(reward + delta)
    return out, magnitude
