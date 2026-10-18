"""Timer-derived decision budget for the VGC Timer (spike for Redmine #678).

Pure functions only. The semantics mirror the pinned Showdown server timer
(``server/room-battle.ts`` ``RoomBattleTimer`` at 7244fe8); see
``docs/research/678-timer-deadline-budget.md`` for the line-level citations.

All durations are seconds. ``bank`` is the VGC "Your Time" total, excluding
grace, exactly as the server prints it in ``|inactive|Time left: ...``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Literal

TimerSource = Literal["server", "shadow"]


@dataclass(frozen=True)
class TimerRules:
    """Showdown ``GameTimerSettings`` subset that the VGC Timer rule sets."""

    starting: int = 420
    grace: int = 90
    add_per_turn: int = 0
    max_per_turn: int = 55
    max_first_turn: int = 90
    tick: int = 5


VGC_TIMER = TimerRules()


@dataclass(frozen=True)
class TimerSnapshot:
    """The player's own timer state at the moment a request arrives."""

    turn_seconds_left: int
    bank_seconds_left: int
    grace_seconds_left: int
    source: TimerSource


_TIME_LEFT = re.compile(
    r"^\|inactive\|Time left: (?P<turn>\d+) sec this turn \| (?P<bank>\d+) sec total"
    r"(?: \| (?P<grace>\d+) sec grace)?$"
)


def parse_time_left(line: str) -> TimerSnapshot | None:
    """Parse the per-player ``|inactive|Time left`` line the server sends after a request."""
    match = _TIME_LEFT.match(line)
    if match is None:
        return None
    return TimerSnapshot(
        turn_seconds_left=int(match["turn"]),
        bank_seconds_left=int(match["bank"]),
        grace_seconds_left=int(match["grace"] or 0),
        source="server",
    )


@dataclass(frozen=True)
class BudgetConfig:
    """Areneon budget policy. Caps are upper bounds (ADR-005), never targets."""

    preview_cap: float = 60.0
    battle_cap: float = 20.0
    safety_margin: float = 3.0
    fallback_reserve: float = 1.0
    bank_reserve: float = 15.0
    min_remaining_decisions: int = 4
    turns_per_pokemon: float = 1.5


DEFAULT_CONFIG = BudgetConfig()


@dataclass(frozen=True)
class Budget:
    """Wall-clock allowances measured from local receipt of the request.

    ``submit_within``: the final choice (policy or fallback) must be sent by then.
    ``policy_budget``: the policy must answer by then; afterwards the fallback is sent.
    """

    submit_within: float
    policy_budget: float

    @property
    def skip_policy(self) -> bool:
        return self.policy_budget <= 0.0


def estimate_remaining_decisions(
    own_alive: int, opp_alive: int, own_bench: int, config: BudgetConfig = DEFAULT_CONFIG
) -> int:
    """Conservative count of decisions this player still has to make.

    Inputs are player-perspective only: ``own_alive`` and ``own_bench`` come from the
    player's own request, ``opp_alive`` from public faint messages against the brought
    count. Each own bench Pokemon can force at most one extra switch request.
    """
    turns = math.ceil(config.turns_per_pokemon * (own_alive + opp_alive))
    return max(config.min_remaining_decisions, turns + own_bench)


def _budget(turn_left: float, allocation: float, config: BudgetConfig) -> Budget:
    hard = max(0.0, turn_left - config.safety_margin)
    submit_within = min(hard, max(config.fallback_reserve, allocation))
    return Budget(
        submit_within=submit_within,
        policy_budget=max(0.0, submit_within - config.fallback_reserve),
    )


def preview_budget(snapshot: TimerSnapshot, config: BudgetConfig = DEFAULT_CONFIG) -> Budget:
    """Team Preview: bounded by the first-request cap only.

    Preview time is drawn from grace; the server clips the total back to ``starting``
    on the first battle turn, so preview never touches the battle bank.
    """
    return _budget(snapshot.turn_seconds_left, config.preview_cap, config)


def battle_budget(
    snapshot: TimerSnapshot, remaining_decisions: int, config: BudgetConfig = DEFAULT_CONFIG
) -> Budget:
    """Battle request: min(cap, turn cap - margin, (bank - reserve) / remaining decisions)."""
    if remaining_decisions < 1:
        raise ValueError("remaining_decisions must be >= 1")
    spendable = max(0.0, snapshot.bank_seconds_left - config.bank_reserve)
    allocation = min(config.battle_cap, spendable / remaining_decisions)
    return _budget(snapshot.turn_seconds_left, allocation, config)


@dataclass(frozen=True)
class ShadowTimer:
    """Local model of the server timer (A1 headless path, or server timer off).

    ``total`` includes grace, like the server's ``secondsLeft``.
    """

    rules: TimerRules = VGC_TIMER
    total: int = VGC_TIMER.starting + VGC_TIMER.grace
    turn: int | None = None
    first_request: bool = True

    @classmethod
    def start(cls, rules: TimerRules = VGC_TIMER) -> ShadowTimer:
        return cls(rules=rules, total=rules.starting + rules.grace)

    def request(self, battle_turn: int) -> tuple[ShadowTimer, TimerSnapshot]:
        """A new (non-update) request for this player at ``battle_turn``."""
        timer = self
        if timer.turn is None:
            timer = replace(timer, turn=battle_turn)
        elif battle_turn > timer.turn:
            timer = replace(
                timer,
                turn=battle_turn,
                first_request=False,
                total=min(timer.total + timer.rules.add_per_turn, timer.rules.starting),
            )
        else:
            timer = replace(timer, first_request=False)
        cap = timer.rules.max_first_turn if timer.first_request else timer.rules.max_per_turn
        grace = max(0, timer.total - timer.rules.starting)
        snapshot = TimerSnapshot(
            turn_seconds_left=min(timer.total, cap),
            bank_seconds_left=timer.total - grace,
            grace_seconds_left=grace,
            source="shadow",
        )
        return timer, snapshot

    def charge(self, server_elapsed: float) -> ShadowTimer:
        """Charge whole ticks only, as ``nextTick`` does."""
        ticks = math.floor(server_elapsed / self.rules.tick)
        return replace(self, total=self.total - ticks * self.rules.tick)


@dataclass(frozen=True)
class DecisionRow:
    index: int
    battle_turn: int
    turn_left: int
    bank_before: int
    remaining_estimate: int
    submit_within: float
    bank_after: int


def worked_example(
    schedule: list[tuple[int, int, int, int]],
    latency: float = 0.25,
    config: BudgetConfig = DEFAULT_CONFIG,
) -> list[DecisionRow]:
    """Replay a battle where every decision uses its full ``submit_within`` (worst case).

    ``schedule`` rows are ``(battle_turn, own_alive, opp_alive, own_bench)``; the first
    row is Team Preview (turn 0). Server-side elapsed is ``submit_within + latency``.
    """
    timer = ShadowTimer.start()
    rows: list[DecisionRow] = []
    for index, (battle_turn, own_alive, opp_alive, own_bench) in enumerate(schedule):
        timer, snap = timer.request(battle_turn)
        remaining = estimate_remaining_decisions(own_alive, opp_alive, own_bench, config)
        if battle_turn == 0:
            budget = preview_budget(snap, config)
        else:
            budget = battle_budget(snap, remaining, config)
        elapsed = budget.submit_within + latency
        if elapsed >= snap.turn_seconds_left:
            raise AssertionError(f"decision {index} would let auto-choose fire")
        timer = timer.charge(elapsed)
        rows.append(
            DecisionRow(
                index=index,
                battle_turn=battle_turn,
                turn_left=snap.turn_seconds_left,
                bank_before=snap.bank_seconds_left,
                remaining_estimate=remaining,
                submit_within=budget.submit_within,
                bank_after=min(timer.total, timer.rules.starting),
            )
        )
    return rows


def long_battle_schedule(turns: int = 40) -> list[tuple[int, int, int, int]]:
    """Synthetic long (stall-like) battle: one faint every ~5 turns, plus forced switches."""
    own_faint_turn, opp_faint_turn = 5, 0  # turn number modulo 10
    schedule = [(0, 4, 4, 2)]
    own_alive, opp_alive, own_bench = 4, 4, 2
    for turn in range(1, turns + 1):
        schedule.append((turn, own_alive, opp_alive, own_bench))
        if turn % 10 == own_faint_turn and own_bench > 0:
            own_alive -= 1
            own_bench -= 1
            schedule.append((turn, own_alive, opp_alive, own_bench))  # forced switch
        elif turn % 10 == opp_faint_turn and opp_alive > 1:
            opp_alive -= 1
    return schedule


if __name__ == "__main__":  # pragma: no cover - documentation helper
    print("| # | turn | T | bank before | R | submit within | bank after |")
    print("|---|---|---|---|---|---|---|")
    for row in worked_example(long_battle_schedule()):
        print(
            f"| {row.index} | {row.battle_turn} | {row.turn_left} | {row.bank_before} | "
            f"{row.remaining_estimate} | {row.submit_within:.2f} | {row.bank_after} |"
        )
