"""Tests for the #678 timer budget spike."""

# Expected timer values are literal seconds from the VGC Timer rule; naming each is noise.
# ruff: noqa: PLR2004

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from timer_budget import (
    DEFAULT_CONFIG,
    ShadowTimer,
    TimerSnapshot,
    battle_budget,
    estimate_remaining_decisions,
    long_battle_schedule,
    parse_time_left,
    preview_budget,
    worked_example,
)


def snap(turn: int, bank: int, grace: int = 0) -> TimerSnapshot:
    return TimerSnapshot(turn, bank, grace, "server")


def test_parse_time_left_preview_line_with_grace() -> None:
    line = "|inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace"
    assert parse_time_left(line) == snap(90, 420, 90)


def test_parse_time_left_battle_line_without_grace() -> None:
    line = "|inactive|Time left: 55 sec this turn | 385 sec total"
    assert parse_time_left(line) == snap(55, 385)


def test_parse_time_left_ignores_other_inactive_lines() -> None:
    assert parse_time_left("|inactive|Bot has 20 seconds left.") is None


def test_shadow_timer_matches_server_preview_and_grace_clip() -> None:
    timer, preview = ShadowTimer.start().request(battle_turn=0)
    assert preview == TimerSnapshot(90, 420, 90, "shadow")
    # 85 s of preview time comes out of grace: the turn-1 clip restores 420.
    timer, turn1 = timer.charge(85.0).request(battle_turn=1)
    assert turn1 == TimerSnapshot(55, 420, 0, "shadow")
    # Mid-turn (forced switch) request on the same turn: 55 s cap, no refill.
    timer, forced = timer.charge(12.0).request(battle_turn=1)
    assert forced == TimerSnapshot(55, 410, 0, "shadow")


def test_shadow_timer_charges_whole_ticks_only() -> None:
    timer = ShadowTimer.start()
    assert timer.charge(4.99).total == 510
    assert timer.charge(5.0).total == 505
    assert timer.charge(14.9).total == 500


def test_turn_cap_follows_bank_when_bank_is_low() -> None:
    timer, _ = ShadowTimer.start().request(0)
    timer, _ = timer.request(1)
    timer = timer.charge(400.0)
    _, low = timer.request(2)
    assert low == TimerSnapshot(20, 20, 0, "shadow")


def test_preview_budget_is_capped_at_sixty_seconds() -> None:
    budget = preview_budget(snap(90, 420, 90))
    assert budget.submit_within == 60.0
    assert budget.policy_budget == 59.0


def test_preview_budget_respects_short_first_request() -> None:
    budget = preview_budget(snap(30, 420, 90))
    assert budget.submit_within == 27.0


def test_battle_budget_capped_at_twenty_seconds_early() -> None:
    budget = battle_budget(snap(55, 420), remaining_decisions=14)
    assert budget.submit_within == 20.0
    assert budget.policy_budget == 19.0


def test_battle_budget_shares_bank_over_remaining_decisions() -> None:
    budget = battle_budget(snap(55, 115), remaining_decisions=10)
    assert budget.submit_within == pytest.approx(10.0)


def test_battle_budget_skips_policy_when_only_reserve_is_left() -> None:
    budget = battle_budget(snap(15, 15), remaining_decisions=6)
    assert budget.submit_within == DEFAULT_CONFIG.fallback_reserve
    assert budget.skip_policy


def test_battle_budget_never_exceeds_turn_cap_minus_margin() -> None:
    budget = battle_budget(snap(5, 5), remaining_decisions=1)
    assert budget.submit_within == 1.0
    tiny = battle_budget(snap(2, 2), remaining_decisions=1)
    assert tiny.submit_within == 0.0


def test_battle_budget_rejects_zero_remaining() -> None:
    with pytest.raises(ValueError, match="remaining_decisions"):
        battle_budget(snap(55, 420), remaining_decisions=0)


def test_remaining_decisions_estimate() -> None:
    assert estimate_remaining_decisions(4, 4, 2) == 14
    assert estimate_remaining_decisions(1, 1, 0) == DEFAULT_CONFIG.min_remaining_decisions


def test_fixed_twenty_seconds_exhausts_bank() -> None:
    """The ticket's premise: a flat 20 s per decision runs out of bank."""
    timer, _ = ShadowTimer.start().request(0)
    turn = 0
    while True:
        turn += 1
        timer, s = timer.request(turn)
        if s.turn_seconds_left <= 20:
            break
        timer = timer.charge(20.25)
    assert turn == 21


def test_worked_example_long_battle_keeps_bank() -> None:
    rows = worked_example(long_battle_schedule(40))
    assert len(rows) > 40
    assert min(r.bank_after for r in rows) >= DEFAULT_CONFIG.bank_reserve
    assert all(r.submit_within + 0.25 < r.turn_left for r in rows)


@given(
    turns=st.integers(min_value=1, max_value=300),
    latency=st.floats(min_value=0.0, max_value=1.5),
    remaining=st.lists(st.integers(min_value=1, max_value=40), min_size=300, max_size=300),
)
def test_bank_never_exhausted_and_autochoose_never_fires(
    turns: int, latency: float, remaining: list[int]
) -> None:
    """Any estimator output (even 1) and any latency below the margin is safe."""
    timer, preview = ShadowTimer.start().request(0)
    timer = timer.charge(preview_budget(preview).submit_within + latency)
    for turn in range(1, turns + 1):
        timer, s = timer.request(turn)
        assert s.bank_seconds_left > 0
        budget = battle_budget(s, remaining[turn - 1])
        elapsed = budget.submit_within + latency
        assert elapsed < s.turn_seconds_left
        timer = timer.charge(elapsed)
