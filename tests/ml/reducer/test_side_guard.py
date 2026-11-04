"""A second Wide Guard on one side prints nothing (#1281).

`Side.addSideCondition` returns false without a message when the side's condition is already up,
while the move's `onHitSide` still adds the user's `stall` volatile and the move does not fail
(`sim/battle-actions.ts` `moveHit`, `data/moves.ts` `wideguard`). So the partner's Wide Guard in
the same turn is a `|move|` followed by neither `-singleturn` nor `-fail` (run
public-main-df3c-002, generation 21, game 747: `MalformedLineError: a stalling move with no
-singleturn or -fail`). It counts as a success for the Protect streak.

`fixtures/showdown/<format>/side-guard/battle.json` is a seeded random-legal battle that reaches
the line (`packages/showdown-bridge/scripts/capture-side-guard-battle.ts`; the captured spec is
committed and never re-searched). The state is unchanged for every other log, so no reduced
golden was rewritten.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.reducer import MalformedLineError, reduce_channel

from .support import FORMAT_DIR, VOCAB

BATTLE: dict[str, Any] = json.loads((FORMAT_DIR / "side-guard" / "battle.json").read_text())
PLAYERS = ("p1", "p2")


def _lines(player: str) -> list[str]:
    return list(BATTLE["channels"][player]["lines"])


def _reduce(player: str, lines: list[str]) -> Any:
    return reduce_channel(player, lines, BATTLE["teams"][player], VOCAB)


def _silent(lines: list[str]) -> int:
    """Index of the Wide Guard `|move|` line that is followed by neither outcome."""
    return next(
        at
        for at, line in enumerate(lines)
        if "|Wide Guard|" in line
        and line.startswith("|move|p1")
        and not lines[at + 1].startswith(("|-singleturn|", "|-fail|"))
    )


@pytest.mark.parametrize("player", PLAYERS)
def test_the_whole_battle_reduces(player: str) -> None:
    assert _reduce(player, _lines(player)).winner is not None


@pytest.mark.parametrize("player", PLAYERS)
def test_the_silent_use_counts_as_a_success_for_the_streak(player: str) -> None:
    lines = _lines(player)
    at = _silent(lines)
    user = lines[at].split("|")[2].split(": ")[1]
    partner = lines[at - 1] if lines[at - 1].startswith("|-singleturn|") else lines[at - 2]
    assert "|-singleturn|" in partner
    state = _reduce(player, lines[: at + 2])  # the line after the move is the next action
    side = state.own if player == "p1" else state.opponent
    mon = next(m for m in side.roster if m.name == user)
    assert (mon.recent.last_move, mon.recent.protect_streak) == ("wideguard", 1)


def test_without_the_partners_singleturn_the_silence_is_refused() -> None:
    lines = _lines("p1")
    at = _silent(lines)
    # the partner's line is the one before the move: remove it and the silence is unexplained
    assert lines[at - 1].startswith("|-singleturn|")
    broken = [*lines[: at - 1], *lines[at:]]
    with pytest.raises(MalformedLineError, match="stalling move"):
        _reduce("p1", broken)


def test_ally_side_moves_meet_the_stalling_moves_in_exactly_the_two_side_guards() -> None:
    """Mirrors the TS parity assertion: only Quick Guard and Wide Guard are both."""
    assert VOCAB.stall_moves is not None
    assert VOCAB.ally_side_moves & VOCAB.stall_moves == {"quickguard", "wideguard"}
