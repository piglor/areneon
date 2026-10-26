"""Baton Pass through the reducer (#1099).

The engine's `copyVolatileFrom` gives the Pokemon that comes in after a Baton Pass the stat
stages of the one that went out and prints no line for them; the only sign is the switch line's
`[from] Baton Pass`. A reducer that reset the stages at every switch lost them, so a later
`|-boost|` took the Pokemon past the engine's cap of 6 (`boost 7 is outside -6..6`, run
public-main-009). Shed Tail passes none.

`fixtures/showdown/<format>/baton-pass/battle.json` is a seeded random-legal battle on a team of
three golden Pokemon and three Swords Dance + Baton Pass users
(`packages/showdown-bridge/scripts/capture-baton-pass-battle.ts`; CI re-plays it). Rewrite the
reduced snapshots deliberately with
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_baton_pass.py --no-cov` and review
the diff.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from areneon_ml.reducer import MalformedLineError, reduce_channel

from .support import FORMAT_DIR, VOCAB

BATTLE: dict[str, Any] = json.loads((FORMAT_DIR / "baton-pass" / "battle.json").read_text())
PROVENANCE = "synthetic: reduced state of baton-pass/battle.json {player}, Baton Pass users"
PLAYERS = ("p1", "p2")


def _lines(player: str) -> list[str]:
    return list(BATTLE["channels"][player]["lines"])


def _reduce(player: str, lines: list[str]) -> Any:
    return reduce_channel(player, lines, BATTLE["teams"][player], VOCAB)


def _passes(lines: list[str]) -> list[int]:
    return [
        i
        for i, line in enumerate(lines)
        if line.startswith("|switch|") and "[from] Baton Pass" in line
    ]


def _active(state: Any, line: str) -> Any:
    ref = line.split("|")[2]
    side = state.own if ref.startswith(state.player) else state.opponent
    return next(m for m in side.roster if m.active_slot == ref[2])


@pytest.mark.parametrize("player", PLAYERS)
def test_the_incoming_pokemon_has_the_outgoing_one_s_stages(player: str) -> None:
    lines = _lines(player)
    passes = _passes(lines)
    assert len(passes) == 3
    first = passes[0]
    before = _reduce(player, lines[:first])
    sender = _active(before, "|switch|" + lines[first].split("|")[2])
    assert sender.boosts["atk"] == 4  # two Swords Dances
    after = _reduce(player, lines[: first + 1])
    receiver = _active(after, lines[first])
    assert receiver.boosts["atk"] == 4
    assert receiver.name == "Leafeon"
    side = after.own if lines[first].split("|")[2].startswith(player) else after.opponent
    gone = next(m for m in side.roster if m.name == sender.name)
    assert gone.active_slot is None
    assert gone.boosts["atk"] == 0  # the one that left keeps none: the engine moved them


@pytest.mark.parametrize("player", PLAYERS)
def test_stages_pass_on_again_and_end_at_the_next_plain_switch(player: str) -> None:
    lines = _lines(player)
    passes = _passes(lines)
    last = _reduce(player, lines[: passes[2] + 1])
    assert _active(last, lines[passes[2]]).boosts["atk"] == 4
    plain = next(i for i in range(passes[2], len(lines)) if lines[i].startswith("|switch|p1"))
    if plain != passes[2]:
        assert _active(_reduce(player, lines[: plain + 1]), lines[plain]).boosts["atk"] == 0


def test_a_raise_after_a_pass_is_taken_from_the_passed_stage() -> None:
    lines = _lines("p1")
    first = _passes(lines)[0]
    sender = lines[first].split("|")[2][:3]
    carried = [*lines[: first + 1], f"|-boost|{sender}: Leafeon|atk|2"]
    assert _active(_reduce("p1", carried), lines[first]).boosts["atk"] == 6
    over = [*carried, f"|-boost|{sender}: Leafeon|atk|1"]
    with pytest.raises(MalformedLineError, match=r"boost 7 is outside -6\.\.6"):
        _reduce("p1", over)


def test_a_switch_without_the_baton_pass_tag_starts_from_zero() -> None:
    lines = _lines("p1")
    first = _passes(lines)[0]
    plain = lines[first].replace("|[from] Baton Pass", "")
    assert _active(_reduce("p1", [*lines[:first], plain]), plain).boosts["atk"] == 0
    shed = lines[first].replace("Baton Pass", "Shed Tail")
    assert _active(_reduce("p1", [*lines[:first], shed]), shed).boosts["atk"] == 0


@pytest.mark.parametrize("player", PLAYERS)
def test_the_reduced_state_matches_its_golden_snapshot(player: str) -> None:
    document = (
        json.dumps(
            {
                "schema_version": 1,
                "data_class": "synthetic",
                "provenance": PROVENANCE.format(player=player),
                "state": _reduce(player, _lines(player)).to_dict(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    snapshot = FORMAT_DIR / "reduced" / f"baton-pass-{player}.json"
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.write_text(document)
    assert snapshot.read_text() == document
