"""A Fling whose item is already gone through the reducer (#1162).

The engine's Fling hits first and ends the thrown item in the move's own `onUpdate`, which
reads the holder's item *then*. When the hit starts a terrain through Seed Sower and the
Pokemon's Grassy Seed is eaten at once, the item is already gone and the engine prints
`|-enditem|POKEMON||[from] move: Fling` with no item name (run public-main-010, generation 19,
game 390: `VocabularyError: unknown item ''`). The line leaves the Pokemon with no
item, lost (what the engine's `setItem('')` does). Any other empty item name stays refused.

`fixtures/showdown/<format>/fling/battle.json` is a seeded random-legal battle that reaches the
line (`packages/showdown-bridge/scripts/capture-fling-battle.ts`; CI re-plays it). Rewrite the
reduced snapshots deliberately with
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_fling.py --no-cov` and review the diff.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from areneon_ml.reducer import VocabularyError, reduce_channel

from .support import FORMAT_DIR, VOCAB

BATTLE: dict[str, Any] = json.loads((FORMAT_DIR / "fling" / "battle.json").read_text())
PROVENANCE = "synthetic: reduced state of fling/battle.json {player}, Fling of a consumed item"
PLAYERS = ("p1", "p2")
EMPTY = "|-enditem|p1b: Dedenne||[from] move: Fling"


def _lines(player: str) -> list[str]:
    return list(BATTLE["channels"][player]["lines"])


def _reduce(player: str, lines: list[str]) -> Any:
    return reduce_channel(player, lines, BATTLE["teams"][player], VOCAB)


def _dedenne(state: Any, player: str) -> Any:
    side = state.own if player == "p1" else state.opponent
    return next(m for m in side.roster if m.name == "Dedenne")


@pytest.mark.parametrize("player", PLAYERS)
def test_the_unnamed_item_end_leaves_the_holder_without_its_item(player: str) -> None:
    lines = _lines(player)
    at = lines.index(EMPTY)
    before = _dedenne(_reduce(player, lines[:at]), player)
    after = _dedenne(_reduce(player, lines[: at + 1]), player)
    assert before.item_lost  # the seed's own item end came first
    assert (after.item, after.item_known, after.item_lost) == (None, True, True)


def test_the_item_end_alone_is_taken_for_a_held_item_that_is_gone() -> None:
    lines = _lines("p1")
    at = lines.index(EMPTY)
    without_seed = [line for line in lines[:at] if line != "|-enditem|p1b: Dedenne|Grassy Seed"]
    without_seed = [line for line in without_seed if "[from] item: Grassy Seed" not in line]
    state = _reduce("p1", [*without_seed, EMPTY])
    held = _dedenne(state, "p1")
    assert (held.item, held.item_known, held.item_lost) == (None, True, True)


@pytest.mark.parametrize(
    "line",
    [
        "|-enditem|p1b: Dedenne|",  # no tag
        "|-enditem|p1b: Dedenne||[from] move: Knock Off",
        "|-enditem|p1b: Dedenne||[from] item: Grassy Seed",
    ],
)
def test_any_other_empty_item_name_is_refused(line: str) -> None:
    lines = _lines("p1")
    at = lines.index(EMPTY)
    with pytest.raises(VocabularyError, match="unknown item ''"):
        _reduce("p1", [*lines[:at], line])


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
    snapshot = FORMAT_DIR / "reduced" / f"fling-{player}.json"
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.write_text(document)
    assert snapshot.read_text() == document
