"""`moves_revealed` stays what a Pokemon can have at one time (#938).

Transform replaces the whole move list with the target's, and the Pokemon gets its own back on
switching out or fainting (`sim/pokemon.ts` `transformInto` and `clearVolatile`). Mimic and
Sketch also change a slot, but neither is a move of the format (the engine-data export has
neither, so a `|move|` of them is a `VocabularyError`); the Transform lines here are the
engine's own, from the seeded battle that stopped run public-main-002, in which a Ditto
transformed twice showed nine moves.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from areneon_ml.model.encoder import Encoder
from areneon_ml.reducer import reduce_channel

from tests.ml.model.support import TINY_ENCODER, game, observation

from .support import FORMAT_DIR, VOCAB, opp, run

TRANSFORM = (
    "|move|p2a: Venusaur|Transform|p1b: Charizard",
    "|-transform|p2a: Venusaur|p1b: Charizard",
)
SWEET_KISS = "|move|p2a: Venusaur|Sweet Kiss|p1a: Venusaur"
PROTECT = "|move|p2a: Venusaur|Protect|p2a: Venusaur"
SWITCH_IN = "|switch|p2a: Venusaur|Venusaur, L50, M|100/100"
SWITCH_OUT = "|switch|p2a: Blastoise|Blastoise, L50, M|100/100"


def _moves(*lines: str) -> list[str]:
    return opp(run(*lines), "Venusaur").moves_revealed


def test_transform_replaces_the_moves_shown_so_far() -> None:
    assert _moves(PROTECT, *TRANSFORM, SWEET_KISS) == ["sweetkiss"]


def test_the_moves_come_back_when_the_pokemon_switches_out() -> None:
    assert _moves(PROTECT, *TRANSFORM, SWEET_KISS, SWITCH_OUT) == ["protect", "transform"]


def test_a_pokemon_that_faints_while_transformed_gets_its_moves_back() -> None:
    assert _moves(PROTECT, *TRANSFORM, "|faint|p2a: Venusaur") == ["protect", "transform"]


def test_the_copy_starts_empty_on_every_stay() -> None:
    again = [*TRANSFORM, SWEET_KISS, SWITCH_OUT, SWITCH_IN, *TRANSFORM]
    assert _moves(PROTECT, *again) == []


def test_a_move_list_stays_within_four_slots_and_struggle() -> None:
    lines = [PROTECT]
    for _ in range(3):
        lines += [
            *TRANSFORM,
            SWEET_KISS,
            "|move|p2a: Venusaur|Roost|p2a: Venusaur",
            SWITCH_OUT,
            SWITCH_IN,
        ]
    lines += [
        *TRANSFORM,
        "|move|p2a: Venusaur|Protect|p2a: Venusaur",
        "|move|p2a: Venusaur|Roost|p2a: Venusaur",
        "|move|p2a: Venusaur|Tailwind|p2a: Venusaur",
        "|move|p2a: Venusaur|Baton Pass||[still]",
        "|-activate|p2a: Venusaur|move: Struggle",
        "|move|p2a: Venusaur|Struggle|p1a: Venusaur",
    ]
    moves = _moves(*lines)
    assert moves == ["protect", "roost", "tailwind", "batonpass", "struggle"]


BATTLE: dict[str, Any] = json.loads((FORMAT_DIR / "transform" / "battle.json").read_text())


def _requests(lines: list[str]) -> list[int]:
    return [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]


@pytest.mark.parametrize("player", ["p1", "p2"])
def test_the_battle_that_stopped_the_run_encodes_at_every_decision(player: str) -> None:
    # Each seat's channel up to the request the encoder refused (max_moves 8: the run's value).
    lines = list(BATTLE["channels"][player]["lines"])
    encoder = Encoder(game(), dataclasses.replace(TINY_ENCODER, max_moves=8))
    for end in _requests(lines):
        state = reduce_channel(player, lines[:end], BATTLE["teams"][player], VOCAB)
        if state.request_kind == "wait":
            continue
        kind = "preview" if state.request_kind == "preview" else "battle"
        encoder.encode(
            {
                "observation": observation(player, lines[:end], BATTLE["teams"][player], kind),
                "candidates": [],
            }
        )
    state = reduce_channel(player, lines, BATTLE["teams"][player], VOCAB)
    rosters = [*state.own.roster, *state.opponent.roster]
    assert max(len(mon.moves_revealed) for mon in rosters) <= 5
    ditto = next(
        mon
        for half in (state.own, state.opponent)
        for mon in half.roster
        if mon.base_species == "ditto"
    )
    assert ditto.moves_revealed == ["roost", "batonpass", "protect", "tailwind", "struggle"]
    assert ditto.moves_base == ["transform"]
