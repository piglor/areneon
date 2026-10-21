"""An Illusion battle through the reducer and the encoder (#851).

`fixtures/showdown/<format>/illusion/battle.json` is a seeded random-legal battle on the golden
team with a Zoroark (Illusion) in the last place of both teams
(`packages/showdown-bridge/scripts/capture-illusion-battle.ts`; CI re-plays it). In it each
side's disguise breaks (`|replace|`): p1's own Zoroark was shown as Pikachu, p2's as Pikachu
too. Rewrite the reduced snapshots deliberately with
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_illusion_battle.py --no-cov`
and review the diff.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest
from areneon_ml.model.encoder import Encoder
from areneon_ml.reducer import BattleState, reduce_channel

from tests.ml.model.support import TINY_ENCODER, game, observation

from .support import FORMAT_DIR, VOCAB, leaks, unattributed, unrevealed_ids

BATTLE: dict[str, Any] = json.loads((FORMAT_DIR / "illusion" / "battle.json").read_text())
PLAYERS = (("p1", "p2"), ("p2", "p1"))
PROVENANCE = "synthetic: reduced state of illusion/battle.json {player}, golden team with a Zoroark"


def _lines(player: str) -> list[str]:
    return list(BATTLE["channels"][player]["lines"])


def _reduce(player: str, lines: list[str]) -> BattleState:
    return reduce_channel(player, lines, BATTLE["teams"][player], VOCAB)


def _requests(lines: list[str]) -> list[int]:
    return [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]


def _replace_at(lines: list[str], player: str) -> int:
    return next(i for i, line in enumerate(lines) if line.startswith(f"|replace|{player}"))


def test_each_side_s_disguise_breaks_in_the_battle() -> None:
    for player, _ in PLAYERS:
        text = "\n".join(_lines(player))
        assert "|replace|p1a: Zoroark|Zoroark, L50, M" in text
        assert "|replace|p2b: Zoroark|Zoroark, L50, M" in text
        assert "|-end|" not in text or "Illusion" in text  # the line that follows a replace


@pytest.mark.parametrize(("player", "other"), PLAYERS)
def test_the_opponent_s_illusion_user_is_unnamed_until_its_replace(player: str, other: str) -> None:
    lines = _lines(player)
    at = _replace_at(lines, other)
    pristine = _reduce(player, lines[: _requests(lines)[0]]).opponent.roster
    before = _reduce(player, lines[:at]).opponent  # the disguise is up, the replace not yet read
    zoroark = next(m for m in before.roster if m.base_species == "zoroark")
    pikachu = next(m for m in before.roster if m.base_species == "pikachu")
    assert zoroark.to_dict() == next(m for m in pristine if m.base_species == "zoroark").to_dict()
    assert pikachu.name == "Pikachu"
    assert pikachu.brought
    after = _reduce(player, lines[: at + 1]).opponent
    zoroark = next(m for m in after.roster if m.base_species == "zoroark")
    pikachu = next(m for m in after.roster if m.base_species == "pikachu")
    assert zoroark.name == "Zoroark"
    assert zoroark.brought
    assert zoroark.ability == "illusion"
    assert pikachu.name is None
    assert pikachu.brought is None
    assert not pikachu.moves_revealed


@pytest.mark.parametrize(("player", "other"), PLAYERS)
def test_the_canary_holds_at_every_request_point(player: str, other: str) -> None:
    lines = _lines(player)
    for end in _requests(lines):
        prefix = lines[:end]
        opponent = _reduce(player, prefix).to_dict()["opponent"]
        secrets = unrevealed_ids(BATTLE["teams"][other], "\n".join(prefix))
        assert leaks(opponent, secrets) == [], (player, end)
        assert unattributed(opponent, prefix, other) == [], (player, end)


def test_the_real_species_is_in_no_observation_before_the_replace() -> None:
    # The preview names Zoroark among the six (`|poke|`), so the species alone is no secret; what
    # the replace reveals is that it is the Pokemon in play: no position, name, hit points, move
    # or ability of it in the state before that line, nor any line that names it.
    for player, other in PLAYERS:
        lines = _lines(player)
        at = _replace_at(lines, other)
        for end in [e for e in _requests(lines) if e <= at] + [at]:
            state = _reduce(player, lines[:end])
            zoroark = next(m for m in state.opponent.roster if m.base_species == "zoroark")
            assert (zoroark.name, zoroark.active_slot, zoroark.hp_num, zoroark.brought) == (
                None,
                None,
                None,
                None,
            )
            assert zoroark.moves_revealed == []
            assert zoroark.ability is None
        assert not [x for x in lines[:at] if re.search(rf"{other}[a-c]?: Zoroark", x)]


def test_own_illusion_user_is_the_real_pokemon_from_the_request_on() -> None:
    lines = _lines("p1")
    at = _replace_at(lines, "p1")
    state = _reduce("p1", lines[: _requests(lines)[1]])  # the first move request, disguise up
    assert state.own.slots["a"] == next(
        i for i, m in enumerate(state.own.roster) if m.name == "Zoroark"
    )
    assert next(m for m in state.own.roster if m.name == "Pikachu").active_slot is None
    assert at > _requests(lines)[1]


@pytest.mark.parametrize(("player", "other"), PLAYERS)
def test_every_decision_encodes(player: str, other: str) -> None:
    lines = _lines(player)
    encoder = Encoder(game(), TINY_ENCODER)
    for end in _requests(lines):
        prefix = lines[:end]
        state = _reduce(player, prefix)
        if state.request_kind == "wait":
            continue
        kind = "preview" if state.request_kind == "preview" else "battle"
        encoder.encode(
            {
                "observation": observation(player, prefix, BATTLE["teams"][player], kind),
                "candidates": [],
            }
        )


def test_an_unresolved_pair_is_encoded_as_what_the_player_sees() -> None:
    lines = _lines("p1")
    first_turn = _requests(lines)[1]
    prefix = lines[:first_turn]
    start = next(i for i, line in enumerate(prefix) if line.startswith("|switch|p2b"))
    shown_twice = [
        *prefix[: start + 1],
        "|switch|p2a: Blastoise|Blastoise, L50, M|100/100",
        *prefix[start + 1 :],
    ]
    state = _reduce("p1", shown_twice)
    assert len(state.opponent.roster) == 7
    assert state.opponent.roster[-1].phantom
    encoded = Encoder(game(), TINY_ENCODER).encode(
        {
            "observation": observation("p1", shown_twice, BATTLE["teams"]["p1"], "battle"),
            "candidates": [],
        }
    )
    assert encoded.mon_cat.shape[0] == 6 + 7


@pytest.mark.parametrize("player", ["p1", "p2"])
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
    snapshot = FORMAT_DIR / "reduced" / f"illusion-{player}.json"
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.write_text(document)
    assert snapshot.read_text() == document
