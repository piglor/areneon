"""The ADR-008 canary for the reducer: no opponent secret in any reduced state.

The secrets are everything in the OTHER player's team that the channel never
names: unrevealed items, abilities and moves, and every exact stat or HP. The
checks scan only the `opponent` half of the state. Each has a negative control:
the same scan over a state that does hold secrets must fire, so a silent
detector cannot pass the suite. The random-team corpus gets the same canary in
`test_corpus.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.reducer import OpponentSecretError, reduce_channel

from .support import (
    FORMAT_DIR,
    HEADER,
    TEAM,
    VOCAB,
    leaks,
    run,
    unrevealed_ids,
)


def _channel(player: str) -> list[str]:
    path = FORMAT_DIR / "channels" / f"{player}.json"
    return list(json.loads(path.read_text())["channel"]["lines"])


@pytest.mark.parametrize("player", ["p1", "p2"])
def test_no_opponent_secret_at_any_request_point_of_the_golden_channels(player: str) -> None:
    lines = _channel(player)
    for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
        prefix = lines[:end]
        state = reduce_channel(player, prefix, TEAM, VOCAB)
        secrets = unrevealed_ids(TEAM, "\n".join(prefix))
        assert secrets  # the golden team keeps most of its set hidden at every point
        assert leaks(state.to_dict()["opponent"], secrets) == []


@pytest.mark.parametrize("player", ["p1", "p2"])
def test_negative_control_the_own_half_of_the_same_state_trips_the_detector(player: str) -> None:
    state = reduce_channel(player, _channel(player), TEAM, VOCAB).to_dict()
    assert leaks(state["own"], unrevealed_ids(TEAM, "")), "the detector must fire on exact values"


def test_negative_control_a_contaminated_opponent_half_is_caught() -> None:
    state = run(
        "|move|p2a: Venusaur|Protect|p2a: Venusaur", "|-singleturn|p2a: Venusaur|Protect"
    ).to_dict()
    secrets = {"leftovers", "gigadrain"}
    assert leaks(state["opponent"], secrets) == []
    contaminated = json.loads(json.dumps(state["opponent"]))
    contaminated["roster"][0]["item"] = "leftovers"
    contaminated["roster"][0]["hp_num"] = 159
    contaminated["roster"][0]["stats"] = {"atk": 102}
    found = leaks(contaminated, secrets)
    assert "secret leftovers" in found
    assert "integer 159" in found
    assert any("exact own-side fields" in leak for leak in found)


def _opponent(species: str, item: str, ability: str, moves: list[str]) -> dict[str, Any]:
    return {**TEAM[0], "species": species, "item": item, "ability": ability, "moves": moves}


# A team that shares no item, ability or move with the golden one, with its own exact numbers.
OPPONENT_TEAM = [
    _opponent(
        "Garchomp",
        "Life Orb",
        "Rough Skin",
        ["Earthquake", "Outrage", "Stealth Rock", "Rock Slide"],
    ),
    _opponent(
        "Clefable",
        "Choice Scarf",
        "Magic Guard",
        ["Moonblast", "Helping Hand", "Life Dew", "Coaching"],
    ),
    _opponent(
        "Talonflame", "Sharp Beak", "Gale Wings", ["Brave Bird", "Tailwind", "Flare Blitz", "Howl"]
    ),
    _opponent(
        "Arcanine",
        "Scope Lens",
        "Intimidate",
        ["Flamethrower", "Extreme Speed", "Counter", "Acupressure"],
    ),
]
EXACT_OPPONENT_HP = {"Garchomp": 183, "Clefable": 196, "Talonflame": 153, "Arcanine": 177}


def _asymmetric_channel() -> list[str]:
    """p1's view: its own golden team against OPPONENT_TEAM, who reveal very little."""
    preview = [f"|poke|p2|{s['species']}, L50, M|" for s in OPPONENT_TEAM]
    header = [
        line for line in HEADER if not line.startswith(("|poke|p2", "|teamsize|p2", "|switch|p2"))
    ]
    at = header.index("|teampreview|4")
    return [
        *header[:at],
        *preview,
        *header[at:-1],
        "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
        "|switch|p2b: Clefable|Clefable, L50, M|100/100",
        "|turn|1",
        "|move|p2a: Garchomp|Earthquake|p1a: Venusaur|[spread] p1a,p1b,p2b",
        "|-damage|p2b: Clefable|63/100",
        "|-damage|p2a: Garchomp|50/100g|[from] item: Life Orb",
        "|move|p2b: Clefable|Moonblast|p1b: Charizard",
        "|-unboost|p1b: Charizard|spa|1",
    ]


def test_a_different_opponent_team_stays_out_of_the_state() -> None:
    lines = _asymmetric_channel()
    state = reduce_channel("p1", lines, TEAM, VOCAB)
    opponent = state.to_dict()["opponent"]
    text = json.dumps(opponent)
    secrets = unrevealed_ids(OPPONENT_TEAM, "\n".join(lines))
    assert {
        "stealthrock",
        "rockslide",
        "outrage",
        "roughskin",
        "magicguard",
        "choicescarf",
    } <= secrets
    assert leaks(opponent, secrets) == []
    for secret in secrets:
        assert f'"{secret}"' not in text
    for exact in EXACT_OPPONENT_HP.values():
        assert str(exact) not in text
    garchomp, clefable = state.opponent.roster[0], state.opponent.roster[1]
    assert (garchomp.moves_revealed, garchomp.item) == (["earthquake"], "lifeorb")
    assert clefable.moves_revealed == ["moonblast"]
    assert (clefable.ability, clefable.item_known) == (None, False)


def test_negative_control_the_asymmetric_detector_fires_on_the_full_set() -> None:
    state = reduce_channel("p1", _asymmetric_channel(), TEAM, VOCAB).to_dict()["opponent"]
    state["roster"][1]["ability"] = "magicguard"
    state["roster"][0]["hp_num"] = EXACT_OPPONENT_HP["Garchomp"]
    found = leaks(state, unrevealed_ids(OPPONENT_TEAM, "\n".join(_asymmetric_channel())))
    assert "secret magicguard" in found
    assert "integer 183" in found


@pytest.mark.parametrize(
    "line",
    [
        "|split|p2",
        "|-damage|p2a: Venusaur|311/311",
        "|-heal|p2b: Charizard|250/311",
    ],
)
def test_negative_control_omniscient_lines_are_refused_not_reduced(line: str) -> None:
    with pytest.raises(OpponentSecretError):
        run(line)
