"""The ADR-008 canary for the reducer: no opponent secret in any reduced state.

The secrets are everything in the OTHER player's team that the channel never
names: unrevealed items, abilities and moves, and every exact stat or HP. The
checks scan only the `opponent` half of the state. Each has a negative control:
the same scan over a state that does hold secrets must fire, so a silent
detector cannot pass the suite.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.reducer import OpponentSecretError, reduce_channel

from .support import FORMAT_DIR, TEAM, VOCAB, run

PERCENT = 100


def _channel(player: str) -> list[str]:
    return list(
        json.loads((FORMAT_DIR / "channels" / f"{player}.json").read_text())["channel"]["lines"]
    )


def _integers(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, dict):
        return [n for item in value.values() for n in _integers(item)]
    if isinstance(value, list):
        return [n for item in value for n in _integers(item)]
    return []


def _unrevealed_ids(team: list[dict[str, Any]], channel_text: str) -> set[str]:
    """Item, ability and move ids of `team` whose names the channel never prints."""
    text = channel_text.lower()
    secrets: set[str] = set()
    for pokemon_set in team:
        for name in [pokemon_set["item"], pokemon_set["ability"], *pokemon_set["moves"]]:
            if name.lower() not in text:
                secrets.add(name.replace(" ", "").lower())
    return secrets


def _leaks(opponent: dict[str, Any], secret_ids: set[str]) -> list[str]:
    """What in the opponent half of a state is an exact value or an unrevealed id."""
    found = []
    for mon in opponent["roster"]:
        if mon["stats"] is not None or mon["moves_own"] is not None or mon["pp"]:
            found.append(f"exact own-side fields on {mon['name']}")
        known = {mon["item"], mon["ability"], *mon["moves_revealed"], *mon["volatiles"]}
        found.extend(f"secret {name}" for name in known & secret_ids)
    found.extend(f"integer {n}" for n in _integers(opponent) if n > PERCENT)
    return found


@pytest.mark.parametrize("player", ["p1", "p2"])
def test_no_opponent_secret_at_any_request_point_of_the_golden_channels(player: str) -> None:
    lines = _channel(player)
    for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
        prefix = lines[:end]
        state = reduce_channel(player, prefix, TEAM, VOCAB)
        secrets = _unrevealed_ids(TEAM, "\n".join(prefix))
        assert secrets  # the golden team keeps most of its set hidden at every point
        assert _leaks(state.to_dict()["opponent"], secrets) == []


@pytest.mark.parametrize("player", ["p1", "p2"])
def test_negative_control_the_own_half_of_the_same_state_trips_the_detector(player: str) -> None:
    lines = _channel(player)
    state = reduce_channel(player, lines, TEAM, VOCAB).to_dict()
    secrets = _unrevealed_ids(TEAM, "")
    assert _leaks(state["own"], secrets), (
        "the detector must fire on a state that holds exact values"
    )


def test_negative_control_a_contaminated_opponent_half_is_caught() -> None:
    state = run("|move|p2a: Venusaur|Protect|p2a: Venusaur").to_dict()
    secrets = {"leftovers", "gigadrain"}
    assert _leaks(state["opponent"], secrets) == []
    mon = state["opponent"]["roster"][0]
    contaminated = json.loads(json.dumps(state["opponent"]))
    contaminated["roster"][0]["item"] = "leftovers"
    contaminated["roster"][0]["hp_num"] = 159
    contaminated["roster"][0]["stats"] = {"atk": 102}
    leaks = _leaks(contaminated, secrets)
    assert "secret leftovers" in leaks
    assert "integer 159" in leaks
    assert any("exact own-side fields" in leak for leak in leaks)
    assert mon["item"] is None


def test_a_hidden_opponent_set_stays_out_of_the_state() -> None:
    """Asymmetric teams: the opponent holds values p1 never sees; the channel names only some."""
    secret_item, secret_ability, secret_move = "Rocky Helmet", "Unaware", "Giga Drain"
    state = run(
        "|move|p2a: Venusaur|Protect|p2a: Venusaur",
        "|-damage|p2b: Charizard|63/100",
        "|-heal|p2b: Charizard|70/100|[from] item: Leftovers",
    )
    text = json.dumps(state.to_dict()["opponent"])
    for hidden in (secret_item, secret_ability, secret_move):
        assert hidden.replace(" ", "").lower() not in text
    assert "311" not in text  # an exact opponent maximum HP never appears
    assert next(m for m in state.opponent.roster if m.name == "Charizard").item == "leftovers"


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
