"""Shared fixtures for the reducer tests: the committed golden format and a small battle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from areneon_ml.reducer import BattleState, PokemonState, Vocabulary, reduce_channel
from areneon_ml.reducer.vocab import to_id

ROOT = Path(__file__).resolve().parents[3]
FORMAT_DIR = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc"
VOCAB = Vocabulary.from_engine_data(json.loads((FORMAT_DIR / "engine-data.json").read_text()))
TEAM: list[dict[str, Any]] = json.loads((FORMAT_DIR / "synthetic-team.json").read_text())

_SPECIES = ("Venusaur", "Charizard", "Blastoise", "Pikachu", "Snorlax", "Gengar")

HEADER = [
    "|gametype|doubles",
    "|player|p1|Alice||",
    "|player|p2|Bob||",
    "|gen|9",
    "|tier|[Gen 9 Champions] VGC 2026 Reg M-C",
    "|clearpoke",
    *(f"|poke|p1|{name}, L50, F|" for name in _SPECIES),
    *(f"|poke|p2|{name}, L50, M|" for name in _SPECIES),
    "|teampreview|4",
    "|teamsize|p1|4",
    "|teamsize|p2|4",
    "|start",
    "|switch|p1a: Venusaur|Venusaur, L50, F|159/159",
    "|switch|p1b: Charizard|Charizard, L50, F|157/157",
    "|switch|p2a: Venusaur|Venusaur, L50, M|100/100",
    "|switch|p2b: Charizard|Charizard, L50, F|100/100",
    "|turn|1",
]
# The same battle as p2 sees it: its own Pokemon exact, Alice's as percentages.
HEADER_P2 = [
    "|switch|p1a: Venusaur|Venusaur, L50, F|100/100"
    if line.startswith("|switch|p1a")
    else "|switch|p1b: Charizard|Charizard, L50, F|100/100"
    if line.startswith("|switch|p1b")
    else "|switch|p2a: Venusaur|Venusaur, L50, M|159/159"
    if line.startswith("|switch|p2a")
    else "|switch|p2b: Charizard|Charizard, L50, F|157/157"
    if line.startswith("|switch|p2b")
    else line
    for line in HEADER
]


def run(*lines: str, player: str = "p1", header: bool = True) -> BattleState:
    """Reduce the shared header and `lines` for `player` (p1 sees Alice's side as own)."""
    base = (HEADER if player == "p1" else HEADER_P2) if header else []
    return reduce_channel(player, [*base, *lines], TEAM, VOCAB)


def opp(state: BattleState, name: str) -> PokemonState:
    return next(mon for mon in state.opponent.roster if mon.name == name)


def own(state: BattleState, name: str) -> PokemonState:
    return next(mon for mon in state.own.roster if mon.name == name)


PERCENT = 100


def integers(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, dict):
        return [n for item in value.values() for n in integers(item)]
    if isinstance(value, list):
        return [n for item in value for n in integers(item)]
    return []


def unrevealed_ids(team: list[dict[str, Any]], channel_text: str) -> set[str]:
    """Item, ability and move ids of `team` whose names the channel never prints."""
    text = channel_text.lower()
    secrets: set[str] = set()
    for pokemon_set in team:
        for name in [pokemon_set["item"], pokemon_set["ability"], *pokemon_set["moves"]]:
            if name.lower() not in text:
                secrets.add(name.replace(" ", "").lower())
    return secrets


def unattributed(opponent: dict[str, Any], lines: list[str], opponent_id: str) -> list[str]:
    """Revealed opponent items, abilities and moves that no line about the opponent names.

    A value must come from a line that mentions one of the opponent's Pokemon; a value
    only the opponent's team holds and no such line prints is a leak.
    """
    about = "\n".join(
        to_id(line)
        for line in lines
        if any(f"{opponent_id}{slot}: " in line for slot in ("a", "b", "c", ""))
    )
    found = []
    for mon in opponent["roster"]:
        for value in {mon["item"], mon["ability"], *mon["moves_revealed"]} - {None}:
            if value not in about:
                found.append(f"{value} on {mon['name']}")
    return found


def leaks(opponent: dict[str, Any], secret_ids: set[str]) -> list[str]:
    """What in the opponent half of a state is an exact value or an unrevealed id."""
    found = []
    for mon in opponent["roster"]:
        if mon["stats"] is not None or mon["moves_own"] is not None or mon["pp"]:
            found.append(f"exact own-side fields on {mon['name']}")
        known = {mon["item"], mon["ability"], *mon["moves_revealed"], *mon["volatiles"]}
        found.extend(f"secret {name}" for name in known & secret_ids)
    found.extend(f"integer {n}" for n in integers(opponent) if n > PERCENT)
    return found
