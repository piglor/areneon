"""Shared fixtures for the reducer tests: the committed golden format and a small battle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from areneon_ml.reducer import BattleState, PokemonState, Vocabulary, reduce_channel

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
