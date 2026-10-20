"""Fail-closed tests: every unknown or forbidden input raises a typed error, never a skip."""

from __future__ import annotations

import pytest
from areneon_ml.reducer import (
    MalformedLineError,
    OpponentSecretError,
    ReducerError,
    UnknownFieldError,
    UnknownLineError,
    VocabularyError,
    reduce_channel,
)

from .support import HEADER, TEAM, VOCAB, run


@pytest.mark.parametrize(
    "line",
    [
        "|c|Bob|hello",
        "|-terastallize|p2a: Venusaur|Fire",
        "|replace|p2a: Venusaur|Zoroark, L50, M|100/100",
        "|-zpower|p2a: Venusaur",
        "|inactive|Alice has 30 seconds left.",
        "|t:|1700000000",
        "|error|[Invalid choice] nope",
        "|somethingnew|p2a: Venusaur",
    ],
)
def test_unknown_line_types_raise(line: str) -> None:
    with pytest.raises(UnknownLineError) as error:
        run(line)
    assert error.value.line == line
    assert line in str(error.value)


@pytest.mark.parametrize(
    "line",
    [
        "|-damage|p2a: Venusaur|50/100|[bogus] x",
        "|-supereffective|p1a: Venusaur|1|2",
        "|-boost|p2a: Venusaur|atk|1|2",
        "|-boost|p2a: Venusaur|luck|1",
        "|-weather|RainDance|x|[upkeep]",
        "|-ability|p2a: Venusaur|Intimidate|surprise",
        "|poke|p2|Venusaur, L50, M|unknown",
        "|switch|p2a: Venusaur|Venusaur, L50, M, tera:Fire|100/100",
        "|gametype|multi",
        "|player|p3|Carol||",
        "|-fieldstart|item: Leftovers",
        "|-sidestart|p2: Bob|item: Leftovers",
        "|upkeep|extra",
        "|switch|p2a: Stranger|Pikachu, L50, M|100/100|extra",
    ],
)
def test_unknown_fields_raise(line: str) -> None:
    with pytest.raises(UnknownFieldError):
        run(line)


@pytest.mark.parametrize(
    "line",
    [
        "no pipe",
        "|turn|soon",
        "|turn",
        "|-damage|p2a: Venusaur|lots",
        "|-damage|p2a: Venusaur|10",
        "|-damage|p2a: Venusaur|50/100 fnt",
        "|-damage|p2a: Venusaur|120/100",
        "|-damage|p1a: Venusaur|1/0",
        "|-damage|Venusaur|50/100",
        "|-damage|p2a: Nobody|50/100",
        "|-boost|p2a: Venusaur|atk|7",
        "|-boost|p2a: Venusaur|atk|many",
        "|-damage|p2a: Venusaur|50/100|[from] x|[from] y",
        "|-damage|p2a: Venusaur|50/100|[from] x|y",
        "|switch|p2: Venusaur|Venusaur, L50, M|100/100",
        "|switch|p2c: Venusaur|Venusaur, L50, M|100/100",
        "|win|Carol",
        "|-sidestart|Bob|move: Tailwind",
        "|swap|p2a: Venusaur|9",
        "|swap|p2a: Charizard|1",
        "|swap|p2: Venusaur|1",
        "|request|{not json",
        "|request|{}",
        "|request|[]",
        '|request|{"side": 3}',
        "|move|p2a: Venusaur",
    ],
)
def test_malformed_lines_raise(line: str) -> None:
    with pytest.raises(MalformedLineError):
        run(line)


@pytest.mark.parametrize(
    "line",
    [
        "|poke|p2|Missingno, L50, M|",
        "|move|p2a: Venusaur|Sketchy Move|p1a: Venusaur",
        "|-status|p2a: Venusaur|zzz",
        "|-item|p2a: Venusaur|Mystery Berry",
        "|-mega|p2a: Venusaur|Mystery Stone",
        "|-weather|Smog",
        "|-start|p2a: Venusaur|protosynthesisatk",
        "|-start|p2a: Venusaur|move: Nonexistent",
        "|-fieldstart|move: Unknown Room",
        "|-formechange|p2a: Venusaur|Nothing-Mega",
    ],
)
def test_unknown_names_raise(line: str) -> None:
    with pytest.raises(VocabularyError):
        run(line)


def test_refusals_share_one_base_and_carry_the_line() -> None:
    with pytest.raises(ReducerError) as error:
        run("|-weather|Smog")
    assert error.value.line == "|-weather|Smog"
    assert str(ReducerError("plain")) == "plain"


@pytest.mark.parametrize(
    "line",
    [
        "|split|p2",
        "|-damage|p2a: Venusaur|120/159",
        "|-heal|p1a: Venusaur|50/157",
        "|switch|p2a: Blastoise|Blastoise, L50, M|158/158",
        '|request|{"wait":true,"side":{"name":"p2","id":"p2","pokemon":[]}}',
    ],
)
def test_opponent_secrets_raise(line: str) -> None:
    # p1's own Pokemon (p1a: Venusaur) at 120/159 is fine; the opponent's is not.
    if line.startswith("|-heal|p1a"):
        run(line)
        return
    with pytest.raises(OpponentSecretError):
        run(line)


def test_a_pokemon_the_log_never_introduced_is_refused() -> None:
    with pytest.raises(MalformedLineError):
        run("|-damage|p2a: Gengar|50/100")


def test_opponent_beyond_team_size_is_refused() -> None:
    lines = [
        "|teamsize|p2|2",
        "|switch|p2a: Venusaur|Venusaur, L50, M|100/100",
        "|switch|p2b: Charizard|Charizard, L50, F|100/100",
        "|switch|p2a: Blastoise|Blastoise, L50, M|100/100",
    ]
    with pytest.raises(MalformedLineError):
        reduce_channel("p1", HEADER[:6] + lines, TEAM, VOCAB)


def test_preview_pokemon_outside_the_own_team_is_refused() -> None:
    with pytest.raises(MalformedLineError):
        run("|poke|p1|Ditto, L50, F|", header=False)


def test_own_team_must_be_complete_and_unambiguous() -> None:
    with pytest.raises(MalformedLineError):
        reduce_channel("p1", [], [{"species": "Venusaur"}], VOCAB)
    with pytest.raises(MalformedLineError):
        reduce_channel("p1", [], [TEAM[0], TEAM[0]], VOCAB)
    with pytest.raises(MalformedLineError):
        reduce_channel("p3", [], TEAM, VOCAB)
    with pytest.raises(VocabularyError):
        reduce_channel("p1", [], [{**TEAM[0], "item": "Mystery Berry"}], VOCAB)


def test_a_switch_without_a_known_name_for_the_own_side_is_refused() -> None:
    with pytest.raises(MalformedLineError):
        run("|switch|p1a: Stranger|Pikachu, L50, F|114/114")
