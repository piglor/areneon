"""The engine-sent team sheet in the reducer (ADR-027 section 5, #1150)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from areneon_ml.reducer import (
    MalformedLineError,
    UnknownFieldError,
    VocabularyError,
    reduce_channel,
)

from .support import FORMAT_DIR, ROOT, VOCAB

OPEN_DIR = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmcbo3" / "open-regime"
GAMES = [json.loads(path.read_text()) for path in sorted(OPEN_DIR.glob("o[0-9]*.json"))]

TEAM: list[dict[str, Any]] = [
    {
        "name": "",
        "species": "Altaria",
        "item": "Altarianite",
        "ability": "Natural Cure",
        "moves": ["Substitute", "Feather Dance", "Hyper Beam", "Fire Spin"],
        "nature": "Serious",
        "gender": "F",
        "level": 50,
    },
    {
        "name": "",
        "species": "Delphox",
        "item": "Pidgeotite",
        "ability": "Blaze",
        "moves": ["Psych Up", "Thunder Punch", "Sunny Day", "Blast Burn"],
        "nature": "Serious",
        "gender": "",
        "level": 50,
    },
]
BLANK = ["Altaria", "", "", "NaturalCure", "Substitute", "", "", "", "", "", "", ""]


def mon(*fields: str) -> str:
    return "|".join(fields)


SHEET = (
    "|showteam|p2|Altaria||Altarianite|NaturalCure|Substitute,FeatherDance,HyperBeam,FireSpin"
    "|Serious||F|||50|]Delphox||Pidgeotite|Blaze|PsychUp,ThunderPunch,SunnyDay,BlastBurn"
    "|Serious||M|||50|"
)
OWN_SHEET = SHEET.replace("|p2|", "|p1|")
PREVIEW = [
    "|gametype|doubles",
    "|player|p1|p1||",
    "|player|p2|p2||",
    "|clearpoke",
    "|poke|p1|Altaria, L50, F|",
    "|poke|p1|Delphox, L50, M|",
    "|poke|p2|Altaria, L50, F|",
    "|poke|p2|Delphox, L50, M|",
    "|teampreview|2",
]


def edited(field: int, value: str, sheet: str = SHEET) -> str:
    head, *others = sheet.split("]")
    parts = head.split("|")
    parts[3 + field] = value
    return "]".join(["|".join(parts), *others])


def read(*lines: str, team: Any = TEAM, header: Any = PREVIEW) -> Any:
    return reduce_channel("p1", [*header, *lines], team, VOCAB)


@pytest.mark.parametrize("game", GAMES, ids=[g["id"] for g in GAMES])
@pytest.mark.parametrize("player", ["p1", "p2"])
def test_open_regime_channels_hold_one_sheet_per_side(game: dict[str, Any], player: str) -> None:
    state = reduce_channel(player, game["channels"][player]["lines"], game["teams"][player], VOCAB)
    assert state.own.sheet is not None
    assert len(state.own.sheet) == 6
    assert state.opponent.sheet is not None
    assert len(state.opponent.sheet) == 6
    dumped = state.to_dict()
    assert [m["species"] for m in dumped["own"]["sheet"]] == [
        VOCAB.species(s["species"]) for s in game["teams"][player]
    ]


def test_the_sheet_is_kept_apart_from_what_the_battle_reveals() -> None:
    state = read(OWN_SHEET, SHEET)
    assert state.opponent.sheet is not None
    assert state.opponent.sheet[0].to_dict() == {
        "species": "altaria",
        "item": "altarianite",
        "ability": "naturalcure",
        "moves": ["substitute", "featherdance", "hyperbeam", "firespin"],
        "nature": "serious",
        "gender": "F",
        "level": 50,
    }
    assert [(m.item, m.ability, m.moves_revealed) for m in state.opponent.roster] == [
        (None, None, []),
        (None, None, []),
    ]


def test_a_closed_game_has_no_sheet_key() -> None:
    golden = json.loads((FORMAT_DIR / "reduced" / "p1.json").read_text())["state"]
    assert "sheet" not in golden["own"]
    assert "sheet" not in golden["opponent"]
    state = read().to_dict()
    assert "sheet" not in state["own"]
    assert "sheet" not in state["opponent"]


def test_missing_item_nature_gender_and_level_read_as_unset() -> None:
    second_mon = mon("Delphox", "", "Leftovers", "Blaze", "Psychup", "", "", "M", "", "", "", "")
    sheet = f"|showteam|p2|{mon(*BLANK)}]{second_mon}"
    opponent_sheet = read(sheet).opponent.sheet
    assert opponent_sheet is not None
    first, second = opponent_sheet
    assert (first.item, first.nature, first.gender, first.level) == (None, None, None, 100)
    assert (second.item, second.gender, second.level) == ("leftovers", "M", 100)


def test_a_sheet_before_the_preview_has_nothing_to_match() -> None:
    state = read(SHEET, header=PREVIEW[:3])
    assert state.opponent.sheet is not None


@pytest.mark.parametrize(
    ("lines", "error", "text"),
    [
        ([SHEET, SHEET], MalformedLineError, "second team sheet"),
        ([OWN_SHEET, OWN_SHEET], MalformedLineError, "second team sheet"),
        (["|start", SHEET], MalformedLineError, "after |start"),
        ([SHEET.replace("|p2|", "|p3|")], UnknownFieldError, "unsupported side"),
        (["|showteam|p2"], MalformedLineError, "needs at least 2 fields"),
    ],
)
def test_sheet_lines_that_are_refused(lines: list[str], error: type, text: str) -> None:
    with pytest.raises(error, match=text):
        read(*lines)


@pytest.mark.parametrize(
    ("sheet", "error", "text"),
    [
        (edited(1, "Birdy"), UnknownFieldError, "no nickname"),
        (edited(6, "252,,,,,4"), UnknownFieldError, "no Stat Points"),
        (edited(8, ",,,30,30,"), UnknownFieldError, "no Stat Points"),
        (edited(9, "S"), UnknownFieldError, "no shiny flag"),
        (edited(11, ",,,,,tera"), UnknownFieldError, "no shiny flag"),
        (edited(7, "X"), UnknownFieldError, "unknown sheet gender"),
        (SHEET + "|extra", MalformedLineError, "has 12 fields"),
        (SHEET + "]", MalformedLineError, "has 12 fields"),
        (edited(0, ""), MalformedLineError, "lacks its species"),
        (edited(3, ""), MalformedLineError, "lacks its species"),
        (edited(4, ""), MalformedLineError, "lacks its species"),
        (edited(4, "Substitute,,Protect"), MalformedLineError, "empty move"),
        (edited(5, "Ser1ous"), MalformedLineError, "nature is a name"),
        (edited(10, "0"), MalformedLineError, "level is a number"),
        (edited(10, "101"), MalformedLineError, "level is a number"),
        (edited(10, "5x"), MalformedLineError, "level is a number"),
        (edited(0, "Notamon"), VocabularyError, "unknown species"),
        (edited(2, "Notanitem"), VocabularyError, "unknown item"),
        (edited(3, "Notability"), VocabularyError, "unknown ability"),
        (edited(4, "Substitute,Notamove"), VocabularyError, "unknown move"),
    ],
)
def test_a_sheet_pokemon_the_grammar_does_not_take(sheet: str, error: type, text: str) -> None:
    with pytest.raises(error, match=text):
        read(sheet, header=PREVIEW[:3])


def test_a_sheet_lists_between_one_and_24_pokemon() -> None:
    many = "|showteam|p2|" + "]".join(mon(*BLANK) for _ in range(25))
    with pytest.raises(MalformedLineError, match="between one and 24"):
        read(many, header=PREVIEW[:3])


def test_the_opponent_sheet_must_be_the_previews() -> None:
    with pytest.raises(MalformedLineError, match="not the team preview's"):
        read(SHEET.split("]", maxsplit=1)[0])
    swapped = SHEET.replace("Altaria||", "Arbok||", 1)
    with pytest.raises(MalformedLineError, match="not the team preview's"):
        read(swapped)


def test_the_own_sheet_equals_the_own_team_without_points_ivs_and_nickname() -> None:
    assert read(OWN_SHEET).own.sheet is not None
    nicknamed = copy.deepcopy(TEAM)
    nicknamed[0]["name"] = "Birdy"  # a nickname is not on the sheet and not compared
    nicknamed[1]["gender"] = "M"
    assert read(OWN_SHEET, team=nicknamed).own.sheet is not None


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t[0].update(item="Leftovers"),
        lambda t: t[0].update(item=""),
        lambda t: t[0].update(ability="Cloud Nine"),
        lambda t: t[0].update(moves=["Substitute", "Feather Dance", "Hyper Beam", "Protect"]),
        lambda t: t[0].update(moves=["Feather Dance", "Substitute", "Hyper Beam", "Fire Spin"]),
        lambda t: t[0].update(level=100),
        lambda t: t[0].update(nature="Adamant"),
        lambda t: t[0].update(gender="M"),
    ],
)
def test_an_own_sheet_that_differs_from_the_own_team_is_refused(change: Any) -> None:
    team = copy.deepcopy(TEAM)
    change(team)
    with pytest.raises(MalformedLineError, match="not the own team"):
        read(OWN_SHEET, team=team)


def test_a_genderless_own_pokemon_shows_no_gender() -> None:
    team = copy.deepcopy(TEAM)
    team[0]["gender"] = "N"
    sheet = edited(7, "", OWN_SHEET)
    assert read(sheet, team=team).own.sheet is not None
    with pytest.raises(MalformedLineError, match="not the own team"):
        read(OWN_SHEET, team=team)


@pytest.mark.parametrize("change", [lambda t: t[1].update(species="Arbok"), lambda t: t.pop()])
def test_an_own_sheet_of_another_team_is_refused(change: Any) -> None:
    team = copy.deepcopy(TEAM)
    change(team)
    with pytest.raises(MalformedLineError, match="not the own team"):
        read(OWN_SHEET, team=team, header=PREVIEW[:3])
