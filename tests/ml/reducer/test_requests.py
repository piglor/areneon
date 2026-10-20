"""Own-side exact values from the player's own `|request|` lines, and the key-set drift check."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.reducer import MalformedLineError, OpponentSecretError, UnknownFieldError
from areneon_ml.reducer import reducer as reducer_module

from .support import ROOT, own, run

_STATS = {"atk": 100, "def": 90, "spa": 80, "spd": 70, "spe": 60}


def _pokemon(name: str, condition: str, *, active: bool, moves: list[str]) -> dict[str, Any]:
    return {
        "ident": f"p1: {name}",
        "details": f"{name}, L50, F",
        "condition": condition,
        "active": active,
        "stats": _STATS,
        "moves": moves,
        "baseAbility": "overgrow",
        "item": "leftovers",
        "pokeball": "pokeball",
        "ability": "overgrow",
        "commanding": False,
        "reviving": False,
    }


def _body(**extra: Any) -> dict[str, Any]:
    side = {
        "name": "p1",
        "id": "p1",
        "pokemon": [
            _pokemon("Venusaur", "100/159 par", active=True, moves=["gigadrain", "protect"]),
            _pokemon("Charizard", "157/157", active=True, moves=["flamethrower"]),
            _pokemon("Pikachu", "0 fnt", active=False, moves=["thunderbolt"]),
        ],
    }
    return {"side": side, **extra}


def _request(body: dict[str, Any]) -> str:
    return "|request|" + json.dumps(body)


def _active() -> list[dict[str, Any]]:
    return [
        {
            "moves": [
                {"move": "Giga Drain", "id": "gigadrain", "pp": 7, "maxpp": 12, "disabled": False},
                {"move": "Protect", "id": "protect", "disabled": False},
            ],
            "canMegaEvoX": True,
        },
        {"moves": [{"move": "Flamethrower", "id": "flamethrower", "pp": 3, "maxpp": 16}]},
    ]


def test_a_move_request_updates_the_own_side() -> None:
    state = run(_request(_body(active=_active())))
    venusaur, charizard, pikachu = (
        own(state, "Venusaur"),
        own(state, "Charizard"),
        own(state, "Pikachu"),
    )
    assert state.request_kind == "move"
    assert (venusaur.hp_num, venusaur.hp_den, venusaur.status) == (100, 159, "par")
    assert venusaur.stats == _STATS
    assert venusaur.moves_own == ["gigadrain", "protect"]
    assert venusaur.pp == {"gigadrain": 7}
    assert (venusaur.can_mega_evo, charizard.can_mega_evo) == (True, False)
    assert charizard.pp == {"flamethrower": 3}
    assert (pikachu.hp_num, pikachu.fainted) == (0, True)
    brought = {mon.name: mon.brought for mon in state.own.roster}
    assert brought == {
        "Venusaur": True,
        "Charizard": True,
        "Blastoise": False,
        "Pikachu": True,
        "Snorlax": False,
        "Gengar": False,
    }


@pytest.mark.parametrize(
    ("extra", "kind"),
    [
        ({"wait": True}, "wait"),
        ({"forceSwitch": [True, False]}, "forceSwitch"),
        ({"teamPreview": True, "maxChosenTeamSize": 4}, "preview"),
    ],
)
def test_request_phases(extra: dict[str, Any], kind: str) -> None:
    state = run(_request(_body(**extra)))
    assert state.request_kind == kind
    # Only a request after Team Preview shows who was brought.
    assert (own(state, "Gengar").brought is False) == (kind != "preview")


def test_a_request_for_the_other_player_is_a_secret() -> None:
    body = _body(wait=True)
    body["side"]["id"] = "p2"
    with pytest.raises(OpponentSecretError):
        run(_request(body))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(surprise=True),
        lambda b: b["side"].update(surprise=True),
        lambda b: b["side"]["pokemon"][0].update(terastallized="Fire"),
        lambda b: b["side"]["pokemon"][0].update(teraType="Fire"),
        lambda b: b.update(active=[{"moves": [], "canTerastallize": "Fire"}]),
        lambda b: b.update(active=[{"moves": [{"move": "x", "id": "gigadrain", "boom": 1}]}]),
    ],
)
def test_unknown_request_keys_raise(mutate: Any) -> None:
    body = _body(wait=True)
    mutate(body)
    with pytest.raises(UnknownFieldError):
        run(_request(body))


def test_a_request_must_match_the_own_team() -> None:
    body = _body(wait=True)
    body["side"]["pokemon"][0]["ident"] = "p1: Stranger"
    with pytest.raises(MalformedLineError):
        run(_request(body))
    body = _body(active=_active() + _active() + _active())
    with pytest.raises(MalformedLineError):
        run(_request(body))
    with pytest.raises(MalformedLineError):
        run(_request(_body()))


def test_request_key_sets_match_the_closed_engine_schema() -> None:
    schema = json.loads(
        (ROOT / "packages/schema/contracts/v1/engine-request.schema.json").read_text()
    )
    defs = schema["$defs"]
    tera = {"teraType", "terastallized", "canTerastallize"}  # no rule: they fail closed
    assert set(schema["properties"]) == reducer_module._REQUEST_KEYS
    assert set(defs["SideRequestData"]["properties"]) == reducer_module._SIDE_KEYS
    assert (
        set(defs["PokemonSwitchRequestData"]["properties"]) - tera == reducer_module._POKEMON_KEYS
    )
    assert set(defs["PokemonMoveRequestData"]["properties"]) - tera == reducer_module._ACTIVE_KEYS
    assert set(defs["MoveRequestData"]["properties"]) == reducer_module._MOVE_KEYS
