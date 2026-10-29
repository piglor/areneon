"""Illusion: `|replace|` and the disguises before it (ADR-010 §3, R10, #851).

The channel shows an Illusion user as the Pokemon it copies until the disguise breaks. The
reducer binds each switch-in to what it is shown as; `|replace|` (or, for the player's own
Pokemon, the next request) says who it was. Opponent hidden information must not move before
that line: the shared header here gives the opponent's six as a team preview does, and the
Illusion user is the one never named (Gengar), shown as Blastoise.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from areneon_ml.reducer import (
    BattleState,
    IllusionError,
    MalformedLineError,
    PokemonState,
    reduce_channel,
)

from .support import HEADER, HEADER_P2, TEAM, VOCAB, opp, own, run

SHOWN_AS_BLASTOISE = "|switch|p2a: Blastoise|Blastoise, L50, M|100/100"
REPLACE = ("|replace|p2a: Gengar|Gengar, L50, M", "|-end|p2a: Gengar|Illusion")


def _entry(state: BattleState, species: str) -> PokemonState:
    return next(m for m in state.opponent.roster if m.base_species == species and not m.phantom)


# -- the opponent's Illusion user --------------------------------------------------------


def test_a_type_change_seen_under_the_disguise_moves_to_the_real_pokemon() -> None:
    state = run(SHOWN_AS_BLASTOISE, "|-start|p2a: Blastoise|typechange|Fire", *REPLACE)
    assert _entry(state, "gengar").types == ["fire"]
    assert _entry(state, "blastoise").types is None


def test_what_was_seen_under_the_disguise_moves_to_the_real_pokemon() -> None:
    shown = [
        SHOWN_AS_BLASTOISE,
        "|-damage|p2a: Blastoise|60/100",
        "|-boost|p2a: Blastoise|atk|2",
        "|-status|p2a: Blastoise|par",
        "|move|p2a: Blastoise|Protect|p2a: Blastoise",
        "|-enditem|p2a: Blastoise|Sitrus Berry",
        "|-endability|p2a: Blastoise",
        "|-mega|p2a: Blastoise|Blastoise|Blastoisinite",
    ]
    state = run(*shown, *REPLACE)
    gengar, blastoise = _entry(state, "gengar"), _entry(state, "blastoise")
    assert (gengar.name, gengar.species, gengar.level, gengar.gender) == (
        "Gengar",
        "gengar",
        50,
        "M",
    )
    assert (gengar.hp_num, gengar.hp_den, gengar.status) == (60, 100, "par")
    assert gengar.boosts["atk"] == 2
    assert gengar.moves_revealed == ["protect"]
    assert (gengar.item, gengar.item_lost, gengar.ability_suppressed) == (None, True, True)
    assert gengar.ability == "illusion"
    assert gengar.mega
    assert gengar.brought
    assert gengar.active_slot == "a"
    assert state.opponent.slots["a"] == state.opponent.roster.index(gengar)
    # The Pokemon it was shown as is as if it had never appeared.
    assert blastoise.name is None
    assert blastoise.brought is None
    assert blastoise.active_slot is None
    assert blastoise.hp_num is None
    assert blastoise.moves_revealed == []
    assert not blastoise.mega
    assert blastoise.boosts == {key: 0 for key in blastoise.boosts}
    assert blastoise.status is None
    # Later lines name the real Pokemon.
    later = run(
        *shown, *REPLACE, "|-damage|p2a: Gengar|30/100", "|move|p2a: Gengar|Protect|p2a: Gengar"
    )
    assert opp(later, "Gengar").hp_num == 30


def test_nothing_of_the_real_pokemon_is_in_the_state_before_the_replace() -> None:
    lines = [
        SHOWN_AS_BLASTOISE,
        "|-damage|p2a: Blastoise|60/100",
        "|move|p2a: Blastoise|Protect|p2a: Blastoise",
    ]
    before = run(*lines)
    gengar = _entry(before, "gengar")
    pristine = _entry(run(), "gengar")
    assert gengar.to_dict() == pristine.to_dict()  # the roster entry the preview gave, untouched
    assert before.opponent.slots["a"] == before.opponent.roster.index(_entry(before, "blastoise"))
    text = json.dumps(before.opponent.to_dict()["slots"]) + "".join(lines)
    assert "Gengar" not in text
    assert "gengar" not in text
    after = run(*lines, *REPLACE)
    assert _entry(after, "gengar").to_dict() != pristine.to_dict()


def test_a_replace_may_carry_the_hit_points_and_is_the_channels_own_for_both_players() -> None:
    state = run(SHOWN_AS_BLASTOISE, "|replace|p2a: Gengar|Gengar, L50, M|40/100")
    assert _entry(state, "gengar").hp_num == 40
    as_p2 = run(
        "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
        "|replace|p1a: Gengar|Gengar, L50, M",
        player="p2",
    )
    assert next(m for m in as_p2.opponent.roster if m.name == "Gengar").species == "gengar"


def test_the_pokemon_it_was_shown_as_keeps_what_was_seen_before_the_disguise() -> None:
    # Blastoise (real) was seen first, then the Illusion user is shown as Blastoise.
    state = run(
        SHOWN_AS_BLASTOISE,
        "|move|p2a: Blastoise|Surf|p2b: Charizard",
        "|-damage|p2a: Blastoise|70/100",
        "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2a: Blastoise|Blastoise, L50, M|90/100",
        "|-boost|p2a: Blastoise|spa|1",
        "|move|p2a: Blastoise|Protect|p2a: Blastoise",
        *REPLACE,
    )
    blastoise, gengar = _entry(state, "blastoise"), _entry(state, "gengar")
    assert blastoise.moves_revealed == ["surf"]
    assert blastoise.hp_num == 70
    assert blastoise.boosts["spa"] == 0
    assert blastoise.active_slot is None
    assert gengar.moves_revealed == ["protect"]
    assert gengar.boosts["spa"] == 1
    assert gengar.hp_num == 90


def test_a_drag_that_takes_the_place_of_the_real_pokemon_puts_it_back_out_of_play() -> None:
    state = run(
        SHOWN_AS_BLASTOISE,
        "|-boost|p2a: Blastoise|atk|1",
        "|drag|p2a: Blastoise|Blastoise, L50, M|80/100",  # the same name, dragged in again
        *REPLACE,
    )
    assert _entry(state, "blastoise").active_slot is None
    assert _entry(state, "blastoise").boosts["atk"] == 0
    assert state.opponent.slots["a"] == state.opponent.roster.index(_entry(state, "gengar"))


def test_a_swap_keeps_the_disguise_with_its_position() -> None:
    state = run(
        SHOWN_AS_BLASTOISE,
        "|swap|p2a: Blastoise|1",
        "|-boost|p2b: Blastoise|atk|1",
        "|replace|p2b: Gengar|Gengar, L50, M",
    )
    assert _entry(state, "gengar").boosts["atk"] == 1
    assert _entry(state, "gengar").active_slot == "b"
    assert opp(state, "Charizard").active_slot == "a"


def test_a_swap_of_a_pokemon_that_is_not_where_the_line_says_is_refused() -> None:
    with pytest.raises(MalformedLineError):
        run(SHOWN_AS_BLASTOISE, "|swap|p2a: Venusaur|1")


def test_a_pokemon_that_fainted_under_its_disguise_is_not_confused_with_the_real_one() -> None:
    # The Illusion user fainted shown as Blastoise (no replace); Blastoise itself shows up later.
    state = run(
        SHOWN_AS_BLASTOISE,
        "|-damage|p2a: Blastoise|0 fnt",
        "|faint|p2a: Blastoise",
        "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2a: Blastoise|Blastoise, L50, M|100/100",
    )
    dead = [m for m in state.opponent.roster if m.phantom]
    assert len(dead) == 1
    assert dead[0].fainted
    assert dead[0].base_species == "blastoise"
    assert dead[0].to_dict()["phantom"] is True
    live = _entry(state, "blastoise")
    assert not live.fainted
    assert live.hp_num == 100
    assert live.active_slot == "a"
    assert "phantom" not in live.to_dict()


def test_a_fainted_own_pokemon_may_switch_in_again_after_a_revival() -> None:
    # Own entries come from the team and the request tells their state: nothing is added.
    state = run(
        "|-damage|p1a: Venusaur|0 fnt",
        "|faint|p1a: Venusaur",
        "|switch|p1a: Pikachu|Pikachu, L50, F|120/120",
    )
    assert not any(m.phantom for m in state.own.roster)


# -- refusals -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((SHOWN_AS_BLASTOISE, "|replace|p2a: Venusaur|Gengar, L50, M"), "not the roster's"),
        (("|replace|p2a: Venusaur|Venusaur, L50, M",), "no disguise"),  # nothing was bound
        ((SHOWN_AS_BLASTOISE, "|replace|p2a: Absol|Absol, L50, M"), "outside the roster"),
    ],
)
def test_a_replace_the_roster_cannot_explain_is_refused(
    lines: tuple[str, ...], message: str
) -> None:
    with pytest.raises(IllusionError):
        run(*lines)


def test_a_replace_at_an_empty_position_is_malformed() -> None:
    with pytest.raises(MalformedLineError, match="nobody holds"):
        run("|faint|p2a: Venusaur", "|replace|p2a: Gengar|Gengar, L50, M")
    with pytest.raises(MalformedLineError, match="nobody holds"):
        run("|replace|p2: Gengar|Gengar, L50, M")


def test_a_replace_that_fits_two_pokemon_is_refused() -> None:
    lines = [*HEADER]
    lines.insert(lines.index("|clearpoke") + 1, "|poke|p2|Gengar, L50, F|")  # a second Gengar
    with pytest.raises(IllusionError, match="more than one"):
        reduce_channel("p1", [*lines, SHOWN_AS_BLASTOISE, *REPLACE], TEAM, VOCAB)


def test_a_replace_naming_a_pokemon_already_in_play_is_refused() -> None:
    with pytest.raises(IllusionError, match="already in play"):
        run(
            "|switch|p2a: Gengar|Gengar, L50, M|100/100",
            "|switch|p2b: Blastoise|Blastoise, L50, M|100/100",
            "|replace|p2b: Gengar|Gengar, L50, M",
        )


def test_a_disguise_the_roster_did_not_hold_is_refused() -> None:
    # A species the preview never listed: the switch adds an entry, so there is nothing to put back.
    lines = [line for line in HEADER if "Pikachu" not in line or not line.startswith("|poke|p2")]
    lines[lines.index("|teamsize|p2|4")] = "|teamsize|p2|6"
    with pytest.raises(IllusionError, match="not on the roster"):
        reduce_channel(
            "p1",
            [*lines, "|switch|p2a: Pikachu|Pikachu, L50, M|100/100", *REPLACE],
            TEAM,
            VOCAB,
        )


# -- the same Pokemon shown twice --------------------------------------------------------

TWICE = (SHOWN_AS_BLASTOISE, "|switch|p2b: Blastoise|Blastoise, L50, M|100/100")


def test_a_pokemon_shown_twice_gets_an_entry_of_its_own_until_one_breaks() -> None:
    state = run(*TWICE, "|-damage|p2a: Blastoise|50/100", "|-damage|p2b: Blastoise|70/100")
    blastoises = [m for m in state.opponent.roster if m.base_species == "blastoise"]
    assert [(m.hp_num, m.phantom, m.active_slot) for m in blastoises] == [
        (50, False, "a"),
        (70, True, "b"),
    ]
    assert len(state.opponent.roster) == 7
    assert state.opponent.roster[-1].to_dict()["phantom"] is True
    assert state.opponent.slots["b"] == 6


def test_the_replace_names_the_first_one_shown() -> None:
    state = run(
        *TWICE, "|-damage|p2a: Blastoise|50/100", "|-damage|p2b: Blastoise|70/100", *REPLACE
    )
    assert len(state.opponent.roster) == 6
    assert not any(m.phantom for m in state.opponent.roster)
    gengar, blastoise = _entry(state, "gengar"), _entry(state, "blastoise")
    assert (gengar.hp_num, gengar.active_slot) == (50, "a")
    assert (blastoise.hp_num, blastoise.active_slot, blastoise.name) == (70, "b", "Blastoise")
    assert state.opponent.slots["b"] == state.opponent.roster.index(blastoise)
    assert run(*TWICE, *REPLACE, "|-damage|p2b: Blastoise|10/100")  # the name still works


def test_the_replace_names_the_added_one() -> None:
    state = run(
        *TWICE,
        "|-damage|p2a: Blastoise|50/100",
        "|-damage|p2b: Blastoise|70/100",
        "|replace|p2b: Gengar|Gengar, L50, M",
    )
    assert len(state.opponent.roster) == 6
    assert not any(m.phantom for m in state.opponent.roster)
    gengar, blastoise = _entry(state, "gengar"), _entry(state, "blastoise")
    assert (gengar.hp_num, gengar.active_slot) == (70, "b")
    assert (blastoise.hp_num, blastoise.active_slot) == (50, "a")


def test_two_pairs_at_once_are_resolved_one_at_a_time() -> None:
    state = run(
        *TWICE,
        "|-ability|p2b: Blastoise|Torrent",
        "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2a: Pikachu|Pikachu, L50, M|100/100",
        "|switch|p2b: Pikachu|Pikachu, L50, M|90/100",
        "|-damage|p2b: Pikachu|45/100",
        "|replace|p2b: Gengar|Gengar, L50, M",
    )
    assert [m.phantom for m in state.opponent.roster].count(True) == 1
    pikachu = next(
        m for m in state.opponent.roster if m.base_species == "pikachu" and not m.phantom
    )
    assert pikachu.hp_num == 100
    assert pikachu.active_slot == "a"
    assert len([m for m in state.opponent.roster if m.base_species == "gengar"]) == 1
    # The Illusion user was shown second: the first Blastoise is real, and its ability is known.
    assert _entry(state, "blastoise").ability is None
    assert _entry(state, "gengar").hp_num == 45


def test_the_ability_the_added_one_showed_is_the_real_ones() -> None:
    state = run(*TWICE, "|-ability|p2b: Blastoise|Torrent", *REPLACE)
    assert _entry(state, "blastoise").ability == "torrent"


def test_the_one_that_left_and_came_back_is_the_added_entry() -> None:
    state = run(
        *TWICE,
        "|switch|p2b: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2b: Blastoise|Blastoise, L50, M|65/100",
        "|replace|p2a: Gengar|Gengar, L50, M",
    )
    blastoise = _entry(state, "blastoise")
    assert not any(m.phantom for m in state.opponent.roster)
    assert (blastoise.hp_num, blastoise.active_slot) == (65, "b")


def test_the_replace_after_the_other_one_left_still_finds_the_real_one() -> None:
    state = run(
        *TWICE,
        "|-damage|p2b: Blastoise|20/100",
        "|switch|p2b: Snorlax|Snorlax, L50, M|100/100",  # the added one leaves
        "|replace|p2a: Gengar|Gengar, L50, M",
    )
    blastoise = _entry(state, "blastoise")
    assert (blastoise.hp_num, blastoise.active_slot) == (20, None)
    assert not any(m.phantom for m in state.opponent.roster)


def test_a_pair_that_never_breaks_is_kept_and_a_fainted_one_stays_unnamed() -> None:
    state = run(*TWICE, "|-damage|p2b: Blastoise|0 fnt", "|faint|p2b: Blastoise")
    assert len(state.opponent.roster) == 7
    assert [m.fainted for m in state.opponent.roster if m.base_species == "blastoise"] == [
        False,
        True,
    ]


def test_the_pair_is_forgotten_with_the_battle_of_a_new_preview() -> None:
    state = run(*TWICE, "|clearpoke")
    assert state.opponent.roster == []


def test_the_added_one_may_come_back() -> None:
    state = run(
        *TWICE,
        "|switch|p2b: Snorlax|Snorlax, L50, M|100/100",
        "|switch|p2b: Blastoise|Blastoise, L50, M|100/100",
    )
    assert len(state.opponent.roster) == 7  # the same added entry, not another


def test_a_third_copy_is_refused() -> None:
    triples = [line.replace("doubles", "triples") for line in HEADER[: HEADER.index("|start")]]
    lines = [
        *triples,
        "|start",
        *(f"|switch|p2{slot}: Blastoise|Blastoise, L50, M|100/100" for slot in "abc"),
    ]
    with pytest.raises(IllusionError, match="three positions"):
        reduce_channel("p1", lines, TEAM, VOCAB)


# -- the player's own Illusion user ------------------------------------------------------

OWN_TEAM = copy.deepcopy(TEAM)
OWN_TEAM[5]["ability"] = "Illusion"  # Gengar holds Illusion


def _own_pokemon(
    name: str, *, active: bool, condition: str = "150/150", ability: str = "static"
) -> dict[str, Any]:
    return {
        "ident": f"p1: {name}",
        "details": f"{name}, L50, F",
        "condition": condition,
        "active": active,
        "stats": {"atk": 100, "def": 90, "spa": 80, "spd": 70, "spe": 60},
        "moves": ["protect"],
        "baseAbility": ability,
        "item": "leftovers",
        "pokeball": "pokeball",
        "ability": ability,
        "commanding": False,
        "reviving": False,
    }


def _own_request(*pokemon: dict[str, Any]) -> str:
    body = {
        "side": {"name": "p1", "id": "p1", "pokemon": list(pokemon)},
        "active": [{"moves": [{"move": "Protect", "id": "protect"}]}] * 2,
    }
    return "|request|" + json.dumps(body)


def _gengar(active: bool = True, condition: str = "120/150") -> dict[str, Any]:
    return _own_pokemon("Gengar", active=active, condition=condition, ability="illusion")


def _own(*lines: str) -> BattleState:
    return reduce_channel("p1", [*HEADER, *lines], OWN_TEAM, VOCAB)


OWN_SHOWN = (
    "|switch|p1a: Blastoise|Blastoise, L50, F|120/150",
    "|-boost|p1a: Blastoise|atk|1",
    "|move|p1a: Blastoise|Protect|p1a: Blastoise",
)


def test_an_own_replace_moves_what_was_seen_to_the_real_pokemon() -> None:
    state = _own(*OWN_SHOWN, "|replace|p1a: Gengar|Gengar, L50, F", "|-end|p1a: Gengar|Illusion")
    gengar, blastoise = own(state, "Gengar"), own(state, "Blastoise")
    assert (gengar.hp_num, gengar.boosts["atk"], gengar.moves_revealed) == (120, 1, ["protect"])
    assert gengar.active_slot == "a"
    assert gengar.ability == "illusion"
    assert blastoise.boosts["atk"] == 0
    assert blastoise.moves_revealed == []
    assert blastoise.active_slot is None
    assert not any(m.phantom for m in state.own.roster)


def test_an_own_request_finds_the_illusion_user_before_its_replace() -> None:
    request = _own_request(
        _gengar(), _own_pokemon("Charizard", active=True), _own_pokemon("Blastoise", active=False)
    )
    state = _own(
        *OWN_SHOWN, request, "|-boost|p1a: Blastoise|spa|1"
    )  # the disguise's name still works
    gengar, blastoise = own(state, "Gengar"), own(state, "Blastoise")
    assert gengar.boosts == {**gengar.boosts, "atk": 1, "spa": 1}
    assert gengar.active_slot == "a"
    assert blastoise.boosts["atk"] == 0
    assert blastoise.active_slot is None
    assert state.own.slots["a"] == state.own.roster.index(gengar)
    # Its replace line then names the real one.
    done = _own(
        *OWN_SHOWN, request, "|replace|p1a: Gengar|Gengar, L50, F", "|-damage|p1a: Gengar|100/150"
    )
    assert own(done, "Gengar").hp_num == 100


def test_an_own_replace_after_a_request_must_name_the_pokemon_the_request_found() -> None:
    request = _own_request(
        _gengar(), _own_pokemon("Charizard", active=True), _own_pokemon("Blastoise", active=False)
    )
    with pytest.raises(IllusionError, match="no disguise"):
        _own(*OWN_SHOWN, request, "|replace|p1a: Charizard|Charizard, L50, F")


def test_a_request_that_shows_the_pokemon_as_shown_changes_nothing() -> None:
    request = _own_request(
        _own_pokemon("Blastoise", active=True),
        _own_pokemon("Charizard", active=True),
        _gengar(active=False),
    )
    state = _own(*OWN_SHOWN, request)
    assert own(state, "Blastoise").boosts["atk"] == 1
    assert own(state, "Blastoise").active_slot == "a"
    assert own(state, "Gengar").boosts["atk"] == 0


def test_a_request_with_another_pokemon_that_is_no_illusion_user_changes_nothing() -> None:
    # A mismatch no Illusion explains is left to the log, as before this ticket.
    request = _own_request(
        _own_pokemon("Pikachu", active=True), _own_pokemon("Charizard", active=True)
    )
    state = _own(*OWN_SHOWN, request)
    assert own(state, "Blastoise").boosts["atk"] == 1


def test_a_request_with_the_shown_pokemon_out_of_play_drops_its_disguise() -> None:
    request = _own_request(
        _own_pokemon("Blastoise", active=False), _own_pokemon("Charizard", active=True)
    )
    state = _own(*OWN_SHOWN, request)
    assert state.request_kind == "move"
    with pytest.raises(IllusionError, match="no disguise"):
        _own(*OWN_SHOWN, request, "|replace|p1a: Gengar|Gengar, L50, F")


def test_an_own_pokemon_shown_twice_is_settled_by_the_request_either_way() -> None:
    both = (
        "|switch|p1a: Blastoise|Blastoise, L50, F|150/150",
        "|switch|p1b: Blastoise|Blastoise, L50, F|120/150",
    )
    # The first one shown is the Illusion user.
    first = _own(
        *both,
        "|-boost|p1a: Blastoise|atk|2",
        "|-boost|p1b: Blastoise|def|1",
        _own_request(
            _gengar(),
            _own_pokemon("Blastoise", active=True),
            _own_pokemon("Charizard", active=False),
        ),
    )
    gengar, blastoise = own(first, "Gengar"), own(first, "Blastoise")
    assert (gengar.boosts["atk"], gengar.active_slot) == (2, "a")
    assert (blastoise.boosts["def"], blastoise.active_slot) == (1, "b")
    assert not any(m.phantom for m in first.own.roster)
    assert len(first.own.roster) == 6
    # The second one shown is the Illusion user.
    second = _own(
        *both,
        "|-boost|p1a: Blastoise|atk|2",
        "|-boost|p1b: Blastoise|def|1",
        _own_request(
            _own_pokemon("Blastoise", active=True),
            _gengar(),
            _own_pokemon("Charizard", active=False),
        ),
    )
    gengar, blastoise = own(second, "Gengar"), own(second, "Blastoise")
    assert (gengar.boosts["def"], gengar.active_slot) == (1, "b")
    assert (blastoise.boosts["atk"], blastoise.active_slot) == (2, "a")
    assert not any(m.phantom for m in second.own.roster)


def test_an_own_pair_a_request_cannot_settle_is_refused() -> None:
    both = (
        "|switch|p1a: Blastoise|Blastoise, L50, F|150/150",
        "|switch|p1b: Blastoise|Blastoise, L50, F|120/150",
    )
    plain = _own_request(
        _own_pokemon("Blastoise", active=True), _own_pokemon("Charizard", active=True)
    )
    with pytest.raises(IllusionError, match="cannot tell"):
        _own(*both, plain)
    # A request that shows the same Pokemon in both of the positions settles neither.
    with pytest.raises(IllusionError, match="cannot tell"):
        _own(
            *both,
            _own_request(
                _own_pokemon("Blastoise", active=True), _own_pokemon("Blastoise", active=True)
            ),
        )


def test_a_preview_request_settles_nothing() -> None:
    body = {
        "teamPreview": True,
        "maxChosenTeamSize": 4,
        "side": {"name": "p1", "id": "p1", "pokemon": [_own_pokemon("Venusaur", active=True)]},
    }
    state = reduce_channel(
        "p1",
        [*HEADER[: HEADER.index("|teamsize|p1|4")], "|request|" + json.dumps(body)],
        OWN_TEAM,
        VOCAB,
    )
    assert state.request_kind == "preview"


def test_p2_sees_the_same_channel_as_its_own_illusion_user() -> None:
    state = reduce_channel(
        "p2",
        [
            *HEADER_P2,
            "|switch|p2a: Blastoise|Blastoise, L50, M|150/150",
            "|-boost|p2a: Blastoise|atk|1",
            "|replace|p2a: Gengar|Gengar, L50, M",
        ],
        OWN_TEAM,
        VOCAB,
    )
    assert own(state, "Gengar").boosts["atk"] == 1


def test_an_own_illusion_user_that_left_play_is_found_by_the_pair_and_the_team() -> None:
    lines = (
        "|switch|p1a: Blastoise|Blastoise, L50, F|82/144",  # Gengar, shown as Blastoise
        "|-boost|p1a: Blastoise|atk|2",
        "|move|p1a: Blastoise|Protect|p1a: Blastoise",
        "|switch|p1b: Blastoise|Blastoise, L50, F|150/150",  # the real one, next to it
        "|-boost|p1b: Blastoise|def|1",
        "|switch|p1a: Pikachu|Pikachu, L50, F|120/120",  # the first one leaves
    )
    request = _own_request(
        _own_pokemon("Pikachu", active=True),
        _own_pokemon("Blastoise", active=True),
        _gengar(active=False, condition="82/144"),
    )
    state = _own(*lines, request)
    gengar, blastoise = own(state, "Gengar"), own(state, "Blastoise")
    assert gengar.moves_revealed == ["protect"]
    assert gengar.active_slot is None
    assert (blastoise.boosts["def"], blastoise.active_slot) == (1, "b")
    assert not any(m.phantom for m in state.own.roster)


@pytest.mark.parametrize("what", ["two users", "twin gone", "another there"])
def test_an_own_pair_the_request_leaves_open_is_refused(what: str) -> None:
    team = copy.deepcopy(OWN_TEAM)
    if what == "two users":
        team[4]["ability"] = "Illusion"  # Snorlax too
    lines = [
        "|switch|p1a: Blastoise|Blastoise, L50, F|82/144",
        "|switch|p1b: Blastoise|Blastoise, L50, F|150/150",
        "|switch|p1a: Pikachu|Pikachu, L50, F|120/120",
    ]
    if what == "twin gone":
        lines.append("|switch|p1b: Charizard|Charizard, L50, F|157/157")
    third = _own_pokemon("Charizard" if what == "another there" else "Blastoise", active=True)
    request = _own_request(_own_pokemon("Pikachu", active=True), third, _gengar(active=False))
    with pytest.raises(IllusionError, match="cannot tell"):
        reduce_channel("p1", [*HEADER, *lines, request], team, VOCAB)
