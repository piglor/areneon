"""Unit tests per line type: what each rule changes in the reduced state."""

from __future__ import annotations

import pytest

from .support import opp, own, run


def test_header_lines_set_the_battle_facts() -> None:
    state = run()
    assert state.format_name == "[Gen 9 Champions] VGC 2026 Reg M-C"
    assert state.gametype == "doubles"
    assert (state.own.username, state.opponent.username) == ("Alice", "Bob")
    assert state.own.teamsize == state.opponent.teamsize == 4
    assert state.started
    assert state.turn == 1
    assert state.own.slots == {"a": 0, "b": 1}
    assert [mon.base_species for mon in state.opponent.roster][:2] == ["venusaur", "charizard"]
    assert opp(state, "Venusaur").brought is True
    unseen = state.opponent.roster[4]
    assert (unseen.base_species, unseen.name, unseen.brought) == ("snorlax", None, None)


def test_perspective_flips_for_p2() -> None:
    state = run(player="p2")
    assert state.own.player == "p2"
    assert state.opponent.player == "p1"
    assert state.opponent.slots == {"a": 0, "b": 1}


@pytest.mark.parametrize("gametype", ["singles", "triples"])
def test_other_gametypes_set_their_slots(gametype: str) -> None:
    state = run(f"|gametype|{gametype}", header=False)
    assert list(state.own.slots) == list("a" if gametype == "singles" else "abc")


def test_switch_places_and_resets_the_leaver() -> None:
    state = run(
        "|-boost|p2a: Venusaur|atk|2",
        "|-start|p2a: Venusaur|confusion",
        "|switch|p2a: Blastoise|Blastoise, L50, M|100/100",
    )
    venusaur, blastoise = opp(state, "Venusaur"), opp(state, "Blastoise")
    assert venusaur.active_slot is None
    assert venusaur.boosts["atk"] == 0
    assert venusaur.volatiles == set()
    assert (blastoise.active_slot, blastoise.brought) == ("a", True)
    assert state.opponent.slots["a"] == state.opponent.roster.index(blastoise)


def test_drag_is_a_switch() -> None:
    state = run("|drag|p2b: Pikachu|Pikachu, L50, M|55/100 par")
    pikachu = opp(state, "Pikachu")
    assert (pikachu.active_slot, pikachu.hp_num, pikachu.status) == ("b", 55, "par")


def test_own_switch_in_keeps_the_exact_hp() -> None:
    state = run("|switch|p1a: Blastoise|Blastoise, L50, M|158/158")
    blastoise = own(state, "Blastoise")
    assert (blastoise.hp_num, blastoise.hp_den, blastoise.hp_fraction) == (158, 158, 1.0)
    assert blastoise.item == "mysticwater"


def test_move_reveals_the_move_once_unless_called_by_another_effect() -> None:
    state = run(
        "|move|p2b: Charizard|Flamethrower|p1a: Venusaur",
        "|move|p2b: Charizard|Flamethrower|p1a: Venusaur|[spread] p1a,p1b",
        "|move|p2a: Venusaur|Protect|p2a: Venusaur|[still]",
        "|move|p2a: Venusaur|Giga Drain||[from] lockedmove",
        "|move|p2a: Venusaur|Sludge Bomb|",
    )
    assert opp(state, "Charizard").moves_revealed == ["flamethrower"]
    assert opp(state, "Venusaur").moves_revealed == ["protect", "sludgebomb"]


def test_damage_heal_and_sethp_move_the_percentage() -> None:
    state = run(
        "|-damage|p2a: Venusaur|40/100",
        "|-heal|p2a: Venusaur|65/100",
        "|-sethp|p2b: Charizard|50/100",
    )
    assert opp(state, "Venusaur").hp_fraction == 0.65
    assert opp(state, "Charizard").hp_fraction == 0.5


def test_damage_to_zero_then_faint_empties_the_slot() -> None:
    state = run("|-damage|p2a: Venusaur|0 fnt", "|faint|p2a: Venusaur")
    venusaur = opp(state, "Venusaur")
    assert (venusaur.hp_num, venusaur.fainted, venusaur.active_slot, venusaur.status) == (
        0,
        True,
        None,
        None,
    )
    assert state.opponent.slots["a"] is None


def test_hp_with_status_sets_and_clears_it() -> None:
    state = run("|-damage|p2a: Venusaur|70/100 brn")
    assert opp(state, "Venusaur").status == "brn"
    state = run("|-damage|p2a: Venusaur|70/100 brn", "|-heal|p2a: Venusaur|80/100")
    assert opp(state, "Venusaur").status is None


def test_status_lines() -> None:
    state = run(
        "|-status|p2a: Venusaur|par",
        "|-status|p2b: Charizard|slp",
        "|-curestatus|p2b: Charizard|slp",
        "|-status|p1a: Venusaur|psn",
        "|-cureteam|p1a: Venusaur",
    )
    assert opp(state, "Venusaur").status == "par"
    assert opp(state, "Charizard").status is None
    assert own(state, "Venusaur").status is None


def test_item_is_revealed_by_its_own_effect_only_without_of() -> None:
    state = run(
        "|-heal|p2a: Venusaur|90/100|[from] item: Leftovers",
        "|-damage|p2b: Charizard|90/100|[from] item: Life Orb",
        "|-damage|p1a: Venusaur|100/159|[from] item: Rocky Helmet|[of] p2b: Charizard",
        "|-damage|p1b: Charizard|100/157|[from] drain|[of] p2b: Charizard",
        "|-heal|p2b: Charizard|95/100|[from] ability: Water Absorb",
    )
    assert opp(state, "Venusaur").item == "leftovers"
    assert opp(state, "Charizard").item == "lifeorb"
    assert opp(state, "Charizard").ability == "waterabsorb"
    assert opp(state, "Charizard").item_known


def test_item_ability_lines() -> None:
    state = run(
        "|-item|p2a: Venusaur|Sitrus Berry|[from] move: Trick|[of] p1a: Venusaur",
        "|-enditem|p2a: Venusaur|Sitrus Berry|[eat]",
        "|-ability|p2b: Charizard|Intimidate|boost",
        "|-ability|p2b: Charizard|Trace|[from] ability: Trace|[of] p1a: Venusaur",
        "|-endability|p2b: Charizard",
    )
    venusaur, charizard = opp(state, "Venusaur"), opp(state, "Charizard")
    assert (venusaur.item, venusaur.item_known, venusaur.item_lost) == (None, True, True)
    assert (charizard.ability, charizard.ability_suppressed) == ("trace", True)
    state = run("|-item|p2a: Venusaur|Leftovers")
    assert (opp(state, "Venusaur").item, opp(state, "Venusaur").item_lost) == ("leftovers", False)


def test_boost_lines() -> None:
    state = run(
        "|-boost|p2a: Venusaur|atk|2",
        "|-unboost|p2a: Venusaur|spe|1",
        "|-setboost|p2b: Charizard|atk|6|[from] move: Belly Drum",
        "|-boost|p2b: Charizard|accuracy|1",
    )
    assert opp(state, "Venusaur").boosts["atk"] == 2
    assert opp(state, "Venusaur").boosts["spe"] == -1
    assert opp(state, "Charizard").boosts["atk"] == 6
    assert opp(state, "Charizard").boosts["accuracy"] == 1


def test_boost_manipulation_lines() -> None:
    state = run(
        "|-boost|p2a: Venusaur|atk|2",
        "|-unboost|p2a: Venusaur|def|1",
        "|-swapboost|p2a: Venusaur|p2b: Charizard|atk, def",
        "|-boost|p2b: Charizard|spe|3",
        "|-copyboost|p2b: Charizard|p1a: Venusaur",
        "|-invertboost|p2b: Charizard",
    )
    venusaur, charizard = opp(state, "Venusaur"), opp(state, "Charizard")
    assert (venusaur.boosts["atk"], venusaur.boosts["def"]) == (0, 0)
    assert (charizard.boosts["atk"], charizard.boosts["def"], charizard.boosts["spe"]) == (
        -2,
        1,
        -3,
    )
    assert own(state, "Venusaur").boosts["atk"] == 2


def test_swapboost_without_a_stat_list_swaps_everything() -> None:
    state = run("|-boost|p2a: Venusaur|spa|2", "|-swapboost|p2a: Venusaur|p2b: Charizard")
    assert opp(state, "Charizard").boosts["spa"] == 2
    assert opp(state, "Venusaur").boosts["spa"] == 0


def test_clearing_boosts() -> None:
    state = run(
        "|-boost|p2a: Venusaur|atk|2",
        "|-unboost|p2a: Venusaur|def|2",
        "|-clearpositiveboost|p2a: Venusaur|p1a: Venusaur|move: Spectral Thief",
    )
    assert opp(state, "Venusaur").boosts["atk"] == 0
    assert opp(state, "Venusaur").boosts["def"] == -2
    state = run(
        "|-boost|p2a: Venusaur|atk|2",
        "|-unboost|p2a: Venusaur|def|2",
        "|-clearnegativeboost|p2a: Venusaur",
    )
    assert (opp(state, "Venusaur").boosts["atk"], opp(state, "Venusaur").boosts["def"]) == (2, 0)
    state = run("|-boost|p2a: Venusaur|atk|2", "|-clearboost|p2a: Venusaur")
    assert opp(state, "Venusaur").boosts["atk"] == 0
    state = run("|-boost|p2a: Venusaur|atk|2", "|-boost|p1a: Venusaur|atk|2", "|-clearallboost")
    assert opp(state, "Venusaur").boosts["atk"] == own(state, "Venusaur").boosts["atk"] == 0


def test_weather_and_terrain() -> None:
    state = run(
        "|-weather|RainDance", "|-weather|RainDance|[upkeep]", "|-fieldstart|move: Electric Terrain"
    )
    assert (state.weather, state.terrain) == ("raindance", "electricterrain")
    state = run(
        "|-weather|SunnyDay",
        "|-weather|none",
        "|-fieldstart|move: Grassy Terrain",
        "|-fieldend|move: Psychic Terrain",
        "|-fieldend|move: Grassy Terrain",
    )
    assert (state.weather, state.terrain) == (None, None)


def test_field_conditions() -> None:
    state = run("|-fieldstart|move: Trick Room|[of] p1a: Venusaur", "|-fieldstart|move: Gravity")
    assert state.field_conditions == {"trickroom", "gravity"}
    state = run(
        "|-fieldstart|move: Trick Room",
        "|-fieldend|move: Trick Room",
        "|-fieldactivate|move: Pay Day",
    )
    assert state.field_conditions == set()


def test_side_conditions() -> None:
    state = run(
        "|-sidestart|p2: Bob|move: Tailwind",
        "|-sidestart|p1: Alice|move: Spikes",
        "|-sidestart|p1: Alice|Spikes",
        "|-sideend|p2: Bob|move: Tailwind",
        "|-sidestart|p2: Bob|move: Reflect",
    )
    assert state.own.conditions == {"spikes": 2}
    assert state.opponent.conditions == {"reflect": 1}
    swapped = run("|-sidestart|p2: Bob|move: Tailwind", "|-swapsideconditions")
    assert swapped.own.conditions == {"tailwind": 1}
    assert swapped.opponent.conditions == {}


def test_volatiles() -> None:
    state = run(
        "|-start|p2a: Venusaur|confusion",
        "|-start|p2a: Venusaur|move: Taunt",
        "|-start|p2a: Venusaur|Substitute",
        "|-start|p2b: Charizard|ability: Flash Fire",
        "|-start|p2b: Charizard|typechange|Fire/Flying|[from] ability: Protean",
        "|-end|p2a: Venusaur|Substitute",
        "|-end|p2a: Venusaur|move: Encore",
    )
    assert opp(state, "Venusaur").volatiles == {"confusion", "taunt"}
    assert opp(state, "Charizard").volatiles == {"flashfire", "typechange"}
    assert opp(state, "Charizard").ability == "flashfire"


def test_mega_forme_and_transform_lines() -> None:
    state = run(
        "|-mega|p2b: Charizard|Charizardite X",
        "|detailschange|p2b: Charizard|Charizard-Mega-X, L50, F|100/100",
        "|-formechange|p2a: Venusaur|Venusaur-Mega|80/100",
        "|-transform|p1a: Venusaur|Charizard",
        "|detailschange|p1b: Charizard|Charizard-Mega-X, L50, F",
    )
    charizard = opp(state, "Charizard")
    assert (charizard.mega, charizard.mega_stone, charizard.species) == (
        True,
        "charizarditex",
        "charizardmegax",
    )
    assert charizard.base_species == "charizard"
    assert (opp(state, "Venusaur").temp_species, opp(state, "Venusaur").hp_num) == (
        "venusaurmega",
        80,
    )
    assert own(state, "Venusaur").temp_species == "charizard"
    assert own(state, "Charizard").species == "charizardmegax"


def test_swap_exchanges_active_positions() -> None:
    state = run("|swap|p2a: Venusaur|1")
    assert state.opponent.slots == {"a": 1, "b": 0}
    assert (opp(state, "Venusaur").active_slot, opp(state, "Charizard").active_slot) == ("b", "a")


def test_turn_and_results() -> None:
    assert run("|turn|2").turn == 2
    assert run("|win|Alice").winner == "own"
    assert run("|win|Bob").winner == "opponent"
    assert run("|tie").winner == "tie"


@pytest.mark.parametrize(
    "line",
    [
        "|",
        "|upkeep",
        "|rated",
        "|rated|Tournament",
        "|rule|Species Clause: Limit one of each Pokemon",
        "|uhtml|otsrequest|<button>x</button>",
        "|uhtmlchange|otsrequest|",
        "|cant|p2a: Venusaur|par",
        "|cant|p2a: Venusaur|Disable|Flamethrower",
        "|-crit|p1a: Venusaur",
        "|-supereffective|p1a: Venusaur|1",
        "|-resisted|p1a: Venusaur",
        "|-immune|p1a: Venusaur",
        "|-fail|p1a: Venusaur|move: Protect",
        "|-block|p1a: Venusaur|move: Protect|Tackle|p2a: Venusaur",
        "|-notarget|p1a: Venusaur",
        "|-miss|p2a: Venusaur|p1a: Venusaur",
        "|-hint|Fake Out only works on the first turn.",
        "|-message|[Sleep Clause] fired",
        "|-activate|p1a: Venusaur|move: Protect",
        "|-center",
        "|-combine",
        "|-waiting|p1a: Venusaur|p1b: Charizard",
        "|-prepare|p1a: Venusaur|Solar Beam",
        "|-mustrecharge|p1a: Venusaur",
        "|-nothing",
        "|-hitcount|p1a: Venusaur|2",
        "|-singlemove|p1a: Venusaur|Destiny Bond",
        "|-singleturn|p1a: Venusaur|Protect",
        "|-fieldactivate|move: Pay Day",
        "|-ohko",
    ],
)
def test_lines_without_state_effect_leave_the_state_alone(line: str) -> None:
    assert run(line).to_dict() == run().to_dict()


def test_bench_style_ids_and_shiny_and_zero_hp_forms() -> None:
    state = run(
        "|-damage|p2: Venusaur|30/100", "|switch|p2a: Blastoise|Blastoise, L50, M, shiny|0/100 fnt"
    )
    assert opp(state, "Venusaur").hp_num == 30
    blastoise = opp(state, "Blastoise")
    assert (blastoise.shiny, blastoise.hp_num, blastoise.fainted, blastoise.status) == (
        True,
        0,
        True,
        None,
    )


def test_a_from_tag_that_names_no_item_or_ability_reveals_nothing() -> None:
    state = run(
        "|-damage|p2a: Venusaur|90/100|[from] psn",
        "|-damage|p2b: Charizard|90/100|[from] move: Flail",
    )
    assert (opp(state, "Venusaur").item, opp(state, "Charizard").item) == (None, None)
