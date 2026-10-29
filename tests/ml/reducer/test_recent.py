"""Own-channel recent-history tracking (ADR-010 erratum A4 item 7, #1013).

Hand-written logs for each field, the fail-closed refusal, and the canary with negative controls.
The definitions agree with the world facts (`packages/public-state/src/world-facts.ts`) where they
overlap: `active_turns` counts `|turn|` lines since the switch-in, the Protect streak counts
consecutive turns of successful stalling uses, the last move leaves out a move another effect
called (`[from]`) but keeps `[from]lockedmove`, and a condition's start turn is the turn it began.
"""

from __future__ import annotations

import copy
import json

import pytest
from areneon_ml.reducer import reduce_channel
from areneon_ml.reducer.state import STATE_SCHEMA_VERSION

from .support import HEADER, PERCENT, TEAM, VOCAB, leaks, opp, own, run


def test_the_state_schema_version_names_the_recent_fields() -> None:
    state = run().to_dict()
    assert state["schema_version"] == STATE_SCHEMA_VERSION == 2
    assert {"elapsed"} <= set(state)
    assert "condition_turns" in state["own"]
    assert state["own"]["roster"][0]["recent"]["active_turns"] == 1


def test_last_move_target_turn_and_the_place_among_the_turns_actions() -> None:
    state = run(
        "|move|p2a: Venusaur|Earthquake|p1a: Venusaur",
        "|move|p1b: Charizard|Protect|p1b: Charizard",
        "|-singleturn|p1b: Charizard|Protect",
        "|cant|p1a: Venusaur|slp",
        "|move|p2b: Charizard|Quick Attack|p1a: Venusaur|[from]lockedmove",
    )
    foe, ally = opp(state, "Venusaur").recent, own(state, "Charizard").recent
    assert (foe.last_move, foe.last_target, foe.last_move_turn) == ("earthquake", "own:a", 1)
    assert (foe.last_action_turn, foe.last_action_order) == (1, 1)
    assert (ally.last_move, ally.last_target, ally.last_action_order) == ("protect", "own:b", 2)
    sleeper = own(state, "Venusaur").recent
    assert (sleeper.last_move, sleeper.last_action_order) == (None, 3)  # a `|cant|` is an action
    locked = opp(state, "Charizard").recent  # `[from]lockedmove` is the Pokemon's own move
    assert (locked.last_move, locked.last_action_order) == ("quickattack", 4)


def test_a_move_another_effect_called_is_not_the_pokemons_last_move() -> None:
    state = run(
        "|move|p2a: Venusaur|Earthquake|p1a: Venusaur|[from]move: Metronome",
        "|move|p2b: Charizard|Recover|p2b: Charizard|[still]",
    )
    assert opp(state, "Venusaur").recent.last_move is None
    assert opp(state, "Charizard").recent.last_target == "opponent:b"


def test_a_move_with_no_target_has_no_last_target() -> None:
    state = run("|move|p1a: Venusaur|Sunny Day|null")
    assert own(state, "Venusaur").recent.last_target is None


def test_the_order_of_action_restarts_each_turn() -> None:
    state = run(
        "|move|p2a: Venusaur|Recover|p2a: Venusaur",
        "|turn|2",
        "|move|p1a: Venusaur|Recover|p1a: Venusaur",
    )
    assert opp(state, "Venusaur").recent.last_action_turn == 1
    assert (
        own(state, "Venusaur").recent.last_action_turn,
        own(state, "Venusaur").recent.last_action_order,
    ) == (2, 1)


def test_hp_dealt_and_taken_last_turn_are_in_the_units_the_player_sees() -> None:
    lines = (
        "|move|p2a: Venusaur|Earthquake|p1a: Venusaur",
        "|-damage|p1a: Venusaur|120/159",
        "|move|p1a: Venusaur|Earthquake|p2a: Venusaur",
        "|-damage|p2a: Venusaur|60/100",
        "|-damage|p1b: Charizard|150/157|[from] psn",
    )
    mid = run(*lines)
    assert own(mid, "Venusaur").recent.hp_taken == 39  # this turn so far, exact
    state = run(*lines, "|turn|2")
    mine, theirs = own(state, "Venusaur").recent, opp(state, "Venusaur").recent
    assert (mine.hp_taken_last_turn, mine.hp_dealt_last_turn) == (
        39,
        40,
    )  # taken exact, dealt percent
    assert (theirs.hp_taken_last_turn, theirs.hp_dealt_last_turn) == (
        40,
        39,
    )  # taken percent, dealt exact
    assert (
        own(state, "Charizard").recent.hp_taken_last_turn == 7
    )  # residual damage is taken, dealt by nobody
    assert (mine.hp_taken, mine.hp_dealt) == (0, 0)
    later = run(*lines, "|turn|2", "|turn|3")
    assert own(later, "Venusaur").recent.hp_taken_last_turn == 0


def test_a_heal_and_a_tagged_damage_line_are_not_dealt() -> None:
    state = run(
        "|move|p1a: Venusaur|Giga Drain|p2a: Venusaur",
        "|-damage|p2a: Venusaur|70/100",
        "|-heal|p1a: Venusaur|159/159|[from] drain|[of] p2a: Venusaur",
        "|-damage|p1a: Venusaur|150/159|[from] item: Life Orb",
        "|turn|2",
    )
    mine = own(state, "Venusaur").recent
    assert (mine.hp_dealt_last_turn, mine.hp_taken_last_turn) == (30, 9)


def test_turns_since_switch_in() -> None:
    state = run("|turn|2", "|turn|3")
    assert own(state, "Venusaur").recent.active_turns == 3  # the lead is 1 at turn 1


def test_a_switch_in_restarts_the_count_and_a_switch_out_clears_the_history() -> None:
    lines = [
        *HEADER,
        "|move|p1a: Venusaur|Protect|p1a: Venusaur",
        "|-singleturn|p1a: Venusaur|Protect",
    ]
    state = reduce_channel("p1", [*lines, "|turn|2"], TEAM, VOCAB)
    assert own(state, "Venusaur").recent.protect_streak == 1
    assert own(state, "Venusaur").recent.last_move == "protect"
    swapped = reduce_channel(
        "p1", [*lines, "|turn|2", "|switch|p1a: Blastoise|Blastoise, L50, F|100/100"], TEAM, VOCAB
    )
    gone = own(swapped, "Venusaur").recent
    assert (gone.active_turns, gone.last_move, gone.protect_streak) == (0, None, 0)
    assert own(swapped, "Blastoise").recent.active_turns == 0
    later = reduce_channel(
        "p1",
        [*lines, "|turn|2", "|switch|p1a: Blastoise|Blastoise, L50, F|100/100", "|turn|3"],
        TEAM,
        VOCAB,
    )
    assert own(later, "Blastoise").recent.active_turns == 1


def _protect(name: str, ok: bool = True) -> list[str]:
    user = f"p1a: {name}"
    return [
        f"|move|{user}|Protect|{user}",
        f"|-singleturn|{user}|Protect" if ok else f"|-fail|{user}",
    ]


def test_the_protect_streak_counts_consecutive_turns_of_successful_stalling_moves() -> None:
    state = run(*_protect("Venusaur"), "|turn|2", *_protect("Venusaur"))
    assert own(state, "Venusaur").recent.protect_streak == 2
    failed = run(*_protect("Venusaur"), "|turn|2", *_protect("Venusaur", ok=False))
    assert own(failed, "Venusaur").recent.protect_streak == 0
    gap = run(*_protect("Venusaur"), "|turn|2", "|turn|3", *_protect("Venusaur"))
    assert own(gap, "Venusaur").recent.protect_streak == 1
    expired = run(*_protect("Venusaur"), "|turn|2", "|turn|3")
    assert own(expired, "Venusaur").recent.protect_streak == 0


def test_the_streak_ignores_a_called_stalling_move_and_other_pokemon() -> None:
    state = run(
        "|move|p1a: Venusaur|Protect|p1a: Venusaur|[from]move: Metronome",
        "|-singleturn|p1a: Venusaur|Protect",
    )
    assert own(state, "Venusaur").recent.protect_streak == 0


def test_a_stalling_move_with_no_stated_outcome_leaves_the_streak_alone() -> None:
    state = run("|move|p1a: Venusaur|Protect|p1a: Venusaur", "|turn|2")
    assert own(state, "Venusaur").recent.protect_streak == 0
    assert own(state, "Venusaur").recent.last_move == "protect"


def test_weather_terrain_and_field_conditions_count_turns_since_they_began() -> None:
    state = run(
        "|-weather|SunnyDay",
        "|-fieldstart|move: Grassy Terrain",
        "|-fieldstart|move: Trick Room",
        "|turn|2",
        "|-weather|SunnyDay|[upkeep]",
        "|turn|3",
    )
    elapsed = state.to_dict()["elapsed"]
    assert (elapsed["weather"], elapsed["terrain"]) == (2, 2)
    assert elapsed["field_conditions"] == {"trickroom": 2}
    ended = run(
        "|-weather|SunnyDay",
        "|turn|2",
        "|-weather|none",
        "|-fieldstart|move: Grassy Terrain",
        "|-fieldend|move: Grassy Terrain",
    )
    assert ended.to_dict()["elapsed"] == {"weather": None, "terrain": None, "field_conditions": {}}
    changed = run("|-weather|SunnyDay", "|turn|2", "|-weather|RainDance", "|turn|3")
    assert changed.to_dict()["elapsed"]["weather"] == 1


def test_side_conditions_count_turns_and_follow_a_swap() -> None:
    lines = ["|-sidestart|p1: Alice|move: Tailwind", "|turn|2", "|turn|3"]
    assert run(*lines).to_dict()["own"]["condition_turns"] == {"tailwind": 2}
    swapped = run(*lines, "|-swapsideconditions")
    assert swapped.to_dict()["opponent"]["condition_turns"] == {"tailwind": 2}
    assert swapped.to_dict()["own"]["condition_turns"] == {}
    layered = run("|-sidestart|p2: Bob|Spikes", "|turn|2", "|-sidestart|p2: Bob|Spikes", "|turn|3")
    assert layered.to_dict()["opponent"]["condition_turns"] == {
        "spikes": 2
    }  # the first layer's turn
    gone = run("|-sidestart|p2: Bob|Spikes", "|-sideend|p2: Bob|Spikes")
    assert gone.to_dict()["opponent"]["condition_turns"] == {}


def test_a_vocabulary_without_the_stall_table_refuses_a_move() -> None:
    bare = copy.copy(VOCAB)
    object.__setattr__(bare, "stall_moves", None)
    with pytest.raises(Exception, match="stall-moves"):
        reduce_channel("p1", [*HEADER, "|move|p1a: Venusaur|Recover|p1a: Venusaur"], TEAM, bare)


def test_nothing_hidden_enters_the_opponents_recent_history() -> None:
    state = run(
        "|move|p2a: Venusaur|Earthquake|p1a: Venusaur",
        "|-damage|p1a: Venusaur|120/159",
        "|move|p1a: Venusaur|Earthquake|p2a: Venusaur",
        "|-damage|p2a: Venusaur|60/100",
        "|turn|2",
    ).to_dict()
    side = state["opponent"]
    text = json.dumps([mon["recent"] for mon in side["roster"]])
    secrets = {"gigadrain", "sludgebomb", "sleeppowder", "leftovers"}
    assert leaks(side, secrets) == []
    for secret in secrets:
        assert secret not in text
    taken = [
        mon["recent"][key]
        for mon in side["roster"]
        for key in ("hp_taken_last_turn", "hp_taken_this_turn")
    ]
    assert all(value <= PERCENT for value in taken)  # the opponent's own HP is only ever percent


def test_negative_control_an_exact_opponent_hp_in_the_history_is_caught() -> None:
    side = run(
        "|move|p1a: Venusaur|Earthquake|p2a: Venusaur", "|-damage|p2a: Venusaur|60/100", "|turn|2"
    ).to_dict()["opponent"]
    assert leaks(side, set()) == []
    side["roster"][0]["recent"]["hp_taken_last_turn"] = (
        183  # an exact value where only percent is allowed
    )
    assert "integer 183" in leaks(side, set())
    side["roster"][0]["recent"]["last_move"] = "gigadrain"
    side["roster"][0]["moves_revealed"] = ["gigadrain"]
    assert "secret gigadrain" in leaks(side, {"gigadrain"})
