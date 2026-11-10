"""The history facts of the reduced state (ADR-022 erratum E-22.5.2, slice H1): `HistoryTracker`.

Hand-written lines for every fact, reset and refusal, the same cases as the TS module's tests
(`packages/public-state/test/history-facts.test.ts`); the TS/Python equality on every committed
channel is `history-facts-parity.test.ts`, the equality with the live engine's fields
`packages/showdown-bridge/test/history-facts-parity.test.ts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.reducer import ChannelReducer, Vocabulary, VocabularyError
from areneon_ml.reducer import vocab as vocab_module
from areneon_ml.reducer.history import HISTORY_FACTS_FORMAT, UNSETTLED, HistoryTracker
from areneon_ml.reducer.state import STATE_SCHEMA_VERSION

from .support import FORMAT_DIR, VOCAB

DATA: dict[str, Any] = {
    "stall_moves": ["protect", "detect", "quickguard", "wideguard"],
    "ally_side_moves": ["quickguard", "wideguard"],
    "stall_counter": {"start": 3, "factor": 3, "max": 27},
    "action_sources": ["move: Round", "ability: Dancer"],
    "external_action_sources": ["ability: Dancer"],
    "item_use_sources": ["move: Fling"],
    "eat_tags": ["eat", "weaken"],
    "eaten_berries": ["oranberry"],
    "last_item_restorers": ["move: Recycle"],
    "last_item_takers": ["ability: Pickup"],
    "direct_end_items": ["airballoon"],
    "null_cant_reasons": ["recharge"],
    "protect_breaker_moves": ["feint"],
    "slot_conditions": [
        {"id": "futuremove", "moves": ["futuresight"], "residual_turns": 3},
        {"id": "wish", "moves": ["wish"], "residual_turns": 2},
    ],
    "volatile_not_copied": ["torment"],
    "shed_tail_copies": ["substitute"],
}
HEAD = [
    "|player|p1|Ann|1|",
    "|player|p2|Bob|2|",
    "|start",
    "|switch|p1a: Alpha|Alpha, L50|100/100",
    "|switch|p1b: Beta|Beta, L50|100/100",
    "|switch|p2a: Gamma|Gamma, L50|100/100",
    "|switch|p2b: Delta|Delta, L50|100/100",
]
END = ["|", "|upkeep"]


def turn(n: int) -> list[str]:
    return ["|", f"|turn|{n}"]


def request(moves: list[dict[str, Any]], second: list[dict[str, Any]] | None = None) -> str:
    return "|request|" + json.dumps(
        {"active": [{"moves": moves}, {"moves": moves if second is None else second}]}
    )


REQUEST = request(
    [{"id": "tackle", "maxpp": 10}, {"id": "protect", "maxpp": 10}, {"id": "struggle"}]
)


def facts(
    lines: list[str], player: str = "p1", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    tracker = HistoryTracker(player, data or DATA)
    for line in lines:
        tracker.feed(line)
    return tracker.facts()


def mon(state: dict[str, Any], side: str, letter: str) -> dict[str, Any]:
    found = state[side]["active"][letter]
    assert found is not None
    assert not found["unknown"]
    return found  # type: ignore[no-any-return]


def protect(n: int, user: str = "p1a: Alpha") -> list[str]:
    return [*turn(n), f"|move|{user}|Protect|{user}", f"|-singleturn|{user}|Protect", *END]


def test_the_state_carries_the_facts_and_the_format() -> None:
    state = facts(HEAD)
    assert state["history_facts_format"] == HISTORY_FACTS_FORMAT == "history-facts/1"
    assert STATE_SCHEMA_VERSION == 3
    assert state["turn"] == 0
    assert state["field_last_move"] is None


def test_move_actions() -> None:
    state = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|-damage|p2a: Gamma|80/100",
            "|move|p1b: Beta|Metronome|p1b: Beta",
            "|move|p1b: Beta|Tackle|p2a: Gamma|[from] move: Metronome",
            "|cant|p2a: Gamma|par",
            "|-activate|p2b: Delta|confusion",
            "|-damage|p2b: Delta|90/100|[from] confusion",
            *END,
        ]
    )
    assert [
        mon(state, s, k)["move_actions"]
        for s, k in [("own", "a"), ("own", "b"), ("opponent", "a"), ("opponent", "b")]
    ] == [1, 1, 1, 1]
    state = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|move|p1b: Beta|Round|p2a: Gamma|[from] move: Round",
            "|cant|p2b: Delta|slp",
            "|move|p2b: Delta|Snore|p1a: Alpha",
            *END,
        ]
    )
    assert mon(state, "own", "b")["move_actions"] == 1
    assert mon(state, "opponent", "b")["move_actions"] == 1
    state = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            *END,
            *turn(2),
            "|switch|p1a: Alpha|Alpha, L50|90/100",
        ]
    )
    assert mon(state, "own", "a")["move_actions"] == 0


def test_the_stall_counter() -> None:
    use = ["|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect"]
    assert mon(facts([*HEAD, *protect(1)]), "own", "a")["stall_counter"] == 3
    assert mon(facts([*HEAD, *turn(1), *use]), "own", "a")["stall_counter"] == 3
    assert mon(facts([*HEAD, *protect(1), *turn(2), *use]), "own", "a")["stall_counter"] == 9
    assert mon(facts([*HEAD, *protect(1), *turn(2)]), "own", "a")["stall_counter"] == 3
    assert (
        mon(facts([*HEAD, *protect(1), *turn(2), *END, *turn(3)]), "own", "a")["stall_counter"]
        is None
    )
    capped = facts([*HEAD, *protect(1), *protect(2), *protect(3), *protect(4), *turn(5), *use])
    assert mon(capped, "own", "a")["stall_counter"] == 27
    failed = facts(
        [*HEAD, *protect(1), *turn(2), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-fail|p1a: Alpha"]
    )
    assert mon(failed, "own", "a")["stall_counter"] is None
    called = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Metronome|p1a: Alpha",
            "|move|p1a: Alpha|Protect|p1a: Alpha|[from] move: Metronome",
            "|-singleturn|p1a: Alpha|Protect",
        ]
    )
    assert mon(called, "own", "a")["stall_counter"] == 3
    alone = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Protect|p1a: Alpha|[from] move: Metronome",
            "|-singleturn|p1a: Alpha|Protect",
        ]
    )
    assert mon(alone, "own", "a")["stall_counter"] == 3
    broken = facts(
        [
            *HEAD,
            *protect(1),
            *turn(2),
            "|move|p2a: Gamma|Feint|p1a: Alpha",
            "|-activate|p1a: Alpha|move: Feint",
        ]
    )
    assert mon(broken, "own", "a")["stall_counter"] is None
    other = facts(
        [
            *HEAD,
            *protect(1),
            *turn(2),
            "|move|p2a: Gamma|Tackle|p1a: Alpha",
            "|-activate|p1a: Alpha|move: Tackle",
            "|-activate|p1b: Beta|ability: Static",
            "|-activate|p1c: Nobody|move: Feint",
        ]
    )
    assert mon(other, "own", "a")["stall_counter"] == 3
    guard = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Wide Guard|p1a: Alpha",
            "|-singleturn|p1a: Alpha|Wide Guard",
            "|move|p1b: Beta|Wide Guard|p1b: Beta",
            "|move|p2a: Gamma|Tackle|p1a: Alpha",
        ]
    )
    assert mon(guard, "own", "b")["stall_counter"] == 3


def test_a_stalling_move_with_no_outcome_is_a_refusal() -> None:
    lines = [
        *HEAD,
        *turn(1),
        "|move|p1a: Alpha|Protect|p1a: Alpha",
        "|move|p2a: Gamma|Tackle|p1a: Alpha",
    ]
    assert facts(lines) == {"refused": "stall_unresolved"}
    assert facts(lines[:-1]) == {"refused": "stall_unresolved"}


def test_a_baton_pass_hands_the_counter_over() -> None:
    base = [*HEAD, *protect(1), *turn(2)]
    assert (
        mon(facts([*base, "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Baton Pass"]), "own", "a")[
            "stall_counter"
        ]
        == 3
    )
    assert (
        mon(facts([*base, "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Shed Tail"]), "own", "a")[
            "stall_counter"
        ]
        is None
    )
    assert (
        mon(facts([*base, "|switch|p1a: Alpha|Alpha, L50|100/100"]), "own", "a")["stall_counter"]
        is None
    )
    kept = {**DATA, "volatile_not_copied": ["stall"], "shed_tail_copies": ["stall"]}
    assert (
        mon(
            facts([*base, "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Baton Pass"], data=kept),
            "own",
            "a",
        )["stall_counter"]
        is None
    )
    assert (
        mon(
            facts([*base, "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Shed Tail"], data=kept),
            "own",
            "a",
        )["stall_counter"]
        == 3
    )


def test_hits_hurt_and_attackers() -> None:
    state = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Double Hit|p2a: Gamma",
            "|-damage|p2a: Gamma|80/100",
            "|-damage|p2a: Gamma|60/100",
            "|-hitcount|p2a: Gamma|2",
            "|move|p1b: Beta|Tackle|p2a: Gamma",
            "|-damage|p2a: Gamma|50/100",
        ]
    )
    gamma = mon(state, "opponent", "a")
    assert gamma["times_attacked"] == 3
    assert gamma["hurt_this_turn"] is True
    assert gamma["attackers_this_turn"] == ["own:a", "own:b"]
    later = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|-damage|p2a: Gamma|70/100",
            *END,
            *turn(2),
        ]
    )
    assert mon(later, "opponent", "a")["times_attacked"] == 1
    assert mon(later, "opponent", "a")["hurt_this_turn"] is False
    assert mon(later, "opponent", "a")["attackers_this_turn"] == []
    early = facts(
        [*HEAD, "|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|70/100", *turn(1)]
    )
    assert mon(early, "opponent", "a")["hurt_this_turn"] is True
    out = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|-damage|p2a: Gamma|70/100",
            *turn(2),
            "|switch|p2a: Gamma|Gamma, L50|70/100",
        ]
    )
    assert mon(out, "opponent", "a")["times_attacked"] == 0
    direct = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Substitute|p1a: Alpha",
            "|-start|p1a: Alpha|Substitute",
            "|-damage|p1a: Alpha|75/100",
            "|move|p2a: Gamma|Tackle|p1a: Alpha",
            "|-activate|p1a: Alpha|move: Substitute|[damage]",
            "|move|p2b: Delta|Struggle|p1b: Beta",
            "|-damage|p1b: Beta|90/100",
            "|-damage|p2b: Delta|90/100|[from] recoil",
            "|-activate|p1b: Beta|confusion",
            "|-damage|p1b: Beta|80/100|[from] confusion",
        ]
    )
    assert mon(direct, "own", "a")["times_attacked"] == 0
    assert mon(direct, "own", "a")["hurt_this_turn"] is False
    assert mon(direct, "opponent", "b")["hurt_this_turn"] is False
    assert mon(direct, "own", "b")["times_attacked"] == 1
    assert mon(direct, "own", "b")["hurt_this_turn"] is True


def test_a_transform_copies_the_count_and_clears_the_moves_used() -> None:
    state = facts(
        [
            *HEAD,
            request([{"id": "transform", "maxpp": 10}]),
            *turn(1),
            "|move|p2a: Gamma|Tackle|p1b: Beta",
            "|-damage|p1b: Beta|80/100",
            "|move|p1a: Alpha|Transform|p1b: Beta",
            "|-transform|p1a: Alpha|p1b: Beta|[from] move: Transform",
        ]
    )
    assert mon(state, "own", "a")["times_attacked"] == 1
    assert mon(state, "own", "a")["moves_used"] == []
    assert facts([*HEAD, "|-transform|p1a: Alpha|p2: Nobody"]) == {"refused": "unattributable_line"}


def test_the_last_moves_result() -> None:
    def result(action: list[str]) -> bool | None:
        state = facts([*HEAD, *turn(1), *action, *END, *turn(2)])
        return mon(state, "own", "a")["last_move_failed"]  # type: ignore[no-any-return]

    assert result(["|cant|p1a: Alpha|par"]) is True
    assert result(["|cant|p1a: Alpha|recharge"]) is False
    assert (
        result(["|move|p1a: Alpha|Tackle|p2a: Gamma|[miss]", "|-miss|p1a: Alpha|p2a: Gamma"])
        is True
    )
    assert result(["|move|p1a: Alpha|Thunder Wave|p2a: Gamma", "|-immune|p2a: Gamma"]) is True
    assert result(["|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|50/100"]) is False
    assert result(["|move|p1a: Alpha|Swords Dance|p1a: Alpha", "|-boost|p1a: Alpha|atk|2"]) is False
    assert (
        result(["|-activate|p1a: Alpha|confusion", "|-damage|p1a: Alpha|90/100|[from] confusion"])
        is True
    )
    assert result(["|move|p1a: Alpha|Growl|p2a: Gamma", "|-fail|p1a: Alpha"]) is None
    assert result(["|move|p1a: Alpha|Growl|p2a: Gamma", "|-unboost|p2a: Gamma|atk|0"]) is None
    assert result(["|move|p1a: Alpha|Growl|p2a: Gamma"]) is None
    assert (
        result(
            [
                "|move|p1a: Alpha|Metronome|p1a: Alpha",
                "|move|p1a: Alpha|Tackle|p2a: Gamma|[from] move: Metronome",
            ]
        )
        is None
    )
    assert result([]) is False
    reflected = facts(
        [
            *HEAD,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|-damage|p2a: Gamma|90/100",
            "|move|p2b: Delta|Growl|p1a: Alpha",
            "|move|p1a: Alpha|Growl|p2b: Delta|[from] ability: Magic Bounce",
            *END,
            *turn(2),
        ]
    )
    assert mon(reflected, "own", "a")["last_move_failed"] is None


def test_stat_changes_this_turn() -> None:
    state = facts([*HEAD, "|-unboost|p1a: Alpha|atk|1", *turn(1), "|-boost|p1b: Beta|spe|1"])
    assert mon(state, "own", "a")["stats_lowered_this_turn"] is True
    assert mon(state, "own", "b")["stats_raised_this_turn"] is True
    later = facts([*HEAD, "|-unboost|p1a: Alpha|atk|1", *turn(1), *END, *turn(2)])
    assert mon(later, "own", "a")["stats_lowered_this_turn"] is False
    out = facts([*HEAD, "|-unboost|p1a: Alpha|atk|1", "|switch|p1a: Alpha|Alpha, L50|100/100"])
    assert mon(out, "own", "a")["stats_lowered_this_turn"] is False
    assert facts([*HEAD, "|-boost|p2: Nobody|atk|1"])["turn"] == 0


def roster(lines: list[str], side: str = "own", name: str = "Alpha") -> dict[str, Any]:
    return facts([*HEAD, *turn(1), *lines])[side]["roster"][name]  # type: ignore[no-any-return]


def test_berries_and_the_last_item() -> None:
    assert roster(["|-enditem|p1a: Alpha|Sitrus Berry|[eat]"]) == {
        "ate_berry": True,
        "last_item": "sitrusberry",
    }
    assert roster(["|-enditem|p1a: Alpha|Chilan Berry|[weaken]"]) == {
        "ate_berry": True,
        "last_item": "chilanberry",
    }
    assert roster(["|-enditem|p1a: Alpha|White Herb"]) == {
        "ate_berry": False,
        "last_item": "whiteherb",
    }
    assert roster(["|-enditem|p1a: Alpha|Normal Gem|[from] gem"]) == {
        "ate_berry": False,
        "last_item": "normalgem",
    }
    assert roster(["|-enditem|p1a: Alpha|Air Balloon"]) == {"ate_berry": False, "last_item": None}
    none = {"ate_berry": False, "last_item": None}
    assert roster(["|-enditem|p1a: Alpha|Leftovers|[from] move: Knock Off|[of] p2a: Gamma"]) == none
    assert roster(
        ["|-enditem|p1a: Alpha|Oran Berry|[from] stealeat|[move] Bug Bite|[of] p2a: Gamma"],
        "opponent",
        "Gamma",
    ) == {"ate_berry": True, "last_item": None}
    assert (
        roster(["|-enditem|p1a: Alpha|Oran Berry|[from] stealeat|[move] Bug Bite|[of] p3a: Nobody"])
        == none
    )
    thrown = [
        "|move|p1a: Alpha|Fling|p2a: Gamma",
        "|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling",
    ]
    assert roster(thrown) == {"ate_berry": False, "last_item": "oranberry"}
    assert roster(thrown, "opponent", "Gamma") == {"ate_berry": True, "last_item": None}
    assert (
        roster(
            [
                "|move|p1a: Alpha|Fling|p2a: Gamma",
                "|-enditem|p1a: Alpha|Leppa Berry|[from] move: Fling",
            ],
            "opponent",
            "Gamma",
        )
        == none
    )
    assert (
        roster(["|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling"], "opponent", "Gamma") == none
    )
    assert (
        roster(
            [
                "|move|p1a: Alpha|Fling|p2: Fainted",
                "|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling",
            ],
            "opponent",
            "Gamma",
        )
        == none
    )
    assert roster(
        [
            "|-enditem|p1a: Alpha|Sitrus Berry|[eat]",
            "|-item|p1a: Alpha|Sitrus Berry|[from] move: Recycle",
        ]
    ) == {"ate_berry": True, "last_item": None}
    assert roster(["|-item|p1a: Alpha|Sitrus Berry|[from] ability: Static"]) == none
    assert roster(["|-item|p1a: Alpha|Sitrus Berry"]) == none
    taken = [
        "|-enditem|p1a: Alpha|Sitrus Berry|[eat]",
        "|-item|p2a: Gamma|Sitrus Berry|[from] ability: Pickup",
    ]
    assert roster(taken)["last_item"] == UNSETTLED
    assert (
        roster(
            [
                "|-enditem|p1a: Alpha|Sitrus Berry|[eat]",
                "|-item|p2a: Gamma|Lum Berry|[from] ability: Pickup",
            ]
        )["last_item"]
        == "sitrusberry"
    )


def test_own_moves_used() -> None:
    state = facts(
        [
            *HEAD,
            REQUEST,
            *turn(1),
            "|move|p1a: Alpha|Tackle|p2a: Gamma",
            "|move|p1b: Beta|Protect|p1b: Beta",
            "|-singleturn|p1b: Beta|Protect",
            "|move|p2a: Gamma|Tackle|p1a: Alpha",
            "|move|p1b: Beta|Struggle|p2a: Gamma",
            "|move|p1b: Beta|Round|p2a: Gamma|[from] move: Round",
            *END,
        ]
    )
    assert mon(state, "own", "a")["moves_used"] == ["tackle"]
    assert mon(state, "own", "b")["moves_used"] == ["protect"]
    assert mon(state, "opponent", "a")["moves_used"] is None
    own = request([{"id": "round", "maxpp": 10}])

    def used(lines: list[str]) -> list[str]:
        return mon(facts([*HEAD, own, *turn(1), *lines]), "own", "a")["moves_used"]  # type: ignore[no-any-return]

    assert used(["|move|p1a: Alpha|Round|p2a: Gamma|[from] move: Round"]) == ["round"]
    assert used(["|move|p1a: Alpha|Round|p2a: Gamma|[from] ability: Dancer"]) == []
    swapped = facts(
        [
            *HEAD,
            request([{"id": "tackle", "maxpp": 10}], [{"id": "protect", "maxpp": 10}]),
            *turn(1),
            "|swap|p1a: Alpha|1",
            "|move|p1a: Beta|Protect|p1a: Beta",
            "|-singleturn|p1a: Beta|Protect",
        ]
    )
    assert mon(swapped, "own", "a")["moves_used"] == ["protect"]
    assert mon(swapped, "own", "b")["moves_used"] == []
    half = facts(
        [
            *HEAD,
            request([{"id": "tackle", "maxpp": 10}], []),
            *turn(1),
            "|swap|p1a: Alpha|1",
            "|swap|p1a: Beta|1",
        ]
    )
    assert mon(half, "own", "a")["moves_used"] == []
    gone = facts(
        [
            *HEAD,
            own,
            *turn(1),
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|move|p1a: Alpha|Round|p2a: Gamma",
        ]
    )
    assert mon(gone, "own", "a")["moves_used"] == []
    odd = [
        "|request|{}",
        "|request|not json",
        "|request|null",
        "|request|"
        + json.dumps({"active": [{}, {"moves": [{"id": 3}, 4, {"id": "x", "maxpp": True}]}]}),
    ]
    assert (
        mon(facts([*HEAD, *odd, *turn(1), "|move|p1a: Alpha|Tackle|p2a: Gamma"]), "own", "a")[
            "moves_used"
        ]
        == []
    )


def test_faints_the_last_move_pending_slots_and_volatiles() -> None:
    state = facts([*HEAD, *turn(1), "|faint|p2a: Gamma", "|faint|p1b: Beta"])
    assert state["opponent"]["total_fainted"] == 1
    assert state["opponent"]["active"]["a"] is None
    assert facts([*HEAD, "|faint|p2c: Nobody"]) == {"refused": "unattributable_line"}
    assert (
        facts(
            [
                *HEAD,
                *turn(1),
                "|move|p1a: Alpha|Tackle|p2a: Gamma",
                "|move|p2a: Gamma|Growl|p1a: Alpha|[from] move: Metronome",
            ]
        )["field_last_move"]
        == "growl"
    )
    wish = [*HEAD, *turn(1), "|move|p1a: Alpha|Wish|p1a: Alpha", *END]
    assert facts([*wish, *turn(2)])["own"]["pending_slot"] == {
        "a": {"wish": {"started_turn": 1, "after_residual": False, "remaining": 1}}
    }
    assert facts([*wish, *turn(2), *END])["own"]["pending_slot"] == {}
    assert (
        facts([*HEAD, *turn(1), "|move|p1a: Alpha|Wish|p1a: Alpha", "|-fail|p1a: Alpha"])["own"][
            "pending_slot"
        ]
        == {}
    )
    assert (
        facts([*HEAD, *turn(1), "|move|p1a: Alpha|Wish|p1a: Alpha", "|-fail|p1b: Beta"])["own"][
            "pending_slot"
        ]
        == {}
    )
    future = [
        *HEAD,
        *turn(1),
        "|move|p1a: Alpha|Future Sight|p2a: Gamma",
        "|-start|p1a: Alpha|move: Future Sight",
        *END,
    ]
    assert (
        facts([*future, *turn(2), *END, *turn(3)])["opponent"]["pending_slot"]["a"]["futuremove"][
            "remaining"
        ]
        == 1
    )
    landing = ["|", "|-end|p2a: Gamma|move: Future Sight", "|-damage|p2a: Gamma|60/100"]
    landed = facts([*future, *turn(2), *END, *turn(3), *landing, *END, *turn(4)])
    assert mon(landed, "opponent", "a")["times_attacked"] == 1
    assert mon(landed, "opponent", "a")["attackers_this_turn"] == []
    assert landed["opponent"]["pending_slot"] == {}
    mid = facts([*future, *turn(2), *END, *turn(3), *landing])
    assert mon(mid, "opponent", "a")["attackers_this_turn"] == ["own:a"]
    left = facts(
        [*future, *turn(2), "|switch|p1a: Zeta|Zeta, L50|100/100", *END, *turn(3), *landing]
    )
    assert mon(left, "opponent", "a")["attackers_this_turn"] is None
    assert mon(left, "opponent", "a")["times_attacked"] == 1
    assert facts([*HEAD, *turn(1), "|-end|p2a: Gamma|move: Future Sight"])["turn"] == 1
    assert facts([*HEAD, *turn(1), "|-end|p2a: Gamma|Stockpile"])["turn"] == 1
    fainted = facts(
        [*HEAD, *turn(1), "|faint|p2a: Gamma", "|move|p1a: Alpha|Future Sight|p2: Gamma", *END]
    )
    assert list(fainted["opponent"]["pending_slot"]) == ["a"]
    again = facts(
        [*wish, *turn(2), *END, *turn(3), "|move|p1a: Alpha|Wish|p1a: Alpha", *END, *turn(4)]
    )
    assert again["own"]["pending_slot"]["a"]["wish"]["started_turn"] == 3
    early = facts([*HEAD, "|move|p1a: Alpha|Wish|p1a: Alpha", *turn(1)])
    assert early["own"]["pending_slot"]["a"]["wish"] == {
        "started_turn": 0,
        "after_residual": True,
        "remaining": 2,
    }
    vol = facts(
        [
            *HEAD,
            *turn(1),
            "|-start|p1a: Alpha|move: Taunt",
            *END,
            *turn(2),
            "|-start|p1a: Alpha|confusion",
            "|-end|p1a: Alpha|confusion",
            "|-start|p1b: Beta|typechange|Fire",
        ]
    )
    assert mon(vol, "own", "a")["volatile_started"] == {
        "taunt": {"started_turn": 1, "after_residual": False}
    }
    assert mon(vol, "own", "b")["volatile_started"]["typechange"] == {
        "started_turn": 2,
        "after_residual": False,
    }
    assert facts([*HEAD, "|-start|p1a: Alpha|"]) == {"refused": "malformed_line"}
    assert (
        facts([*HEAD, "|-start|p1c: Nobody|confusion", "|-end|p1c: Nobody|confusion"])["turn"] == 0
    )
    twice = facts([*HEAD, "|-start|p1a: Alpha|move: Taunt", "|-start|p1a: Alpha|move: Taunt"])
    assert mon(twice, "own", "a")["volatile_started"]["taunt"]["after_residual"] is True


def test_an_illusion_makes_the_position_unknown_until_it_leaves() -> None:
    lines = [
        *HEAD,
        *turn(1),
        "|move|p2a: Gamma|Tackle|p1a: Alpha",
        "|-damage|p1a: Alpha|90/100",
        "|replace|p2a: Zorua|Zorua, L50",
        "|-end|p2a: Zorua|Illusion",
    ]
    state = facts(lines)
    assert state["opponent"]["active"]["a"] == {"name": "Zorua", "unknown": True}
    assert state["opponent"]["roster"]["Gamma"] == {"ate_berry": None, "last_item": None}
    assert state["opponent"]["roster"]["Zorua"] == {"ate_berry": None, "last_item": None}
    out = facts([*lines, "|", "|upkeep", *turn(2), "|switch|p2a: Gamma|Gamma, L50|100/100"])
    assert mon(out, "opponent", "a")["name"] == "Gamma"
    assert facts([*HEAD, "|replace|p2c: Nobody|Nobody, L50"]) == {"refused": "unattributable_line"}


def test_refusals_are_typed_and_sticky_and_the_other_seat_reads_the_same_channel() -> None:
    assert facts([*HEAD, "|move|p3a: Nobody|Tackle|p2a: Gamma"]) == {
        "refused": "unattributable_line"
    }
    assert facts([*HEAD, "|move|p1a: Alpha||p2a: Gamma"]) == {"refused": "malformed_line"}
    assert facts([*HEAD, "|switch|p1: Alpha|Alpha, L50|100/100"]) == {
        "refused": "unattributable_line"
    }
    assert facts([*HEAD, "|swap|p1a: Alpha|7"]) == {"refused": "unattributable_line"}
    assert facts([*HEAD, "|swap|p1a: Alpha"]) == {"refused": "unattributable_line"}
    assert facts([*HEAD, "|turn|x"]) == {"refused": "malformed_line"}
    assert facts([*HEAD, *turn(3), "|turn|2"]) == {"refused": "malformed_line"}
    assert facts([*HEAD, "|-enditem|nobody|Leftovers"]) == {"refused": "unattributable_line"}
    assert facts([*HEAD, "nonsense"]) == {"refused": "malformed_line"}
    assert facts([*HEAD, "|turn"]) == {"refused": "malformed_line"}
    assert facts([*HEAD, "|faint|p2c: Nobody", "|move|p1a: Alpha|Tackle|p2a: Gamma"]) == {
        "refused": "unattributable_line"
    }
    for line in [
        "|request",
        "|cant",
        "|-fail",
        "|-notarget",
        "|-miss",
        "|-immune",
        "|-boost",
        "|-unboost",
        "|-activate",
        "|-damage",
        "|upkeep",
        "|win|Ann",
        "|-weather|none",
    ]:
        assert "refused" not in facts([*HEAD, line, "|turn|1"]) or line in ("|-damage",), line
    for line in [
        "|switch",
        "|swap",
        "|replace",
        "|faint",
        "|move",
        "|-enditem",
        "|-start",
        "|-end",
        "|-transform",
    ]:
        assert "refused" in facts([*HEAD, line]), line
    state = facts(
        [*HEAD, *turn(1), "|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|50/100"],
        player="p2",
    )
    assert mon(state, "own", "a")["times_attacked"] == 1
    assert mon(state, "opponent", "a")["move_actions"] == 1
    odd = facts(
        [
            *HEAD,
            "|move|p1a: Alpha|Wish",
            "|move|p1b: Beta|Tackle|",
            "|-boost|p1a: Alpha|atk|0",
            "|move|p1a: Alpha|Growl|p1a: Alpha",
            "|-unboost|p1a: Alpha|atk|0",
        ]
    )
    assert mon(odd, "own", "a")["move_actions"] == 2
    assert odd["own"]["pending_slot"] == {}


# -- the reducer carries the facts -------------------------------------------------------------


def test_the_reducer_state_carries_the_facts_of_the_channel() -> None:
    assert VOCAB.history_data is not None
    reducer = ChannelReducer("p1", [], VOCAB)
    assert reducer.state.history is not None
    assert "history" in reducer.state.to_dict()
    battle = json.loads((FORMAT_DIR / "random" / "b00.json").read_text(encoding="utf-8"))
    state = ChannelReducer("p1", battle["teams"]["p1"], VOCAB)
    for raw in battle["channels"]["p1"]["lines"]:
        state.feed(raw)
    history = state.state.to_dict()["history"]
    assert history["history_facts_format"] == HISTORY_FACTS_FORMAT
    assert history["turn"] == state.state.turn


def test_a_state_without_the_data_for_its_pin_has_no_history_key() -> None:
    bare = Vocabulary(**{**VOCAB.__dict__, "history_data": None})
    reducer = ChannelReducer("p1", [], bare)
    assert reducer.state.history is None
    assert "history" not in reducer.state.to_dict()


def test_the_data_file_is_checked() -> None:
    commit, format_id = VOCAB.showdown_commit, VOCAB.format_id
    data = vocab_module.read_history_facts(commit, format_id)
    assert data is not None
    assert data["format_id"] == format_id
    assert vocab_module.read_history_facts("0" * 40, format_id) is None
    assert vocab_module.read_history_facts(commit, "gen9nosuchformat") is None


def test_a_malformed_or_doctored_data_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = vocab_module.DATA_DIR / f"history-facts.{VOCAB.format_id}.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    monkeypatch.setattr(vocab_module, "DATA_DIR", tmp_path)

    def put(document: dict[str, Any]) -> None:
        (tmp_path / source.name).write_text(json.dumps(document), encoding="utf-8")

    put({**original, "extra": 1})
    with pytest.raises(VocabularyError, match="not a history-facts table"):
        vocab_module.read_history_facts(VOCAB.showdown_commit, VOCAB.format_id)
    put({**original, "null_cant_reasons": ["recharge", "other"]})
    with pytest.raises(VocabularyError, match="hash"):
        vocab_module.read_history_facts(VOCAB.showdown_commit, VOCAB.format_id)
    put({**original, "format_id": "gen9other"})
    body = {k: v for k, v in original.items() if k != "history_facts_data_hash"}
    body["format_id"] = "gen9other"
    put({**body, "history_facts_data_hash": sha256_hex(jcs(body))})
    with pytest.raises(VocabularyError, match="another format"):
        vocab_module.read_history_facts(VOCAB.showdown_commit, VOCAB.format_id)
