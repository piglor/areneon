"""Engine data version 2 in the reducer: a condition vocabulary for volatiles, and the type a
`typechange` gives (ADR-010 erratum A4 item 7). A version 1 table keeps today's behaviour."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.reducer import MalformedLineError, Vocabulary, VocabularyError, reduce_channel
from areneon_ml.reducer.vocab import PROTOCOL_VOLATILES

from .support import FORMAT_DIR, HEADER, TEAM, VOCAB, opp

TABLE_2: dict[str, Any] = json.loads((FORMAT_DIR / "engine-data-2.json").read_text())
VOCAB_2 = Vocabulary.from_engine_data(TABLE_2)
CORPUS = sorted((FORMAT_DIR / "random").glob("b*.json"))
# The mons of the shared header: p2a Venusaur, p2b Charizard.
START = "|-start|p2a: Venusaur|"


def run(*lines: str, vocab: Vocabulary = VOCAB_2) -> Any:
    return reduce_channel("p1", [*HEADER, *lines], TEAM, vocab)


def test_a_version_1_table_has_no_condition_list_and_a_version_2_table_does() -> None:
    assert VOCAB.conditions is None
    assert VOCAB_2.conditions == frozenset(c["id"] for c in TABLE_2["conditions"])
    assert VOCAB_2.types == VOCAB.types
    assert VOCAB_2.moves == VOCAB.moves


def test_every_volatile_the_random_corpus_shows_resolves_in_version_2() -> None:
    seen: set[str] = set()
    for path in CORPUS:
        battle = json.loads(path.read_text())
        for player in ("p1", "p2"):
            lines = battle["channels"][player]["lines"]
            reduce_channel(player, lines, battle["teams"][player], VOCAB_2)
            for line in lines:
                parts = line.split("|")
                if len(parts) > 3 and parts[1] in ("-start", "-end"):
                    seen.add(parts[3])
    assert len(seen) > 20  # the corpus really shows many (Yawn, Taunt, Substitute, confusion, ...)
    assert {"confusion", "Substitute", "move: Taunt"} <= seen


def test_version_2_refuses_a_volatile_the_engine_does_not_define() -> None:
    line = f"{START}Zzz Not A Condition"
    assert run(line, vocab=VOCAB).opponent.roster  # version 1: opaque, kept
    with pytest.raises(VocabularyError, match="unknown volatile"):
        run(line)


def test_version_2_resolves_condition_ids_derived_names_and_effect_names() -> None:
    state = run(
        f"{START}confusion",  # a condition id
        f"{START}stockpile1",  # derived: the condition id plus a number
        f"{START}Fire Spin",  # the name of the move that started it
        f"{START}fallen2",  # a protocol word with a count
    )
    assert opp(state, "Venusaur").volatiles >= {"confusion", "stockpile1", "firespin", "fallen2"}
    assert "typechange" in PROTOCOL_VOLATILES


def test_typechange_keeps_the_types_it_names() -> None:
    state = run(f"{START}typechange|Fire/Flying|[silent]")
    assert opp(state, "Venusaur").types == ["fire", "flying"]
    assert opp(state, "Venusaur").to_dict()["types"] == ["fire", "flying"]
    assert "typechange" in opp(state, "Venusaur").volatiles
    # The same under a version 1 table (the types vocabulary is in both).
    assert opp(run(f"{START}typechange|Water", vocab=VOCAB), "Venusaur").types == ["water"]


def test_typechange_to_the_typeless_marker_and_a_copy() -> None:
    assert opp(run(f"{START}typechange|???/Fighting"), "Venusaur").types == ["???", "fighting"]
    assert opp(run(f"{START}typechange|???"), "Venusaur").types == ["???"]
    copied = run(f"{START}typechange|[from] move: Reflect Type|[of] p1a: Venusaur")
    assert opp(copied, "Venusaur").types == []  # the channel does not say which


def test_typechange_fails_closed_on_an_unknown_type() -> None:
    for text in ("Plasma", "Fire/Plasma", "Fire/"):
        with pytest.raises(VocabularyError, match="unknown type"):
            run(f"{START}typechange|{text}")


def test_types_end_with_the_switch_out_and_stay_out_of_a_state_without_a_change() -> None:
    state = run(f"{START}typechange|Water")
    assert "types" not in opp(run(), "Venusaur").to_dict()
    assert opp(state, "Venusaur").types == ["water"]
    out = run(
        f"{START}typechange|Water",
        "|switch|p2a: Charizard|Charizard, L50, F|100/100",
    )
    assert all(mon.types is None for mon in out.opponent.roster)


def test_typeadd_is_a_protocol_word_that_keeps_no_type() -> None:
    state = run(f"{START}typeadd|Ghost|[from] move: Trick-or-Treat")
    assert opp(state, "Venusaur").types is None
    assert "typeadd" in opp(state, "Venusaur").volatiles


def test_a_malformed_volatile_is_still_refused() -> None:
    with pytest.raises(MalformedLineError):
        run(f"{START}###")
