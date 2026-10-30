"""`PokemonState.derived`: the latest numbered or stat-suffixed volatile per condition id (#1019).

The engine prints Stockpile as `stockpile1`..`3` and ends it as `-end|Stockpile`, Perish Song as
`perish3`..`perish0` with no `-end`, Supreme Overlord as `fallenN` and Protosynthesis as
`protosynthesis<stat>`. The raw ids in `volatiles` accumulate or go stale, so the reducer keeps the
latest suffix per condition id (version 2 tables only; not part of `to_dict`).
"""

from __future__ import annotations

import json
from typing import Any

from areneon_ml.reducer import Vocabulary, reduce_channel

from .support import FORMAT_DIR, HEADER, TEAM, VOCAB, opp

VOCAB_2 = Vocabulary.from_engine_data(json.loads((FORMAT_DIR / "engine-data-2.json").read_text()))
START = "|-start|p2a: Venusaur|"
BLASTOISE_IN = "|switch|p2a: Blastoise|Blastoise, L50, M|100/100"


def run(*lines: str, vocab: Vocabulary = VOCAB_2) -> Any:
    return reduce_channel("p1", [*HEADER, *lines], TEAM, vocab)


def test_the_latest_suffix_is_kept_per_condition_id() -> None:
    state = run(f"{START}stockpile1", f"{START}stockpile2", f"{START}perish3", f"{START}perish2")
    assert opp(state, "Venusaur").derived == {"stockpile": "2", "perish": "2"}
    stat = run(f"{START}protosynthesisatk")
    assert opp(stat, "Venusaur").derived == {"protosynthesis": "atk"}
    assert "derived" not in opp(stat, "Venusaur").to_dict()


def test_the_bare_end_clears_it_and_a_version_1_table_tracks_nothing() -> None:
    state = run(f"{START}stockpile3", "|-end|p2a: Venusaur|Stockpile")
    assert opp(state, "Venusaur").derived == {}
    assert (
        opp(
            run(f"{START}fallen4|[silent]", "|-end|p2a: Venusaur|fallen4|[silent]"), "Venusaur"
        ).derived
        == {}
    )
    assert opp(run(f"{START}stockpile2", vocab=VOCAB), "Venusaur").derived == {}


def test_switching_out_clears_it() -> None:
    state = run(f"{START}perish3", BLASTOISE_IN)
    assert opp(state, "Venusaur").derived == {}
    assert opp(state, "Blastoise").derived == {}


def test_fainting_clears_it() -> None:
    state = run(f"{START}perish1", f"{START}stockpile2", "|faint|p2a: Venusaur")
    assert opp(state, "Venusaur").derived == {}


def test_a_baton_pass_hands_over_what_the_volatile_copy_table_passes() -> None:
    # perish is copied by the engine; stockpile and protosynthesis are `noCopy`.
    state = run(
        f"{START}perish3",
        f"{START}stockpile2",
        f"{START}protosynthesisspe",
        f"{BLASTOISE_IN}|[from] Baton Pass",
    )
    assert opp(state, "Blastoise").derived == {"perish": "3"}
    assert opp(state, "Venusaur").derived == {}


def test_a_shed_tail_hands_over_only_a_substitute() -> None:
    state = run(f"{START}perish3", f"{START}stockpile1", f"{BLASTOISE_IN}|[from] Shed Tail")
    assert opp(state, "Blastoise").derived == {}


def test_supreme_overlord_s_count_passes_on_a_baton_pass_as_the_raw_id_does() -> None:
    """Known limit, pinned: `fallenN` is a protocol word whose count lives in the ability's own
    state, not a copied volatile, yet the copy table passes `fallen` (the raw id has always
    passed), so the receiver gets the raw id and the derived count. A later fix is deliberate."""
    state = run(f"{START}fallen4|[silent]", f"{BLASTOISE_IN}|[from] Baton Pass")
    receiver = opp(state, "Blastoise")
    assert receiver.volatiles == {"fallen4"}
    assert receiver.derived == {"fallen": "4"}


def test_an_illusion_settle_carries_it_to_the_real_pokemon() -> None:
    state = run(
        BLASTOISE_IN,
        "|-start|p2a: Blastoise|perish2",
        "|-start|p2a: Blastoise|protosynthesisdef",
        "|replace|p2a: Gengar|Gengar, L50, M",
        "|-end|p2a: Gengar|Illusion",
    )
    real = next(m for m in state.opponent.roster if m.base_species == "gengar" and not m.phantom)
    shown = next(
        m for m in state.opponent.roster if m.base_species == "blastoise" and not m.phantom
    )
    assert real.derived == {"perish": "2", "protosynthesis": "def"}
    assert shown.derived == {}
