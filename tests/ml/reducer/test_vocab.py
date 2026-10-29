from __future__ import annotations

import pytest
from areneon_ml.reducer import Vocabulary, VocabularyError
from areneon_ml.reducer.vocab import to_id

from .support import VOCAB


def test_to_id_keeps_letters_and_digits() -> None:
    assert to_id("Mr. Mime-Galar") == "mrmimegalar"


def test_lookups_return_ids() -> None:
    assert VOCAB.species("Charizard-Mega-X") == "charizardmegax"
    assert VOCAB.base_species("charizardmegax") == "charizard"
    assert VOCAB.move("Flamethrower") == "flamethrower"
    assert VOCAB.item("Sitrus Berry") == "sitrusberry"
    assert VOCAB.ability("Thick Fat") == "thickfat"
    assert VOCAB.status("par") == "par"
    assert VOCAB.weather("RainDance") == "raindance"


@pytest.mark.parametrize("kind", ["species", "move", "item", "ability", "status", "weather"])
def test_unknown_names_raise(kind: str) -> None:
    with pytest.raises(VocabularyError, match="unknown"):
        getattr(VOCAB, kind)("Not A Thing")


def test_an_unsupported_export_version_is_refused() -> None:
    with pytest.raises(VocabularyError, match="schema_version"):
        Vocabulary.from_engine_data({"schema_version": 3})


def test_the_vocabulary_knows_its_engine() -> None:
    assert VOCAB.format_id == "gen9championsvgc2026regmc"
    assert len(VOCAB.showdown_commit) == 40
