"""PROTOCOL_VOLATILES against the pinned engine's source (ADR-010 erratum A4 item 7).

The set holds the `-start` / `-end` words the engine prints that are no condition id and no name
of a move, ability or item. Scanning the vendored source for literal volatile arguments keeps it
honest both ways: a pin bump that prints a new such word fails, and so does a stale entry."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from areneon_ml.reducer import Vocabulary, VocabularyError
from areneon_ml.reducer.vocab import PROTOCOL_VOLATILES, to_id

from .support import FORMAT_DIR, ROOT

ENGINE = ROOT / "vendor" / "pokemon-showdown"
MOD = "champions"
# add('-start', POKEMON, 'literal' ...) or a template literal such as `fallen${n}`.
CALL = re.compile(r"""add\('-(?:start|end)',\s*[^,()]+(?:\([^)]*\))?,\s*(['`])([^'`]*)\1""")
KIND = re.compile(r"^(?:move|ability|item): ")

HAVE_ENGINE = ENGINE.joinpath("sim").is_dir()
pytestmark = pytest.mark.skipif(
    not HAVE_ENGINE and not os.environ.get("CI"), reason="no engine checkout"
)


@pytest.fixture(autouse=True)
def _engine_in_ci() -> None:
    """Under CI the checkout must be there: a drift test that always skips guards nothing."""
    if os.environ.get("CI") and not HAVE_ENGINE:
        pytest.fail("the engine source is missing under CI: check out the submodule")


def _files() -> list[Path]:
    paths = sorted(ENGINE.joinpath("sim").glob("*.ts")) + sorted(
        ENGINE.joinpath("data").glob("*.ts")
    )
    return paths + sorted(ENGINE.joinpath("data", "mods", MOD).glob("*.ts"))


def _literals() -> set[str]:
    found: set[str] = set()
    for path in _files():
        for _, text in CALL.findall(path.read_text(encoding="utf-8")):
            found.add(to_id(KIND.sub("", text.split("${")[0])))
    found.discard("")
    return found


V2 = Vocabulary.from_engine_data(
    json.loads((FORMAT_DIR / "engine-data-2.json").read_text(encoding="utf-8"))
)


def _engine_effect_ids() -> set[str]:
    """Ids of every move, ability and item in the engine's data files, the format's or not: a
    literal naming one that the format lacks (Mimic, Doom Desire) can never be printed in it."""
    key = re.compile(r"^\t(\w+): \{", re.MULTILINE)
    return {
        found
        for name in ("moves", "abilities", "items")
        for found in key.findall(ENGINE.joinpath("data", f"{name}.ts").read_text(encoding="utf-8"))
    }


def _resolves(word: str) -> bool:
    try:
        V2.volatile(word)
    except VocabularyError:
        return False
    return True


def test_every_literal_volatile_the_engine_prints_resolves_in_version_2() -> None:
    literals = _literals()
    assert {"typechange", "substitute", "confusion"} <= literals  # the scan really finds calls
    # A literal that only resolves through PROTOCOL_VOLATILES is what the set is for; any
    # other unresolved one is a word the set (or the table) lacks.
    unresolved = {w for w in literals - _engine_effect_ids() if not _resolves(w)}
    assert unresolved == set(), unresolved


def test_every_protocol_volatile_is_still_printed_and_still_not_a_condition() -> None:
    literals = _literals()
    for word in PROTOCOL_VOLATILES:
        assert any(lit == word or lit.rstrip("0123456789") == word for lit in literals), word
        assert V2.conditions is not None
        assert word not in V2.conditions
        assert word not in V2.moves | V2.abilities | V2.items
