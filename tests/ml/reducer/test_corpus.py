"""The reducer on ordinary play: the random-legal corpus of `fixtures/showdown/<format>/random/`.

Seeded random-legal vs random-legal battles on random synthetic teams
(`packages/showdown-bridge/src/random-corpus.ts`; CI re-plays each one).
Every channel must reduce without a refusal, except at an allowlisted line
type; the canary runs over every request point of every channel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.reducer import ReducerError, UnknownLineError, reduce_channel

from .support import FORMAT_DIR, VOCAB, leaks, unrevealed_ids

CORPUS = sorted((FORMAT_DIR / "random").glob("b*.json"))
# Line types the reducer names as unsupported: the channel is not reduced past them.
UNSUPPORTED = {"replace"}  # Illusion ending: the disguise's bindings would be wrong


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _first_refusal(player: str, battle: dict[str, Any]) -> ReducerError | None:
    lines = battle["channels"][player]["lines"]
    try:
        reduce_channel(player, lines, battle["teams"][player], VOCAB)
    except ReducerError as error:
        return error
    return None


def test_the_corpus_is_there() -> None:
    manifest = json.loads((FORMAT_DIR / "random" / "manifest.json").read_text())
    assert len(CORPUS) == len(manifest["battles"]) >= 40
    assert all(_load(p)["data_class"] == "synthetic" for p in CORPUS)


def test_every_channel_reduces_except_at_a_named_unsupported_line_type() -> None:
    refused = []
    for path in CORPUS:
        battle = _load(path)
        for player in ("p1", "p2"):
            error = _first_refusal(player, battle)
            if error is None:
                continue
            assert isinstance(error, UnknownLineError), f"{path.name} {player}: {error}"
            assert error.line is not None
            assert error.line.split("|")[1] in UNSUPPORTED, f"{path.name} {player}: {error}"
            refused.append((path.name, player))
    # Illusion is rare in random teams: nearly every channel is reduced to its end.
    assert len(refused) <= len(CORPUS) // 4


def test_the_corpus_exercises_the_lines_that_broke_the_first_version() -> None:
    text = "\n".join(line for p in CORPUS for line in _load(p)["channels"]["p1"]["lines"])
    for marker in ("|-mega|", "|-formechange|", "[notarget]", "/100g", "/100y", "|-fail|"):
        assert marker in text, marker


def test_the_canary_holds_at_every_request_point_of_every_channel() -> None:
    checked = 0
    for path in CORPUS:
        battle = _load(path)
        for player, other in (("p1", "p2"), ("p2", "p1")):
            if _first_refusal(player, battle) is not None:
                continue
            lines = battle["channels"][player]["lines"]
            for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
                prefix = lines[:end]
                state = reduce_channel(player, prefix, battle["teams"][player], VOCAB)
                secrets = unrevealed_ids(battle["teams"][other], "\n".join(prefix))
                assert leaks(state.to_dict()["opponent"], secrets) == [], (path.name, player, end)
                checked += 1
    assert checked > 1000


@pytest.mark.parametrize("path", CORPUS[:3], ids=lambda p: p.name)
def test_reduction_is_deterministic(path: Path) -> None:
    battle = _load(path)
    first = reduce_channel("p1", battle["channels"]["p1"]["lines"], battle["teams"]["p1"], VOCAB)
    again = reduce_channel("p1", battle["channels"]["p1"]["lines"], battle["teams"]["p1"], VOCAB)
    assert first.to_dict() == again.to_dict()
