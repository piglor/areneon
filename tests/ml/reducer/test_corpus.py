"""The reducer on ordinary play: the random-legal corpus of `fixtures/showdown/<format>/random/`.

Seeded random-legal vs random-legal battles on random synthetic teams
(`packages/showdown-bridge/src/random-corpus.ts`; CI re-plays each one).
Every channel must reduce without a refusal (the Illusion ending `|replace|` included, #851);
the canary runs over every request point of every channel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.reducer import ReducerError, reduce_channel
from areneon_ml.reducer.vocab import to_id

from .support import FORMAT_DIR, VOCAB, leaks, unattributed, unrevealed_ids

CORPUS = sorted((FORMAT_DIR / "random").glob("b*.json"))


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


def test_every_channel_reduces() -> None:
    for path in CORPUS:
        battle = _load(path)
        for player in ("p1", "p2"):
            assert _first_refusal(player, battle) is None, f"{path.name} {player}"


def test_the_corpus_exercises_the_lines_that_broke_the_first_version() -> None:
    text = "\n".join(line for p in CORPUS for line in _load(p)["channels"]["p1"]["lines"])
    markers = ("|-mega|", "|-formechange|", "[notarget]", "/100g", "/100y", "|-fail|", "|replace|")
    for marker in markers:
        assert marker in text, marker


def test_the_canary_holds_at_every_request_point_of_every_channel() -> None:
    # Two checks per state: no id of the opponent's team that the channel never names, and
    # every revealed value is named by a line about an opponent Pokemon. It cannot tell a
    # value the player was legitimately shown from the same value read off another line
    # (a Protect the own side used, say): it catches values from nowhere, not misattribution.
    checked = 0
    for path in CORPUS:
        battle = _load(path)
        for player, other in (("p1", "p2"), ("p2", "p1")):
            lines = battle["channels"][player]["lines"]
            for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
                prefix = lines[:end]
                state = reduce_channel(player, prefix, battle["teams"][player], VOCAB)
                secrets = unrevealed_ids(battle["teams"][other], "\n".join(prefix))
                opponent = state.to_dict()["opponent"]
                assert leaks(opponent, secrets) == [], (path.name, player, end)
                assert unattributed(opponent, prefix, other) == [], (path.name, player, end)
                checked += 1
    assert checked > 1000


@pytest.mark.parametrize("path", CORPUS[:3], ids=lambda p: p.name)
def test_reduction_is_deterministic(path: Path) -> None:
    battle = _load(path)
    first = reduce_channel("p1", battle["channels"]["p1"]["lines"], battle["teams"]["p1"], VOCAB)
    again = reduce_channel("p1", battle["channels"]["p1"]["lines"], battle["teams"]["p1"], VOCAB)
    assert first.to_dict() == again.to_dict()


def _half(state: Any, player: str, side: str) -> Any:
    return state.own if side == player else state.opponent


def _mon(state: Any, player: str, side: str, name: str) -> Any:
    return next(m for m in _half(state, player, side).roster if m.name == name)


def test_frisk_credits_the_item_to_the_target_and_the_ability_to_the_frisker() -> None:
    """#1171, on the committed channels: `|-item|TARGET|ITEM|[from] ability: Frisk|[of] FRISKER`."""
    frisk = re.compile(
        r"^\|-item\|(p[12])[ab]: ([^|]+)\|([^|]+)\|\[from\] ability: Frisk"
        r"\|\[of\] (p[12])[ab]: ([^|]+)"
    )
    reveals = 0
    for path in CORPUS:
        battle = _load(path)
        for player in ("p1", "p2"):
            lines = battle["channels"][player]["lines"]
            team = battle["teams"][player]
            for index, line in enumerate(lines):
                match = frisk.match(line)
                if match is None:
                    continue
                reveals += 1
                t_side, t_name, item, f_side, f_name = match.groups()
                where = (path.name, player, line)
                before = reduce_channel(player, lines[:index], team, VOCAB)
                after = reduce_channel(player, lines[: index + 1], team, VOCAB)
                target = _mon(after, player, t_side, t_name)
                assert target.item == to_id(item), where
                assert _mon(after, player, f_side, f_name).ability == "frisk", where
                assert target.ability == _mon(before, player, t_side, t_name).ability, where
    assert reveals > 10
