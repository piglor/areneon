"""The reducer and encoder on a player's own `|error|` lines (#909).

`fixtures/showdown/<format>/unavailable/` holds seeded battles in which a player
submitted a pending candidate that the engine refused as `[Unavailable choice]`
(`packages/showdown-bridge/src/random-corpus.ts`, `pendingFirstActor`; CI re-plays each
one). The refusal reaches only that player's own channel, and a new request follows it.
Rewrite the reduced snapshots deliberately with
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_unavailable.py --no-cov`
and review the diff.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.model.encoder import Encoder
from areneon_ml.reducer import reduce_channel

from tests.ml.model.support import TINY_ENCODER, game, observation

from .support import FORMAT_DIR, VOCAB, leaks, unattributed, unrevealed_ids

CORPUS = sorted((FORMAT_DIR / "unavailable").glob("b*.json"))
PLAYERS = ("p1", "p2")
PROVENANCE = (
    "synthetic: reduced state of {channel}, battles of fixtures-v1 teams, pending-first play"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _lines(battle: dict[str, Any], player: str) -> list[str]:
    return list(battle["channels"][player]["lines"])


def _error_lines(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.startswith("|error|")]


def _reduce(battle: dict[str, Any], player: str, lines: list[str]) -> dict[str, Any]:
    return reduce_channel(player, lines, battle["teams"][player], VOCAB).to_dict()


def test_the_corpus_reaches_an_unavailable_choice() -> None:
    manifest = json.loads((FORMAT_DIR / "unavailable" / "manifest.json").read_text())
    assert len(CORPUS) == len(manifest["battles"]) >= 2
    for path in CORPUS:
        battle = _load(path)
        assert battle["data_class"] == "synthetic"
        errors = [line for p in PLAYERS for line in _lines(battle, p) if line.startswith("|error|")]
        assert errors
        assert all(line.startswith("|error|[Unavailable choice] ") for line in errors)


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
@pytest.mark.parametrize("player", PLAYERS)
def test_every_request_point_reduces_and_an_error_line_changes_nothing(
    path: Path, player: str
) -> None:
    battle = _load(path)
    lines = _lines(battle, player)
    for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
        prefix = lines[:end]
        with_errors = _reduce(battle, player, prefix)
        without = _reduce(battle, player, [x for x in prefix if not x.startswith("|error|")])
        assert with_errors == without, (path.name, player, end)


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
@pytest.mark.parametrize("player", PLAYERS)
def test_the_encoder_takes_every_decision_with_the_error_lines_in_the_log(
    path: Path, player: str
) -> None:
    battle = _load(path)
    lines = _lines(battle, player)
    encoder = Encoder(game(), dataclasses.replace(TINY_ENCODER, max_moves=8))
    encoded = 0
    for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
        prefix = lines[:end]
        state = reduce_channel(player, prefix, battle["teams"][player], VOCAB)
        if state.request_kind == "wait":
            continue
        kind = "preview" if state.request_kind == "preview" else "battle"
        encoder.encode(
            {
                "observation": observation(player, prefix, battle["teams"][player], kind),
                "candidates": [],
            }
        )
        encoded += 1
    assert encoded


def test_the_player_sees_the_refusal_before_the_request_that_follows() -> None:
    battle = _load(CORPUS[0])
    seen = 0
    for player in PLAYERS:
        lines = _lines(battle, player)
        for i in _error_lines(lines):
            assert lines[i - 1].startswith("|request|") or lines[i - 1].startswith("|error|")
            assert lines[i + 1].startswith("|request|")
            seen += 1
    assert seen


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
def test_the_canary_holds_at_every_request_point_with_error_lines(path: Path) -> None:
    battle = _load(path)
    for player, other in (("p1", "p2"), ("p2", "p1")):
        lines = _lines(battle, player)
        for end in [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]:
            prefix = lines[:end]
            opponent = _reduce(battle, player, prefix)["opponent"]
            secrets = unrevealed_ids(battle["teams"][other], "\n".join(prefix))
            assert leaks(opponent, secrets) == []
            assert unattributed(opponent, prefix, other) == []


def test_negative_control_the_canary_scan_sees_a_contaminated_state() -> None:
    battle = _load(CORPUS[0])
    lines = _lines(battle, "p1")
    state = _reduce(
        battle, "p1", lines[: max(i for i, x in enumerate(lines) if x.startswith("|request|")) + 1]
    )
    secrets = unrevealed_ids(battle["teams"]["p2"], "")
    assert leaks(state["own"], secrets), "the scan must fire on exact own-side values"


def _snapshots() -> list[tuple[Path, str]]:
    return [(p, player) for p in CORPUS for player in PLAYERS]


@pytest.mark.parametrize(
    ("path", "player"), _snapshots(), ids=lambda v: v.name if isinstance(v, Path) else v
)
def test_the_reduced_state_matches_its_golden_snapshot(path: Path, player: str) -> None:
    battle = _load(path)
    state = _reduce(battle, player, _lines(battle, player))
    document = (
        json.dumps(
            {
                "schema_version": 1,
                "data_class": "synthetic",
                "provenance": PROVENANCE.format(channel=f"unavailable/{path.name} {player}"),
                "state": state,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    snapshot = FORMAT_DIR / "reduced" / f"unavailable-{path.stem}-{player}.json"
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.write_text(document)
    assert snapshot.read_text() == document
