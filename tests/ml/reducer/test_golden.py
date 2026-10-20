"""Golden reduced-state snapshots of every committed channel fixture (ADR-010 §3, R10).

Regenerate deliberately, and review the diff, with:
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_golden.py --no-cov`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.reducer import BattleState, Vocabulary, reduce_channel

from .support import ROOT

CHANNELS = sorted((ROOT / "fixtures" / "showdown").glob("*/channels/p*.json"))
PROVENANCE = (
    "synthetic: reduced state of the channel fixture {channel}, own team synthetic-team.json"
)


def _inputs(path: Path) -> tuple[str, list[str], list[dict[str, Any]], Vocabulary]:
    fixture_dir = path.parent.parent
    log = json.loads(path.read_text())["channel"]
    team = json.loads((fixture_dir / "synthetic-team.json").read_text())
    vocab = Vocabulary.from_engine_data(json.loads((fixture_dir / "engine-data.json").read_text()))
    return log["player"], log["lines"], team, vocab


def _reduce(path: Path) -> BattleState:
    player, lines, team, vocab = _inputs(path)
    return reduce_channel(player, lines, team, vocab)


def _snapshot_path(path: Path) -> Path:
    return path.parent.parent / "reduced" / path.name


def _document(path: Path) -> str:
    document = {
        "schema_version": 1,
        "data_class": "synthetic",
        "provenance": PROVENANCE.format(channel=f"channels/{path.name}"),
        "state": _reduce(path).to_dict(),
    }
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_every_format_has_both_players_channels() -> None:
    assert [p.name for p in CHANNELS] == ["p1.json", "p2.json"] * (len(CHANNELS) // 2)
    assert CHANNELS


@pytest.mark.parametrize("path", CHANNELS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_the_reduced_state_matches_its_golden_snapshot(path: Path) -> None:
    snapshot = _snapshot_path(path)
    document = _document(path)
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(document)
    assert snapshot.read_text() == document


@pytest.mark.parametrize("path", CHANNELS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_request_point_reduces_and_the_result_is_deterministic(path: Path) -> None:
    player, lines, team, vocab = _inputs(path)
    cuts = [i + 1 for i, line in enumerate(lines) if line.startswith("|request|")]
    assert cuts
    for end in cuts:
        first = reduce_channel(player, lines[:end], team, vocab).to_dict()
        again = reduce_channel(player, lines[:end], team, vocab).to_dict()
        assert first == again
        assert json.dumps(first)  # plain JSON data, nothing else


@pytest.mark.parametrize("path", CHANNELS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_the_golden_battle_ends_as_recorded(path: Path) -> None:
    summary = json.loads((path.parent.parent / "battle.summary.json").read_text())
    state = _reduce(path)
    assert state.turn == summary["turns"]
    assert state.winner == ("own" if summary["winner"] == state.own.player else "opponent")
