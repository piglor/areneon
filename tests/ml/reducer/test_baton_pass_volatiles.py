"""The volatiles a Baton Pass or Shed Tail hands over, through the reducer (#1102).

The engine's `copyVolatileFrom` copies every volatile of the Pokemon that goes out except the
ones whose condition data says `noCopy` (a Shed Tail switch copies Substitute alone) and prints no
line for the copy; only the switch's `[from] Baton Pass` / `[from] Shed Tail` says so. The rule is
the committed table `ml/areneon_ml/reducer/data/volatile-copy.<format>.json`, exported from the
pinned engine (`packages/showdown-bridge/scripts/export-volatile-copy.ts`).

`fixtures/showdown/<format>/baton-pass-volatiles/battle.json` is a seeded random-legal battle on a
team of three golden Pokemon and three Substitute + Baton Pass users
(`packages/showdown-bridge/scripts/capture-baton-pass-battle.ts`; CI re-plays it). Rewrite the
reduced snapshots deliberately with
`ARENEON_UPDATE_GOLDEN=1 uv run pytest tests/ml/reducer/test_baton_pass_volatiles.py --no-cov`
and review the diff.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.inspector.replay_page import dex_for_ids
from areneon_ml.reducer import Vocabulary, VocabularyError, reduce_channel
from areneon_ml.reducer import vocab as vocab_module

from .support import FORMAT_DIR, VOCAB

BATTLE: dict[str, Any] = json.loads(
    (FORMAT_DIR / "baton-pass-volatiles" / "battle.json").read_text()
)
PROVENANCE = "synthetic: reduced state of baton-pass-volatiles/battle.json {player}"
PLAYERS = ("p1", "p2")
TABLE = vocab_module.DATA_DIR / f"volatile-copy.{VOCAB.format_id}.json"


def _lines(player: str) -> list[str]:
    return list(BATTLE["channels"][player]["lines"])


def _reduce(player: str, lines: list[str], vocab: Vocabulary = VOCAB) -> Any:
    return reduce_channel(player, lines, BATTLE["teams"][player], vocab)


def _active(state: Any, switch: str) -> Any:
    ref = switch.split("|")[2]
    side = state.own if ref.startswith(state.player) else state.opponent
    return next(m for m in side.roster if m.active_slot == ref[2])


def _first_pass(lines: list[str]) -> int:
    return next(i for i, ln in enumerate(lines) if ln.startswith("|switch|") and "Baton Pass" in ln)


def _outgoing(lines: list[str], index: int) -> str:
    return lines[index].split("|")[2][:3]


@pytest.mark.parametrize("player", PLAYERS)
def test_a_standing_substitute_is_passed_and_then_ends_on_the_receiver(player: str) -> None:
    lines = _lines(player)
    index = _first_pass(lines)
    before = _reduce(player, lines[:index])
    ref = lines[index].split("|")[2][:3]
    side = before.own if ref.startswith(player) else before.opponent
    sender = next(m for m in side.roster if m.active_slot == ref[2])
    assert sender.name == "Leafeon"
    assert "substitute" in sender.volatiles
    after = _reduce(player, lines[: index + 1])
    receiver = _active(after, lines[index])
    assert receiver.name == "Charizard"
    assert "substitute" in receiver.volatiles
    assert sender.volatiles  # the earlier state is a copy: the one that left keeps none below
    side_after = after.own if ref.startswith(player) else after.opponent
    gone = next(m for m in side_after.roster if m.name == "Leafeon")
    assert gone.active_slot is None
    assert gone.volatiles == set()
    ended = next(
        i for i, ln in enumerate(lines) if ln.startswith("|-end|p1b: Charizard|Substitute")
    )
    assert "substitute" in _active(_reduce(player, lines[:ended]), lines[index]).volatiles
    assert "substitute" not in _active(_reduce(player, lines[: ended + 1]), lines[index]).volatiles


def _with(player: str, extra: list[str], cause: str) -> tuple[list[str], str]:
    lines = _lines(player)
    index = _first_pass(lines)
    switch = lines[index].replace("Baton Pass", cause)
    ref = _outgoing(lines, index)
    # `extra` lines name the passer by its position, `{ref}`
    return [*lines[:index], *(e.format(ref=ref) for e in extra), switch], switch


def test_a_baton_pass_keeps_every_volatile_the_engine_does_not_block() -> None:
    lines, switch = _with(
        "p1",
        ["|-start|{ref}: Leafeon|move: Leech Seed", "|-start|{ref}: Leafeon|Encore"],
        "Baton Pass",
    )
    got = _active(_reduce("p1", lines), switch).volatiles
    assert got == {"substitute", "leechseed"}  # Encore is `noCopy` in the engine's data


def test_a_shed_tail_passes_the_substitute_alone() -> None:
    lines, switch = _with("p1", ["|-start|{ref}: Leafeon|move: Leech Seed"], "Shed Tail")
    assert _active(_reduce("p1", lines), switch).volatiles == {"substitute"}


def test_a_plain_switch_and_a_baton_pass_of_a_pokemon_with_none_pass_none() -> None:
    lines = _lines("p1")
    index = _first_pass(lines)
    plain = lines[index].replace("|[from] Baton Pass", "")
    assert _active(_reduce("p1", [*lines[:index], plain]), plain).volatiles == set()
    # the second pass of the battle: nothing stands on the passer
    passer = _outgoing(lines, index)
    cleared = [*lines[:index], f"|-end|{passer}: Leafeon|Substitute", lines[index]]
    assert _active(_reduce("p1", cleared), lines[index]).volatiles == set()


def test_a_suppressed_ability_goes_with_gastro_acid() -> None:
    lines, switch = _with("p1", ["|-endability|{ref}: Leafeon"], "Baton Pass")
    assert _active(_reduce("p1", lines), switch).ability_suppressed is True
    lines, switch = _with("p1", ["|-endability|{ref}: Leafeon"], "Shed Tail")
    assert _active(_reduce("p1", lines), switch).ability_suppressed is False


def test_a_vocabulary_without_the_table_refuses_the_pass() -> None:
    bare = dataclasses.replace(VOCAB, not_copied=None, shed_tail_copies=None)
    lines, _ = _with("p1", [], "Baton Pass")
    with pytest.raises(VocabularyError, match="volatile-copy"):
        _reduce("p1", lines, bare)
    assert _reduce("p1", lines[:-1], bare)  # nothing else needs it


@pytest.mark.parametrize("player", PLAYERS)
def test_the_reduced_state_matches_its_golden_snapshot(player: str) -> None:
    document = (
        json.dumps(
            {
                "schema_version": 1,
                "data_class": "synthetic",
                "provenance": PROVENANCE.format(player=player),
                "state": _reduce(player, _lines(player)).to_dict(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    snapshot = FORMAT_DIR / "reduced" / f"baton-pass-volatiles-{player}.json"
    if os.environ.get("ARENEON_UPDATE_GOLDEN") == "1":
        snapshot.write_text(document)
    assert snapshot.read_text() == document


# -- the table ----------------------------------------------------------------------------------


def _table() -> dict[str, Any]:
    table: dict[str, Any] = json.loads(TABLE.read_text())
    return table


def _seal(table: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in table.items() if k != "volatile_copy_hash"}
    return {**body, "volatile_copy_hash": sha256_hex(jcs(body))}


def test_the_committed_table_is_sealed_and_loaded() -> None:
    table = _table()
    body = {k: v for k, v in table.items() if k != "volatile_copy_hash"}
    assert sha256_hex(jcs(body)) == table["volatile_copy_hash"]
    assert table["showdown_commit"] == VOCAB.showdown_commit
    assert VOCAB.not_copied == frozenset(table["not_copied"])
    assert VOCAB.shed_tail_copies == frozenset({"substitute"})
    assert "encore" in VOCAB.not_copied
    assert "leechseed" not in VOCAB.not_copied


def _point_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, table: dict[str, Any]) -> None:
    (tmp_path / TABLE.name).write_text(json.dumps(table))
    monkeypatch.setattr(vocab_module, "DATA_DIR", tmp_path)


def test_a_table_for_another_pin_is_not_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_at(monkeypatch, tmp_path, _seal({**_table(), "showdown_commit": "0" * 40}))
    assert vocab_module.read_volatile_copy(VOCAB.showdown_commit, VOCAB.format_id) is None


def test_a_missing_table_is_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vocab_module, "DATA_DIR", tmp_path)
    assert vocab_module.read_volatile_copy(VOCAB.showdown_commit, VOCAB.format_id) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda t: {**t, "not_copied": [*t["not_copied"], "extra"]},  # hash no longer matches
        lambda t: {**t, "unexpected": 1},
        lambda t: _seal({**t, "schema_version": 2}),
        lambda t: _seal({**t, "format_id": "gen9other"}),
    ],
)
def test_a_bad_table_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: Any
) -> None:
    _point_at(monkeypatch, tmp_path, change(_table()))
    with pytest.raises(VocabularyError):
        vocab_module.read_volatile_copy(VOCAB.showdown_commit, VOCAB.format_id)


def test_the_replay_page_carries_the_table_for_its_interpreter() -> None:
    table = json.loads((FORMAT_DIR / "engine-data.json").read_text())
    dex = dex_for_ids(table, set())
    assert dex["volatile_copy"] == {
        "not_copied": _table()["not_copied"],
        "shed_tail_copies": ["substitute"],
    }
    assert "volatile_copy" not in dex_for_ids({**table, "showdown_commit": "0" * 40}, set())
