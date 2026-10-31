"""The worker and the opponent sheet source (ADR-027 slice 4, ADR-025 erratum E4, #1152).

A worker whose checkpoints were made with the ``opponent_sheet`` option states the source in
``ready`` and reads a query only with a block of that source; a worker that states none refuses a
block. An ``oracle-diagnostic`` block is accepted only in the diagnostic mode.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.model.network import ModelConfig
from areneon_ml.model.options import OPPONENT_SHEET, read_options
from areneon_ml.worker.__main__ import build
from areneon_ml.worker.protocol import QueryError, check
from areneon_ml.worker.serve import EXIT_OK, SheetSupport, Worker, read_query

from tests.ml.helpers import MAX_LINE, battle_query, hello, line, make_run_dir, make_worker, session
from tests.ml.model.support import TINY_ENCODER, TINY_MODEL, sample
from tests.ml.model.test_sheet import block
from tests.ml.test_worker_features import cli

SOURCE = {
    "feature_set": "opponent-sheet/1",
    "feature_set_hash": hashlib.sha256(b"opponent-sheet sample").hexdigest(),
}


def valid() -> dict[str, Any]:
    return sample("policy-query/valid/battle-opponent-sheet.json")


def sheet_model(source: dict[str, str]) -> ModelConfig:
    options = read_options("encoder", {OPPONENT_SHEET: source})
    return replace(TINY_MODEL, encoder=replace(TINY_ENCODER, options=options))


def sheet_worker(tmp_path: Path, source: dict[str, str] | None) -> Worker:
    plain = make_worker(tmp_path, model=TINY_MODEL if source is None else sheet_model(source))
    return Worker(
        plain._scorer,
        plain._ready["message"]["engine_data_hash"],
        plain._ready["message"]["checkpoints"],
        MAX_LINE,
        sheet=None if source is None else SheetSupport(source),
    )


def test_the_ready_message_states_the_source_only_when_given(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    code, frames = session(sheet_worker(tmp_path / "a", SOURCE), line(hello()))
    assert code == EXIT_OK
    assert frames[0]["message"]["opponent_sheet"] == SOURCE
    check("policy-handshake", frames[0])
    _, plain = session(make_worker(tmp_path / "b"), line(hello()))
    assert "opponent_sheet" not in plain[0]["message"]
    check("policy-handshake", sample("policy-handshake/valid/ready-sheet.json"))


def test_a_block_goes_with_a_stated_source_only() -> None:
    query = valid()
    assert read_query(query, None, SheetSupport(SOURCE)) == query
    with pytest.raises(QueryError, match="stated no source"):
        read_query(query)
    with pytest.raises(QueryError, match="has no opponent_sheet"):
        read_query(battle_query(), None, SheetSupport(SOURCE))
    other = {**SOURCE, "feature_set_hash": "0" * 64}
    with pytest.raises(QueryError, match="another source"):
        read_query(query, None, SheetSupport(other))
    assert read_query(battle_query())["query_id"] == 1


def test_the_reader_rules_apply() -> None:
    query = sample("policy-query/reader-invalid/opponent-sheet-unsorted.json")
    with pytest.raises(QueryError, match="opponent_sheet: slot 0 item: the entries must be sorted"):
        read_query(query, None, SheetSupport(SOURCE))


def test_an_oracle_block_needs_the_diagnostic_mode() -> None:
    oracle = sample("policy-query/valid/battle-opponent-sheet-oracle.json")
    with pytest.raises(QueryError, match="needs the diagnostic mode"):
        read_query(oracle, None, SheetSupport(SOURCE))
    assert read_query(oracle, None, SheetSupport(SOURCE, diagnostic=True)) == oracle


def test_a_query_with_its_block_is_answered_and_a_wrong_block_is_an_invalid_query(
    tmp_path: Path,
) -> None:
    query = battle_query()
    query["opponent_sheet"] = block(query)
    query["opponent_sheet"]["source"]["hash"] = SOURCE["feature_set_hash"]
    worker = sheet_worker(tmp_path, SOURCE)
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    wrong = copy.deepcopy(query)
    wrong["opponent_sheet"]["slots"][0]["species"] = "notaspecies"
    (tmp_path / "w").mkdir()
    _, frames = session(sheet_worker(tmp_path / "w", SOURCE), line(hello()), line(wrong))
    assert frames[1]["reply"]["error"] == "invalid_query"


def test_the_worker_states_the_source_its_checkpoints_were_made_for(tmp_path: Path) -> None:
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    run_dir, preview, battle = make_run_dir(tmp_path / "on", sheet_model(SOURCE))
    worker = build(cli(run_dir, preview, battle))
    assert worker._ready["message"]["opponent_sheet"] == SOURCE
    query = battle_query()
    query["opponent_sheet"] = block(query)
    query["opponent_sheet"]["source"]["hash"] = SOURCE["feature_set_hash"]
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    _, frames = session(worker, line(hello()), line(battle_query()))
    assert frames[1]["reply"]["error"] == "invalid_query"
    run_dir, preview, battle = make_run_dir(tmp_path / "off")
    assert "opponent_sheet" not in build(cli(run_dir, preview, battle))._ready["message"]


def test_the_diagnostic_flag_lets_the_worker_read_an_oracle_block(tmp_path: Path) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path, sheet_model(SOURCE))
    assert build(cli(run_dir, preview, battle))._sheet == SheetSupport(SOURCE)
    flagged = build([*cli(run_dir, preview, battle), "--diagnostic-sheet"])
    assert flagged._sheet == SheetSupport(SOURCE, diagnostic=True)
