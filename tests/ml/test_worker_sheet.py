"""The worker applies the opponent_sheet reader rules to a query (ADR-027, ADR-025 erratum E4).

The encoder does not read the block yet (the encoder option is a later slice), so a valid block is
accepted and changes nothing; a block the reader rules refuse is an ``invalid_query``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.worker.protocol import QueryError
from areneon_ml.worker.serve import read_query

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts/v1/policy-query"


def sample(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    return loaded


def test_a_valid_block_is_accepted_and_the_query_unchanged() -> None:
    query = sample("valid/battle-opponent-sheet.json")
    assert read_query(query) is query
    assert read_query(sample("valid/battle-opponent-sheet-oracle.json"))["opponent_sheet"]


def test_a_block_the_reader_rules_refuse_is_an_invalid_query() -> None:
    with pytest.raises(QueryError, match="opponent_sheet: slot 0 item: the entries must be sorted"):
        read_query(sample("reader-invalid/opponent-sheet-unsorted.json"))
