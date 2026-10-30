"""The committed usage weight table of 2026-09, cutoff 1630 (#976).

It is reviewed aggregate data. These tests pin it: it is the JCS bytes of a table that passes the
reader against the golden team vocabulary, its hash is the stated one, its sources are the input
record's, and every list kind passed the ADR-020 P1 gate (the report is journaled on the ticket).
"""

from __future__ import annotations

import json
from pathlib import Path

from areneon_contracts.contracts import jcs
from areneon_contracts.usage_weight_table import read_usage_weight_table, usage_table_hash

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9championsvgc2026regmc"
TABLE = ROOT / "config" / "teams" / FORMAT / "usage" / "2026-09-1630.json"
INPUTS = ROOT / "config" / "opponent-model" / FORMAT / "smogon-chaos.2026-09-1630.inputs.json"
VOCABULARY = json.loads(
    (ROOT / "fixtures" / "showdown" / FORMAT / "team-vocabulary.json").read_text(encoding="utf-8")
)
TABLE_HASH = "f42f3dbacd5913229bfabf874ed1f6cbc8a94488966b8643bf96c64c94b6a13b"


def test_the_table_is_its_jcs_bytes_with_the_pinned_hash_and_passes_the_reader() -> None:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    assert TABLE.read_bytes() == jcs(table) + b"\n"
    assert usage_table_hash(table) == TABLE_HASH
    read_usage_weight_table(table, VOCABULARY)


def test_the_sources_are_the_input_records_files() -> None:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    record = json.loads(INPUTS.read_text(encoding="utf-8"))
    wanted = {
        f["source_id"]: (f["url"], f["retrieved_at"], f["size"], f["sha256"])
        for f in record["files"]
    }
    got = {
        s["source_id"]: (s["url"], s["retrieved_at"], s["size"], s["sha256"])
        for s in table["sources"]
    }
    assert got == wanted
    assert (table["month"], table["cutoff"]) == (record["month"], record["cutoff"])
    assert (TABLE.parent / "2026-09-1630.NOTICE.txt").is_file()
