"""The CheckpointRecord contract and its reader (ADR-010 §5)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, parse, sha256_hex
from areneon_ml.checkpoint.record import CheckpointRecord, RecordError, read_record

from .support import SAMPLES

DIR = SAMPLES / "checkpoint-record"


def load(kind: str, name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((DIR / kind / f"{name}.json").read_text())
    return loaded


def names(kind: str) -> list[str]:
    return sorted(p.stem for p in (DIR / kind).glob("*.json"))


@pytest.mark.parametrize("name", names("valid"))
def test_a_valid_record_reads_and_round_trips(name: str) -> None:
    raw = load("valid", name)
    record = read_record(raw)
    assert isinstance(record, CheckpointRecord)
    assert jcs(record.to_json()) == jcs(raw)
    assert record.record_hash == sha256_hex(jcs(raw))
    parse("checkpoint-record", raw)


def test_the_valid_samples_cover_a_warm_started_a_fresh_and_an_exploiter_record() -> None:
    records = {n: read_record(load("valid", n)) for n in names("valid")}
    assert records["battle"].parent is not None
    assert records["preview-fresh"].parent is None
    assert records["preview-fresh"].role == "preview"
    assert records["exploiter"].exploiter_target == records["exploiter"].league_opponents[0]


@pytest.mark.parametrize("name", names("invalid"))
def test_an_invalid_record_is_refused_by_the_contract(name: str) -> None:
    with pytest.raises(ContractError):
        read_record(load("invalid", name))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("unsorted-league-opponents", "league_opponents must be sorted and unique"),
        ("duplicate-training-sets", "training_set_hashes must be sorted and unique"),
        ("parent-is-self", "its own parent"),
        ("opponent-is-self", "train against itself"),
        ("exploiter-of-self", "its own exploiter target"),
    ],
)
def test_a_record_the_profile_cannot_judge_is_refused_by_the_reader(
    name: str, message: str
) -> None:
    raw = load("reader-invalid", name)
    parse("checkpoint-record", raw)  # schema-valid
    with pytest.raises(RecordError, match=message):
        read_record(raw)


def test_the_reader_invalid_samples_are_exactly_the_ones_tested() -> None:
    assert len(names("reader-invalid")) == 5
