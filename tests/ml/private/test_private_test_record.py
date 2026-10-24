"""The PrivateTestRecord reader and its interval (Python twin of ``private-test-record.ts``)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.private.private_test_record import (
    INTERVAL_LABEL,
    LABEL,
    PrivateTestRecordError,
    bootstrap_key,
    match_interval,
    opponent_name,
    pair_scores,
    private_test_allowed,
    read_private_test_record,
    record_hash,
    summary_of,
)
from areneon_ml.progress.stats import pair_interval

DIR = Path(__file__).resolve().parents[3] / "fixtures/contracts/v1/private-test-record"


def sample(kind: str, name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DIR / kind / name).read_text(encoding="utf-8"))
    return found


def names(kind: str) -> list[str]:
    return sorted(p.name for p in (DIR / kind).iterdir())


def test_reads_every_valid_sample_and_refuses_every_reader_invalid_one() -> None:
    assert len(names("valid")) == 3
    for name in names("valid"):
        assert read_private_test_record(sample("valid", name))["data_class"] == "private"
    assert len(names("reader-invalid")) > 15
    for name in names("reader-invalid"):
        with pytest.raises(PrivateTestRecordError):
            read_private_test_record(sample("reader-invalid", name))


def test_refuses_every_schema_invalid_sample_also_with_the_private_label_put_back() -> None:
    # synthetic-class is a sound record with the wrong label, so it is invalid for that alone.
    for name in names("invalid"):
        value = sample("invalid", name)
        with pytest.raises(PrivateTestRecordError):
            read_private_test_record(value)
        if name != "synthetic-class.json":
            with pytest.raises(PrivateTestRecordError):
                read_private_test_record({**value, "data_class": "private"})
    flipped = {**sample("invalid", "synthetic-class.json"), "data_class": "private"}
    assert read_private_test_record(flipped)["data_class"] == "private"


def test_a_message_states_a_rule_and_never_repeats_a_value() -> None:
    canary = "CanaryValue123456"
    for broken in (
        {**sample("valid", "staging.json"), "run_id": canary, "extra": canary},
        {**sample("valid", "staging.json"), "test_set": {"commitment": canary}},
    ):
        with pytest.raises(PrivateTestRecordError) as caught:
            read_private_test_record(broken)
        assert canary not in str(caught.value)
    with pytest.raises(PrivateTestRecordError):
        read_private_test_record("not an object")


def test_names_and_the_sources_a_private_pair_may_use() -> None:
    assert opponent_name("actor-rule@v1") == "actor-rule"
    assert opponent_name("policy:" + "a" * 64) == "pair"
    assert private_test_allowed("snapshot", "holdout")
    assert private_test_allowed("dataset", "all")
    assert not private_test_allowed("snapshot", "all")
    assert not private_test_allowed("staging", "all")


def test_the_interval_is_derived_from_the_record_alone_and_is_repeatable() -> None:
    record = read_private_test_record(sample("valid", "staging.json"))
    first = summary_of(record)
    assert first == summary_of(copy.deepcopy(record))
    match = record["matches"][0]
    # Three pairs scored 2, 2 and 4 half-games: the mean is 8 / 12 of a score.
    assert pair_scores(match) == [2, 2, 4]
    assert first[0]["pair_score_sum"] == 8
    assert first[0]["score_ppm"] == 8 * 1_000_000 // 12
    assert first[0]["low_ppm"] <= first[0]["score_ppm"] <= first[0]["high_ppm"]
    assert (first[0]["low_ppm"], first[0]["high_ppm"]) == (
        match_interval(record, match).low_ppm,
        match_interval(record, match).high_ppm,
    )
    # The stream depends on the commitment and the match: change either and the key changes.
    commitment = record["test_set"]["commitment"]
    assert bootstrap_key(commitment, "a-b") != bootstrap_key(commitment, "a-c")
    assert bootstrap_key(commitment, "a-b") != bootstrap_key("0" * 64, "a-b")
    assert match_interval(record, match) == pair_interval(
        pair_scores(match), bootstrap_key(commitment, match["match_id"])
    )


def test_a_record_has_a_stable_hash_and_the_label_is_the_owners() -> None:
    record = read_private_test_record(sample("valid", "staging.json"))
    assert len(record_hash(record)) == 64
    assert record_hash(record) == record_hash(json.loads(jcs(record)))
    assert LABEL == "private test set; not the Training gate; not validation"
    assert "bootstrap" in INTERVAL_LABEL
