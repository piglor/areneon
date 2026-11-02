"""The LadderRecord reader (ADR-021 sections 4, 6, 8) on the shared corpus.

Python-jsonschema and the Python reader must classify the corpus as the TypeScript reader does:
``valid`` passes, ``invalid`` fails the schema, ``reader-invalid`` passes the schema and is refused
by the reader with the words the TS test (``ladder-record.test.ts``) holds for the same sample.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import validate
from areneon_ml.loop.ladder import (
    REFERENCE_ANCHOR,
    REFERENCE_STRENGTH,
    LadderRecordError,
    ladder_record_hash,
    read_ladder_record,
)

from tests.ml.learner.support import tiny_config
from tests.ml.model.support import SAMPLES

LADDER = SAMPLES / "ladder-record"

# The first refusal each reader-invalid sample draws: the same words as the TS test.
MESSAGES: dict[str, str] = {
    "anchors-unsorted": "anchors must be sorted by anchor_id, each once",
    "cell-pairs-not-sum": "a cell's pairs must equal the sum of its scores",
    "cell-plays-itself": "a member does not play itself in the table",
    "cell-score-sum-off-step": "score_sum_ppm must be 250000 times",
    "cell-score-sum-wrong": "score_sum_ppm must be 250000 times",
    "cell-unknown-member": "a cell names a member the record does not have",
    "cell-unknown-opponent": "a cell's opponent is not in the record",
    "cells-unsorted": "cells must be sorted by member, opponent kind and opponent id, each once",
    "copy-after-health": "at most one copy",
    "copy-anchor-minimum-wrong": "a copy's anchor minima must be the least anchor pair scores",
    "copy-donor-anchor-below-loser": (
        "a copy's donor cannot have a lower anchor minimum than its loser"
    ),
    "copy-donor-weaker": "a copy's donor cannot be weaker than its loser",
    "copy-loser-is-donor": "a copy's loser must differ from its donor",
    "copy-lower-bound-too-low": "a copy needs a lower bound above 500000",
    "copy-pairs-not-cell": "a copy's pairs must equal the donor-against-loser cell's pairs",
    "copy-roots-after-wrong": "a copy's roots must be counted from the members' roots",
    "copy-roots-before-wrong": "a copy's roots must be counted from the members' roots",
    "copy-unknown-member": "a copy names a member the record does not have",
    "copy-without-anchor-cells": "a copy needs anchor cells for its donor and its loser",
    "cyclic-more-than-triples": "cyclic_triples cannot exceed the number of member triples",
    "duplicate-cell": "cells must be sorted by member, opponent kind and opponent id, each once",
    "first-selection-health": "the first selection must be the copy test's outcome",
    "health-duplicate": "health replacements must be sorted by member, each once",
    "health-names-loser": "a health replacement cannot name a copy's loser or donor",
    "health-unknown-member": "a health replacement names a member the record does not have",
    "health-unsorted": "health replacements must be sorted by member, each once",
    "lower-above-upper": "a strength's lower bound cannot exceed its upper bound",
    "members-unsorted": "members must be sorted by member, each once",
    "reference-anchor-missing": "the reference anchor must be present at exactly 1000000000",
    "reference-anchor-moved": "the reference anchor must be present at exactly 1000000000",
    "shares-do-not-sum": "a member's opponent shares must add up to 1000000",
    "snapshot-id-mismatch": "a snapshot's id must be its member and generation",
    "snapshot-lower-above-upper": "a strength's lower bound cannot exceed its upper bound",
    "snapshot-unknown-member": "a snapshot names a member the record does not have",
    "snapshots-unsorted": "snapshots must be sorted by snapshot_id, each once",
    "two-copies": "at most one copy",
    "window-end-not-generation": "the window must end at the record's generation",
    "window-start-wrong": "the window must start size generations before its end",
}


def files(kind: str) -> list[Path]:
    return sorted((LADDER / kind).glob("*.json"))


def raw(name: str, kind: str = "valid") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((LADDER / kind / f"{name}.json").read_text())
    return loaded


@pytest.mark.parametrize("path", files("valid"), ids=lambda p: p.stem)
def test_a_valid_sample_passes_the_schema_and_the_reader(path: Path) -> None:
    value = json.loads(path.read_text())
    validate("ladder-record", value)
    record = read_ladder_record(value)
    assert record.raw == value


@pytest.mark.parametrize("path", files("invalid"), ids=lambda p: p.stem)
def test_an_invalid_sample_is_refused_as_invalid(path: Path) -> None:
    with pytest.raises(LadderRecordError, match="invalid LadderRecord"):
        read_ladder_record(json.loads(path.read_text()))


def test_every_reader_invalid_sample_has_its_message() -> None:
    assert sorted(p.stem for p in files("reader-invalid")) == sorted(MESSAGES)


@pytest.mark.parametrize(("name", "message"), sorted(MESSAGES.items()))
def test_a_reader_invalid_sample_passes_the_schema_and_fails_the_reader(
    name: str, message: str
) -> None:
    value = raw(name, "reader-invalid")
    validate("ladder-record", value)
    with pytest.raises(LadderRecordError, match=message.replace("(", r"\(").replace(")", r"\)")):
        read_ladder_record(value)


def test_the_reference_anchor_and_the_hash() -> None:
    assert (REFERENCE_ANCHOR, REFERENCE_STRENGTH) == ("actor-rule@v1", 1_000_000_000)
    value = raw("none")
    reordered = dict(reversed(list(value.items())))
    assert ladder_record_hash(value) == ladder_record_hash(reordered)
    assert ladder_record_hash(value) != ladder_record_hash({**value, "cyclic_triples": 1})


def previous() -> dict[str, Any]:
    value = raw("none")
    value["generation"] = 4
    value["window"] = {"size": 3, "first_generation": 2, "last_generation": 4}
    return value


def test_the_previous_record_is_accepted_and_its_rules_are_enforced() -> None:
    assert read_ladder_record(raw("none"), previous()).generation == 5
    cases: list[tuple[str, Any]] = [
        ("another run or config", lambda p: p.update(run_id="other-run")),
        ("another run or config", lambda p: p.update(run_config_hash="b" * 64)),
        ("the generation before", lambda p: p.update(generation=3)),
        ("anchor strengths are frozen", lambda p: p["anchors"][1].update(strength=1)),
        ("frozen at entry", lambda p: p["snapshots"][0]["strength"].update(strength=1)),
    ]
    for message, change in cases:
        earlier = copy.deepcopy(previous())
        change(earlier)
        with pytest.raises(LadderRecordError, match=message):
            read_ladder_record(raw("none"), earlier)
    earlier = previous()
    earlier["snapshots"] = []
    assert len(read_ladder_record(raw("none"), earlier).raw["snapshots"]) == 1


def test_a_record_is_tied_to_its_config() -> None:
    config = tiny_config()
    value = raw("none")
    value["run_id"] = config.run_key
    value["run_config_hash"] = config.hash
    assert read_ladder_record(value, config=config).run_id == config.run_key
    with pytest.raises(LadderRecordError, match="run key"):
        read_ladder_record({**value, "run_id": "other-run"}, config=config)
    with pytest.raises(LadderRecordError, match="config's hash"):
        read_ladder_record({**value, "run_config_hash": "c" * 64}, config=config)
