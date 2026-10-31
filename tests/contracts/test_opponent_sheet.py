"""The ``opponent_sheet`` reader rules (ADR-027 section 2, ADR-025 erratum E4) on the corpus.

Python-jsonschema and the Python reader must classify the corpus as the TypeScript side does:
``valid`` passes both, ``invalid`` fails the schema, ``reader-invalid`` passes the schema and
fails ``sheet_problem``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, validate
from areneon_contracts.opponent_sheet import sheet_problem
from areneon_contracts.training import StreamChecker, TrainingStreamError

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts/v1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def files(kind: str) -> list[Path]:
    return sorted((SAMPLES / "opponent-sheet" / kind).glob("*.json"))


MESSAGES = {
    "unsorted-entries": "sorted by ppm descending then id, and unique",
    "duplicate-entry": "sorted by ppm descending then id, and unique",
    "other-ppm-wrong": "other_ppm must be max(0, 1000000",
    "revealed-two-entries": "a revealed field holds its one value",
    "unknown-with-entries": "source unknown has no entries",
    "belief-sheet-source": "a belief block has no field of source sheet",
    "refusal-not-unknown": "brought_ppm does not fit brought_source",
    "refusal-on-sheet": "only a belief block can carry a refusal",
    "engine-sheet-predicted-field": "move field at source sheet",
    "revealed-only-predicted": "revealed or unknown fields only",
    "slots-known-over-certain": "slots_known counts revealed moves",
    "brought-unknown-nonzero": "brought_ppm does not fit brought_source",
}


@pytest.mark.parametrize("path", files("valid"), ids=lambda p: p.stem)
def test_a_valid_sample_passes_the_schema_and_the_reader(path: Path) -> None:
    block = load(path)
    validate("opponent-sheet", block)
    assert sheet_problem(block) is None


@pytest.mark.parametrize("path", files("invalid"), ids=lambda p: p.stem)
def test_an_invalid_sample_fails_the_schema(path: Path) -> None:
    with pytest.raises(ContractError):
        validate("opponent-sheet", load(path))


@pytest.mark.parametrize("path", files("reader-invalid"), ids=lambda p: p.stem)
def test_a_reader_invalid_sample_passes_the_schema_and_fails_the_reader(path: Path) -> None:
    block = load(path)
    validate("opponent-sheet", block)
    problem = sheet_problem(block)
    assert problem is not None
    assert MESSAGES[path.stem] in problem


def test_every_reader_invalid_sample_has_its_message() -> None:
    assert sorted(p.stem for p in files("reader-invalid")) == sorted(MESSAGES)


def test_the_block_travels_in_a_query_and_an_example() -> None:
    validate("policy-query", load(SAMPLES / "policy-query/valid/battle-opponent-sheet.json"))
    validate("policy-query", load(SAMPLES / "policy-query/valid/battle-opponent-sheet-oracle.json"))
    validate(
        "training-example", load(SAMPLES / "training-example/valid/battle-opponent-sheet.json")
    )
    with pytest.raises(ContractError):
        validate("policy-query", load(SAMPLES / "policy-query/invalid/opponent-sheet-null.json"))
    # An oracle block is not a value of a TrainingExample: the schema itself refuses it.
    with pytest.raises(ContractError):
        validate(
            "training-example",
            load(SAMPLES / "training-example/invalid/opponent-sheet-oracle-kind.json"),
        )


def test_the_stream_reader_applies_the_rules_and_one_source_per_stream() -> None:
    manifest = load(SAMPLES / "training-set/valid/league.json")
    checker = StreamChecker(manifest)
    example = load(SAMPLES / "training-example/valid/battle-opponent-sheet.json")
    checker.line(1, json.dumps(example))
    bad = load(SAMPLES / "training-example/reader-invalid/opponent-sheet-other-wrong.json")
    with pytest.raises(TrainingStreamError, match="opponent_sheet: slot 0 item: other_ppm must be"):
        StreamChecker(manifest).line(1, json.dumps(bad))
    other = json.loads(json.dumps(example))
    other["opponent_sheet"]["source"]["hash"] = "b" * 64
    mixed = StreamChecker(manifest)
    mixed.line(1, json.dumps(example))
    mixed.line(2, json.dumps(other))
    mixed.seen = mixed.known
    with pytest.raises(TrainingStreamError, match="same opponent sheet source"):
        mixed.finish()


def _lead(block: dict[str, Any]) -> dict[str, Any]:
    lead: dict[str, Any] = block["slots"][0]
    return lead


def _set(name: str, value: Any) -> Any:
    def change(block: dict[str, Any]) -> None:
        _lead(block)[name] = value

    return change


def _reverse_moves(block: dict[str, Any]) -> None:
    _lead(block)["moves"]["entries"].reverse()


def _bring_revealed(ppm: int) -> Any:
    def change(block: dict[str, Any]) -> None:
        _lead(block)["brought_source"] = "revealed"
        _lead(block)["brought_ppm"] = ppm

    return change


def _ids(source: str, *entries: tuple[str | None, int], other: int = 0) -> dict[str, Any]:
    return {
        "source": source,
        "entries": [{"id": i, "ppm": p} for i, p in entries],
        "other_ppm": other,
    }


def _moves(source: str, known: int, *entries: tuple[str, int]) -> dict[str, Any]:
    return {
        "source": source,
        "entries": [{"id": i, "inclusion_ppm": p} for i, p in entries],
        "slots_known": known,
    }


FULL = 1_000_000
THREE_FULL = (("a", FULL), ("b", FULL), ("c", FULL))
RULES = [
    (
        "belief",
        _set("nature", _ids("unknown", other=5)),
        "nature: a field of source unknown has other_ppm 0",
    ),
    (
        "belief",
        _set("moves", _moves("unknown", 2)),
        "moves: a field of source unknown has slots_known 0",
    ),
    (
        "belief",
        _set("item", _ids("predicted")),
        "item: a field with a source other than unknown has entries",
    ),
    (
        "belief",
        _set("ability", _ids("predicted", ("a1", 600000), ("a2", 600000))),
        "ability: the entries sum to more than 1000000",
    ),
    (
        "belief",
        _set("item", _ids("predicted", (None, 500000), ("ix", 500000))),
        "item: the entries must be sorted",
    ),
    (
        "belief",
        _set("moves", _moves("revealed", 3, *THREE_FULL)),
        "a revealed move field has all four moves known",
    ),
    ("belief", _reverse_moves, "moves: the entries must be sorted"),
    ("engine-sheet", _bring_revealed(FULL), "an engine-sheet block takes brought_ppm"),
    ("belief", _bring_revealed(999999), "brought_ppm does not fit brought_source"),
]


@pytest.mark.parametrize(("name", "change", "message"), RULES)
def test_each_rule_has_its_message(name: str, change: Any, message: str) -> None:
    block = load(SAMPLES / "opponent-sheet/valid" / f"{name}.json")
    assert sheet_problem(block) is None
    change(block)
    validate("opponent-sheet", block)
    problem = sheet_problem(block)
    assert problem is not None
    assert message in problem


def test_a_sum_a_few_ppm_above_a_million_is_accepted() -> None:
    block = load(SAMPLES / "opponent-sheet/valid/belief.json")
    _lead(block)["ability"] = _ids("predicted", ("a1", 500001), ("a2", 500001))
    assert sheet_problem(block) is None
