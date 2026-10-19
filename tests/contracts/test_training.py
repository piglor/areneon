"""The Python stream check (ADR-009 §4): a TrainingExample stream is readable
only together with a TrainingSet that matches it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, sha256_hex
from areneon_contracts.training import TrainingStreamError, read_stream

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts/v1"


def sample(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    return loaded


def stream_of(*lines: str) -> str:
    return "".join(f"{line}\n" for line in lines)


EXAMPLES = [
    sample("training-example/valid/preview.json"),
    sample("training-example/valid/battle.json"),
]
STREAM = stream_of(*(jcs(e).decode("utf-8") for e in EXAMPLES))
SET = sample("training-set/valid/league.json")


def manifest_for(stream: str, **changes: Any) -> dict[str, Any]:
    return {
        **SET,
        "examples": len(stream.splitlines()),
        "stream_sha256": sha256_hex(stream.encode("utf-8")),
        **changes,
    }


def test_reads_the_stream_its_manifest_matches() -> None:
    examples = read_stream(STREAM, manifest_for(STREAM))
    assert len(examples) == len(EXAMPLES)
    assert [e.model_dump(mode="json", by_alias=True) for e in examples] == EXAMPLES


def test_the_corpus_stream_manifest_is_a_valid_manifest() -> None:
    assert SET["episode_hashes"] == [EXAMPLES[0]["episode_hash"]]


def test_refuses_a_stream_that_does_not_hash_to_its_manifest() -> None:
    manifest = manifest_for(STREAM)
    with pytest.raises(TrainingStreamError, match="stream_sha256"):
        read_stream(STREAM + "\n", manifest)


def test_refuses_a_missing_final_newline() -> None:
    stream = STREAM.rstrip("\n")
    with pytest.raises(TrainingStreamError, match="newline"):
        read_stream(stream, manifest_for(stream))


def test_refuses_a_line_count_that_differs() -> None:
    with pytest.raises(TrainingStreamError, match="2 lines, its TrainingSet says 3"):
        read_stream(STREAM, manifest_for(STREAM, examples=3))


def test_refuses_an_example_of_an_episode_the_manifest_does_not_list() -> None:
    other = {**EXAMPLES[0], "episode_hash": "0" * 64}
    stream = stream_of(jcs(other).decode("utf-8"))
    with pytest.raises(TrainingStreamError, match="line 1: its Episode is not in the TrainingSet"):
        read_stream(stream, manifest_for(stream))


@pytest.mark.parametrize(
    "line", ["not json", json.dumps({**EXAMPLES[0], "provenance": {"seed": "sodium,1"}})]
)
def test_refuses_a_line_that_is_not_a_training_example(line: str) -> None:
    stream = stream_of(line)
    with pytest.raises(TrainingStreamError, match="line 1"):
        read_stream(stream, manifest_for(stream))


def test_refuses_a_manifest_off_the_schema_or_unsorted() -> None:
    with pytest.raises(ContractError, match="training-set"):
        read_stream(STREAM, {**manifest_for(STREAM), "examples": 0})
    for field in ("episode_hashes", "evaluation_manifest_hashes"):
        with pytest.raises(TrainingStreamError, match=f"{field} must be sorted and unique"):
            read_stream(STREAM, manifest_for(STREAM, **{field: ["b" * 64, "a" * 64]}))
