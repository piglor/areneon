"""The Python stream check (ADR-009 §4): a TrainingExample stream is readable
only together with a TrainingSet that matches it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, sha256_hex
from areneon_contracts.training import (
    TrainingStreamError,
    check_stream_file,
    read_stream,
    stream_lines,
)

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


def test_refuses_a_manifest_episode_the_stream_never_mentions() -> None:
    hashes = sorted([EXAMPLES[0]["episode_hash"], "f" * 64])
    with pytest.raises(TrainingStreamError, match=f"no example of Episode {'f' * 64}"):
        read_stream(STREAM, manifest_for(STREAM, episode_hashes=hashes))


def test_refuses_a_manifest_class_below_its_examples() -> None:
    private = {
        **EXAMPLES[0],
        "observation": {**EXAMPLES[0]["observation"], "data_class": "private"},
    }
    stream = stream_of(jcs(private).decode("utf-8"))
    with pytest.raises(TrainingStreamError, match="says synthetic, its examples are private"):
        read_stream(stream, manifest_for(stream))
    assert read_stream(stream, manifest_for(stream, data_class="private"))


def test_refuses_a_manifest_that_is_not_an_object() -> None:
    with pytest.raises(ContractError):
        read_stream(STREAM, [])


def test_reads_a_legacy_stream_without_candidates_or_outcome() -> None:
    legacy = sample("training-example/valid/legacy-battle.json")
    stream = stream_of(jcs(legacy).decode("utf-8"))
    manifest = sample("training-set/valid/legacy.json")
    manifest = {
        **manifest,
        "examples": 1,
        "stream_sha256": sha256_hex(stream.encode("utf-8")),
        "episode_hashes": [legacy["episode_hash"]],
    }
    assert "learner_ready" not in manifest
    assert len(read_stream(stream, manifest)) == 1


def test_a_learner_ready_set_names_its_policy_and_its_examples_are_complete() -> None:
    assert SET["learner_ready"] is True
    assert SET["behaviour_policy_id"]
    without_policy = {k: v for k, v in manifest_for(STREAM).items() if k != "behaviour_policy_id"}
    with pytest.raises(TrainingStreamError, match="must name its behaviour_policy_id"):
        read_stream(STREAM, without_policy)
    # The reader-invalid corpus sample is the same set, as stored.
    stored = sample("training-set/reader-invalid/learner-ready-no-policy.json")
    with pytest.raises(TrainingStreamError, match="must name its behaviour_policy_id"):
        read_stream(STREAM, stored)


@pytest.mark.parametrize("field", ["candidates", "outcome"])
def test_refuses_a_learner_ready_example_missing_a_field(field: str) -> None:
    incomplete = {k: v for k, v in EXAMPLES[1].items() if k != field}
    stream = stream_of(jcs(EXAMPLES[0]).decode("utf-8"), jcs(incomplete).decode("utf-8"))
    with pytest.raises(TrainingStreamError, match=f"line 2: .*has no {field}"):
        read_stream(stream, manifest_for(stream))


def test_refuses_a_choice_that_is_not_among_the_candidates() -> None:
    stored = sample("training-example/reader-invalid/choice-not-in-candidates.json")
    stream = stream_of(jcs(stored).decode("utf-8"))
    for ready in (True, False):
        manifest = manifest_for(stream)
        if not ready:
            manifest = {
                k: v
                for k, v in manifest.items()
                if k not in ("learner_ready", "behaviour_policy_id")
            }
        with pytest.raises(TrainingStreamError, match="line 1: its choice is not among"):
            read_stream(stream, manifest)


def test_the_corpus_examples_carry_candidates_and_an_outcome() -> None:
    for example in EXAMPLES:
        assert example["outcome"] in ("win", "loss", "tie")
        assert example["choice"] in [c["choice"] for c in example["candidates"]]


# The file reader (#998): the same checks as read_stream, on a stream file read in chunks.


def check_file(tmp_path: Path, stream: str, manifest: dict[str, Any]) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_bytes(stream.encode("utf-8"))
    check_stream_file(path, manifest)


def test_the_file_reader_accepts_what_the_string_reader_accepts(tmp_path: Path) -> None:
    check_file(tmp_path, STREAM, manifest_for(STREAM))
    lines = list(stream_lines(tmp_path / "stream.jsonl"))
    assert [json.loads(line) for line in lines] == EXAMPLES
    assert "".join(lines) == STREAM


def test_the_file_reader_reads_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("areneon_contracts.training.CHUNK_BYTES", 7)
    check_file(tmp_path, STREAM, manifest_for(STREAM))


def test_the_file_reader_refuses_what_the_string_reader_refuses(tmp_path: Path) -> None:
    other = {**EXAMPLES[0], "episode_hash": "0" * 64}
    foreign = stream_of(jcs(other).decode("utf-8"))
    private = {
        **EXAMPLES[0],
        "observation": {**EXAMPLES[0]["observation"], "data_class": "private"},
    }
    private_stream = stream_of(jcs(private).decode("utf-8"))
    cases: list[tuple[str, dict[str, Any], str]] = [
        (STREAM + "\n", manifest_for(STREAM), "stream_sha256"),
        (STREAM.rstrip("\n"), manifest_for(STREAM.rstrip("\n")), "newline"),
        (STREAM, manifest_for(STREAM, examples=3), "2 lines, its TrainingSet says 3"),
        (foreign, manifest_for(foreign), "line 1: its Episode is not in the TrainingSet"),
        (stream_of("not json"), manifest_for(stream_of("not json")), "line 1"),
        (
            STREAM,
            manifest_for(STREAM, episode_hashes=sorted([EXAMPLES[0]["episode_hash"], "f" * 64])),
            f"no example of Episode {'f' * 64}",
        ),
        (private_stream, manifest_for(private_stream), "says synthetic, its examples are private"),
        (STREAM, manifest_for(STREAM, episode_hashes=["b" * 64, "a" * 64]), "must be sorted"),
    ]
    for stream, manifest, message in cases:
        with pytest.raises(TrainingStreamError, match=message):
            check_file(tmp_path, stream, manifest)
