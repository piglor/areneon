"""The search label stream's reader rules (ADR-026 erratum E2 decision 4, #1288).

A label stream has ``search_target`` on every line and ``search_labels`` in its manifest; a PPO
stream has neither. The corpus samples are the same ones the TS readers refuse.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, sha256_hex
from areneon_contracts.training import (
    StreamChecker,
    TrainingStreamError,
    check_stream_file,
    read_stream,
)

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts/v1"


def sample(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    return loaded


LABELLED = sample("training-example/valid/battle-search-target.json")
PLAIN = sample("training-example/valid/battle.json")
SET = sample("training-set/valid/search-labels.json")


def stream_of(*examples: dict[str, Any]) -> str:
    return "".join(jcs(e).decode("utf-8") + "\n" for e in examples)


def manifest_for(stream: str, **changes: Any) -> dict[str, Any]:
    count = len(stream.splitlines())
    labels = {**SET["search_labels"], "labelled": count}
    base = {**SET, "examples": count, "stream_sha256": sha256_hex(stream.encode("utf-8"))}
    return {**base, "search_labels": labels, **changes}


def without_labels(manifest: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "search_labels"}


def test_reads_a_label_stream() -> None:
    stream = stream_of(LABELLED, LABELLED)
    assert len(read_stream(stream, manifest_for(stream))) == 2


def test_a_ppo_stream_has_neither() -> None:
    stream = stream_of(PLAIN)
    assert len(read_stream(stream, without_labels(manifest_for(stream)))) == 1


def test_a_label_stream_needs_its_manifest_block_and_the_reverse() -> None:
    stream = stream_of(LABELLED)
    with pytest.raises(TrainingStreamError, match="search_labels exactly when"):
        read_stream(stream, without_labels(manifest_for(stream)))
    plain = stream_of(PLAIN)
    with pytest.raises(TrainingStreamError, match="search_labels exactly when"):
        read_stream(plain, manifest_for(plain))


def test_a_stream_mixing_labelled_and_plain_lines_is_refused() -> None:
    stream = stream_of(LABELLED, PLAIN)
    with pytest.raises(TrainingStreamError, match="on every line"):
        read_stream(stream, manifest_for(stream))


def mutated(change: Any) -> dict[str, Any]:
    example = copy.deepcopy(LABELLED)
    change(example)
    return example


CASES = [
    ("prior-short", lambda e: e["search_target"]["prior_ppm"].pop(), "one entry per candidate"),
    (
        "unsorted",
        lambda e: e["search_target"]["entries"].reverse(),
        "ascending by candidate index",
    ),
    (
        "duplicate",
        lambda e: e["search_target"]["entries"].__setitem__(
            slice(None), [{"candidate": 3, "q_ppm": 1}, {"candidate": 3, "q_ppm": 2}]
        ),
        "ascending by candidate index",
    ),
    (
        "out-of-range",
        lambda e: e["search_target"]["entries"].__setitem__(
            -1, {"candidate": len(e["candidates"]), "q_ppm": 1}
        ),
        "does not have",
    ),
    (
        "pending",
        lambda e: e["candidates"][3].__setitem__("pending", True),
        "names a pending candidate",
    ),
    ("no-candidates", lambda e: e.pop("candidates"), "needs the example's candidates"),
]


@pytest.mark.parametrize(("name", "change", "message"), CASES, ids=[c[0] for c in CASES])
def test_refuses_a_bad_search_target(name: str, change: Any, message: str) -> None:
    example = mutated(change)
    stream = stream_of(example)
    manifest = manifest_for(stream)
    if name == "no-candidates":
        manifest = {k: v for k, v in manifest.items() if k not in ("learner_ready",)}
        manifest.pop("behaviour_policy_id", None)
    with pytest.raises(TrainingStreamError, match=message):
        read_stream(stream, manifest)


def test_refuses_more_than_k_max_entries_at_the_reader() -> None:
    example = mutated(
        lambda e: e["search_target"].__setitem__(
            "entries", [{"candidate": i, "q_ppm": 5} for i in range(17)]
        )
    )
    # The schema's maxItems says the same; the reader's own bound holds if the schema were loosened.
    with pytest.raises((TrainingStreamError, ContractError)):
        read_stream(stream_of(example), manifest_for(stream_of(example)))


def test_the_manifest_block_rules() -> None:
    stream = stream_of(LABELLED)
    bad_count = manifest_for(stream, search_labels={**SET["search_labels"], "labelled": 5})
    with pytest.raises(TrainingStreamError, match="labelled must equal examples"):
        read_stream(stream, bad_count)
    unsorted = manifest_for(
        stream,
        search_labels={
            **SET["search_labels"],
            "labelled": 1,
            "refused": list(reversed(SET["search_labels"]["refused"])),
        },
    )
    with pytest.raises(TrainingStreamError, match="sorted by reason"):
        read_stream(stream, unsorted)


def test_reader_invalid_samples_are_refused() -> None:
    for name in ("search-labels-unsorted", "search-labels-duplicate-reason"):
        stored = sample(f"training-set/reader-invalid/{name}.json")
        with pytest.raises(TrainingStreamError, match="sorted by reason"):
            read_stream(stream_of(LABELLED), stored)
    stored = sample("training-set/reader-invalid/search-labels-count-mismatch.json")
    with pytest.raises(TrainingStreamError, match="labelled must equal"):
        read_stream(stream_of(LABELLED), stored)


def test_pieces_join_their_label_class(tmp_path: Path) -> None:
    stream = stream_of(LABELLED)
    manifest = manifest_for(stream)
    first, second = StreamChecker(manifest), StreamChecker(manifest)
    first.line(1, stream_of(LABELLED))
    second.line(2, stream_of(PLAIN))
    assert first.classes != second.classes
    first.classes |= second.classes
    first.seen |= second.seen
    with pytest.raises(TrainingStreamError, match="on every line"):
        first.finish()
    path = tmp_path / "stream.jsonl"
    path.write_bytes(stream.encode("utf-8"))
    check_stream_file(path, manifest)
