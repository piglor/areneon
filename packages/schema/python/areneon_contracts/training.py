"""Check a TrainingExample stream against its TrainingSet (ADR-009 §4).

The TS training loader writes the stream and its manifest. Python training
reads only that stream: ``read_stream`` validates the manifest, refuses a
stream whose digest or line count differs from it, then validates and parses
every line as a ``training-example``. What the contract profile cannot say
(it has no if/then) is checked here: a ``learner_ready`` set names its
``behaviour_policy_id`` and every one of its examples carries ``candidates``
and ``outcome`` (ADR-010 section 5), and an example's ``choice`` is among its
candidates. A stream with no matching manifest is
not readable at all: there is no function that reads one alone.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from areneon_contracts.contracts import ContractError, parse, sha256_hex


class TrainingStreamError(ValueError):
    """A stream does not match its TrainingSet manifest."""


def _strictly_ascending(hashes: list[str]) -> bool:
    return all(a < b for a, b in itertools.pairwise(hashes))


def _check_candidates(number: int, raw: dict[str, Any], ready: bool) -> None:
    """The cross-field rules of one example that the profile cannot say."""
    if ready:
        for field in ("candidates", "outcome"):
            if field not in raw:
                raise TrainingStreamError(
                    f"line {number}: the set is learner_ready, its example has no {field}"
                )
    candidates = raw.get("candidates")
    if candidates is not None and raw["choice"] not in [c["choice"] for c in candidates]:
        raise TrainingStreamError(f"line {number}: its choice is not among its candidates")


def _check_manifest(manifest: object) -> bool:
    """The manifest's own rules; True when the set is learner_ready."""
    parse("training-set", manifest)
    if not isinstance(manifest, dict):  # parse() has validated an object; this narrows it
        raise TrainingStreamError("training-set: not an object")
    for field in ("episode_hashes", "evaluation_manifest_hashes"):
        if not _strictly_ascending(manifest[field]):
            raise TrainingStreamError(f"training-set: {field} must be sorted and unique")
    ready = manifest.get("learner_ready") is True
    if ready and "behaviour_policy_id" not in manifest:
        raise TrainingStreamError("a learner_ready TrainingSet must name its behaviour_policy_id")
    return ready


def _check_lines(
    lines: Iterable[str], manifest: dict[str, Any], ready: bool, keep: bool
) -> list[BaseModel]:
    """Validate every line against the manifest; the examples are kept only when asked."""
    known = set(manifest["episode_hashes"])
    examples: list[BaseModel] = []
    seen: set[str] = set()
    classes: set[str] = set()
    for number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
            example = parse("training-example", raw)
        except (json.JSONDecodeError, ContractError) as error:
            raise TrainingStreamError(f"line {number}: {error}") from error
        _check_candidates(number, raw, ready)
        if raw["episode_hash"] not in known:
            raise TrainingStreamError(f"line {number}: its Episode is not in the TrainingSet")
        seen.add(raw["episode_hash"])
        classes.add(raw["observation"]["data_class"])
        if keep:
            examples.append(example)
    if seen != known:
        missing = sorted(known - seen)
        raise TrainingStreamError(f"the stream holds no example of Episode {missing[0]}")
    highest = "private" if "private" in classes else "synthetic"
    if manifest["data_class"] != highest:
        raise TrainingStreamError(
            f"the TrainingSet says {manifest['data_class']}, its examples are {highest}"
        )
    return examples


def read_stream(stream: str, manifest: object) -> list[BaseModel]:
    """The stream's examples, once the manifest and every line check out."""
    ready = _check_manifest(manifest)
    assert isinstance(manifest, dict)
    if sha256_hex(stream.encode("utf-8")) != manifest["stream_sha256"]:
        raise TrainingStreamError("the stream does not hash to its TrainingSet's stream_sha256")
    if not stream.endswith("\n"):
        raise TrainingStreamError("the stream's last line must end with a newline")
    lines = stream.split("\n")[:-1]
    if len(lines) != manifest["examples"]:
        raise TrainingStreamError(
            f"the stream has {len(lines)} lines, its TrainingSet says {manifest['examples']}"
        )
    return _check_lines(lines, manifest, ready, keep=True)


CHUNK_BYTES = 1 << 20


def stream_lines(path: Path) -> Iterator[str]:
    """The lines of a file, each with its newline, split on newline only."""
    with path.open("rb") as handle:
        for raw in handle:
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TrainingStreamError("the stream is not UTF-8") from error


def check_stream_file(path: Path, manifest: object) -> None:
    """The checks of ``read_stream`` on a stream file, read in chunks and line by line.

    A generation's stream can be larger than a gigabyte, so no string holds it and no
    example is kept: the file is hashed and its newlines counted in one pass, then each
    line is validated in a second. The refusals and their order are ``read_stream``'s.
    """
    ready = _check_manifest(manifest)
    assert isinstance(manifest, dict)
    digest = hashlib.sha256()
    newlines = 0
    last = b""
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    if digest.hexdigest() != manifest["stream_sha256"]:
        raise TrainingStreamError("the stream does not hash to its TrainingSet's stream_sha256")
    if last != b"\n":
        raise TrainingStreamError("the stream's last line must end with a newline")
    if newlines != manifest["examples"]:
        raise TrainingStreamError(
            f"the stream has {newlines} lines, its TrainingSet says {manifest['examples']}"
        )
    _check_lines(stream_lines(path), manifest, ready, keep=False)
