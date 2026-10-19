"""Check a TrainingExample stream against its TrainingSet (ADR-009 §4).

The TS training loader writes the stream and its manifest. Python training
reads only that stream: ``read_stream`` validates the manifest, refuses a
stream whose digest or line count differs from it, then validates and parses
every line as a ``training-example``. A stream with no matching manifest is
not readable at all: there is no function that reads one alone.
"""

from __future__ import annotations

import itertools
import json

from pydantic import BaseModel

from areneon_contracts.contracts import ContractError, parse, sha256_hex


class TrainingStreamError(ValueError):
    """A stream does not match its TrainingSet manifest."""


def _strictly_ascending(hashes: list[str]) -> bool:
    return all(a < b for a, b in itertools.pairwise(hashes))


def read_stream(stream: str, manifest: object) -> list[BaseModel]:
    """The stream's examples, once the manifest and every line check out."""
    parse("training-set", manifest)
    assert isinstance(manifest, dict)  # parse() validated an object
    for field in ("episode_hashes", "evaluation_manifest_hashes"):
        if not _strictly_ascending(manifest[field]):
            raise TrainingStreamError(f"training-set: {field} must be sorted and unique")
    if sha256_hex(stream.encode("utf-8")) != manifest["stream_sha256"]:
        raise TrainingStreamError("the stream does not hash to its TrainingSet's stream_sha256")
    if not stream.endswith("\n"):
        raise TrainingStreamError("the stream's last line must end with a newline")
    lines = stream.split("\n")[:-1]
    if len(lines) != manifest["examples"]:
        raise TrainingStreamError(
            f"the stream has {len(lines)} lines, its TrainingSet says {manifest['examples']}"
        )
    known = set(manifest["episode_hashes"])
    examples: list[BaseModel] = []
    for number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
            example = parse("training-example", raw)
        except (json.JSONDecodeError, ContractError) as error:
            raise TrainingStreamError(f"line {number}: {error}") from error
        if raw["episode_hash"] not in known:
            raise TrainingStreamError(f"line {number}: its Episode is not in the TrainingSet")
        examples.append(example)
    return examples
