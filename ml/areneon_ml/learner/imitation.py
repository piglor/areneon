"""The teacher imitation mix: the teacher stream and its draw (ADR-010 erratum A11.3, #1367).

``ppo.imitation`` mixes a supervised term on the fixed states of a scripted teacher's
``TrainingSet`` into the Battle update. This module holds the two things the term needs that are
not the loss: reading the teacher stream (``open_teacher``, which refuses everything the erratum
lists) and the draw of one update's teacher steps (``DrawStream``, a generator of its own, so the
learner's main generator, and with it every run without the block and the Preview role's orders,
is untouched).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.training import TrainingStreamError, check_stream_digest, check_stream_file

from areneon_ml.learner import teacher_marker
from areneon_ml.learner.data import LearnerDataError, Step, _Collector
from areneon_ml.model.encoder import EncodedQuery, Encoder, EncodingError
from areneon_ml.worker.view import IncrementalView

#: The purpose string of the draw's seed (A11.3).
PURPOSE = "imitation/1"

#: The largest integer JCS (RFC 8785) holds exactly, plus one.
SAFE_INTEGER = 2**53

#: A scripted teacher's policy id, ``<name>@<version>`` (the clone config's pattern); a learned
#: pair's ``policy:<hash>`` is not a teacher.
TEACHER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*@[A-Za-z0-9._-]+$")


def check_teacher_manifest(manifest: Mapping[str, Any], expected_hash: str | None) -> str:
    """The manifest's hash after the checks that need no stream; refuses what A11.3 lists.

    ``expected_hash`` is ``ppo.imitation.training_set_hash`` (``None`` where no config is at hand:
    the import before a run's config exists). Returns the manifest's own hash.
    """
    found = sha256_hex(jcs(dict(manifest)))
    if expected_hash is not None and found != expected_hash:
        raise LearnerDataError(
            "the teacher TrainingSet's manifest does not hash to ppo.imitation.training_set_hash"
        )
    if manifest.get("data_class") != "synthetic":
        raise LearnerDataError("the teacher TrainingSet is not synthetic")
    if manifest.get("learner_ready") is not True:
        raise LearnerDataError("the teacher TrainingSet is not learner_ready")
    behaviour = manifest.get("behaviour_policy_id")
    if not isinstance(behaviour, str) or TEACHER_ID.match(behaviour) is None:
        raise LearnerDataError(
            "the teacher TrainingSet was not played by a scripted teacher "
            f"(<name>@<version>): {behaviour!r}"
        )
    return found


class TeacherSource(Protocol):
    """What the trainer needs of the teacher's Battle steps: how many there are and, for the
    indices an update drew, their encoded steps (held only for that update)."""

    def __len__(self) -> int: ...

    def fetch(self, indices: Sequence[int]) -> tuple[Step, ...]: ...


class TeacherSet:
    """The teacher stream's Battle examples as a light index: their byte offsets in the stream
    file, in stream order. Nothing encoded is kept: ``fetch`` reads and encodes only the examples
    asked for, so memory is bounded by the draw (K), not by the set's size (A11.3, #1367)."""

    def __init__(self, stream: Path, offsets: Sequence[int], encoder: Encoder) -> None:
        self._stream = stream
        self._offsets = tuple(offsets)
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._offsets)

    def fetch(self, indices: Sequence[int]) -> tuple[Step, ...]:
        out = []
        with self._stream.open("rb") as handle:
            for i in indices:
                handle.seek(self._offsets[i])
                example = json.loads(handle.readline())
                chosen = [c["choice"] for c in example["candidates"]].index(example["choice"])
                out.append(Step(_encode(self._encoder, example), chosen))
        return tuple(out)


def _encode(encoder: Encoder, example: dict[str, Any]) -> EncodedQuery:
    """One example's tensors, equal to what the incremental reader gives it inside its Episode
    (the encoder session's documented property: it restarts on a log that does not extend)."""
    return encoder.session().encode(
        IncrementalView().encoder_input(
            {
                "observation": example["observation"],
                "candidates": example["candidates"],
                **{k: example[k] for k in ("candidate_features", "opponent_sheet") if k in example},
            }
        )
    )


def _full_pass(
    stream: Path, manifest: Mapping[str, Any], encoders: dict[str, Encoder]
) -> list[int]:
    """The check and the encode of every example, and the byte offsets of the Battle examples."""
    try:
        check_stream_file(stream, dict(manifest), False)
    except TrainingStreamError as exc:
        raise LearnerDataError(str(exc)) from exc
    collector = _Collector(encoders)
    offsets: list[int] = []
    position = 0
    try:
        with stream.open("rb") as handle:
            for raw in handle:
                example = json.loads(raw)
                if example["observation"]["kind"] == "battle":
                    offsets.append(position)
                collector.add(example)
                collector.steps.clear()  # encoded to check it, kept for nobody
                position += len(raw)
    except EncodingError as exc:
        raise LearnerDataError(f"the teacher TrainingSet does not encode: {exc}") from exc
    return offsets


def _marker_key(
    stream: Path, manifest: Mapping[str, Any], found: str, encoders: dict[str, Encoder]
) -> dict[str, Any] | None:
    """The key of the stream's marker, or ``None`` where there can be none (a manifest or a stream
    the full pass refuses with its own message)."""
    digest = manifest.get("stream_sha256")
    if not isinstance(digest, str):
        return None
    try:
        size = stream.stat().st_size
    except OSError:
        return None
    return teacher_marker.marker_key(
        training_set_hash=found, stream_sha256=digest, stream_bytes=size, encoders=encoders
    )


def _digest_holds(stream: Path, manifest: Mapping[str, Any]) -> bool:
    """The file-level checks of the full pass (the manifest, the digest, the last newline and the
    line count): the one read of the whole stream a marker keeps. Whatever is wrong with the
    stream is the full pass's to refuse, in its own words."""
    try:
        check_stream_digest(stream, dict(manifest), private_pool=False)
    except (ValueError, OSError):  # TrainingStreamError and ContractError are ValueErrors
        return False
    return True


def open_teacher(
    stream: Path,
    manifest: Mapping[str, Any],
    expected_hash: str,
    encoders: dict[str, Encoder],
) -> TeacherSet:
    """The teacher's Battle examples as a ``TeacherSet``, after every refusal of A11.3.

    The manifest checks and ``check_stream_file`` (digest, line count, every line) run once here,
    and every example, Previews included, is encoded once and dropped at once so a stream the
    run's encoder refuses fails at the first update (a ``LearnerDataError``, the encoder's own
    message wrapped). Only the Battle examples' byte offsets are kept.

    That full pass is minutes for a large stream and a queue starts a driver process per
    generation, so a pass that succeeded leaves a marker beside the stream
    (``teacher_marker``, #1501). A later call with a marker for the same key keeps only the
    digest check: the stream is read once and hashed, and the offsets come from the marker. A
    marker that does not match, or a stream that fails the digest, takes the full pass, with its
    refusals and messages, and a pass that succeeds writes the marker again.
    """
    found = check_teacher_manifest(manifest, expected_hash)
    if "search_labels" in manifest:
        raise LearnerDataError("the teacher TrainingSet must not carry search_labels")
    key = _marker_key(stream, manifest, found, encoders)
    if key is not None:
        offsets = teacher_marker.read_marker(stream.parent, key)
        if offsets is not None and _digest_holds(stream, manifest):
            return TeacherSet(stream, offsets, encoders["battle"])
    offsets = _full_pass(stream, manifest, encoders)
    if key is not None and stream.stat().st_size == key["stream_bytes"]:
        teacher_marker.write_marker(stream.parent, key, offsets)
    return TeacherSet(stream, offsets, encoders["battle"])


class DrawStream:
    """A generator for one update's teacher draw, seeded from the learner seed, the update index
    and the purpose (the first 8 bytes of SHA-256 over their JCS bytes).

    A population member's learner seed is a ``population-seed/1`` draw up to 2^63, beyond the
    integers JCS can hold; such a seed enters the preimage as its decimal string (a seed within
    the safe range, every single run's, keeps the integer and so its bytes)."""

    def __init__(self, learner_seed: int, update: int) -> None:
        seed: int | str = learner_seed if learner_seed < SAFE_INTEGER else str(learner_seed)
        digest = hashlib.sha256(jcs({"purpose": PURPOSE, "seed": seed, "update": update})).digest()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int.from_bytes(digest[:8], "big") % 2**63)

    def draw(self, size: int, count: int) -> list[int]:
        """``count`` of ``size`` indices without replacement, in drawn order."""
        return torch.randperm(size, generator=self.generator).tolist()[:count]

    def shuffle(self, items: list[int]) -> list[int]:
        """``items`` in a new order (one epoch's shuffle), from this stream."""
        order = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[i] for i in order]
