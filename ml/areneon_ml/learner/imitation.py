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
from areneon_contracts.training import TrainingStreamError, check_stream_file

from areneon_ml.learner.data import LearnerDataError, Step, _Collector
from areneon_ml.model.encoder import EncodedQuery, Encoder, EncodingError
from areneon_ml.worker.view import IncrementalView

#: The purpose string of the draw's seed (A11.3).
PURPOSE = "imitation/1"

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
    """
    check_teacher_manifest(manifest, expected_hash)
    if "search_labels" in manifest:
        raise LearnerDataError("the teacher TrainingSet must not carry search_labels")
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
    return TeacherSet(stream, offsets, encoders["battle"])


class DrawStream:
    """A generator for one update's teacher draw, seeded from the learner seed, the update index
    and the purpose (the first 8 bytes of SHA-256 over their JCS bytes)."""

    def __init__(self, learner_seed: int, update: int) -> None:
        digest = hashlib.sha256(
            jcs({"purpose": PURPOSE, "seed": learner_seed, "update": update})
        ).digest()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int.from_bytes(digest[:8], "big") % 2**63)

    def draw(self, size: int, count: int) -> list[int]:
        """``count`` of ``size`` indices without replacement, in drawn order."""
        return torch.randperm(size, generator=self.generator).tolist()[:count]

    def shuffle(self, items: list[int]) -> list[int]:
        """``items`` in a new order (one epoch's shuffle), from this stream."""
        order = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[i] for i in order]
