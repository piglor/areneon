"""The teacher imitation mix: the teacher stream and its draw (ADR-010 erratum A11.3, #1367).

``ppo.imitation`` mixes a supervised term on the fixed states of a scripted teacher's
``TrainingSet`` into the Battle update. This module holds the two things the term needs that are
not the loss: reading the teacher stream (``read_teacher``, which refuses everything the erratum
lists) and the draw of one update's teacher steps (``DrawStream``, a generator of its own, so the
learner's main generator, and with it every run without the block and the Preview role's orders,
is untouched).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.learner.data import LearnerDataError, Step, trajectories
from areneon_ml.model.encoder import Encoder, EncodingError

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


def read_teacher(
    stream: Path,
    manifest: Mapping[str, Any],
    expected_hash: str,
    encoders: dict[str, Encoder],
    processes: int = 1,
) -> tuple[Step, ...]:
    """The teacher's Battle steps in stream order, encoded under the run's encoder.

    The stream goes through ``trajectories`` (the PPO path's reader: ``check_stream_file`` with the
    manifest, so the digest, the line count and every line are checked, by path and streamed). The
    teacher's Team Preview examples are skipped. Any refusal is a ``LearnerDataError``, the
    encoder's own (a feature set the run does not read, rows missing) included.
    """
    check_teacher_manifest(manifest, expected_hash)
    try:
        found = trajectories(
            stream, dict(manifest), encoders, str(manifest["behaviour_policy_id"]), processes
        )
    except EncodingError as exc:
        raise LearnerDataError(f"the teacher TrainingSet does not encode: {exc}") from exc
    return tuple(step for t in found if t.role == "battle" for step in t.steps)


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
