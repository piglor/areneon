"""The scorer seam: what turns an encoder input into one probability per candidate.

The real networks (ADR-010 §3) come with their own slice. ``StubScorer``
stands in for them: deterministic weights derived from the checkpoint bytes
and each candidate, so the protocol, the sampler and batch invariance can be
tested without a network. It reads only the encoder's view of a query.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from areneon_contracts.contracts import jcs

from areneon_ml.worker.sampler import to_float32

ENCODER_VERSION = "stub-1"


class Scorer(Protocol):
    """One probability per candidate, in the candidates' order, for one query at a time.

    Evaluation and sealed runs call it with a single query per pass (batch
    invariance, ADR-010 §2): the result must not depend on what was scored before.
    """

    encoder_version: str

    def probabilities(self, view: dict[str, Any]) -> list[float]: ...


class StubScorer:
    """Weights from SHA-256(checkpoint bytes, candidate), normalised: no network, no state."""

    encoder_version = ENCODER_VERSION

    def __init__(self, checkpoints: dict[str, bytes]) -> None:
        self._checkpoints = checkpoints

    def probabilities(self, view: dict[str, Any]) -> list[float]:
        weights = []
        data = self._checkpoints[view["observation"]["kind"]]
        for candidate in view["candidates"]:
            digest = hashlib.sha256(data + b"\x00" + jcs(candidate)).digest()
            weights.append(1 + int.from_bytes(digest[:4], "big"))
        total = sum(weights)
        return [to_float32(weight / total) for weight in weights]
