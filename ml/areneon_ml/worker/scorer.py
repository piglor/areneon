"""The scorer seam: what turns an encoder input into one probability per candidate.

The implementation is ``areneon_ml.model.scorer.NetworkScorer`` (ADR-010 §3),
which loads the Preview and Battle checkpoints. The seam stays a protocol so
the session, the sampler and the failure rules can be tested with scorers that
return chosen values.
"""

from __future__ import annotations

from typing import Any, Protocol


class Scorer(Protocol):
    """One probability per candidate, in the candidates' order, for one query at a time.

    Evaluation and sealed runs call it with a single query per pass (batch
    invariance, ADR-010 §2): the result must not depend on what was scored before.
    """

    encoder_version: str

    def probabilities(self, view: dict[str, Any], temperature_ppm: int = ...) -> list[float]:
        """``softmax(logits * 10^6 / temperature_ppm)`` in float32 (ADR-023 §1).

        The worker passes ``temperature_ppm`` only when a query carries one; without it the
        call is ``probabilities(view)``, temperature 1.
        """
        ...
