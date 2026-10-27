"""The real scorer behind the worker (ADR-010 §2, §3): networks in, probabilities out.

It is stateless: a query's probabilities depend on the loaded weights and on
that query alone. Each query is its own forward pass (batch of one) on the CPU
with the process set up by ``configure_cpu_inference``, so the result does not
depend on what was scored before or beside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES, PolicyNetwork

# Temperature 1 in millionths (ADR-023 §1); the worker's sampler states the same constant.
FULL_TEMPERATURE_PPM = 1_000_000


class ScorerError(ValueError):
    """The loaded networks cannot serve one session together."""


class NetworkScorer:
    """Scores a Team Preview or a Battle query with the matching network."""

    def __init__(self, game: GameData, networks: Mapping[str, PolicyNetwork]) -> None:
        if set(networks) != set(ROLES):
            raise ScorerError(f"one network per role is needed: {sorted(ROLES)}")
        for role, network in networks.items():
            if network.role != role:
                raise ScorerError(f"the {role} slot holds a {network.role} network")
        self._networks = dict(networks)
        self._encoders = {role: Encoder(game, net.config.encoder) for role, net in networks.items()}
        versions = {encoder.version for encoder in self._encoders.values()}
        if len(versions) != 1:
            raise ScorerError(
                "the Preview and Battle networks were trained with different encoders"
            )
        self.encoder_version = versions.pop()
        for network in self._networks.values():
            network.eval()

    def probabilities(
        self, view: dict[str, Any], temperature_ppm: int = FULL_TEMPERATURE_PPM
    ) -> list[float]:
        """One probability per candidate, in the candidates' order (float32 values).

        ``softmax(logits * 10^6 / temperature_ppm)``: the scale is computed once in float64
        and applied to the float32 logits (exactly 1 at the default, which skips the product,
        so the default's bytes are those of the untempered scorer).
        """
        role = view["observation"]["kind"]
        encoded = self._encoders[role].encode(view)
        with torch.inference_mode():
            logits = self._networks[role](collate([encoded])).logits[0]
            if temperature_ppm != FULL_TEMPERATURE_PPM:
                logits = logits * (1_000_000 / temperature_ppm)
            probabilities = torch.softmax(logits, dim=-1)
        return [float(p) for p in probabilities.tolist()]
