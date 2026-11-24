"""The real scorer behind the worker (ADR-010 §2, §3): networks in, probabilities out.

It is stateless: a query's probabilities depend on the loaded weights and on
that query alone. Each query is its own forward pass (batch of one) on the CPU
with the process set up by ``configure_cpu_inference``, so the result does not
depend on what was scored before or beside it.

A GPU-inference worker of training self-play (ADR-030 step 4) gives the scorer an
``InferencePlacement``: every forward pass then runs on the device copy of the
networks, or on the CPU networks after an allocation failure or under the rollback
flag. Without one (every evaluation and sealed path) nothing here changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import torch

from areneon_ml import stopwatch
from areneon_ml.model.batch import Batch, collate
from areneon_ml.model.candidates import UnsupportedCandidate
from areneon_ml.model.encoder import EncodedQuery, Encoder, EncodingError
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import (
    ROLES,
    PolicyNetwork,
    TrunkError,
    check_shared_trunk,
)
from areneon_ml.model.placement import InferencePlacement, batch_to
from areneon_ml.reducer import ReducerError

# Temperature 1 in millionths (ADR-023 §1); the worker's sampler states the same constant.
FULL_TEMPERATURE_PPM = 1_000_000

_CPU = torch.device("cpu")
T = TypeVar("T")


def _on(batch: Batch, device: torch.device) -> Batch:
    """The batch on ``device``: the very same object on the CPU, so that path is unchanged."""
    return batch if device == _CPU else batch_to(batch, device)


class ScorerError(ValueError):
    """The loaded networks cannot serve one session together."""


class NetworkScorer:
    """Scores a Team Preview or a Battle query with the matching network."""

    def __init__(
        self,
        game: GameData,
        networks: Mapping[str, PolicyNetwork],
        placement: InferencePlacement | None = None,
    ) -> None:
        if set(networks) != set(ROLES):
            raise ScorerError(f"one network per role is needed: {sorted(ROLES)}")
        for role, network in networks.items():
            if network.role != role:
                raise ScorerError(f"the {role} slot holds a {network.role} network")
        try:
            check_shared_trunk(networks)  # a shared_trunk pair must hold equal trunk tensors
        except TrunkError as exc:
            raise ScorerError(str(exc)) from exc
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
        self._placement = placement
        if placement is not None:
            placement.adopt(self._networks)

    def placement_stats(self) -> dict[str, Any] | None:
        """A GPU-inference worker's device counts (ADR-030 step 4); None without a placement."""
        return None if self._placement is None else self._placement.stats()

    def _apply(self, role: str, call: Callable[[PolicyNetwork, torch.device], T]) -> T:
        """Run ``call`` with ``role``'s network: on the CPU, or through the placement."""
        if self._placement is None:
            return call(self._networks[role], _CPU)
        return self._placement.run(role, call)

    def probabilities(
        self, view: dict[str, Any], temperature_ppm: int = FULL_TEMPERATURE_PPM
    ) -> list[float]:
        """One probability per candidate, in the candidates' order (float32 values).

        ``softmax(logits * 10^6 / temperature_ppm)``: the scale is computed once in float64
        and applied to the float32 logits (exactly 1 at the default, which skips the product,
        so the default's bytes are those of the untempered scorer).
        """
        return self._forward(view, temperature_ppm, with_value=False)[0]

    def evaluate(self, view: dict[str, Any]) -> tuple[list[float], float]:
        """The probabilities (temperature 1) and the value head's output, one forward pass.

        The probabilities are those ``probabilities(view)`` gives; the value is the float32
        output for the state (ADR-026 §5).
        """
        return self._forward(view, FULL_TEMPERATURE_PPM, with_value=True)

    @property
    def value_query(self) -> bool:
        """Whether value-only queries are served: neither encoder fills a token or field column
        from candidate rows or the opponent sheet (``Layout.value_reads_candidates``, ADR-026
        erratum E2 decision 7). Read from the encoders' layouts, never from a list of sets."""
        return not any(e.layout.value_reads_candidates for e in self._encoders.values())

    def value(self, view: dict[str, Any]) -> float:
        """The value head's float32 output for the state alone: trunk and value head, no candidates.

        ``view`` carries no candidates or rows. Batch of one on the CPU, so as independent of what
        was scored before as ``evaluate`` is; the value equals the one ``evaluate`` gives for the
        same state (a test), since the layout the scorer serves it for reads nothing of the rows.
        """
        role = view["observation"]["kind"]
        encoded = self._encoders[role].encode(view, value_only=True)
        batch = collate([encoded])

        def call(network: PolicyNetwork, device: torch.device) -> float:
            with torch.inference_mode():
                output = network.value_output(_on(batch, device))
            return float(output[0].item())

        return self._apply(role, call)

    def score_batch(
        self, items: Sequence[tuple[dict[str, Any], int | None]]
    ) -> list[tuple[list[float], float] | Exception]:
        """Score several queries in padded forward passes (ADR-028 section 2 F2, self-play).

        Each item is ``(view, temperature_ppm)`` (``None`` or the default is temperature 1); the
        result holds, in the items' order, the probabilities and the value head's output of each,
        or the ``ReducerError`` / ``EncodingError`` / ``UnsupportedCandidate`` that refused that
        query alone (any other error is a defect and propagates). Only the forward pass batches:
        each query is reduced and encoded alone, and the queries of one role go through the
        network together, padded by ``collate``. The result of a query is NOT bit-identical to its
        batch-1 result (padding changes the matmul shapes); a batch of one is, and the worker uses
        this method only for runs whose config says ``generation.inference`` (evaluation and
        sealed paths score one query per pass).
        """
        results: list[tuple[list[float], float] | Exception | None] = [None] * len(items)
        groups: dict[str, list[tuple[int, EncodedQuery, int | None]]] = {}
        for index, (view, temperature_ppm) in enumerate(items):
            role = view["observation"]["kind"]
            mark = stopwatch.begin()
            try:
                encoded = self._encoders[role].encode(view)
            except (ReducerError, EncodingError, UnsupportedCandidate) as exc:
                results[index] = exc
                continue
            finally:
                stopwatch.end("w_enc", mark)
            groups.setdefault(role, []).append((index, encoded, temperature_ppm))
        for role, members in groups.items():
            mark = stopwatch.begin()
            batch = collate([encoded for _, encoded, _ in members])

            def call(
                network: PolicyNetwork,
                device: torch.device,
                batch: Batch = batch,
                members: list[tuple[int, EncodedQuery, int | None]] = members,
            ) -> list[tuple[int, tuple[list[float], float]]]:
                found: list[tuple[int, tuple[list[float], float]]] = []
                with torch.inference_mode():
                    output = network(_on(batch, device))
                    for row, (index, encoded, temperature_ppm) in enumerate(members):
                        logits = output.logits[row, : encoded.pending.shape[0]]
                        if temperature_ppm is not None and temperature_ppm != FULL_TEMPERATURE_PPM:
                            logits = logits * (1_000_000 / temperature_ppm)
                        probabilities = torch.softmax(logits, dim=-1)
                        found.append(
                            (
                                index,
                                (
                                    [float(p) for p in probabilities.tolist()],
                                    float(output.value[row].item()),
                                ),
                            )
                        )
                return found

            for index, answer in self._apply(role, call):
                results[index] = answer
            stopwatch.end("w_fwd", mark)
        return [found for found in results if found is not None]

    def _forward(
        self, view: dict[str, Any], temperature_ppm: int, with_value: bool
    ) -> tuple[list[float], float]:
        """The probabilities and, when asked, the value head's float32 output (batch of one)."""
        role = view["observation"]["kind"]
        mark = stopwatch.begin()
        encoded = self._encoders[role].encode(view)
        stopwatch.end("w_enc", mark)
        mark = stopwatch.begin()
        batch = collate([encoded])

        def call(network: PolicyNetwork, device: torch.device) -> tuple[list[float], float]:
            with torch.inference_mode():
                output = network(_on(batch, device))
                logits = output.logits[0]
                if temperature_ppm != FULL_TEMPERATURE_PPM:
                    logits = logits * (1_000_000 / temperature_ppm)
                probabilities = torch.softmax(logits, dim=-1)
                value = float(output.value[0].item()) if with_value else 0.0
                return [float(p) for p in probabilities.tolist()], value

        found = self._apply(role, call)
        stopwatch.end("w_fwd", mark)
        return found
