"""Search labels as distillation targets (ADR-026 erratum E2 decisions 4, 5 and 8, #1291).

A label stream (``labels.jsonl`` with its manifest ``label-set.json``) holds one ``TrainingExample``
per labelled Battle decision, each with a ``search_target`` (the pair's temperature-1 prior in ppm
and the searched candidates' scores ``q`` in ppm). The learner reads such a stream through its
own path (the PPO path refuses a set with ``search_labels``) and turns each stored label into a
target distribution from the stored numbers alone:

* ``z(c) = ln(max(prior_ppm(c), 1) / 10^6)`` (a floor of 1 ppm: ``evaluate`` rounds and can give 0);
* ``q_hat(c) = q(c)`` for a searched candidate, and for every other one (pending ones included)
  the prior-weighted mean ``sum(prior_ppm(a) * q(a)) / sum(prior_ppm(a))`` over the searched set;
* ``x(c) = (q_hat(c) - q_min) / (q_max - q_min)`` over the searched set;
* no label when ``q_max - q_min`` is below ``min_spread_ppm``;
* ``pi' = softmax(z / tau + c_scale * x)`` over every candidate, ``tau`` the run's play
  temperature (1 when it declares none). Play-time truncation is ignored.

The target is computed in float64 with ``math`` so each value can be checked by hand.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.training import TrainingStreamError, check_stream_file, stream_lines

from areneon_ml.learner.config import MICRO, SearchDistillSettings
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.model.encoder import EncodedQuery, Encoder
from areneon_ml.worker.view import IncrementalView

PRIOR_FLOOR_PPM = 1


@dataclass(frozen=True)
class LabelStep:
    """One labelled decision: its encoded query and the log of its target distribution."""

    query: EncodedQuery
    log_target: tuple[float, ...]  # ln pi'(c), one entry per candidate, in candidate order
    episode: str
    request_seq: int


@dataclass(frozen=True)
class LabelSet:
    """The usable labels of one or more label streams, and what was left out."""

    steps: tuple[LabelStep, ...]
    below_spread: int  # labels whose q spread is under ``min_spread_ppm`` (not used)


def distill_target(
    prior_ppm: Sequence[int],
    entries: Sequence[tuple[int, int]],
    temperature_ppm: int | None,
    c_scale_micro: int,
    min_spread_ppm: int,
) -> tuple[float, ...] | None:
    """``ln pi'`` of one stored label, or ``None`` when its spread is below ``min_spread_ppm``.

    ``entries`` are the searched ``(candidate index, q_ppm)`` pairs. See the module docstring.
    """
    qs = [q for _, q in entries]
    q_min, q_max = min(qs), max(qs)
    if q_max - q_min < min_spread_ppm or q_max == q_min:
        return None
    weight = sum(prior_ppm[c] for c, _ in entries)
    if weight < 1:
        raise LearnerDataError("a label's searched candidates carry no prior mass")
    completed = sum(prior_ppm[c] * q for c, q in entries) / weight
    searched = dict(entries)
    scale = MICRO / (MICRO if temperature_ppm is None else temperature_ppm)
    c_scale = c_scale_micro / MICRO
    logits = []
    for c, p in enumerate(prior_ppm):
        z = math.log(max(p, PRIOR_FLOOR_PPM) / MICRO)
        x = (searched.get(c, completed) - q_min) / (q_max - q_min)
        logits.append(z * scale + c_scale * x)
    top = max(logits)
    log_norm = top + math.log(sum(math.exp(v - top) for v in logits))
    return tuple(v - log_norm for v in logits)


class LabelReader:
    """Reads label streams for one run: the encoder and settings are fixed at build."""

    def __init__(
        self,
        encoder: Encoder,
        settings: SearchDistillSettings,
        temperature_ppm: int | None,
    ) -> None:
        self._encoder = encoder
        self._settings = settings
        self._temperature_ppm = temperature_ppm
        self._seen: dict[tuple[str, int], tuple[str, ...]] = {}

    def read(
        self,
        stream: Path,
        manifest: dict[str, Any],
        behaviour_policy_id: str,
        search_identity_hash: str,
    ) -> LabelSet:
        """The usable labels of one label stream (checked against its manifest).

        ``search_identity_hash`` is the identity the stream's own generation states.
        """
        block = manifest.get("search_labels")
        if block is None:
            raise LearnerDataError("a label stream needs a TrainingSet with search_labels")
        if block["search_identity_hash"] != search_identity_hash:
            raise LearnerDataError(
                "a label stream was made under another search identity than the run's label config"
            )
        if manifest.get("learner_ready") is not True:
            raise LearnerDataError("the label TrainingSet is not learner_ready")
        if manifest.get("behaviour_policy_id") != behaviour_policy_id:
            raise LearnerDataError(
                "the label TrainingSet was not played by the policy being trained: "
                f"{manifest.get('behaviour_policy_id')!r}, not {behaviour_policy_id!r}"
            )
        try:
            check_stream_file(stream, manifest, False)
        except TrainingStreamError as exc:
            raise LearnerDataError(str(exc)) from exc
        steps: list[LabelStep] = []
        below = 0
        views: dict[str, tuple[IncrementalView, Any]] = {}
        for line in stream_lines(stream):
            example = json.loads(line)
            found = self._one(example, views)
            if found is None:
                below += 1
            else:
                steps.append(found)
        return LabelSet(tuple(steps), below)

    def _one(
        self, example: dict[str, Any], views: dict[str, tuple[IncrementalView, Any]]
    ) -> LabelStep | None:
        observation = example["observation"]
        if observation["kind"] != "battle":
            raise LearnerDataError("a label names a decision that is not a Battle decision")
        episode = example["episode_hash"]
        seq = observation["identity"]["request_seq"]
        choices = tuple(c["choice"] for c in example["candidates"])
        key = (episode, seq)
        if key in self._seen:
            raise LearnerDataError(
                "a label names a decision twice"
                + ("" if self._seen[key] == choices else " with other candidates")
            )
        self._seen[key] = choices
        target = example["search_target"]
        log_target = distill_target(
            target["prior_ppm"],
            [(e["candidate"], e["q_ppm"]) for e in target["entries"]],
            self._temperature_ppm,
            self._settings.c_scale_micro,
            self._settings.min_spread_ppm,
        )
        if log_target is None:
            return None
        if episode not in views:
            views.clear()  # a stream lists an Episode's labels together, in request order
            views[episode] = (IncrementalView(), self._encoder.session())
        view, session = views[episode]
        query = session.encode(
            view.encoder_input(
                {
                    "observation": observation,
                    "candidates": example["candidates"],
                    **{
                        k: example[k]
                        for k in ("candidate_features", "opponent_sheet")
                        if k in example
                    },
                }
            )
        )
        if int(query.pending.shape[0]) != len(log_target):
            raise LearnerDataError("a label's candidates differ from the example's")
        return LabelStep(query, log_target, episode, seq)
