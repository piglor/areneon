"""A TrainingSet stream as per-role trajectories (ADR-010 §1, §5).

The stream is read only with its manifest (``read_stream`` checks the digest,
the line count and every line against the contracts). The learner then refuses
a set that is not ``learner_ready`` and names its behaviour policy: it trains
only on the stream of the policy it was given, so a frozen opponent's or a
baseline's decisions cannot reach an update. Each Episode is one player's
battle, so it is one trajectory of the Preview network (one step) and one of
the Battle network (every later decision, in request order). Every step is
encoded from the example's observation and candidates alone.

The game score is the reward of the last step of each trajectory; all earlier
steps, a refused pending choice among them, have zero immediate reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from areneon_contracts.training import TrainingStreamError, read_stream

from areneon_ml.learner.ppo import GAME_SCORE
from areneon_ml.model.encoder import EncodedQuery, Encoder
from areneon_ml.model.network import ROLES
from areneon_ml.worker.view import encoder_input


class LearnerDataError(ValueError):
    """A TrainingSet cannot train this learner."""


@dataclass(frozen=True)
class Step:
    """One decision: its encoded query and the index of the candidate the policy chose."""

    query: EncodedQuery
    chosen: int


@dataclass(frozen=True)
class Trajectory:
    """The steps of one Episode that one role's network decided, and the game score."""

    role: str
    steps: tuple[Step, ...]
    score: float


def trajectories(
    stream: str, manifest: dict[str, Any], encoders: dict[str, Encoder], behaviour_policy_id: str
) -> list[Trajectory]:
    """The trajectories of a learner-ready stream played by ``behaviour_policy_id``."""
    if manifest.get("learner_ready") is not True:
        raise LearnerDataError("the TrainingSet is not learner_ready: the learner refuses it")
    if manifest.get("behaviour_policy_id") != behaviour_policy_id:
        raise LearnerDataError(
            "the TrainingSet was not played by the policy being trained: "
            f"{manifest.get('behaviour_policy_id')!r}, not {behaviour_policy_id!r}"
        )
    try:
        read_stream(stream, manifest)
    except TrainingStreamError as exc:
        raise LearnerDataError(str(exc)) from exc
    order: list[str] = []
    steps: dict[tuple[str, str], list[Step]] = {}
    scores: dict[str, float] = {}
    for line in stream.split("\n")[:-1]:
        example = json.loads(line)
        observation = example["observation"]
        role = observation["kind"]  # "preview" or "battle": the same words as the roles
        episode = example["episode_hash"]
        if episode not in scores:
            order.append(episode)
            scores[episode] = GAME_SCORE[example["outcome"]]
        elif GAME_SCORE[example["outcome"]] != scores[episode]:
            raise LearnerDataError(f"Episode {episode} has two outcomes")
        candidates = example["candidates"]
        chosen = [candidate["choice"] for candidate in candidates].index(example["choice"])
        query = encoders[role].encode(
            encoder_input({"observation": observation, "candidates": candidates})
        )
        steps.setdefault((episode, role), []).append(Step(query, chosen))
    return [
        Trajectory(role, tuple(steps[(episode, role)]), scores[episode])
        for episode in order
        for role in ROLES
        if (episode, role) in steps
    ]
