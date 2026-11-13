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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.training import (
    TrainingStreamError,
    check_stream_file,
    read_stream,
    stream_lines,
)

from areneon_ml import stopwatch
from areneon_ml.learner.ppo import GAME_SCORE
from areneon_ml.learner.shaping import ShapingError, differential
from areneon_ml.learner.situation import threatened_flag
from areneon_ml.model.encoder import EncodedQuery, Encoder, EncoderSession
from areneon_ml.model.network import ROLES
from areneon_ml.worker.view import IncrementalView


class LearnerDataError(ValueError):
    """A TrainingSet cannot train this learner."""


@dataclass(frozen=True)
class Step:
    """One decision: its encoded query and the index of the candidate the policy chose."""

    query: EncodedQuery
    chosen: int
    #: ``d(o)`` of the decision's observation (ppm, ``shaping.differential``): set only on a Battle
    #: step read under ``ppo.shaping`` (ADR-010 erratum A11.1), ``None`` otherwise.
    potential_d: int | None = None
    #: Whether the decision is Threatened (ADR-010 erratum A11.2, ``learner/situation.py``), a
    #: pure function of the example; ``None`` for a Preview step and for a Battle example whose
    #: ``candidate_features`` lack the threat rows (the flag cannot be read there).
    threatened: bool | None = None


@dataclass(frozen=True)
class Trajectory:
    """The steps of one Episode that one role's network decided, and the game score."""

    role: str
    steps: tuple[Step, ...]
    score: float
    # The Episode the steps came from (the cloning learner splits by it); PPO ignores it.
    episode: str = ""


class _Collector:
    """The steps of a run of stream lines, grouped by Episode and role, in first-seen order."""

    def __init__(self, encoders: dict[str, Encoder], shaping: bool = False) -> None:
        self._encoders = encoders
        self._shaping = shaping
        self.order: list[str] = []
        self.steps: dict[tuple[str, str], list[Step]] = {}
        self.scores: dict[str, float] = {}
        # One reduction and one rqid clean per channel: the stream lists an Episode's examples
        # together, in request order, each log a prefix of the next. Anything else restarts.
        self._views: dict[tuple[str, str], tuple[IncrementalView, EncoderSession]] = {}
        self._current: str | None = None

    def add(self, example: dict[str, Any]) -> None:
        observation = example["observation"]
        role = observation["kind"]  # "preview" or "battle": the same words as the roles
        episode = example["episode_hash"]
        if episode not in self.scores:
            self.order.append(episode)
            self.scores[episode] = GAME_SCORE[example["outcome"]]
        elif GAME_SCORE[example["outcome"]] != self.scores[episode]:
            raise LearnerDataError(f"Episode {episode} has two outcomes")
        candidates = example["candidates"]
        chosen = [candidate["choice"] for candidate in candidates].index(example["choice"])
        if episode != self._current:
            self._views.clear()
            self._current = episode
        if (episode, role) not in self._views:
            self._views[(episode, role)] = (IncrementalView(), self._encoders[role].session())
        view, session = self._views[(episode, role)]
        mark = stopwatch.begin()
        encoder_input = view.encoder_input(
            {
                "observation": observation,
                "candidates": candidates,
                **{
                    key: example[key]
                    for key in ("candidate_features", "opponent_sheet")
                    if key in example
                },
            }
        )
        stopwatch.end("reduce", mark)  # the view's rqid clean is part of reading the channel
        query = session.encode(encoder_input)
        potential = None
        if self._shaping and role == "battle":
            state = session.state
            if state is None:  # pragma: no cover - encode just fed the reducer
                raise LearnerDataError("no reduced state to read the shaping potential from")
            try:
                potential = differential(state, session.last_request)
            except ShapingError as error:
                raise LearnerDataError(f"Episode {episode}: {error}") from error
        features = example.get("candidate_features")
        threatened = threatened_flag(observation["log"], features) if role == "battle" else None
        self.steps.setdefault((episode, role), []).append(
            Step(query, chosen, potential, threatened)
        )

    def collected(self) -> Collected:
        """What was seen, without the encoders and sessions (it is sent between processes)."""
        return Collected(self.order, self.scores, self.steps)

    def trajectories(self) -> list[Trajectory]:
        return self.collected().trajectories()


@dataclass
class Collected:
    """The steps of a run of stream lines, by Episode and role, in first-seen order."""

    order: list[str]
    scores: dict[str, float]
    steps: dict[tuple[str, str], list[Step]]

    def absorb(self, later: Collected) -> bool:
        """Append what ``later`` saw next; False, changing nothing, if they share an Episode."""
        if any(episode in self.scores for episode in later.order):
            return False
        self.order.extend(later.order)
        self.scores.update(later.scores)
        self.steps.update(later.steps)
        return True

    def trajectories(self) -> list[Trajectory]:
        return [
            Trajectory(role, tuple(self.steps[(episode, role)]), self.scores[episode], episode)
            for episode in self.order
            for role in ROLES
            if (episode, role) in self.steps
        ]


def _ready(manifest: dict[str, Any], behaviour_policy_id: str) -> None:
    if "search_labels" in manifest:
        raise LearnerDataError(
            "the PPO path refuses a TrainingSet with search_labels: label streams have their own"
            " reader"
        )
    if manifest.get("learner_ready") is not True:
        raise LearnerDataError("the TrainingSet is not learner_ready: the learner refuses it")
    if manifest.get("behaviour_policy_id") != behaviour_policy_id:
        raise LearnerDataError(
            "the TrainingSet was not played by the policy being trained: "
            f"{manifest.get('behaviour_policy_id')!r}, not {behaviour_policy_id!r}"
        )


def trajectories(  # noqa: PLR0913
    stream: str | Path,
    manifest: dict[str, Any],
    encoders: dict[str, Encoder],
    behaviour_policy_id: str,
    processes: int = 1,
    *,
    private_pool: bool = False,
    shaping: bool = False,
) -> list[Trajectory]:
    """The trajectories of a learner-ready stream played by ``behaviour_policy_id``.

    ``stream`` is the text or, for a stream too large to hold as one string (#998), the path
    of its file, which is checked and read line by line. With ``processes`` above 1 a stream
    file is checked and encoded in that many worker processes, in order, to the same
    trajectories, tensor for tensor (``learner/parallel.py``); a stream that cannot be cut
    between Episodes is read by this process alone.

    ``private_pool`` states whether the run config has a private pool (ADR-015 erratum E1 item 6;
    the config field belongs to the private run driver, which passes it): False, the default,
    refuses a private set; True accepts a private set only and refuses a synthetic one. The
    manifest has no run key, so a set of another run cannot be told here yet (the driver adds that).

    ``shaping`` (``ppo.shaping``, ADR-010 erratum A11.1) reads the potential differential of each
    Battle step from the reduced state of its observation (``Step.potential_d``).
    """
    _ready(manifest, behaviour_policy_id)
    if processes > 1 and isinstance(stream, Path):
        from areneon_ml.learner.parallel import parallel_trajectories  # noqa: PLC0415

        found = parallel_trajectories(
            stream, manifest, encoders, processes, private_pool=private_pool, shaping=shaping
        )
        if found is not None:
            return found
    mark = stopwatch.begin()
    try:
        if isinstance(stream, Path):
            check_stream_file(stream, manifest, private_pool)
        else:
            read_stream(stream, manifest, private_pool)
    except TrainingStreamError as exc:
        raise LearnerDataError(str(exc)) from exc
    stopwatch.end("validate", mark)
    collector = _Collector(encoders, shaping)
    lines: Iterator[str] = (
        stream_lines(stream) if isinstance(stream, Path) else iter(stream.split("\n")[:-1])
    )
    for line in lines:
        collector.add(json.loads(line))
    return collector.trajectories()
