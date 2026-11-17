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

import dataclasses
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
from areneon_ml.learner import aux_moves as aux_labels
from areneon_ml.learner import aux_opponent as opp_labels
from areneon_ml.learner.ppo import GAME_SCORE
from areneon_ml.learner.shaping import ShapingError, differential
from areneon_ml.learner.situation import (
    Groups,
    stall_candidates,
    streak_slots,
    streak_unmatched,
    threatened_flag,
)
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
    #: The unmasked move-reveal targets of the decision (ADR-010 erratum A12.2, ``aux_moves.py``):
    #: set on a Battle step read under ``ppo.aux_moves``, ``None`` otherwise. A pure function of
    #: the player's own channel; never an input.
    aux: tuple[aux_labels.AuxTarget, ...] | None = None
    #: The opponent entries of the decision that have no label (no later reveal, Transformed or
    #: shrunk list): beside ``aux`` so a run can read the masked share. ``None`` with ``aux``.
    aux_masked: int | None = None
    #: The situation group of a Battle step read under ``clone.situation_weights`` (ADR-010
    #: erratum A13.2: ``streak``, ``threatened`` or ``other``, ``learner/situation.py``), a pure
    #: function of the example; ``None`` for a Preview step and without the block.
    group: str | None = None
    #: On a ``streak`` step, per candidate: whether it uses a stalling move in a streak slot
    #: (the report's ``stall_teacher`` and ``stall_clone``); ``None`` on every other step.
    stall: tuple[bool, ...] | None = None
    #: On a grouped step: the request slots whose ident matches no roster entry of the reduced
    #: state (the streak rule cannot judge them); 0 without a group.
    unmatched: int = 0
    #: The unmasked opponent-head targets of the decision (ADR-010 erratum A14.2,
    #: ``aux_opponent.py``): set on a Battle step read under ``ppo.aux_opponent``, ``None``
    #: otherwise. A pure function of the player's own channel; never an input.
    opp: opp_labels.OppLabels | None = None


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

    def __init__(
        self,
        encoders: dict[str, Encoder],
        shaping: bool = False,
        aux: bool = False,
        groups: Groups | None = None,
        opp: tuple[str, ...] = (),
    ) -> None:
        self._encoders = encoders
        self._opp = opp
        # Per Battle trajectory, the opponent half of the state at each decision and the longest
        # log of the player's own channel (the labels read the lines after each decision).
        self._opp_snaps: dict[str, list[opp_labels.Snap]] = {}
        self._opp_logs: dict[str, list[str]] = {}
        self._opp_sides: dict[str, str] = {}
        self._groups = groups
        self._shaping = shaping
        self._aux = aux
        # Per Battle trajectory, the opponent entries at each decision (labels need later steps).
        self._seen: dict[str, list[tuple[aux_labels.Seen, ...]]] = {}
        self._finished = False
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
        if self._aux and role == "battle":
            state = session.state
            if state is None:  # pragma: no cover - encode just fed the reducer
                raise LearnerDataError("no reduced state to read the move-reveal labels from")
            self._seen.setdefault(episode, []).append(aux_labels.snapshot(state, query.n_own))
        if self._opp and role == "battle":
            state = session.state
            if state is None:  # pragma: no cover - encode just fed the reducer
                raise LearnerDataError("no reduced state to read the opponent-head labels from")
            log = observation["log"]
            self._opp_snaps.setdefault(episode, []).append(
                opp_labels.snapshot(state, query.n_own, len(log))
            )
            self._opp_logs[episode] = log
            self._opp_sides[episode] = "p2" if observation["player"] == "p1" else "p1"
        features = example.get("candidate_features")
        threatened = threatened_flag(observation["log"], features) if role == "battle" else None
        group, stall, unmatched = None, None, 0
        if self._groups is not None and role == "battle":
            group, stall, unmatched = self._situation(
                episode, session, observation, candidates, features
            )
        self.steps.setdefault((episode, role), []).append(
            Step(
                query, chosen, potential, threatened, group=group, stall=stall, unmatched=unmatched
            )
        )

    def _situation(
        self,
        episode: str,
        session: EncoderSession,
        observation: dict[str, Any],
        candidates: list[dict[str, Any]],
        features: dict[str, Any] | None,
    ) -> tuple[str, tuple[bool, ...] | None, int]:
        """The group of a Battle decision (ADR-010 erratum A13.2, first match) and, for a streak
        decision, which candidates use a stalling move in a streak slot."""
        assert self._groups is not None
        state, raw = session.state, session.last_request
        stall_moves = self._encoders["battle"].game.vocabulary.stall_moves
        if state is None or raw is None:  # pragma: no cover - encode just fed the reducer
            raise LearnerDataError("no reduced state to read the situation group from")
        if stall_moves is None:
            raise LearnerDataError(
                "clone.situation_weights needs the stalling moves of this engine pin "
                "(reducer/data/stall-moves.<format>.json)"
            )
        request = json.loads(raw[len("|request|") :])
        slots = streak_slots(state, request, stall_moves)
        unmatched = streak_unmatched(state, request)
        if slots:
            choices = [candidate["choice"] for candidate in candidates]
            return "streak", stall_candidates(choices, slots, request, stall_moves), unmatched
        if not self._groups.threatened:
            return "other", None, unmatched
        flag = threatened_flag(observation["log"], features)
        if flag is None:
            raise LearnerDataError(
                f"Episode {episode}: clone.situation_weights.threatened_ppm needs the threats, "
                "speeds and foe_priorities rows in every Battle example"
            )
        return ("threatened" if flag else "other"), None, unmatched

    def finish(self) -> None:
        """Put the move-reveal labels on the Battle steps (``ppo.aux_moves``), in place and once.

        A label needs the LATER decisions of its Episode, so it is made when every line is in;
        ``collected()`` shares these dicts, so what it returned earlier holds the labels too."""
        if self._finished:
            return
        self._finished = True
        self._finish_opponent()
        if not self._aux:
            return
        for episode, seen in self._seen.items():
            steps = self.steps[(episode, "battle")]
            lookup = self._encoders["battle"].game.moves.lookup
            made = aux_labels.labels(seen, lookup)
            self.steps[(episode, "battle")] = [
                dataclasses.replace(step, aux=aux, aux_masked=len(entries) - len(aux))
                for step, aux, entries in zip(steps, made, seen, strict=True)
            ]

    def _finish_opponent(self) -> None:
        """Put the opponent-head labels (``ppo.aux_opponent``) on the Battle steps, in place."""
        if not self._opp:
            return
        game = self._encoders["battle"].game
        lookup = opp_labels.lookup_of(game)
        for episode, snaps in self._opp_snaps.items():
            made = opp_labels.labels(
                snaps, self._opp_logs[episode], self._opp_sides[episode], self._opp, lookup
            )
            steps = self.steps[(episode, "battle")]
            self.steps[(episode, "battle")] = [
                dataclasses.replace(step, opp=found)
                for step, found in zip(steps, made, strict=True)
            ]

    def collected(self) -> Collected:
        """What was seen, without the encoders and sessions (it is sent between processes)."""
        return Collected(self.order, self.scores, self.steps)

    def trajectories(self) -> list[Trajectory]:
        self.finish()
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
    aux_moves: bool = False,
    groups: Groups | None = None,
    aux_opponent: tuple[str, ...] = (),
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

    ``aux_moves`` (``ppo.aux_moves``, ADR-010 erratum A12.2) reads each Battle step's move-reveal
    targets from the player's own later channel (``Step.aux``).

    ``aux_opponent`` (the heads of ``ppo.aux_opponent``, ADR-010 erratum A14.2) reads each Battle
    step's next-action, item and ability targets for those heads from the player's own later
    channel (``Step.opp``).

    ``groups`` (``clone.situation_weights``, ADR-010 erratum A13.2) reads each Battle step's
    situation group from the example alone (``Step.group``).
    """
    _ready(manifest, behaviour_policy_id)
    if processes > 1 and isinstance(stream, Path):
        from areneon_ml.learner.parallel import parallel_trajectories  # noqa: PLC0415

        found = parallel_trajectories(
            stream,
            manifest,
            encoders,
            processes,
            private_pool=private_pool,
            shaping=shaping,
            aux_moves=aux_moves,
            groups=groups,
            aux_opponent=aux_opponent,
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
    collector = _Collector(encoders, shaping, aux_moves, groups, aux_opponent)
    lines: Iterator[str] = (
        stream_lines(stream) if isinstance(stream, Path) else iter(stream.split("\n")[:-1])
    )
    for line in lines:
        collector.add(json.loads(line))
    return collector.trajectories()
