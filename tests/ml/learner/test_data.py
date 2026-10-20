"""A TrainingSet stream as trajectories: what is refused, what a trajectory holds (ADR-010 §1)."""

from __future__ import annotations

import copy
import random

import pytest
from areneon_ml.learner.data import LearnerDataError, trajectories
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import POLICY, episode_lines, make_stream, stream_of, tiny_config
from tests.ml.model.support import game


def encoders() -> dict[str, Encoder]:
    config = tiny_config()
    return {role: Encoder(game(), config.model.encoder) for role in ROLES}


def test_each_episode_is_a_preview_step_and_a_battle_trajectory_with_its_game_score() -> None:
    text, manifest = make_stream([(0, "p1"), (0, "p2")])
    found = trajectories(text, manifest, encoders(), POLICY)
    assert len(found) == 4
    # battle 0 was won by p2: p1's episode scores 0, p2's scores 1
    scores = {(t.role, t.score) for t in found}
    assert scores == {("preview", 0.0), ("battle", 0.0), ("preview", 1.0), ("battle", 1.0)}
    for trajectory in found:
        assert trajectory.steps
        if trajectory.role == "preview":
            assert len(trajectory.steps) == 1
            assert trajectory.steps[0].query.kind == "preview"
        else:
            assert all(step.query.kind == "battle" for step in trajectory.steps)


def test_steps_are_in_request_order_and_the_chosen_index_is_the_recorded_choice() -> None:
    picker = random.Random(5)
    examples = episode_lines(0, "p1", picker)
    text, manifest = stream_of(examples)
    found = trajectories(text, manifest, encoders(), POLICY)
    battle = next(t for t in found if t.role == "battle")
    recorded = [e for e in examples if e["observation"]["kind"] == "battle"]
    assert len(battle.steps) == len(recorded)
    for step, example in zip(battle.steps, recorded, strict=True):
        assert example["candidates"][step.chosen]["choice"] == example["choice"]
        assert step.query.pending.shape[0] == len(example["candidates"])


def test_a_tie_scores_one_half() -> None:
    picker = random.Random(1)
    examples = episode_lines(0, "p1", picker, outcome="tie")
    text, manifest = stream_of(examples)
    assert {t.score for t in trajectories(text, manifest, encoders(), POLICY)} == {0.5}


def test_a_set_that_is_not_learner_ready_is_refused() -> None:
    text, manifest = make_stream()
    legacy = {
        k: v for k, v in manifest.items() if k not in ("learner_ready", "behaviour_policy_id")
    }
    with pytest.raises(LearnerDataError, match="not learner_ready"):
        trajectories(text, legacy, encoders(), POLICY)


def test_a_set_played_by_another_policy_is_refused() -> None:
    text, manifest = make_stream(policy="random-legal@tiny")
    with pytest.raises(LearnerDataError, match="not played by the policy being trained"):
        trajectories(text, manifest, encoders(), POLICY)


def test_a_stream_that_does_not_match_its_manifest_is_refused() -> None:
    text, manifest = make_stream()
    with pytest.raises(LearnerDataError, match="stream_sha256"):
        trajectories(text + "\n", manifest, encoders(), POLICY)
    broken = copy.deepcopy(manifest)
    broken["examples"] += 1
    with pytest.raises(LearnerDataError):
        trajectories(text, broken, encoders(), POLICY)


def test_an_example_without_candidates_is_refused_in_a_learner_ready_set() -> None:
    examples = episode_lines(0, "p1", random.Random(2))
    del examples[1]["candidates"]
    text, manifest = stream_of(examples)
    with pytest.raises(LearnerDataError, match="no candidates"):
        trajectories(text, manifest, encoders(), POLICY)


def test_two_outcomes_in_one_episode_are_refused() -> None:
    examples = episode_lines(0, "p1", random.Random(2))
    examples[-1]["outcome"] = "win" if examples[-1]["outcome"] != "win" else "loss"
    text, manifest = stream_of(examples)
    with pytest.raises(LearnerDataError, match="two outcomes"):
        trajectories(text, manifest, encoders(), POLICY)
