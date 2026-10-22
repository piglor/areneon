"""The Preview ablation's reading: scores and paired differences in integers (ADR-017 E1)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from areneon_ml.progress.ablation import ablation_summary, expected_ablation_matches
from areneon_ml.progress.run import AblationSpec
from areneon_ml.progress.stats import pair_scores_of

from tests.ml.progress.support import sample

BASE = "progress-evaluation-record-2/valid/ablation-match"


def test_a_record_without_ablation_matches_has_no_summary() -> None:
    assert ablation_summary(sample("progress-evaluation-record-2/valid/baseline-match")) == {}


def test_the_sample_ablation_is_read_pair_by_pair() -> None:
    summary = ablation_summary(sample(BASE))
    rows = summary["baselines"]["actor-rule"]
    # Every variant of the sample has the same four games: win, loss, win, tie = pair scores 2, 3.
    # two pairs scoring 2 and 3 half-games: resample means are 1/2, 5/8 or 3/4, so the interval
    # is the extremes (the 2.5% and 97.5% ranks of 4,000 resamples)
    assert rows["learned"] == {
        "pairs": 2,
        "score_ppm": 625_000,
        "low_ppm": 500_000,
        "high_ppm": 750_000,
    }
    # identical pairs: every difference is 0, so there is no spread
    assert rows["random"]["minus_learned"] == {
        "mean_ppm": 0,
        "low_ppm": 0,
        "high_ppm": 0,
        "better": 0,
        "equal": 2,
        "worse": 0,
    }
    assert "descriptive" in summary["interval"]
    assert rows["g0"]["minus_learned"]["equal"] == 2


def test_a_missing_learned_match_leaves_out_the_differences_and_a_forfeit_is_a_loss() -> None:
    raw: dict[str, Any] = copy.deepcopy(sample(BASE))
    raw["matches"] = [m for m in raw["matches"] if "learned" not in m["match_id"]]
    raw["games"] = [g for g in raw["games"] if "learned" not in g["match_id"]]
    raw["games"][0]["outcome"] = {
        "kind": "forfeit",
        "loser": "candidate",
        "cause": "crash",
        "request_seq": 1,
        "turns": 2,
    }
    raw["games"][1]["outcome"] = {
        "kind": "forfeit",
        "loser": "opponent",
        "cause": "crash",
        "request_seq": 1,
        "turns": 2,
    }
    rows = ablation_summary(raw)["baselines"]["actor-rule"]
    assert "minus_learned" not in rows["random"]
    # g0: pair 0 is a candidate forfeit and an opponent forfeit (1/2); pair 1 a win and a tie (3/4).
    assert rows["g0"]["score_ppm"] == 625_000
    assert set(rows) == {"g0", "random"}


def test_the_expected_matches_are_three_previews_by_two_baselines() -> None:
    spec = AblationSpec(0, "a" * 64, "policy:x", "policy:y")
    assert expected_ablation_matches(spec) == [
        "preview-learned-actor-rule-argmax",
        "preview-learned-random-legal-argmax",
        "preview-random-actor-rule-argmax",
        "preview-random-random-legal-argmax",
        "preview-g0-actor-rule-argmax",
        "preview-g0-random-legal-argmax",
    ]


@pytest.mark.parametrize("bad", [[], ["win"], ["win", "win", "loss"], ["win", "draw"]])
def test_pair_scores_refuse_odd_empty_and_unknown_results(bad: list[str]) -> None:
    with pytest.raises(ValueError, match="even number"):
        pair_scores_of(bad)
