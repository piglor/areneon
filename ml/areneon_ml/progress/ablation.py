"""The Team Preview ablation's reading, derived on display and never stored (ticket #940).

One generation's Battle network plays two scripted baselines (argmax) with three Previews: its
own learned one, a uniformly random one and an earlier generation's (generation 0). Every
variant plays the same validation pairs on the same seeds, both seats, so the three can be
compared pair by pair. A pair scores the mean of its two games (a win 1, a tie 1/2, a loss 0),
kept as an integer in half-games (0 to 4). Per baseline and variant this gives the mean pair
score, and for the random and the earlier Preview the paired difference to the learned one:
the mean of ``d_k`` (variant minus learned on pair ``k``) in parts per million and how many
pairs the variant did better, equally or worse. All integer arithmetic. Each score and
difference carries the pair-level bootstrap interval of ``stats.pair_interval`` (#948); like
every number here they are descriptive, conditional on the fixed validation plan and a progress
view, never the Training gate.
"""

from __future__ import annotations

from typing import Any

from areneon_ml.progress.records import preview_variant
from areneon_ml.progress.stats import (
    PAIR_INTERVAL_LABEL,
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
    paired_differences,
)

LEARNED = "learned"


def expected_ablation_matches(spec: Any) -> list[str]:
    """The six match ids an ablation of ``spec`` plays: three Previews against two baselines."""
    variants = (LEARNED, "random", f"g{spec.generation}")
    return [
        f"preview-{variant}-{baseline}-argmax"
        for variant in variants
        for baseline in ("actor-rule", "random-legal")
    ]


def ablation_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The ablation of a record: ``{}`` when it has no ablation match."""
    matches = [m for m in record["matches"] if m.get("candidate_preview") is not None]
    if not matches:
        return {}
    scores: dict[tuple[str, str], list[int]] = {}
    for match in matches:
        baseline = match["opponent"]["policy_id"].split("@")[0]
        scores[(baseline, preview_variant(match["candidate_preview"]))] = match_pair_scores(
            record, match["match_id"]
        )
    out: dict[str, Any] = {"interval": PAIR_INTERVAL_LABEL, "baselines": {}}

    def key(baseline: str, variant: str, what: str) -> bytes:
        return bootstrap_stream_key(
            record["run_id"], record["generation"], f"ablation:{baseline}:{variant}:{what}"
        )

    for baseline in sorted({b for b, _ in scores}):
        learned = scores.get((baseline, LEARNED))
        rows = {}
        for variant in sorted(v for b, v in scores if b == baseline):
            mine = scores[(baseline, variant)]
            interval = pair_interval(mine, key(baseline, variant, "score"))
            row: dict[str, Any] = {
                "pairs": len(mine),
                "score_ppm": interval.mean_ppm,
                "low_ppm": interval.low_ppm,
                "high_ppm": interval.high_ppm,
            }
            if learned is not None and variant != LEARNED and len(learned) == len(mine):
                diffs = paired_differences(mine, learned)
                spread = pair_interval(diffs, key(baseline, variant, "minus-learned"))
                row["minus_learned"] = {
                    "mean_ppm": spread.mean_ppm,
                    "low_ppm": spread.low_ppm,
                    "high_ppm": spread.high_ppm,
                    "better": sum(d > 0 for d in diffs),
                    "equal": sum(d == 0 for d in diffs),
                    "worse": sum(d < 0 for d in diffs),
                }
            rows[variant] = row
        out["baselines"][baseline] = rows
    return out
