"""The Team Preview ablation's reading, derived on display and never stored (ADR-017 erratum E1).

One generation's Battle network plays two scripted baselines (argmax) with three Previews: its
own learned one, a uniformly random one and an earlier generation's (generation 0). Every
variant plays the same validation pairs on the same seeds, both seats, so the three can be
compared pair by pair. A pair scores the mean of its two games (a win 1, a tie 1/2, a loss 0),
kept as an integer in half-games (0 to 4). Per baseline and variant this gives the mean pair
score, and for the random and the earlier Preview the paired difference to the learned one:
the mean of ``d_k`` (variant minus learned on pair ``k``) in parts per million and how many
pairs the variant did better, equally or worse. All integer arithmetic. The interval over pairs
is the pair-level interval's own slice (#948); like every number here these are conditional on
the fixed validation plan and a progress view, never the Training gate.
"""

from __future__ import annotations

from typing import Any

from areneon_ml.progress.stats import CONDITIONAL_LABEL, PPM, pair_scores_of

LEARNED = "learned"
HALVES_PER_PAIR = 4


def variant_name(preview: dict[str, Any]) -> str:
    """``learned``, ``random`` or ``g<N>`` for a candidate Preview."""
    return f"g{preview['generation']}" if preview["kind"] == "generation" else str(preview["kind"])


def expected_ablation_matches(spec: Any, checkpoints: dict[str, str]) -> list[str]:
    """The six match ids an ablation of ``spec`` plays: three Previews against two baselines."""
    del checkpoints
    variants = (LEARNED, "random", f"g{spec.generation}")
    return [
        f"preview-{variant}-{baseline}-argmax"
        for variant in variants
        for baseline in ("actor-rule", "random-legal")
    ]


def _pair_scores(record: dict[str, Any], match_id: str) -> list[int]:
    results = []
    for row in record["games"]:
        if row["match_id"] != match_id:
            continue
        outcome = row["outcome"]
        if outcome["kind"] == "forfeit":
            results.append("loss" if outcome["loser"] == "candidate" else "win")
        else:
            results.append(
                {"candidate": "win", "opponent": "loss", "tie": "tie"}[outcome["winner"]]
            )
    return pair_scores_of(results)


def ablation_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The ablation of a record: ``{}`` when it has no ablation match."""
    matches = [m for m in record["matches"] if m.get("candidate_preview") is not None]
    if not matches:
        return {}
    scores: dict[tuple[str, str], list[int]] = {}
    for match in matches:
        baseline = match["opponent"]["policy_id"].split("@")[0]
        scores[(baseline, variant_name(match["candidate_preview"]))] = _pair_scores(
            record, match["match_id"]
        )
    out: dict[str, Any] = {"interval": CONDITIONAL_LABEL, "baselines": {}}
    for baseline in sorted({b for b, _ in scores}):
        learned = scores.get((baseline, LEARNED))
        rows = {}
        for variant in sorted(v for b, v in scores if b == baseline):
            mine = scores[(baseline, variant)]
            row: dict[str, Any] = {
                "pairs": len(mine),
                "score_ppm": sum(mine) * PPM // (HALVES_PER_PAIR * len(mine)),
            }
            if learned is not None and variant != LEARNED and len(learned) == len(mine):
                diffs = [a - b for a, b in zip(mine, learned, strict=True)]
                row["minus_learned"] = {
                    "mean_ppm": sum(diffs) * PPM // (HALVES_PER_PAIR * len(diffs)),
                    "better": sum(d > 0 for d in diffs),
                    "equal": sum(d == 0 for d in diffs),
                    "worse": sum(d < 0 for d in diffs),
                }
            rows[variant] = row
        out["baselines"][baseline] = rows
    return out
