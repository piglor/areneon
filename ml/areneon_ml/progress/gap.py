"""The play-mode gap, derived on display and never stored (ADR-023 section 2, ticket #1042).

For each scripted baseline and for the cross-run head-to-head, a version 2 record plays argmax
and sample on the same validation pairs, seats and seeds (ticket #1044 adds sample at a declared
temperature or threshold). The gap of a mode is ``d_k``, argmax pair score minus that mode's pair
score on pair ``k`` (``paired_differences``, in half-games), its mean and the pair-level bootstrap
interval (``pair_interval``) on a stream labelled with the opponent and both modes, so a page
shows the same interval every time. All integer arithmetic.

A diagnostic: conditional on the fixed validation plan, descriptive, no multiplicity control; it
is no test, claim or gate. Anchor and previous-generation matches play both sides in the same
mode, so the gap is not read from them. A version 1 record has no gap.
"""

from __future__ import annotations

from typing import Any

from areneon_ml.progress.modes import match_extra, match_stem, mode_label
from areneon_ml.progress.records import SCHEMA_V2
from areneon_ml.progress.stats import (
    PAIR_INTERVAL_LABEL,
    PPM,
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
    paired_differences,
)

GAP_LABEL = "diagnostic: play-mode gap, conditional on the fixed validation plan"
SAME_MODE_NOTE = "both sides play the same mode; the play-mode gap is not read from them"
V1_NOT_AVAILABLE = "not available: version 1 record"
NOT_PLAYED = "not available: argmax and sample were not both played against this opponent"


def _mode_name(match: dict[str, Any]) -> str:
    extra = match_extra(match)
    return "sample" if extra is None else extra.name


def _score_ppm(match: dict[str, Any]) -> int:
    return int((2 * match["wins"] + match["ties"]) * PPM // (2 * match["games"]))


def _gapped(match: dict[str, Any]) -> bool:
    """Whether a match takes part: a scripted baseline or the other run's pair, played by the
    plain candidate (a search or ablation match plays one mode)."""
    return (
        match["opponent"]["kind"] in ("baseline", "run")
        and "search" not in match
        and match.get("candidate_preview") is None
    )


def play_mode_gap(record: dict[str, Any]) -> dict[str, Any]:
    """The gap rows of a record: ``{"label", "interval", "rows"}``; a version 1 record gives
    ``{"label", "status": "not available: version 1 record"}``. A row is
    ``{opponent, match_id, mode, pairs, argmax_ppm, mode_ppm, mean_ppm, low_ppm, high_ppm,
    better, equal, worse}``, the gap being argmax minus the row's mode, so a positive gap means
    argmax scored higher."""
    if record["schema_version"] != SCHEMA_V2:
        return {"label": GAP_LABEL, "status": V1_NOT_AVAILABLE}
    argmax: dict[str, dict[str, Any]] = {}
    others: list[dict[str, Any]] = []
    for match in record["matches"]:
        if not _gapped(match):
            continue
        if match["mode"] == "argmax":
            argmax[match_stem(match)] = match
        else:
            others.append(match)
    rows = []
    for match in others:
        reference = argmax.get(match_stem(match))
        if reference is None:
            continue
        mine = match_pair_scores(record, reference["match_id"])
        theirs = match_pair_scores(record, match["match_id"])
        diffs = paired_differences(mine, theirs)
        stem = match_stem(match)
        interval = pair_interval(
            diffs,
            bootstrap_stream_key(
                record["run_id"],
                record["generation"],
                f"gap:{stem}:argmax:{_mode_name(match)}",
            ),
        )
        rows.append(
            {
                "opponent": stem,
                "match_id": match["match_id"],
                "mode": mode_label(match),
                "pairs": interval.pairs,
                "argmax_ppm": _score_ppm(reference),
                "mode_ppm": _score_ppm(match),
                "mean_ppm": interval.mean_ppm,
                "low_ppm": interval.low_ppm,
                "high_ppm": interval.high_ppm,
                "better": sum(d > 0 for d in diffs),
                "equal": sum(d == 0 for d in diffs),
                "worse": sum(d < 0 for d in diffs),
            }
        )
    rows.sort(key=lambda r: (r["opponent"], r["match_id"]))
    return {"label": GAP_LABEL, "interval": PAIR_INTERVAL_LABEL, "rows": rows}
