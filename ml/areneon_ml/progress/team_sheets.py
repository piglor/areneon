"""Open-sheet matches of the progress evaluation (ADR-027 section 5 and 8, ADR-017 erratum E4).

``--team-sheets open`` plays an open twin (``open-<match id>``, ``team_sheets: "open"``) of every
match against a scripted baseline or another run's pair, on the same validation pairs, seeds and
seats, in the engine's open-sheet variant of the run's format. The closed (target) matches and
their bytes do not change, and the two regimes are NEVER pooled: an open match has its own
interval, and the per-generation view puts the score against ``damage-aware@v1`` in the closed and
in the open format side by side, descriptive and without a difference or a combined number.

All integer arithmetic. The Node side plays the games; this module checks the record against the
request and derives the view on display (never stored).
"""

from __future__ import annotations

from typing import Any

from areneon_ml.progress.modes import match_extra
from areneon_ml.progress.records import OPEN_PREFIX, is_damage_aware, is_open
from areneon_ml.progress.stats import (
    PAIR_INTERVAL_LABEL,
    PPM,
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
)

#: Says what an open row is: the engine's open-sheet format, beside the target's closed one.
OPEN_LABEL = (
    "open-sheet format (the engine's Bo3 variant, one game at a time): beside the closed "
    "target format, never pooled with it"
)
VIEW_LABEL = (
    "descriptive: the score against the damage-aware baseline in the closed and in the open "
    "format, same pairs, seeds and seats; no difference and no combined score"
)
#: The fields an open twin shares with its closed match.
_SAME = ("policy_id", "opponent", "mode", "games", "temperature_ppm", "truncation")


class TeamSheetsError(ValueError):
    """A record that does not hold the open matches asked for."""


def closed_view(record: dict[str, Any]) -> dict[str, Any]:
    """The record without its open matches: what the checks written for closed matches read."""
    return record | {"matches": [m for m in record["matches"] if not is_open(m)]}


def twin_of(match: dict[str, Any]) -> str:
    """The id of a closed match's open twin."""
    return f"{OPEN_PREFIX}{match['match_id']}"


def check_team_sheets(record: dict[str, Any], asked: bool, *, legacy: bool) -> None:
    """The open matches are exactly the open twins of the closed baseline and cross-run matches
    when asked for (a fresh evaluation), and none when not; a committed record (``legacy``) holds
    either all of them or none."""
    open_matches = {m["match_id"]: m for m in record["matches"] if is_open(m)}
    if not asked and not legacy:
        if open_matches:
            raise TeamSheetsError("this evaluation asks for no open-sheet matches")
        return
    if not open_matches:
        if asked:
            raise TeamSheetsError("the open-sheet matches asked for are missing")
        return
    eligible = {
        twin_of(m): m
        for m in record["matches"]
        if not is_open(m) and m["opponent"]["kind"] in ("baseline", "run")
    }
    if set(open_matches) != set(eligible):
        raise TeamSheetsError(
            "open-sheet matches are the open twins of the baseline and cross-run matches"
        )
    for name, twin in open_matches.items():
        closed = eligible[name]
        if any(closed.get(key) != twin.get(key) for key in _SAME):
            raise TeamSheetsError(f"{name}: it is not played like its closed match")
        if twin.get("baseline", {}).get("identity_hash") != closed.get("baseline", {}).get(
            "identity_hash"
        ):
            raise TeamSheetsError(f"{name}: it names another baseline identity than its twin")


def _interval(record: dict[str, Any], match: dict[str, Any]) -> dict[str, int]:
    scores = match_pair_scores(record, match["match_id"])
    interval = pair_interval(
        scores,
        bootstrap_stream_key(record["run_id"], record["generation"], f"match:{match['match_id']}"),
    )
    return {
        "score_ppm": (2 * match["wins"] + match["ties"]) * PPM // (2 * match["games"]),
        "low_ppm": interval.low_ppm,
        "high_ppm": interval.high_ppm,
    }


def team_sheets_view(record: dict[str, Any]) -> dict[str, Any]:
    """The generation's closed-versus-open row: per play mode, the score against the damage-aware
    baseline in the closed and in the open format with each one's own pair-level interval.

    ``{"label", "interval", "rows"}``; no rows (and no key in a printed result) when the record
    plays no open match or no damage-aware baseline. Derived, never stored.
    """
    by_id = {m["match_id"]: m for m in record["matches"]}
    rows = []
    for match in record["matches"]:
        if (
            is_open(match)
            or match["opponent"]["kind"] != "baseline"
            or not is_damage_aware(match["opponent"]["policy_id"])
            or match_extra(match) is not None
        ):
            continue
        twin = by_id.get(twin_of(match))
        if twin is None:
            continue
        rows.append(
            {
                "opponent": match["opponent"]["policy_id"],
                "mode": match["mode"],
                "pairs": match["games"] // 2,
                "closed": _interval(record, match),
                "open": _interval(record, twin),
            }
        )
    rows.sort(key=lambda r: (r["opponent"], r["mode"]))
    return {"label": VIEW_LABEL, "interval": PAIR_INTERVAL_LABEL, "rows": rows}
