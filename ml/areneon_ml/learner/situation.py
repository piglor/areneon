"""The Threatened flag of a Battle decision, from the TrainingExample alone (ADR-010 erratum A11.2).

A Python twin of ``packages/showdown-bridge/src/situation.ts`` (``classifySlot``, #1283's one
definition of "Threatened"), written for the learner's per-decision anchor weight
(``ppo.anchor.threatened``). It reads the example's own ``candidate_features`` arrays ``threats``,
``speeds`` and ``foe_priorities`` (``probe-damage/2`` or ``probe-outcome/1``) and the last
``|request|`` line of the observation's log, and nothing else: a pure function of the example, so a
rerun, a resumed run and a parallel load give the same flags. Nothing is guessed: an unsupported or
missing row never threatens. ``tests/ml/learner/test_situation.py`` holds the twin equal to the TS
module over ``fixtures/situation/rows.json``, which the TS side writes from the committed corpora.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

SITUATIONS = ("threatened", "safe", "exposed_slower")

_REQUEST = "|request|"
_FAINTED = " fnt"
_ROW_KEYS = ("threats", "speeds", "foe_priorities")
_KO_FLOOR_PPM = 1_000_000


def has_threat_rows(features: Mapping[str, Any] | None) -> bool:
    """Whether ``candidate_features`` carries the arrays the reading needs."""
    return features is not None and all(key in features for key in _ROW_KEYS)


def _reaches(rows: Mapping[str, Any], position: int) -> list[tuple[int, str]]:
    """The revealed foe moves ``(foe, move)`` that reach own ``position``'s HP with some no-crit
    roll (``ohko_rolls > 0``, or ``max_ppm >= 1,000,000`` when the KO counts are null)."""
    found = []
    for row in rows["threats"]:
        if row["own"] != position or row["mega"] or row["kind"] != "computed":
            continue
        ohko = row["ohko_rolls"]
        if (row["max_ppm"] >= _KO_FLOOR_PPM) if ohko is None else (ohko > 0):
            found.append((row["foe"], row["move"]))
    return found


def _foe_first(rows: Mapping[str, Any], position: int, foe: int, move: str) -> bool:
    """True when ``foe``'s ``move`` acts before own ``position``'s default-priority action: its
    priority is above 0, or equal to 0 and the speed relation is ``foe_first``."""
    priority = 0
    for row in rows["foe_priorities"]:
        if row["foe"] == foe and row["move"] == move:
            priority = row["priority"] if row["kind"] == "computed" else 0
            break
    if priority != 0:
        return priority > 0
    for row in rows["speeds"]:
        if row["own"] == position and not row["mega"] and row["foe"] == foe:
            return bool(row["kind"] == "computed" and row["relation"] == "foe_first")
    return False


def classify_slot(rows: Mapping[str, Any], position: int) -> str:
    """``threatened``, ``exposed_slower`` or ``safe`` for own ``position`` (``classifySlot``)."""
    reached = _reaches(rows, position)
    if not reached:
        return "safe"
    first = any(_foe_first(rows, position, foe, move) for foe, move in reached)
    return "threatened" if first else "exposed_slower"


def living_active_positions(request: Mapping[str, Any]) -> list[int] | None:
    """The own active positions (indices of ``side.pokemon``) of a MOVE request that are not
    fainted, or None when the request is not a move request (Team Preview, forced switch, wait)."""
    if "teamPreview" in request or "active" not in request:
        return None
    return [
        position
        for position, member in enumerate(request["side"]["pokemon"])
        if member.get("active") is True and not str(member["condition"]).endswith(_FAINTED)
    ]


def last_request(log: Sequence[str]) -> dict[str, Any] | None:
    """The body of the last ``|request|`` line of an observation log, or None."""
    for line in reversed(log):
        if line.startswith(_REQUEST):
            body = json.loads(line[len(_REQUEST) :])
            return body if isinstance(body, dict) else None
    return None


def is_threatened(log: Sequence[str], features: Mapping[str, Any]) -> bool:
    """Whether some living own active slot of the request ``log`` ends in is Threatened under
    ``features`` (which must carry the three arrays: see :func:`has_threat_rows`)."""
    request = last_request(log)
    positions = None if request is None else living_active_positions(request)
    if not positions:
        return False
    return any(classify_slot(features, position) == "threatened" for position in positions)


def threatened_flag(log: Sequence[str], features: Mapping[str, Any] | None) -> bool | None:
    """The flag of one Battle example: :func:`is_threatened`, or None when its
    ``candidate_features`` lack the arrays (the learner refuses that under the option)."""
    if features is None or not has_threat_rows(features):
        return None
    return is_threatened(log, features)
