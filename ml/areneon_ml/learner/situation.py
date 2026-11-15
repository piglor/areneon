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
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from areneon_ml.model.candidates import parse_choice
from areneon_ml.reducer.state import BattleState

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


# -- the groups of the cloning loss (ADR-010 erratum A13.2) ----------------------------------

GROUPS = ("streak", "threatened", "other")

_SIDE = re.compile(r"^p[1-4]: ")


@dataclass(frozen=True)
class Groups:
    """What the cloning learner reads a Battle decision's group with (``clone.situation_weights``):
    ``threatened`` says whether ``threatened_ppm`` is set (otherwise a Threatened decision is
    ``other``)."""

    threatened: bool


def streak_slots(
    state: BattleState, request: Mapping[str, Any], stall_moves: Collection[str]
) -> tuple[int, ...]:
    """The active slots (request positions) of a MOVE request that are in the ``streak`` situation:
    a living own active with ``protect_streak >= 1`` in the reduced state and a stalling move of the
    engine's own list among the request's moves of that slot, not ``disabled``."""
    active = request.get("active")
    if "teamPreview" in request or not active:
        return ()
    roster = {mon.name: mon for mon in state.own.roster}
    found = []
    for slot, entry in enumerate(active):
        member = request["side"]["pokemon"][slot]
        mon = roster.get(_SIDE.sub("", str(member["ident"]), count=1))
        if not entry or str(member["condition"]).endswith(_FAINTED) or mon is None:
            continue
        if mon.recent.protect_streak < 1:
            continue
        if any(m["id"] in stall_moves and not m.get("disabled") for m in entry["moves"]):
            found.append(slot)
    return tuple(found)


def streak_unmatched(state: BattleState, request: Mapping[str, Any]) -> int:
    """How many request slots of a MOVE request hold a living active whose ident matches no roster
    entry of the reduced state: ``streak_slots`` skips them, so this count says how often the streak
    rule could not judge a slot (reported as ``streak_unmatched`` in the clone report)."""
    active = request.get("active")
    if "teamPreview" in request or not active:
        return 0
    roster = {mon.name for mon in state.own.roster}
    count = 0
    for slot, entry in enumerate(active):
        member = request["side"]["pokemon"][slot]
        if not entry or str(member["condition"]).endswith(_FAINTED):
            continue
        count += _SIDE.sub("", str(member["ident"]), count=1) not in roster
    return count


def stall_candidates(
    choices: Sequence[str],
    slots: Collection[int],
    request: Mapping[str, Any],
    stall_moves: Collection[str],
) -> tuple[bool, ...]:
    """Per candidate: whether it uses a stalling move in one of the ``slots`` (the streak slots)."""
    out = []
    for choice in choices:
        parsed = parse_choice(choice)
        uses = False
        for slot in slots:
            action = parsed.slots[slot] if parsed.kind == "joint" else None
            if action is not None and action.kind == "move" and action.index is not None:
                moves = request["active"][slot]["moves"]
                uses = uses or (
                    action.index <= len(moves) and moves[action.index - 1]["id"] in stall_moves
                )
        out.append(uses)
    return tuple(out)
