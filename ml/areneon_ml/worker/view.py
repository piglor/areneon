"""What the encoder reads of a query (ADR-010 §2).

The encoder ignores ``decision_context`` and ``identity.rqid``, which differ
between A1 and A2 by design (ADR-005 gate 3, ADR-008 §4), also inside the
``|request|`` lines of the log. It reads nothing of the query's ``query_id``,
``sampling`` or ``policy_id``. Two queries that differ only in those have the
same view, so the same tensors.
"""

from __future__ import annotations

import json
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

_REQUEST = "|request|"


def _without_rqid(line: str) -> str:
    """A ``|request|`` line without its ``rqid``, reserialised canonically."""
    if not line.startswith(_REQUEST):
        return line
    request = json.loads(line[len(_REQUEST) :])
    request.pop("rqid", None)
    return _REQUEST + jcs(request).decode("utf-8")


def encoder_input(query: dict[str, Any]) -> dict[str, Any]:
    """The observation and the candidates, minus ``decision_context`` and every ``rqid``.

    A query's ``candidate_features`` (ADR-022 §3) and ``opponent_sheet`` (ADR-027) are part of the
    view when it carries them, and the keys are left out when it does not, so a view without them
    is what it always was.
    """
    observation = {k: v for k, v in query["observation"].items() if k != "decision_context"}
    observation["identity"] = {k: v for k, v in observation["identity"].items() if k != "rqid"}
    observation["log"] = [_without_rqid(line) for line in observation["log"]]
    return _with_features({"observation": observation, "candidates": query["candidates"]}, query)


def _with_features(view: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    if query.get("candidate_features") is not None:
        view["candidate_features"] = query["candidate_features"]
    if query.get("opponent_sheet") is not None:
        view["opponent_sheet"] = query["opponent_sheet"]
    return view


class IncrementalView:
    """``encoder_input`` for the growing prefixes of one channel, cleaning each line once.

    The result equals ``encoder_input(query)`` for every query. A log that does
    not extend the lines already cleaned is cleaned from the start.
    """

    def __init__(self) -> None:
        self._raw: list[str] = []
        self._clean: list[str] = []

    def encoder_input(self, query: dict[str, Any]) -> dict[str, Any]:
        observation = {k: v for k, v in query["observation"].items() if k != "decision_context"}
        observation["identity"] = {k: v for k, v in observation["identity"].items() if k != "rqid"}
        log = observation["log"]
        seen = len(self._raw)
        if log[:seen] != self._raw:
            self._raw, self._clean, seen = [], [], 0
        new = log[seen:]
        self._raw.extend(new)
        self._clean.extend(_without_rqid(line) for line in new)
        observation["log"] = list(self._clean)
        return _with_features(
            {"observation": observation, "candidates": query["candidates"]}, query
        )


def view_digest(query: dict[str, Any]) -> str:
    """SHA-256 of the view's JCS bytes: equal for the A1 and A2 forms of one decision."""
    return sha256_hex(jcs(encoder_input(query)))
