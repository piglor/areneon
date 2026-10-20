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
    """The observation and the candidates, minus ``decision_context`` and every ``rqid``."""
    observation = {k: v for k, v in query["observation"].items() if k != "decision_context"}
    observation["identity"] = {k: v for k, v in observation["identity"].items() if k != "rqid"}
    observation["log"] = [_without_rqid(line) for line in observation["log"]]
    return {"observation": observation, "candidates": query["candidates"]}


def view_digest(query: dict[str, Any]) -> str:
    """SHA-256 of the view's JCS bytes: equal for the A1 and A2 forms of one decision."""
    return sha256_hex(jcs(encoder_input(query)))
