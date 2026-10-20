"""The syntax of a candidate choice (ADR-006), parsed so an unknown shape is refused.

The enumerator proposes candidates in the engine's canonical syntax; this
parser reads only that syntax. A shape it does not know is not an invalid
reply: the worker answers ``unsupported_candidate``, which the harness
records as an incident (ADR-010 §2, R11). It judges nothing: legality is the
engine's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Doubles: two slots, each one of these (sim/side.ts input syntax).
_SLOT = re.compile(
    r"^(?:pass|switch [1-9]|move [1-9](?: -?[1-9])?(?: (?:mega|megax|megay|terastallize))?)$"
)
_TEAM = re.compile(r"^team [1-9]{1,9}$")
SLOTS = 2


class UnsupportedCandidate(ValueError):
    """The candidate has a shape this parser does not know."""


@dataclass(frozen=True)
class ParsedChoice:
    """``team`` for a Team Preview order; ``joint`` for the two slots of a turn."""

    kind: str
    parts: tuple[str, ...]


def parse_choice(choice: str) -> ParsedChoice:
    if _TEAM.match(choice):
        return ParsedChoice("team", (choice[len("team ") :],))
    slots = tuple(choice.split(", "))
    if len(slots) == SLOTS and all(_SLOT.match(slot) for slot in slots):
        return ParsedChoice("joint", slots)
    raise UnsupportedCandidate("a candidate has a shape the worker does not know")
