"""The syntax of a candidate choice (ADR-006), parsed into its per-slot parts.

The enumerator proposes candidates in the engine's canonical input syntax;
this parser reads only that syntax. The Battle Policy head embeds a candidate
from the parts it returns (action, move slot, target, switch target, event
flag), the Preview Policy head from the ordered team positions. A shape it
does not know is not an invalid reply: the worker answers
``unsupported_candidate`` and the harness records an incident (ADR-010 §2,
R11). It judges nothing: legality is the engine's. ``ParsedChoice.format``
is the exact inverse of ``parse_choice``, which the golden test checks over
every candidate of the committed choice corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SLOTS = 2  # a Doubles turn choice has two slots

# Engine event flags a move may carry (enumerator: canMegaEvo, canMegaEvoX,
# canMegaEvoY, canTerastallize), in the order the enumerator appends them.
EVENTS = ("mega", "megax", "megay", "terastallize")

_SLOT = re.compile(
    r"(?:(?P<pass>pass)"
    r"|switch (?P<switch>[1-9])"
    r"|move (?P<move>[1-9])(?: (?P<target>-?[1-9]))?(?: (?P<event>mega|megax|megay|terastallize))?)"
)
_TEAM = re.compile(r"team (?P<order>[1-9]{1,9})")


class UnsupportedCandidate(ValueError):
    """The candidate has a shape this parser does not know."""


@dataclass(frozen=True)
class SlotAction:
    """What one active slot does: pass, use a move (with optional target and event), or switch."""

    kind: Literal["pass", "move", "switch"]
    index: int | None = None  # the move slot (1-based) or the team position to switch to
    target: int | None = None  # the engine's target location, e.g. 1, 2, -1; None if untargeted
    event: str | None = None  # one of EVENTS

    def format(self) -> str:
        if self.kind == "pass":
            return "pass"
        if self.kind == "switch":
            return f"switch {self.index}"
        text = f"move {self.index}"
        if self.target is not None:
            text += f" {self.target}"
        if self.event is not None:
            text += f" {self.event}"
        return text


@dataclass(frozen=True)
class ParsedChoice:
    """``team``: a Team Preview order (``team`` positions, 1-based); ``joint``: two slot actions."""

    kind: Literal["team", "joint"]
    team: tuple[int, ...] = ()
    slots: tuple[SlotAction, ...] = ()

    @property
    def parts(self) -> tuple[str, ...]:
        if self.kind == "team":
            return ("".join(str(n) for n in self.team),)
        return tuple(slot.format() for slot in self.slots)

    def format(self) -> str:
        if self.kind == "team":
            return f"team {self.parts[0]}"
        return ", ".join(self.parts)


def _slot(text: str) -> SlotAction | None:
    found = _SLOT.fullmatch(text)
    if found is None:
        return None
    if found["pass"]:
        return SlotAction("pass")
    if found["switch"]:
        return SlotAction("switch", int(found["switch"]))
    target = found["target"]
    return SlotAction(
        "move", int(found["move"]), None if target is None else int(target), found["event"]
    )


def parse_choice(choice: str) -> ParsedChoice:
    """The parts of a canonical candidate; ``UnsupportedCandidate`` for any other shape."""
    team = _TEAM.fullmatch(choice)
    if team is not None:
        order = tuple(int(digit) for digit in team["order"])
        if len(set(order)) == len(order):
            return ParsedChoice("team", team=order)
        raise UnsupportedCandidate("a candidate has a shape the worker does not know")
    slots = [_slot(part) for part in choice.split(", ")]
    if len(slots) == SLOTS and all(slot is not None for slot in slots):
        return ParsedChoice("joint", slots=tuple(slot for slot in slots if slot is not None))
    raise UnsupportedCandidate("a candidate has a shape the worker does not know")
