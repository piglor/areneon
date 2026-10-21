"""Typed refusals of the log-to-state reducer (ADR-010 §3, R10).

The reducer never skips a line it does not understand: every refusal is one of
these, carrying the offending line, so a grammar change in the pinned engine
fails loudly instead of teaching the policy a misread state.
"""

from __future__ import annotations


class ReducerError(Exception):
    """Base of every refusal; `line` is the channel line that caused it."""

    def __init__(self, message: str, line: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return self.message if self.line is None else f"{self.message}: {self.line!r}"


class UnknownLineError(ReducerError):
    """The line type is not one the reducer has a rule for."""


class UnknownFieldError(ReducerError):
    """A known line type carries a field, tag or argument the reducer has no rule for."""


class MalformedLineError(ReducerError):
    """A known field has a value outside its grammar."""


class VocabularyError(ReducerError):
    """An id or name is not in the pinned engine-data vocabulary."""


class OpponentSecretError(ReducerError):
    """The channel holds something only the opponent (or the omniscient log) may see."""


class IllusionError(ReducerError):
    """An Illusion ending that the channel alone cannot attribute without guessing.

    Raised where the player's view of a disguise is ambiguous: a disguise shown more than once,
    an Illusion user that fainted under its disguise, or a Pokemon shown in two positions.
    """
