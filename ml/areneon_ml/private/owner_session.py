"""The owner-session guard (ADR-015 erratum E1, items 7-9), Python twin of ``owner-session.ts``.

Every command that reads private teams calls ``assert_owner_session`` before it
reads any team: it refuses in an agent session (``CLAUDECODE`` or ``AI_AGENT``
set) and, for the top-level command, unless stdin, stdout and stderr are
terminals and ``/dev/tty`` opens. There is no override. A child a guarded
driver starts has pipes for stdio and calls ``assert_guarded_child``, which
checks the environment only. The fixed words equal the TypeScript ones
(``fixtures/private-guard/refusals.json``, pinned by a test on each side); a
message names no path, team or variable value. Accident prevention, not a
boundary: a process that sets up a pty and clears its environment gets past it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

OWNER_SESSION_REFUSALS: Final[dict[str, str]] = {
    "agent_session": "refusing: an agent session is running this; use your own terminal",
    "not_a_terminal": (
        "refusing: stdin, stdout and stderr must all be a terminal; "
        "run it in an interactive terminal"
    ),
    "no_controlling_terminal": "refusing: no controlling terminal to show the detail on",
}


class OwnerSessionError(Exception):
    """A refusal with a fixed code and fixed words."""

    def __init__(self, code: str) -> None:
        super().__init__(OWNER_SESSION_REFUSALS[code])
        self.code = code


def open_controlling_terminal() -> None:
    """Open ``/dev/tty`` and close it again; the fixed refusal if there is none."""
    try:
        fd = os.open("/dev/tty", os.O_WRONLY)
    except OSError:
        raise OwnerSessionError("no_controlling_terminal") from None
    os.close(fd)


def _is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except ValueError:  # a closed stream
        return False


def _stdio_flags() -> tuple[bool, bool, bool]:
    return (_is_tty(sys.stdin), _is_tty(sys.stdout), _is_tty(sys.stderr))


@dataclass(frozen=True)
class OwnerSession:
    """What the guard looks at: a seam for tests; the default reads the process."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    is_terminal: Callable[[], tuple[bool, bool, bool]] = _stdio_flags
    open_terminal: Callable[[], None] = open_controlling_terminal


def agent_refusal_code(env: Mapping[str, str]) -> str | None:
    """The fixed code for an agent environment (any non-empty value), else None."""
    return "agent_session" if env.get("CLAUDECODE") or env.get("AI_AGENT") else None


def assert_owner_session(session: OwnerSession | None = None) -> None:
    """Environment, then the three stdio, then ``/dev/tty``. Call it before reading any team."""
    here = session if session is not None else OwnerSession()
    code = agent_refusal_code(here.env)
    if code is not None:
        raise OwnerSessionError(code)
    if not all(here.is_terminal()):
        raise OwnerSessionError("not_a_terminal")
    here.open_terminal()


def assert_guarded_child(env: Mapping[str, str] | None = None) -> None:
    """A child of a guarded driver: its stdio are pipes, so only the environment is checked."""
    code = agent_refusal_code(env if env is not None else os.environ)
    if code is not None:
        raise OwnerSessionError(code)
