"""Reject PokePaste-shaped team files outside ``fixtures/``.

Real competitive teams are private and must never be committed. Public
fixtures live under ``fixtures/`` and must be synthetic or sanitized. A file
elsewhere may opt out with the marker ``pokepaste-guard: allow`` (for example
documentation that shows a deliberately fake set).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ALLOWED_PREFIXES = ("fixtures/",)
ALLOW_MARKER = "pokepaste-guard: allow"
MIN_MOVES = 2

_ABILITY = re.compile(r"^\s*Ability:\s*\S", re.MULTILINE)
_MOVE = re.compile(r"^\s*-\s+\S", re.MULTILINE)
_STATS = re.compile(r"^\s*(EVs|IVs|SPs|Stat Points):\s*\d", re.MULTILINE)
_NATURE = re.compile(r"^\s*\w+ Nature\s*$", re.MULTILINE)


def looks_like_pokepaste(text: str) -> bool:
    """True when ``text`` contains at least one PokePaste set.

    A set is recognised by an ``Ability:`` line plus at least two move lines
    and either a stat-spread line or a nature line.
    """
    if not _ABILITY.search(text):
        return False
    if len(_MOVE.findall(text)) < MIN_MOVES:
        return False
    return bool(_STATS.search(text) or _NATURE.search(text))


def is_allowed_path(path: str) -> bool:
    posix = PurePosixPath(path).as_posix()
    return posix.startswith(ALLOWED_PREFIXES)


def violations(paths: Sequence[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if is_allowed_path(path):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        if ALLOW_MARKER in text:
            continue
        if looks_like_pokepaste(text):
            found.append(path)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    bad = violations(paths)
    for path in bad:
        print(
            f"{path}: looks like a PokePaste team. Private teams belong in "
            "private-data/ (gitignored); public fixtures in fixtures/.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
