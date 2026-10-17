"""Reject PokePaste-shaped team files outside ``fixtures/``.

Real competitive teams are private and must never be committed. Public
fixtures live under the top-level ``fixtures/`` directory and must be
synthetic or sanitized. Any other exemption must be listed by exact path in
``ALLOWLIST_FILE``, so every exception is visible in review; nothing inside a
file can switch the check off.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ALLOWED_TOP_DIRS = frozenset({"fixtures"})
ALLOWLIST_FILE = ".pokepaste-guard-allow"
MIN_MOVES = 2
MIN_SET_MARKERS = 2

_MOVE = re.compile(r"^\s*-\s+\S", re.MULTILINE)
_SET_MARKERS = (
    re.compile(r"^\s*Ability:\s*\S", re.MULTILINE),
    re.compile(r"^\s*(EVs|IVs|SPs|Stat Points):\s*\d", re.MULTILINE),
    re.compile(r"^\s*\w+ Nature\s*$", re.MULTILINE),
    re.compile(r"^[^\n@-][^\n@]* @ \S[^\n]*$", re.MULTILINE),  # "Species @ Item"
    re.compile(r"^\s*Tera Type:\s*\S", re.MULTILINE),
    re.compile(r"^\s*Level:\s*\d", re.MULTILINE),
)


def looks_like_pokepaste(text: str) -> bool:
    """True when ``text`` contains at least one PokePaste set.

    A set is recognised by at least ``MIN_MOVES`` move lines plus at least
    ``MIN_SET_MARKERS`` distinct set markers (ability, stat spread, nature,
    ``Species @ Item`` header, Tera type, level).
    """
    if len(_MOVE.findall(text)) < MIN_MOVES:
        return False
    markers = sum(1 for pattern in _SET_MARKERS if pattern.search(text))
    return markers >= MIN_SET_MARKERS


def normalise(path: str) -> str:
    parts = PurePosixPath(Path(path).as_posix()).parts
    return "/".join(part for part in parts if part != ".")


def load_allowlist(path: str = ALLOWLIST_FILE) -> frozenset[str]:
    """Exact repo-relative paths exempt from the guard (``#`` starts a comment)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return frozenset()
    entries = (line.split("#", 1)[0].strip() for line in lines)
    return frozenset(normalise(entry) for entry in entries if entry)


def is_allowed_path(path: str, allowlist: frozenset[str] = frozenset()) -> bool:
    normal = normalise(path)
    return normal.split("/", 1)[0] in ALLOWED_TOP_DIRS or normal in allowlist


def violations(paths: Sequence[str], allowlist: frozenset[str] = frozenset()) -> list[str]:
    found: list[str] = []
    for path in paths:
        if is_allowed_path(path, allowlist):
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        if looks_like_pokepaste(text):
            found.append(path)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    bad = violations(paths, load_allowlist())
    for path in bad:
        print(
            f"{path}: looks like a PokePaste team. Private teams belong in "
            "private-data/ (gitignored); public fixtures in fixtures/.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
