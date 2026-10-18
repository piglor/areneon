"""Reject committed JSON that is, or may be, private team or game data.

ADR-008 §5: every PrivateTeamSpec, Episode and PlayerChannelLog carries a
``data_class``: ``synthetic`` (CC0, may be committed) or ``private`` (a real
team or a human's games, kept only in the gitignored ``private-data/``). The
PokePaste guard does not see JSON teams, so this guard rejects a committed
``.json`` or ``.jsonl`` file that

- has ``"data_class": "private"`` anywhere; or
- outside ``fixtures/``, holds a team-shaped object (one with ``species``,
  ``moves`` and ``evs`` keys) without ``"data_class": "synthetic"`` at its
  root (for JSONL, at the root of that line).

A file that is not valid JSON is rejected too: the guard cannot vouch for
what it cannot read. An unlabelled team inside ``fixtures/`` is left to
review and fixture provenance, an accepted residual risk.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

SUFFIXES = (".json", ".jsonl")
TEAM_KEYS = frozenset({"species", "moves", "evs"})
PRIVATE = 'has "data_class": "private"'
UNLABELLED_TEAM = 'holds a team without "data_class": "synthetic" at its root'


def _walk(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def problems_in(document: object, in_fixtures: bool) -> list[str]:
    """Why one parsed JSON document must not be committed (empty if it may)."""
    found: list[str] = []
    objects = list(_walk(document))
    if any(obj.get("data_class") == "private" for obj in objects):
        found.append(PRIVATE)
    synthetic_root = isinstance(document, dict) and document.get("data_class") == "synthetic"
    if not in_fixtures and not synthetic_root and any(obj.keys() >= TEAM_KEYS for obj in objects):
        found.append(UNLABELLED_TEAM)
    return found


def _documents(path: str, text: str) -> list[object]:
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def violations(paths: Sequence[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if not path.endswith(SUFFIXES):
            continue
        try:
            documents = _documents(path, Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, IsADirectoryError):
            continue
        except (json.JSONDecodeError, UnicodeDecodeError):
            found.append(f"{path}: is not valid JSON, so it cannot be checked")
            continue
        in_fixtures = PurePosixPath(Path(path).as_posix()).parts[:1] == ("fixtures",)
        reasons = {reason for doc in documents for reason in problems_in(doc, in_fixtures)}
        found.extend(f"{path}: {reason}" for reason in sorted(reasons))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    bad = violations(paths)
    for line in bad:
        print(
            f"{line}. Private teams and games belong in private-data/ (gitignored); "
            "committed data must be labelled synthetic (ADR-008).",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
