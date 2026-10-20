"""Reject committed JSON that is, or may be, private team or game data.

ADR-008 §5: every PrivateTeamSpec, Episode and PlayerChannelLog carries a
``data_class``: ``synthetic`` (CC0, may be committed) or ``private`` (a real
team or a human's games, kept only in the gitignored ``private-data/``). The
PokePaste guard does not see JSON teams, so this guard rejects a committed
``.json`` or ``.jsonl`` file that

- has ``"data_class": "private"`` anywhere; or
- outside ``fixtures/``, holds a team-shaped object without
  ``"data_class": "synthetic"`` at its root (for JSONL, at the root of that
  line). Team-shaped means a set (``species``, ``moves`` and ``evs`` keys) or
  a Pokémon as a ``|request|`` carries it (``details``, ``stats`` and
  ``moves``), whose stats reveal the spread; in both, ``moves`` is a list; or
- holds a sealed evaluation Episode (ADR-009 rule T2; evaluation protocol I7
  and O13): any object with ``"kind": "evaluation"`` and ``"sealed": true``,
  which is ``provenance.collection`` of an Episode. Sealed material lives only
  in the isolated evaluation store, so this is rejected everywhere, including
  ``fixtures/``, with one exception: a path in ``SEALED_MARKERS`` (an exact
  list; a new marker is a visible, code-owner-reviewed edit) whose root says
  ``"data_class": "synthetic"``. Sealed JSON anywhere else, in JSONL, or with
  any other root label is rejected. The two markers are the corpus samples
  ``valid/evaluation-sealed.json`` and ``invalid/collection-evaluation-no-seed.json``
  (a negative sample that is kind evaluation with sealed true). The guard
  checks structure and path, not content: a real Episode relabelled synthetic
  and placed at an allowlisted path is caught only by review and the O13 hash
  scan (evaluation-protocol §7). Stringified JSON inside a string is not
  inspected. A marker added by another PR means rebasing onto this constant
  (the pinning test fails until it is edited).

- holds a team record or manifest of the team system (ADR-013 §4, §8)
  without ``"data_class": "synthetic"`` on the object itself, wherever it
  is, ``fixtures/`` included. A TeamRecord is an object with a string
  ``paste`` and ``team_hash``; a TeamDatasetManifest has ``dataset_id``,
  ``generator``, ``key`` and ``splits``; a TeamSnapshotManifest has
  ``snapshot_id``, ``shards`` and ``cells``; or
- holds a TeamDatasetManifest whose ``key`` is not ``{"kind": "public", ...}``.
  Only a public generation key may be committed; a sealed set's key lives in
  the isolated evaluation store, so any other key kind, and a missing key, is
  rejected.

A file that is not valid JSON is rejected too, so JSON with comments or
trailing commas (JSONC) cannot be committed: the guard cannot vouch for
what it cannot read. Accepted residual risks: an unlabelled team inside
``fixtures/`` is left to review and fixture provenance, and teams in
non-JSON text (YAML, source code, raw logs) are outside this guard.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

SEALED_MARKERS = frozenset(
    {
        "fixtures/contracts/v1/episode/valid/evaluation-sealed.json",
        "fixtures/contracts/v1/episode/invalid/collection-evaluation-no-seed.json",
    }
)
SUFFIXES = (".json", ".jsonl")
TEAM_SHAPES = (
    frozenset({"species", "moves", "evs"}),  # a set (PokemonSet, PokePaste import)
    frozenset({"details", "stats", "moves"}),  # a Pokémon in a |request| side
)
PRIVATE = 'has "data_class": "private"'
SEALED = 'holds a sealed evaluation Episode ("kind": "evaluation", "sealed": true)'
UNLABELLED_TEAM = 'holds a team without "data_class": "synthetic" at its root'
UNLABELLED_RECORD = (
    'holds a team record or manifest without "data_class": "synthetic" on the object itself'
)
UNKNOWN_KEY_KIND = 'holds a team dataset manifest whose key is not of kind "public"'


def _walk(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _is_team(obj: dict[str, object]) -> bool:
    # Real team data lists its moves; a JSON Schema's `properties` with the
    # same keys holds a subschema there instead.
    return isinstance(obj.get("moves"), list) and any(obj.keys() >= shape for shape in TEAM_SHAPES)


def _is_str(obj: dict[str, object], *keys: str) -> bool:
    return all(isinstance(obj.get(key), str) for key in keys)


def _is_team_record(obj: dict[str, object]) -> bool:
    # A JSON Schema's `properties` has the same keys but subschemas, not strings.
    return _is_str(obj, "paste", "team_hash")


def _is_dataset_manifest(obj: dict[str, object]) -> bool:
    return (
        _is_str(obj, "dataset_id")
        and isinstance(obj.get("generator"), dict)
        and "key" in obj
        and isinstance(obj.get("splits"), list)
    )


def _is_snapshot_manifest(obj: dict[str, object]) -> bool:
    return (
        _is_str(obj, "snapshot_id")
        and isinstance(obj.get("shards"), list)
        and isinstance(obj.get("cells"), list)
    )


def _is_sealed(obj: dict[str, object]) -> bool:
    return obj.get("kind") == "evaluation" and obj.get("sealed") is True


def is_sealed_marker_path(path: str) -> bool:
    """True for the exact paths where a synthetic sealed marker may live (rule T2)."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in SEALED_MARKERS


def problems_in(document: object, in_fixtures: bool, sealed_marker_ok: bool = False) -> list[str]:
    """Why one parsed JSON document must not be committed (empty if it may).

    ``sealed_marker_ok`` is true only for a file at ``is_sealed_marker_path``.
    """
    found: list[str] = []
    objects = list(_walk(document))
    if any(obj.get("data_class") == "private" for obj in objects):
        found.append(PRIVATE)
    if any(_is_sealed(obj) for obj in objects):
        synthetic_marker = (
            sealed_marker_ok
            and isinstance(document, dict)
            and document.get("data_class") == "synthetic"
        )
        if not synthetic_marker:
            found.append(SEALED)
    records = [
        obj
        for obj in objects
        if _is_team_record(obj) or _is_dataset_manifest(obj) or _is_snapshot_manifest(obj)
    ]
    if any(obj.get("data_class") != "synthetic" for obj in records):
        found.append(UNLABELLED_RECORD)
    for obj in objects:
        key = obj.get("key")
        if _is_dataset_manifest(obj) and not (
            isinstance(key, dict) and key.get("kind") == "public"
        ):
            found.append(UNKNOWN_KEY_KIND)
    synthetic_root = isinstance(document, dict) and document.get("data_class") == "synthetic"
    has_team = any(_is_team(obj) for obj in objects)
    if not in_fixtures and not synthetic_root and has_team:
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
        marker_ok = is_sealed_marker_path(path)
        reasons = {
            reason for doc in documents for reason in problems_in(doc, in_fixtures, marker_ok)
        }
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
