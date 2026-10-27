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
  any object with a string ``paste`` member (ADR-015 §6: a PokePaste record
  with none of those keys); or
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

- holds ``"data_class": "private"`` outside a path in ``PRIVATE_SAMPLE_MARKERS``
  (ADR-015 §6). A marker is an exact file, never a directory, for a synthetic
  sample of a private record (the contract corpus needs one; ``data_class``
  stays ``private`` in the type). The guard itself checks the marked file: its
  root says ``"data_class": "private"``, it holds at least one ``paste``, and
  every ``paste`` equals a paste in the committed synthetic team corpus
  (``fixtures/teams/**``: a ``.txt`` file's whole text, or a ``paste`` string
  in a JSON file there). The marker waives only the private-class reason, for
  the root object only (a record at the root is not "unlabelled" for being
  private); the sealed and unlabelled-team rules still apply. A marker added
  by another PR means rebasing onto this constant (the pinning test fails
  until it is edited). The marked root's keys must equal ``PRIVATE_RECORD_KEYS``
  (the PrivateTeamRecord shape), no other object in the file may be team-shaped,
  and no other string (keys included) may look like a paste (a ``Species @ Item``,
  ``Ability:`` or ``- Move`` line), so real team text cannot ride in a marked
  file. Every value is also type-checked against the PrivateTeamRecord (ADR-015
  §2): ``schema_version`` 1, ``data_class`` ``private``, ``team_id`` 32 hex,
  ``revision`` 1 to 2**31-1, ``supersedes_record_hash`` null or 64 hex, both team
  hashes 64 hex, ``status`` active or retired, at most 16 ``tags`` matching
  ``[a-z0-9][a-z0-9-]{0,31}``, at most 64 ``labels`` that are exactly
  ``{label_id, value}`` token pairs (ADR-013), a boolean ``holdout``, and one corpus ``paste``. That
  both hashes equal the paste's hashes needs the engine's ``Teams.import``, so
  the guard cannot check it: the vitest round-trip in private-records.test.ts (this
  slice) does, and the pytest tests here pin the keys and types to the schema; or
- holds ``"data_class": "private"`` as the root of a PrivateSnapshotManifest
  outside a path in ``PRIVATE_MANIFEST_MARKERS`` (ADR-015 §2, §6), or at such a
  path with anything but the strict manifest shape: exactly the manifest's keys
  at every level (``parent_snapshot_hash`` and ``first`` optional), every string
  a 64-hex hash, the 40-hex engine commit, a fixed word or an id token
  (``[A-Za-z0-9][A-Za-z0-9._:-]{0,127}``), every number an integer in the
  schema's range, and no ``paste``. No free text and no team content can ride in
  a manifest; the marker waives only the root's private label. The test files
  pin the keys, types and ranges to the schema; or
- holds ``"data_class": "private"`` as the root of a PrivateTestRecord (ADR-015
  erratum E1) outside a path in ``PRIVATE_TEST_RECORD_MARKERS``, or at such a path
  with anything but the strict record shape (``PRIVATE_TEST_RECORD_SPEC``): the
  schema's keys at every level and no others, every string a 64-hex hash, the
  40-hex engine commit, a fixed word, an id token, a policy id, a match id or a
  forfeit cause, and every number an integer in the schema's range. No team, no
  paste and no free text can ride in a test record; the marker waives only the
  root's private label; or
- has a duplicate key in any JSON object (and, for JSONL, in any line): a
  reader keeps the last value while the earlier one stays in the committed
  bytes, so the guard refuses the file; or
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
- holds a role-set table (ADR-020 §3: an object with a string ``source_path``,
  a string ``source_sha256`` and a ``species`` list) that holds a ``paste`` or
  any team-shaped object, or a ``paste`` or ``evs`` key, anywhere inside it,
  ``fixtures/`` included. A table is engine data (role names, move and ability
  ids), never a team; or
- holds a usage prior (ADR-025 §2: an object with a ``source`` object that has a
  string ``kind``, a ``derivation`` object and a ``species`` list) that holds a
  ``paste``, an ``evs`` key or any team-shaped object anywhere inside it,
  ``fixtures/`` included. A prior is aggregate marginals (ids and integers),
  never a team; or
- holds a sealed-set reveal or a sealed team (ADR-013 slice (g), protocol
  L3 and L5) anywhere but at an exact path in ``REVEAL_ALLOWED``. A reveal
  is a ``RetiredSealedManifest``: an object with a string ``set_id``, a
  ``reveal`` object holding a string ``key`` and ``salt``, and a ``files``
  object. A sealed team is a line ``{k, paste, stratum}`` of a set's
  ``teams.jsonl`` (extra keys do not evade). Both are secret until the set retires, and public
  (CC0) after, so a retired bundle is committed by a visible, code-owner-
  reviewed edit of ``REVEAL_ALLOWED`` (initially the corpus samples of the
  reveal only). A reveal must also say ``"data_class": "synthetic"`` on
  itself. The guard checks structure and path, not retirement: whether the
  register logs the set as retired is the evaluation job's check (O13).

A file that is not valid JSON is rejected too, so JSON with comments or
trailing commas (JSONC) cannot be committed: the guard cannot vouch for
what it cannot read. Accepted residual risks: an unlabelled team inside
``fixtures/`` is left to review and fixture provenance, and teams in
non-JSON text (YAML, source code, raw logs) are outside this guard.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SEALED_MARKERS = frozenset(
    {
        "fixtures/contracts/v1/episode/valid/evaluation-sealed.json",
        "fixtures/contracts/v1/episode/invalid/collection-evaluation-no-seed.json",
    }
)
PRIVATE_SAMPLE_MARKERS = frozenset(
    {
        "fixtures/contracts/v1/private-team-record/reader-invalid/canonical-hash-mismatch.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/duplicate-label-ids.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/duplicate-tags.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/first-revision-supersedes.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/later-revision-no-predecessor.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/team-hash-mismatch.json",
        "fixtures/contracts/v1/private-team-record/reader-invalid/unsorted-tags.json",
        "fixtures/contracts/v1/private-team-record/valid/edited.json",
        "fixtures/contracts/v1/private-team-record/valid/first.json",
        "fixtures/contracts/v1/private-team-record/valid/holdout.json",
        "fixtures/contracts/v1/private-team-record/valid/retired.json",
    }
)
# ADR-015 section 2 field types; labels are ADR-013's (an array of at most 64
# {label_id, value} pairs whose strings match the ADR-013 token pattern).
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")
TAG = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
LABEL_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
MAX_TAGS = 16
MAX_LABELS = 64
MAX_REVISION = 2**31 - 1
MAX_SAFE = 2**53 - 1
HEX40 = re.compile(r"[0-9a-f]{40}")
POLICY_ID = re.compile(r"policy:[0-9a-f]{64}")
OPPONENT_ID = re.compile(r"(?:[a-z0-9][a-z0-9._-]*@[A-Za-z0-9._-]+|policy:[0-9a-f]{64})")
MATCH_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
CAUSE = re.compile(r"[a-z][a-z_]{0,31}")
FORMAT_ID = re.compile(r"[a-z0-9]+")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}")
# Exact files that may hold a synthetic sample of a PrivateSnapshotManifest
# (ADR-015 section 2), each checked against the strict shape below.
PRIVATE_MANIFEST_MARKERS = frozenset(
    {
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/active-rows-over-rows.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/both-parent-and-first.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/cell-not-active-sum.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/cells-out-of-order.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/counts-not-shard-rows.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/duplicate-shard.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/neither-parent-nor-first.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/ranges-do-not-tile.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/ranges-out-of-order.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/records-active-not-cells.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/rows-not-range-sum.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/shard-index-not-position.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/shards-out-of-order.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/unlisted-active-cell.json",
        "fixtures/contracts/v1/private-snapshot-manifest/reader-invalid/wrong-logical-hash.json",
        "fixtures/contracts/v1/private-snapshot-manifest/valid/all-retired.json",
        "fixtures/contracts/v1/private-snapshot-manifest/valid/extended.json",
        "fixtures/contracts/v1/private-snapshot-manifest/valid/first.json",
    }
)
# The strict PrivateSnapshotManifest shape: a spec per object (required keys,
# optional keys). A leaf is a kind name or a tuple, see _leaf_ok; a
# ("list", maximum length, item spec) is an array.
_VERSION_MANIFEST: Any = (
    {
        "showdown_commit": "hex40",
        "format_id": "format",
        "regulation": "token",
        "ruleset_hash": "hex64",
        "format_rules_hash": "hex64",
    },
    {},
)
_RANGE: Any = (
    {
        "stratum": ("int", 0, 65535),
        "first_row": ("int", 0, MAX_SAFE),
        "rows": ("int", 1, MAX_SAFE),
        "active_rows": ("int", 0, MAX_SAFE),
    },
    {},
)
_SHARD: Any = (
    {
        "index": ("int", 0, 4095),
        "file_sha256": "hex64",
        "bytes": ("int", 1, MAX_SAFE),
        "logical_hash": "hex64",
        "rows": ("int", 1, MAX_SAFE),
        "split": ("const", "owner"),
        "cell_ranges": ("list", 65536, _RANGE),
    },
    {},
)
_CELL: Any = (
    {
        "split": ("const", "owner"),
        "stratum": ("int", 0, 65535),
        "rows": ("int", 1, MAX_SAFE),
    },
    {},
)
PRIVATE_MANIFEST_SPEC: Any = (
    {
        "schema_version": ("const", 1),
        "data_class": ("const", "private"),
        "snapshot_id": "token",
        "manifest": _VERSION_MANIFEST,
        "vocabulary_hash": "hex64",
        "label_set_hash": "hex64",
        "logical_snapshot_hash": "hex64",
        "shard_key": "hex64",
        "shards": ("list", 4096, _SHARD),
        "cells": ("list", 65536, _CELL),
        "records": (
            {"active": ("int", 0, MAX_SAFE), "retired": ("int", 0, MAX_SAFE)},
            {},
        ),
        "holdout": (
            {
                "rule": ("const", "hmac-sha256-team-id/1"),
                "key": "hex64",
                "ppm": ("const", 200000),
            },
            {},
        ),
        "writer": (
            {"name": "token", "version": "version", "binary_sha256": "hex64"},
            {},
        ),
        "parquet_schema_version": ("int", 1, 65535),
    },
    {"parent_snapshot_hash": "hex64", "first": ("const", True)},
)
# Exact files that may hold a synthetic sample of a PrivateTestRecord (ADR-015
# erratum E1, item 2), each checked against the strict shape below.
PRIVATE_TEST_RECORD_MARKERS = frozenset(
    {
        "fixtures/contracts/v1/private-test-record/reader-invalid/baseline-with-checkpoints.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/counts-not-games.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/dataset-hash-on-staging.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/forfeits-over-losses.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/forfeits-unsorted.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/games-not-twice-pairs.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/histogram-not-pairs.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/histogram-not-score.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/holdout-on-staging.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/match-games-not-plan.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/match-id-not-opponent-mode.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/match-twice.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/matches-unsorted.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/pair-without-checkpoints.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/pairs-over-half.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/private-pair-on-staging.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/private-pair-on-whole-snapshot.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/snapshot-hash-on-staging.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/snapshot-without-hash.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/turn-cap-over-ties.json",
        "fixtures/contracts/v1/private-test-record/reader-invalid/unknown-baseline.json",
        "fixtures/contracts/v1/private-test-record/valid/dataset.json",
        "fixtures/contracts/v1/private-test-record/valid/snapshot-holdout.json",
        "fixtures/contracts/v1/private-test-record/valid/staging.json",
    }
)
_COUNT: Any = ("int", 0, MAX_SAFE)
_CHECKPOINTS: Any = ({"preview": "hex64", "battle": "hex64"}, {})
_LATENCY: Any = (
    {"preview_ms": ("int", 1, MAX_SAFE), "battle_ms": ("int", 1, MAX_SAFE)},
    {},
)
_TEST_MATCH: Any = (
    {
        "match_id": "match_id",
        "policy_id": "policy_id",
        "mode": ("enum", ("sample", "argmax")),
        "opponent_policy_id": "opponent_id",
        "games": ("int", 2, MAX_SAFE),
        "wins": _COUNT,
        "ties": _COUNT,
        "losses": _COUNT,
        "turn_cap_ties": _COUNT,
        "forfeits": (
            "list",
            1024,
            (
                {
                    "side": ("enum", ("candidate", "opponent")),
                    "cause": "cause",
                    "count": ("int", 1, MAX_SAFE),
                },
                {},
            ),
        ),
        "pairs": ("int", 1, MAX_SAFE),
        "pair_score_counts": ("list", 5, _COUNT),
    },
    {"opponent_checkpoints": _CHECKPOINTS},
)
# The strict PrivateTestRecord shape (same spec language as PRIVATE_MANIFEST_SPEC).
PRIVATE_TEST_RECORD_SPEC: Any = (
    {
        "schema_version": ("const", 1),
        "data_class": ("const", "private"),
        "run_id": "token",
        "run_config_hash": "hex64",
        "generation": _COUNT,
        "generation_manifest_sha256": "hex64",
        "checkpoints": _CHECKPOINTS,
        "candidate_data_class": ("enum", ("synthetic", "private")),
        "version_manifest": _VERSION_MANIFEST,
        "test_set": (
            {
                "commitment": "hex64",
                "team_count": ("int", 2, MAX_SAFE),
                "pairs": ("int", 1, MAX_SAFE),
                "source_kind": ("enum", ("staging", "snapshot", "dataset")),
                "subset": ("enum", ("all", "holdout")),
            },
            {"snapshot_hash": "hex64", "dataset_hash": "hex64"},
        ),
        "plan": (
            {
                "kind": ("const", "disjoint-pairs"),
                "seed_derivation": ("const", "private-test-seed/1"),
                "pairing": ("const", "private-test-pairing/1"),
                "games_per_match": ("int", 2, MAX_SAFE),
                "max_turns": ("int", 1, MAX_SAFE),
                "latency_budget_ms": _LATENCY,
            },
            {},
        ),
        "matches": ("list", 64, _TEST_MATCH),
    },
    {},
)
# The root keys of a PrivateTeamRecord (ADR-015 section 2). A marked sample has
# exactly these keys, so team content cannot ride in others. A test pins this
# set to the properties of private-team-record.schema.json.
PRIVATE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "data_class",
        "team_id",
        "revision",
        "supersedes_record_hash",
        "team_hash",
        "canonical_team_hash",
        "paste",
        "status",
        "tags",
        "labels",
        "holdout",
    }
)
# A PokePaste signature: `Species @ Item`, an `Ability:` line or a `- Move` line.
PASTE_LINE = re.compile(r"^(?:[^\n]+ @ [^\n]+|Ability: [^\n]*|- \S[^\n]*)$", re.MULTILINE)
TEAM_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "teams"
REVEAL_ALLOWED = frozenset(
    {
        "fixtures/contracts/v1/retired-sealed-manifest/valid/revealed.json",
        "fixtures/contracts/v1/retired-sealed-manifest/invalid/short-salt.json",
        "fixtures/contracts/v1/retired-sealed-manifest/invalid/teams.json",
        "fixtures/contracts/v1/retired-sealed-manifest/reader-invalid/seed-is-salt.json",
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
PASTE_NOT_IN_CORPUS = (
    "is a private sample marker, but not a private record whose every paste "
    "(and at least one) is in the synthetic team corpus (fixtures/teams/**)"
)
MANIFEST_SHAPE = (
    "is a private manifest sample marker, but not exactly a PrivateSnapshotManifest: "
    "other keys, strings that are not hashes, id tokens or fixed words, or numbers "
    "out of range"
)
TEST_RECORD_SHAPE = (
    "is a private test record sample marker, but not exactly a PrivateTestRecord: "
    "other keys, strings that are not hashes, id tokens or fixed words, or numbers "
    "out of range"
)
MARKER_SHAPE = (
    "is a private sample marker holding more than a PrivateTeamRecord: other root keys, "
    "values that break the record's types, another team-shaped object, or paste text "
    "outside `paste`"
)
DUPLICATE_KEY = "has a duplicate JSON object key, so a reader would see only the last value"
REVEAL = "holds a sealed-set reveal or a sealed team outside a reviewed path"
UNLABELLED_REVEAL = (
    'holds a sealed-set reveal without "data_class": "synthetic" on the object itself'
)
ROLE_SET_TEAM = "holds a role-set table that holds a paste or a team-shaped object"
USAGE_PRIOR_TEAM = "holds a usage prior that holds a paste, an evs key or a team-shaped object"
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
    if isinstance(obj.get("paste"), str):
        return True
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


def _is_role_set_table(obj: dict[str, object]) -> bool:
    return _is_str(obj, "source_path", "source_sha256") and isinstance(obj.get("species"), list)


def _is_usage_prior(obj: dict[str, object]) -> bool:
    # ADR-025 section 2: a source with a kind, a derivation and a species list.
    source = obj.get("source")
    return (
        isinstance(source, dict)
        and isinstance(source.get("kind"), str)
        and isinstance(obj.get("derivation"), dict)
        and isinstance(obj.get("species"), list)
    )


def _holds_team_text(table: dict[str, object]) -> bool:
    return any(_is_team(obj) or "paste" in obj or "evs" in obj for obj in _walk(table))


def _role_set_problems(objects: list[dict[str, object]]) -> list[str]:
    found = []
    if any(_is_role_set_table(obj) and _holds_team_text(obj) for obj in objects):
        found.append(ROLE_SET_TEAM)
    if any(_is_usage_prior(obj) and _holds_team_text(obj) for obj in objects):
        found.append(USAGE_PRIOR_TEAM)
    return found


def _is_reveal(obj: dict[str, object]) -> bool:
    reveal = obj.get("reveal")
    return (
        _is_str(obj, "set_id")
        and isinstance(obj.get("files"), dict)
        and isinstance(reveal, dict)
        and _is_str(reveal, "key", "salt")
    )


def _is_sealed_team(obj: dict[str, object]) -> bool:
    return (
        obj.keys() >= {"k", "paste", "stratum"}
        and isinstance(obj.get("paste"), str)
        and isinstance(obj.get("k"), int)
        and isinstance(obj.get("stratum"), int)
    )


def _is_sealed(obj: dict[str, object]) -> bool:
    return obj.get("kind") == "evaluation" and obj.get("sealed") is True


def is_sealed_marker_path(path: str) -> bool:
    """True for the exact paths where a synthetic sealed marker may live (rule T2)."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in SEALED_MARKERS


def is_private_marker_path(path: str) -> bool:
    """True for the exact paths where a synthetic private-record sample may live."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in PRIVATE_SAMPLE_MARKERS


def is_private_manifest_path(path: str) -> bool:
    """True for the exact paths where a synthetic private-manifest sample may live."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in PRIVATE_MANIFEST_MARKERS


def is_private_test_record_path(path: str) -> bool:
    """True for the exact paths where a synthetic private-test-record sample may live."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in PRIVATE_TEST_RECORD_MARKERS


def team_corpus(root: Path | None = None) -> frozenset[str]:
    """Every paste string in the committed synthetic team corpus."""
    pastes: set[str] = set()
    for file in sorted((TEAM_CORPUS if root is None else root).rglob("*")):
        if file.suffix == ".txt":
            pastes.add(file.read_text(encoding="utf-8"))
        elif file.suffix in SUFFIXES:
            for doc in _documents(file.name, file.read_text(encoding="utf-8")):
                for obj in _walk(doc):
                    paste = obj.get("paste")
                    if isinstance(paste, str):
                        pastes.add(paste)
    return frozenset(pastes)


def is_reveal_path(path: str) -> bool:
    """True for the exact paths where a sealed-set reveal or sealed team may be committed."""
    return PurePosixPath(Path(path).as_posix()).as_posix() in REVEAL_ALLOWED


def _strict_marker_problems(
    document: object, manifest_marker_ok: bool, test_record_marker_ok: bool
) -> tuple[bool, list[str]]:
    """Whether a marked manifest or test record is exactly its strict shape, and why not."""
    manifest_ok = manifest_marker_ok and _spec_ok(PRIVATE_MANIFEST_SPEC, document)
    test_record_ok = test_record_marker_ok and _spec_ok(PRIVATE_TEST_RECORD_SPEC, document)
    found = []
    if manifest_marker_ok and not manifest_ok:
        found.append(MANIFEST_SHAPE)
    if test_record_marker_ok and not test_record_ok:
        found.append(TEST_RECORD_SHAPE)
    return manifest_ok or test_record_ok, found


def problems_in(  # noqa: PLR0913, PLR0917 - one flag per exact-path allowlist
    document: object,
    in_fixtures: bool,
    sealed_marker_ok: bool = False,
    private_marker_ok: bool = False,
    corpus: frozenset[str] | None = None,
    reveal_ok: bool = False,
    manifest_marker_ok: bool = False,
    test_record_marker_ok: bool = False,
) -> list[str]:
    """Why one parsed JSON document must not be committed (empty if it may).

    ``sealed_marker_ok`` is true only for a file at ``is_sealed_marker_path``;
    ``private_marker_ok`` only for one at ``is_private_marker_path``, and then
    ``corpus`` is the set of pastes the marked file may hold; ``reveal_ok`` only
    for one at ``is_reveal_path``; ``manifest_marker_ok`` only for one at
    ``is_private_manifest_path``, which must then be a strict PrivateSnapshotManifest;
    ``test_record_marker_ok`` only for one at ``is_private_test_record_path``, which must
    then be a strict PrivateTestRecord.
    """
    found: list[str] = []
    objects = list(_walk(document))
    record_ok = (
        isinstance(document, dict)
        and document.get("data_class") == "private"
        and _marker_pastes_ok(objects, corpus)
    )
    shape_ok = _marker_shape_ok(document, objects)
    strict_ok, strict_problems = _strict_marker_problems(
        document, manifest_marker_ok, test_record_marker_ok
    )
    waived = (private_marker_ok and record_ok and shape_ok) or strict_ok
    found.extend(strict_problems)
    if private_marker_ok and not record_ok:
        found.append(PASTE_NOT_IN_CORPUS)
    if private_marker_ok and not shape_ok:
        found.append(MARKER_SHAPE)
    if any(
        obj.get("data_class") == "private" and not (waived and obj is document) for obj in objects
    ):
        found.append(PRIVATE)
    if any(_is_sealed(obj) for obj in objects):
        synthetic_marker = (
            sealed_marker_ok
            and isinstance(document, dict)
            and document.get("data_class") == "synthetic"
        )
        if not synthetic_marker:
            found.append(SEALED)
    found.extend(_role_set_problems(objects))
    if any(_is_reveal(obj) or _is_sealed_team(obj) for obj in objects) and not reveal_ok:
        found.append(REVEAL)
    if any(_is_reveal(obj) and obj.get("data_class") != "synthetic" for obj in objects):
        found.append(UNLABELLED_REVEAL)
    records = [
        obj
        for obj in objects
        if _is_team_record(obj) or _is_dataset_manifest(obj) or _is_snapshot_manifest(obj)
    ]
    if any(
        obj.get("data_class") != "synthetic" and not (waived and obj is document) for obj in records
    ):
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


def _marker_pastes_ok(objects: list[dict[str, object]], corpus: frozenset[str] | None) -> bool:
    pastes = [obj["paste"] for obj in objects if "paste" in obj]
    return bool(pastes) and all(isinstance(p, str) and p in (corpus or frozenset()) for p in pastes)


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _is_int(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _is_match(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _labels_ok(labels: object) -> bool:
    if not isinstance(labels, list) or len(labels) > MAX_LABELS:
        return False
    return all(
        isinstance(label, dict)
        and label.keys() == {"label_id", "value"}
        and _is_match(LABEL_TOKEN, label["label_id"])
        and _is_match(LABEL_TOKEN, label["value"])
        for label in labels
    )


def _record_types_ok(record: dict[str, object]) -> bool:
    tags = record["tags"]
    supersedes = record["supersedes_record_hash"]
    return (
        _is_int(record["schema_version"], 1, 1)
        and record["data_class"] == "private"
        and _is_match(HEX32, record["team_id"])
        and _is_int(record["revision"], 1, MAX_REVISION)
        and (supersedes is None or _is_match(HEX64, supersedes))
        and _is_match(HEX64, record["team_hash"])
        and _is_match(HEX64, record["canonical_team_hash"])
        and record["status"] in ("active", "retired")
        and isinstance(tags, list)
        and len(tags) <= MAX_TAGS
        and all(_is_match(TAG, tag) for tag in tags)
        and _labels_ok(record["labels"])
        and isinstance(record["holdout"], bool)
    )


def _leaf_ok(kind: object, value: object) -> bool:
    """One leaf of PRIVATE_MANIFEST_SPEC: a kind name, ("int", low, high) or ("const", x)."""
    if isinstance(kind, tuple):
        if kind[0] == "const":
            return type(value) is type(kind[1]) and value == kind[1]
        if kind[0] == "enum":
            return isinstance(value, str) and value in kind[1]
        return _is_int(value, kind[1], kind[2])
    patterns = {
        "hex64": HEX64,
        "hex40": HEX40,
        "format": FORMAT_ID,
        "token": LABEL_TOKEN,
        "version": VERSION,
        "policy_id": POLICY_ID,
        "opponent_id": OPPONENT_ID,
        "match_id": MATCH_ID,
        "cause": CAUSE,
    }
    return _is_match(patterns[str(kind)], value)


def _spec_ok(spec: object, value: object) -> bool:
    """True when ``value`` is exactly what ``spec`` describes: no other key, no other leaf."""
    if isinstance(spec, tuple) and isinstance(spec[0], dict):
        required, optional = spec
        if not isinstance(value, dict):
            return False
        keys = required.keys() | optional.keys()
        if not (required.keys() <= value.keys() <= keys):
            return False
        return all(_spec_ok({**required, **optional}[key], item) for key, item in value.items())
    if isinstance(spec, tuple) and spec[0] == "list":
        return (
            isinstance(value, list)
            and len(value) <= spec[1]
            and all(_spec_ok(spec[2], item) for item in value)
        )
    return _leaf_ok(spec, value)


def _marker_shape_ok(document: object, objects: list[dict[str, object]]) -> bool:
    if not isinstance(document, dict) or document.keys() != PRIVATE_RECORD_KEYS:
        return False
    if not _record_types_ok(document):
        return False
    if any(_is_team(obj) for obj in objects if obj is not document):
        return False
    rest = {key: value for key, value in document.items() if key != "paste"}
    return not any(PASTE_LINE.search(text) for text in _strings(rest))


class DuplicateKeyError(ValueError):
    """A JSON object names one key twice."""


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise DuplicateKeyError(keys)
    return dict(pairs)


def _loads(text: str) -> object:
    return json.loads(text, object_pairs_hook=_no_duplicates)


def _documents(path: str, text: str) -> list[object]:
    if path.endswith(".jsonl"):
        return [_loads(line) for line in text.splitlines() if line.strip()]
    return [_loads(text)]


def violations(paths: Sequence[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if not path.endswith(SUFFIXES):
            continue
        try:
            documents = _documents(path, Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, IsADirectoryError):
            continue
        except DuplicateKeyError:
            found.append(f"{path}: {DUPLICATE_KEY}")
            continue
        except (json.JSONDecodeError, UnicodeDecodeError):
            found.append(f"{path}: is not valid JSON, so it cannot be checked")
            continue
        in_fixtures = PurePosixPath(Path(path).as_posix()).parts[:1] == ("fixtures",)
        marker_ok = is_sealed_marker_path(path)
        private_ok = is_private_marker_path(path)
        corpus = team_corpus() if private_ok else None
        reveal_ok = is_reveal_path(path)
        manifest_ok = is_private_manifest_path(path)
        test_record_ok = is_private_test_record_path(path)
        reasons = {
            reason
            for doc in documents
            for reason in problems_in(
                doc,
                in_fixtures,
                marker_ok,
                private_ok,
                corpus,
                reveal_ok,
                manifest_ok,
                test_record_ok,
            )
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
