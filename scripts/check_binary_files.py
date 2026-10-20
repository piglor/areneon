"""Reject committed binary data files.

ADR-013 §8 and ADR-014 §10: team shards are Parquet, the working database is
DuckDB, and the private team store is age-encrypted. All are binary, so
``check_data_class.py`` (JSON and JSONL only) and the PokePaste guard cannot
read them, and an encrypted file cannot be read by any guard. They are kept
out of git, so this guard rejects a tracked file that

- is named like one: ``.parquet`` (also ``.parquet.gz``, ``.zst``, ``.snappy``
  and ``.bz2``),
  ``.age``, ``.duckdb`` (also ``.duckdb.wal``), ``.ddb``, ``.arrow``,
  ``.feather``, ``.sqlite``, ``.sqlite3`` and ``.db``, each also with a
  ``-wal``, ``-shm`` or ``-journal`` suffix where the format has one (whatever the
  content); or
- is not valid UTF-8, or holds a NUL byte (a text file never does), so a
  binary renamed to ``notes.txt`` is caught; or
- starts with a magic number as a second check, for a small file that happens
  to be valid text: Parquet ``PAR1`` at offset 0, ``PARE`` (the encrypted
  footer) at offset 0 when the file also holds a NUL or non-UTF-8 byte (a CSV
  starting ``PARENT,CHILD`` is text; an encrypted Parquet file is binary), the
  SQLite header at offset 0, the age header ``age-encryption.org/v1`` at
  offset 0, DuckDB ``DUCK`` at offset 8; or
- holds the ASCII-armoured age header anywhere (armoured age is text, so no
  NUL byte or offset gives it away); or
- holds a base64 run (standard or URL-safe alphabet, wrapped over lines or
  not, at any 4-character boundary) whose decoded bytes start with one of
  those signatures: a binary pasted into a text file.

Non-goals, accepted evasions: a base64 run that does not start on a 4-character
boundary of its run, wrapping narrower than 8 characters per line, a run split
into quoted chunks or space-separated groups, and any other encoding (hex,
compression). A text file that legitimately starts with ``PAR1`` is refused;
the remedy is an exact entry in ``ALLOWED``.

``ALLOWED`` is an exact list of repo-relative paths that may be binary; it
starts empty, and an entry is a visible, code-owner-reviewed edit (the pinning
test fails until it is edited). Nothing else exempts a file: not a directory,
not a prefix, not ``fixtures/``.
"""

from __future__ import annotations

import base64
import binascii
import re
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ALLOWED: frozenset[str] = frozenset()
BINARY_SUFFIXES = (
    ".parquet",
    ".parquet.gz",
    ".parquet.zst",
    ".parquet.snappy",
    ".parquet.bz2",
    ".age",
    ".duckdb",
    ".duckdb.wal",
    ".ddb",
    ".arrow",
    ".feather",
    ".sqlite",
    ".sqlite3",
    ".sqlite-wal",
    ".sqlite-shm",
    ".sqlite3-wal",
    ".sqlite3-shm",
    ".db",
    ".db-wal",
    ".db-shm",
    ".db-journal",
    ".sqlite-journal",
    ".sqlite3-journal",
)
# (offset, bytes): the format's own file signature.
MAGIC = (
    (0, b"PAR1"),
    (0, b"SQLite format 3\0"),
    (0, b"age-encryption.org/v1"),
    (8, b"DUCK"),
)
# Parquet's encrypted footer: only binary content counts (a CSV may start PARENT).
ENCRYPTED_PARQUET = b"PARE"
# Split so that this file does not contain the header it forbids.
ARMOR_HEADER = b"-----BEGIN AGE " + b"ENCRYPTED FILE-----"
# A base64 run, and its continuation lines when it is wrapped.
_B64_RUN = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}(?:[ \t]*\r?\n[ \t]*[A-Za-z0-9+/_-]{8,}={0,2})*")
_B64_PREFIX = 32  # characters decoded per position: 24 bytes cover every signature


def _wraps_a_signature(text: str) -> bool:
    for run in _B64_RUN.findall(text):
        flat = "".join(run.split()).replace("-", "+").replace("_", "/")
        for start in range(0, len(flat) - 15, 4):
            chunk = flat[start : start + _B64_PREFIX]
            chunk = chunk[: len(chunk) // 4 * 4]
            try:
                decoded = base64.b64decode(chunk, validate=True)
            except binascii.Error:
                continue
            if _has_signature(decoded):
                return True
    return False


def _is_binary(data: bytes) -> bool:
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _has_signature(data: bytes) -> bool:
    if data.startswith(ENCRYPTED_PARQUET) and _is_binary(data):
        return True
    return any(data[offset : offset + len(magic)] == magic for offset, magic in MAGIC)


def reasons_for(path: str, data: bytes) -> list[str]:
    """Why one tracked file must not be committed (empty if it may)."""
    found: list[str] = []
    if path.lower().endswith(BINARY_SUFFIXES):
        found.append("is named like a binary data file")
    try:
        text: str | None = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
        found.append("is not valid UTF-8")
    if b"\0" in data:
        found.append("holds a NUL byte")
    if _has_signature(data):
        found.append("starts with a Parquet, SQLite, age or DuckDB signature")
    if ARMOR_HEADER in data:
        found.append("holds armored age text")
    if text is not None and _wraps_a_signature(text):
        found.append("holds a base64-wrapped Parquet, SQLite, age or DuckDB file")
    return found


def violations(paths: Sequence[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if PurePosixPath(Path(path).as_posix()).as_posix() in ALLOWED:
            continue
        try:
            data = Path(path).read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            # Deleted in the working tree, or a submodule: no file to read.
            # A binary *name* is still refused, since the name is the path.
            data = b""
            if not path.lower().endswith(BINARY_SUFFIXES):
                continue
        found.extend(f"{path}: {reason}" for reason in reasons_for(path, data))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    bad = violations(paths)
    for line in bad:
        print(
            f"{line}. Binary data (Parquet, SQLite, DuckDB, age) is not committed: publish it to "
            "object storage (ADR-013) or the encrypted private store (ADR-014); text files "
            "must be UTF-8 without NUL bytes.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
