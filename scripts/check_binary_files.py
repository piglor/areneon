"""Reject committed binary data files.

ADR-013 §8: team shards are Parquet and the working database is DuckDB; both
are binary, so ``check_data_class.py`` (JSON and JSONL only) and the PokePaste
guard cannot read them, and an encrypted ``.age`` file cannot be read by any
guard. They are published to object storage and kept out of git, so this
guard rejects a tracked file that

- is named like one: ``.parquet``, ``.age``, ``.duckdb``, ``.duckdb.wal``,
  ``.arrow``, ``.feather``, ``.sqlite`` or ``.db`` (whatever its content); or
- is not valid UTF-8, or holds a NUL byte (a text file never does), so a
  binary renamed to ``notes.txt`` is caught; or
- starts with a magic number as a second check, for a small file that happens
  to be valid text: Parquet ``PAR1`` at offset 0, the age header
  ``age-encryption.org/v1`` (or its armored form) at offset 0, DuckDB
  ``DUCK`` at offset 8.

``ALLOWED`` is an exact list of repo-relative paths that may be binary; it
starts empty, and an entry is a visible, code-owner-reviewed edit (the pinning
test fails until it is edited). Nothing else exempts a file: not a directory,
not a prefix, not ``fixtures/``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ALLOWED: frozenset[str] = frozenset()
BINARY_SUFFIXES = (
    ".parquet",
    ".age",
    ".duckdb",
    ".duckdb.wal",
    ".arrow",
    ".feather",
    ".sqlite",
    ".db",
)
# (offset, bytes): the format's own file signature.
MAGIC = (
    (0, b"PAR1"),
    (0, b"age-encryption.org/v1"),
    (0, b"-----BEGIN AGE ENCRYPTED FILE-----"),
    (8, b"DUCK"),
)


def reasons_for(path: str, data: bytes) -> list[str]:
    """Why one tracked file must not be committed (empty if it may)."""
    found: list[str] = []
    if path.lower().endswith(BINARY_SUFFIXES):
        found.append("is named like a binary data file")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        found.append("is not valid UTF-8")
    if b"\0" in data:
        found.append("holds a NUL byte")
    if any(data[offset : offset + len(magic)] == magic for offset, magic in MAGIC):
        found.append("starts with a Parquet, age or DuckDB signature")
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
            f"{line}. Binary data (Parquet, DuckDB, age) is not committed: publish it to "
            "object storage (ADR-013); text files must be UTF-8 without NUL bytes.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
