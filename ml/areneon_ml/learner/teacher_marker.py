"""A verified marker beside the teacher stream (ADR-010 erratum A11.3, #1501).

``open_teacher`` checks the teacher TrainingSet's stream against its manifest (the digest, the
line count and every line against the contracts) and encodes every example once, so a stream the
run's encoder refuses fails at the start. That is minutes for a multi-gigabyte stream, and a queue
of experiments starts a new driver process for every generation, so it was paid at every
generation for a file that never changed.

After a full pass that succeeded, ``open_teacher`` writes a marker next to the stream
(``verified-<hash of the key>.json``): the key, the verdict's inputs, and the byte offsets of the
Battle examples that the pass computed. A later process whose key is the same skips the line
checks and the encode pass; it still reads the whole stream once for its SHA-256 (about two
seconds for 2.6 GB) and compares it with the manifest's ``stream_sha256``, so the file it trusts is
byte for byte the file that was checked, whatever its size, mtime or inode. The marker is a cache
of a verdict and never an input: nothing a run writes depends on whether it was honoured.

The key is everything the verdict is a function of: the manifest (its hash, which holds the
stream's digest), the stream's size, each role's encoder version and engine-data hash, the source
of the code that validates and encodes (``code_identity``) and the versions of the libraries it
runs on. Any other marker (another key, damaged, truncated, of another format version) is not
used; the caller runs the full pass, whose refusals and messages are the pass's own, and writes
the marker again. A marker is written only after a pass that returned, atomically (a temporary
file, fsync, rename, fsync of the directory), so a process killed in the pass leaves none and two
processes writing at once leave one whole file (their bytes are equal).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import areneon_contracts
from areneon_contracts.contracts import CONTRACTS, jcs, sha256_hex

import areneon_ml

#: The marker's format; another value is a marker this code does not know, so it is not used.
FORMAT = "teacher-marker/1"

#: A marker's file name is ``PREFIX`` + the SHA-256 of its key (hex) + ``SUFFIX``.
PREFIX = "verified-"
SUFFIX = ".json"

#: The largest marker read: the real marker of a 66,590-example stream is 705 KB (an offset takes
#: at most 16 bytes), so this is two orders of magnitude beyond any stream this loop reads.
MAX_MARKER_BYTES = 64 * 1024 * 1024

#: A temporary file of ours older than this is the remains of a killed writer (a writer holds one
#: for milliseconds), and the next writer removes it.
STALE_PARTIAL_SECONDS = 24 * 3600

#: The libraries (distribution names) whose behaviour the validation runs on: the contract check
#: (jsonschema and what it resolves references with), the models (pydantic), the hash (rfc8785)
#: and the tensors. A library that is not installed is named with no version.
LIBRARIES = (
    "attrs",
    "jsonschema",
    "jsonschema-specifications",
    "pydantic",
    "pydantic-core",
    "referencing",
    "rfc8785",
    "rpds-py",
    "torch",
)

#: A root of the code identity: a directory and the file patterns under it that count.
Root = tuple[Path, tuple[str, ...]]


def default_roots() -> tuple[Root, ...]:
    """Every file the validation and the encode read: both packages' Python source and package
    data (the reducer's engine tables) and the contract schemas the stream is checked against."""
    return (
        (Path(areneon_ml.__file__).resolve().parent, ("*.py", "*.json")),
        (Path(areneon_contracts.__file__).resolve().parent, ("*.py",)),
        (CONTRACTS, ("*.json",)),
    )


def _entries(roots: Sequence[Root]) -> list[tuple[str, Path]]:
    """Every source file of the roots as (name relative to its root, path), sorted by name."""
    found = {
        f"{root.name}/{path.relative_to(root).as_posix()}": path
        for root, patterns in roots
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(found.items())


def code_files(roots: Sequence[Root] | None = None) -> list[Path]:
    """The files ``code_identity`` hashes, in the order it hashes them."""
    return [path for _, path in _entries(roots if roots is not None else default_roots())]


def _identity(roots: Sequence[Root]) -> str:
    digest = hashlib.sha256()
    for name, path in _entries(roots):
        digest.update(f"{name}\0{sha256_hex(path.read_bytes())}\n".encode())
    return digest.hexdigest()


#: The identity of the code as it was when this module was imported, which is when the process
#: started (``imitation`` imports it). The pass that a marker records runs minutes later, so
#: reading the files at the first use would name whatever the tree holds by then: a worktree whose
#: sources were edited or switched while the process loaded the stream would be recorded with the
#: new code's identity for a verdict the old code gave, and a later process on the new code would
#: skip a validation it never did. Fixed at import, the key names the code the process began with.
#: Residual, stated: the hash is of the files, not of what is in memory. A first-party module that
#: is imported lazily after this point (the generated pydantic models are loaded by name on first
#: use) is read from the tree as it is then, so an edit of such a file during the pass is not
#: caught; the test that every module ``imitation`` imports is in the identity keeps that set to
#: the lazy ones, and the window is the one of a source tree edited under a running process.
_IMPORTED_IDENTITY = _identity(default_roots())


def code_identity(roots: Sequence[Root] | None = None) -> str:
    """SHA-256 over the relative path and the content hash of every file of the roots.

    Under the default roots it is the identity of the code that validates and encodes the stream,
    so a change to any of it, a moved or a new file included, is another key. The paths are
    relative to the roots, so the same tree gives the same identity wherever it is checked out.
    The default is the identity at import (``_IMPORTED_IDENTITY``), not at the call.
    """
    return _identity(roots) if roots is not None else _IMPORTED_IDENTITY


def library_versions() -> dict[str, str]:
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = ""
    return out


def marker_key(
    *,
    training_set_hash: str,
    stream_sha256: str,
    stream_bytes: int,
    encoders: Mapping[str, Any],
) -> dict[str, Any]:
    """The inputs of the verdict: change any of them and the marker is not the same marker.

    ``encoders`` are the run's encoders by role (``version`` and ``game.engine_data_hash`` are
    read; the version already holds the engine-data hash, which is repeated so that a table change
    is named on its own).
    """
    return {
        "code_sha256": code_identity(),
        "encoders": {
            role: {
                "engine_data_hash": encoder.game.engine_data_hash,
                "version": encoder.version,
            }
            for role, encoder in sorted(encoders.items())
        },
        "libraries": library_versions(),
        "stream_bytes": stream_bytes,
        "stream_sha256": stream_sha256,
        "training_set_hash": training_set_hash,
    }


def marker_path(directory: Path, key: Mapping[str, Any]) -> Path:
    return directory / f"{PREFIX}{sha256_hex(jcs(dict(key)))}{SUFFIX}"


def _sealed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "sha256": sha256_hex(jcs(body))}


def _offsets_fit(offsets: object, stream_bytes: int) -> bool:
    """Whether ``offsets`` can be the start of lines of a stream of ``stream_bytes`` bytes: whole
    numbers, strictly ascending, inside the file."""
    if not isinstance(offsets, list):
        return False
    previous = -1
    for offset in offsets:
        if type(offset) is not int or not previous < offset < stream_bytes:
            return False
        previous = offset
    return True


def read_marker(directory: Path, key: Mapping[str, Any]) -> list[int] | None:
    """The Battle offsets of the marker for ``key`` in ``directory``, or ``None`` when there is no
    marker that this code can trust to be the pass's own record (absent, unreadable, damaged, of
    another format or another key). It reads the marker alone: the stream's own check is the
    caller's."""
    path = marker_path(directory, key)
    try:
        if path.stat().st_size > MAX_MARKER_BYTES:
            return None
        found = json.loads(path.read_bytes())
        if not isinstance(found, dict) or set(found) != {"format", "key", "offsets", "sha256"}:
            return None
        body = {name: found[name] for name in ("format", "key", "offsets")}
        if (
            body["format"] != FORMAT
            or body["key"] != dict(key)
            or found["sha256"] != sha256_hex(jcs(body))
            or not _offsets_fit(body["offsets"], int(key["stream_bytes"]))
        ):
            return None
    except (OSError, ValueError, RecursionError):
        return None
    offsets: list[int] = body["offsets"]
    return offsets


def _prune_partials(directory: Path) -> None:
    """Remove the temporary files that killed writers left (older than ``STALE_PARTIAL_SECONDS``);
    a writer at work keeps its own, and nothing outside our prefix is touched."""
    limit = time.time() - STALE_PARTIAL_SECONDS
    with contextlib.suppress(OSError):
        for path in directory.glob(f".{PREFIX}*.partial"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < limit:
                    os.unlink(path)


def write_marker(directory: Path, key: Mapping[str, Any], offsets: Sequence[int]) -> bool:
    """Write the marker for ``key`` atomically; ``False`` (nothing written) where the directory
    cannot take it, which never fails a run: the marker only saves time."""
    _prune_partials(directory)
    data = jcs(_sealed({"format": FORMAT, "key": dict(key), "offsets": list(offsets)})) + b"\n"
    target = marker_path(directory, key)
    partial: str | None = None
    try:
        descriptor, partial = tempfile.mkstemp(
            dir=directory, prefix=f".{PREFIX}", suffix=".partial"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
        partial = None
        flush = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(flush)
        finally:
            os.close(flush)
    except OSError:
        if partial is not None:
            with contextlib.suppress(OSError):
                os.unlink(partial)
        return False
    return True
