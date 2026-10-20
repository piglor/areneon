"""The run directory and its generation commits (ADR-010 §1).

A run directory holds the run config, the checkpoint store (``checkpoints/``,
shared with the worker) and ``generations/g<NNNNNN>/``, one directory per
completed generation. A generation is built in a ``.partial`` sibling and
becomes real by one ``os.replace`` of the directory, after every file in it is
written: the rename is the commit, so a killed run leaves either a complete
generation or a ``.partial`` that the next start deletes and replays from its
plan. Each committed generation carries ``manifest.json``, the SHA-256 of
every other file in it; resuming refuses a generation whose files differ.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import CanonicalError, jcs, sha256_hex

MANIFEST = "manifest.json"
CONFIG = "run-config.json"
_GENERATION = re.compile(r"g(\d{6})")


class RunDirError(ValueError):
    """A run directory is not in a state this build can continue."""


def generation_name(generation: int) -> str:
    return f"g{generation:06d}"


def generations_dir(run_dir: Path) -> Path:
    return run_dir / "generations"


def generation_dir(run_dir: Path, generation: int) -> Path:
    return generations_dir(run_dir) / generation_name(generation)


def partial_dir(run_dir: Path, generation: int) -> Path:
    return generations_dir(run_dir) / f".{generation_name(generation)}.partial"


def write_file(path: Path, data: bytes) -> None:
    """Write ``data`` and flush it to disk, so a later rename commits finished files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` so that ``path`` holds all of it or none: temp file, fsync, rename, fsync."""
    partial = path.with_name(f".{path.name}.partial")
    write_file(partial, data)
    os.replace(partial, path)
    sync_dir(path.parent)


def write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> bytes:
    data = jcs(value) + b"\n"
    write_file(path, data)
    return data


def read_json(path: Path) -> Any:
    """A canonical JSON file (JCS bytes and one newline), parsed; anything else is refused."""
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
        canonical = jcs(value)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, CanonicalError) as exc:
        raise RunDirError(f"{path.name} is not canonical JSON") from exc
    if raw != canonical + b"\n":
        raise RunDirError(f"{path.name} is not in RFC 8785 (JCS) form")
    return value


def sync_dir(path: Path) -> None:
    """Flush a directory's entries (a rename is durable only once its directory is)."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    """Flush every file and every directory under ``root`` (and ``root``) to disk.

    Files written by another process (the Node side) are made durable here, before their
    bytes are hashed into the manifest and before the one rename that commits them.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in [*sorted(p for p in root.rglob("*") if p.is_dir()), root]:
        sync_dir(path)


def hash_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root`` except the manifest, by relative path."""
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST):
        out[path.relative_to(root).as_posix()] = sha256_hex(path.read_bytes())
    return out


def commit_generation(run_dir: Path, generation: int, facts: Mapping[str, Any]) -> Path:
    """Write the manifest into the partial directory, then rename it into place.

    ``facts`` (checkpoint ids and hashes the driver recorded) go into the manifest
    beside the file hashes.
    """
    work = partial_dir(run_dir, generation)
    fsync_tree(work)
    manifest = {"schema_version": 1, "generation": generation, **facts, "files": hash_tree(work)}
    write_json(work / MANIFEST, manifest)
    sync_dir(work)
    target = generation_dir(run_dir, generation)
    if target.exists():
        raise RunDirError(f"generation {generation} is already committed")
    os.replace(work, target)
    sync_dir(generations_dir(run_dir))
    return target


@dataclass(frozen=True)
class Committed:
    """The newest complete generation of a run directory."""

    generation: int
    path: Path
    manifest: dict[str, Any]


def verify_generation(path: Path, generation: int) -> dict[str, Any]:
    """The manifest of a committed generation, once every file matches it."""
    manifest = read_json(path / MANIFEST)
    if not isinstance(manifest, dict) or manifest.get("generation") != generation:
        raise RunDirError(f"{path.name}: its manifest is not generation {generation}'s")
    if manifest.get("files") != hash_tree(path):
        raise RunDirError(f"{path.name}: its files differ from its manifest")
    return manifest


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Hold ``<run_dir>/.lock`` for the whole run: a second driver on the directory fails fast.

    Two drivers would race: the second would delete the first's live ``.partial`` generation.
    The lock is an advisory ``flock`` on an open file, so the operating system releases it when
    the holder dies, however it dies.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunDirError(
                f"another process is running in {run_dir}: its .lock is held"
            ) from exc
        yield


def discard_stale_checkpoint_files(run_dir: Path) -> list[str]:
    """Delete the temp files of checkpoint writes that a killed run left in ``checkpoints/``."""
    found = []
    store = run_dir / "checkpoints"
    if store.is_dir():
        for path in sorted(store.glob(".*.partial")):
            path.unlink()
            found.append(path.name)
    return found


def discard_partials(run_dir: Path) -> list[str]:
    """Delete every unfinished generation; they are replayed from the plan."""
    found = []
    base = generations_dir(run_dir)
    if base.is_dir():
        for path in sorted(base.glob(".g*.partial")):
            shutil.rmtree(path)
            found.append(path.name)
    return found


def latest_complete(run_dir: Path) -> Committed | None:
    """The newest committed generation, if any; the numbers must run 0..n with no gap.

    Only the newest generation's manifest is verified (every file against its hash): resuming
    needs the learner state, records and team set that live there, and an older generation
    is not read again by the loop. ``export_provenance`` verifies all of them.
    """
    base = generations_dir(run_dir)
    names = sorted(p.name for p in base.iterdir()) if base.is_dir() else []
    numbers = []
    for name in names:
        found = _GENERATION.fullmatch(name)
        if found is None or not (base / name).is_dir():
            if name.startswith(".") and name.endswith(".partial"):
                continue
            raise RunDirError(f"generations/{name} is not a generation of this run")
        numbers.append(int(found[1]))
    if not numbers:
        return None
    if numbers != list(range(len(numbers))):
        raise RunDirError(f"the committed generations are {numbers}, not 0..{len(numbers) - 1}")
    newest = numbers[-1]
    path = generation_dir(run_dir, newest)
    return Committed(newest, path, verify_generation(path, newest))
