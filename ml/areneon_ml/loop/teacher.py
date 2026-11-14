"""The teacher stream of ``ppo.imitation`` (ADR-010 erratum A11.3, #1367).

Nothing is fetched. ``import_teacher`` copies a teacher run's ``stream.jsonl`` and
``training-set.json`` (the output of ``teacher-run``) into ``<run dir>/teacher/<hash>/``, where
``<hash>`` is the SHA-256 of the manifest's JCS bytes (the run config's
``ppo.imitation.training_set_hash``). It checks the manifest and the stream against each other,
refuses a manifest the erratum refuses, refuses a hash other than the run config's (once the run
has a config), never replaces a file with other bytes and is a no-op when run again.
``teacher_paths`` is where the learner reads from.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from areneon_contracts.training import TrainingStreamError, check_stream_file

from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.imitation import check_teacher_manifest
from areneon_ml.loop import layout

TEACHER_DIR = "teacher"
STREAM = "stream.jsonl"
MANIFEST = "training-set.json"


class TeacherImportError(LearnerDataError):
    """The teacher stream cannot be imported into the run directory."""


def teacher_dir(run_dir: Path, training_set_hash: str) -> Path:
    return run_dir / TEACHER_DIR / training_set_hash


def teacher_paths(run_dir: Path, training_set_hash: str) -> tuple[Path, Path]:
    """The imported stream and its manifest; the learner refuses a run that lacks them."""
    where = teacher_dir(run_dir, training_set_hash)
    return where / STREAM, where / MANIFEST


def _configured_hash(run_dir: Path) -> str | None:
    """``ppo.imitation.training_set_hash`` of the run's config, or ``None`` before it exists."""
    path = run_dir / layout.CONFIG
    if not path.is_file():
        return None
    ppo = layout.read_json(path).get("ppo", {})
    found = ppo.get("imitation")
    if found is None:
        raise TeacherImportError("the run's config has no ppo.imitation: nothing to import for")
    return str(found["training_set_hash"])


def _place(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` durably; an existing target must hold the same bytes."""
    if target.exists():
        if layout.sha256_file(target) != layout.sha256_file(source):
            raise TeacherImportError(f"{target.name}: the run already holds other bytes")
        return
    partial = target.with_name(f".{target.name}.partial")
    shutil.copyfile(source, partial)
    descriptor = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, target)
    layout.sync_dir(target.parent)


def import_teacher(stream_dir: Path, run_dir: Path) -> dict[str, Any]:
    """Copy ``stream_dir``'s stream and manifest into the run directory; return the hash."""
    try:
        manifest = layout.read_json(stream_dir / MANIFEST)
        stream = stream_dir / STREAM
        if not stream.is_file():
            raise TeacherImportError(f"{STREAM}: not found in the teacher directory")
        expected = _configured_hash(run_dir)
        found = check_teacher_manifest(manifest, expected)
        check_stream_file(stream, manifest, False)
    except TrainingStreamError as exc:
        raise TeacherImportError(str(exc)) from exc
    except layout.RunDirError as exc:
        raise TeacherImportError(str(exc)) from exc
    with layout.run_lock(run_dir):
        layout.ensure_run_dirs(run_dir)
        where = teacher_dir(run_dir, found)
        where.mkdir(parents=True, exist_ok=True)
        _place(stream, where / STREAM)
        _place(stream_dir / MANIFEST, where / MANIFEST)
        layout.sync_dir(where)
    return {"training_set_hash": found}
