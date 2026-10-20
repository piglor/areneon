"""Checkpoints by id, from the run directory only (ADR-010 §2, Isolation).

A checkpoint id is the SHA-256 of its safetensors file's bytes. The worker
resolves an id to ``<run_dir>/checkpoints/<id>.safetensors`` and nothing
else: no path from a query or the command line reaches the file system, and
the bytes must hash to the id they were asked by.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_ID = re.compile(r"[0-9a-f]{64}")


class CheckpointError(ValueError):
    """A checkpoint id is malformed, missing, or does not match its bytes."""


class CheckpointStore:
    def __init__(self, run_dir: Path) -> None:
        self._dir = run_dir / "checkpoints"

    def load(self, checkpoint_id: str) -> bytes:
        if not _ID.fullmatch(checkpoint_id):
            raise CheckpointError("a checkpoint id is 64 lowercase hex digits")
        path = self._dir / f"{checkpoint_id}.safetensors"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CheckpointError(
                f"checkpoint {checkpoint_id} cannot be read from the run directory"
            ) from exc
        if hashlib.sha256(data).hexdigest() != checkpoint_id:
            raise CheckpointError(f"checkpoint {checkpoint_id}: its bytes hash to another id")
        return data
