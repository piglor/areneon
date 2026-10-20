"""A generation's A1 play (ADR-010 §1): a seam, and the Node process behind it.

``GenerationPlayer`` is what the driver needs of the simulator side: the version
manifest of the format, and one generation played into a directory (Episodes,
the loader's TrainingSet and stream, the plan and the per-game summary). The
real one runs ``packages/showdown-bridge/scripts/play-generation.ts``, which
starts the policy worker on CPU torch from the shipped closure and uses the
pinned engine; tests give the driver a fake one.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class PlayError(RuntimeError):
    """The simulator side failed; its message holds the end of its stderr."""


@dataclass(frozen=True)
class PlayRequest:
    generation: int
    policy_id: str
    preview: str
    battle: str
    out_dir: Path


class GenerationPlayer(Protocol):
    def manifest(self, format_id: str) -> dict[str, Any]: ...

    def play(self, request: PlayRequest) -> None: ...


_STDERR_TAIL = 2000


@dataclass(frozen=True)
class NodePlayer:
    """Runs the TS scripts of ``packages/showdown-bridge`` as child processes."""

    repo: Path
    node: str
    worker_python: str
    run_dir: Path
    config_path: Path
    engine_data: Path
    encoder_version: str

    def _run(self, script: str, *args: str) -> str:
        command = [self.node, str(self.repo / "packages/showdown-bridge/scripts" / script), *args]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL
            )
        except OSError as exc:
            raise PlayError(f"cannot start {self.node}: {exc}") from exc
        if done.returncode != 0:
            raise PlayError(
                f"{script} failed (exit {done.returncode}): {done.stderr[-_STDERR_TAIL:]}"
            )
        return done.stdout

    def manifest(self, format_id: str) -> dict[str, Any]:
        found = json.loads(self._run("format-manifest.ts", "--format", format_id))
        if not isinstance(found, dict):
            raise PlayError("format-manifest.ts did not print an object")
        return found

    def play(self, request: PlayRequest) -> None:
        self._run(
            "play-generation.ts",
            *("--config", str(self.config_path)),
            *("--generation", str(request.generation)),
            *("--policy-id", request.policy_id),
            *("--preview", request.preview),
            *("--battle", request.battle),
            *("--run-dir", str(self.run_dir)),
            *("--engine-data", str(self.engine_data)),
            *("--encoder-version", self.encoder_version),
            *("--worker-python", self.worker_python),
            *("--out", str(request.out_dir)),
        )
