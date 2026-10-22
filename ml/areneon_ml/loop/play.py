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
    #: The play script, relative to the repository. A run that names a team dataset
    #: needs the script of ``packages/team-dataset``, which sees both the bridge and
    #: the dataset reader.
    play_script: str = "packages/showdown-bridge/scripts/play-generation.ts"
    #: How many policy worker processes serve a generation's games (#957). Operational only:
    #: it is not in the run config, and the generation's files do not depend on it.
    policy_workers: int = 1
    #: How many Node game processes share a generation's games (#957), each with its own
    #: ``policy_workers``. Operational too.
    game_processes: int = 1

    def _run(self, script: str, *args: str) -> str:
        command = [self.node, str(self.repo / script), *args]
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
        found = json.loads(
            self._run("packages/showdown-bridge/scripts/format-manifest.ts", "--format", format_id)
        )
        if not isinstance(found, dict):
            raise PlayError("format-manifest.ts did not print an object")
        return found

    def play(self, request: PlayRequest) -> None:
        self._run(
            self.play_script,
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
            # Absent at one worker, so the default command line is the one it always was.
            *(("--policy-workers", str(self.policy_workers)) if self.policy_workers != 1 else ()),
            *(("--game-processes", str(self.game_processes)) if self.game_processes != 1 else ()),
        )
