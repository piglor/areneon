"""A generation's A1 play (ADR-010 §1): a seam, and the Node process behind it.

``GenerationPlayer`` is what the driver needs of the simulator side: the version
manifest of the format, and one generation played into a directory (Episodes,
the loader's TrainingSet and stream, the plan and the per-game summary). The
real one runs ``packages/showdown-bridge/scripts/play-generation.ts``, which
starts the policy worker on CPU torch from the shipped closure and uses the
pinned engine; tests give the driver a fake one.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from dataclasses import dataclass, field
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
    #: Caps each Node game process's old-space heap (``--max-old-space-size``, in MB), which
    #: makes V8 collect garbage before the process grows: a game process that loads a 20,000
    #: team pool peaks at about 860 MB without it and about 590 MB with 640 (#1069). Operational:
    #: the generation's files do not depend on it. ``None`` leaves Node's default.
    node_heap_mb: int | None = None
    #: The wall-clock limit of one ``play`` call, in seconds (#1100). Past it the driver kills
    #: the Node process group (the game processes and the policy workers with it) and the
    #: generation fails with a ``PlayError``: nothing is committed. Operational: the
    #: generation's files do not depend on it. ``None`` sets no limit.
    generation_timeout_s: float | None = None
    #: The peak resident set (bytes) of the largest game process and of the largest policy
    #: worker of the last ``play`` call, as the play script reported them; empty when it did not.
    #: Diagnostics for the run-metrics record, never an input to a generation.
    resource_usage: dict[str, int] = field(default_factory=dict, compare=False)

    def _run(
        self,
        script: str,
        *args: str,
        heap_mb: int | None = None,
        timeout_s: float | None = None,
    ) -> str:
        cap = () if heap_mb is None else (f"--max-old-space-size={heap_mb}",)
        command = [self.node, *cap, str(self.repo / script), *args]
        try:
            # Same process group as the driver, so killing the driver's group still ends it.
            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise PlayError(f"cannot start {self.node}: {exc}") from exc
        try:
            stdout, stderr = child.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(child)
            try:
                _, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # a descendant outside the group holds a pipe
                stderr = ""
            raise PlayError(
                f"{script} exceeded the generation time limit of {timeout_s:g} s and was killed "
                f"(--generation-timeout-minutes): {stderr[-_STDERR_TAIL:]}"
            ) from None
        except BaseException:
            _kill_group(child)
            child.wait()
            raise
        if child.returncode != 0:
            raise PlayError(f"{script} failed (exit {child.returncode}): {stderr[-_STDERR_TAIL:]}")
        return stdout

    def manifest(self, format_id: str) -> dict[str, Any]:
        found = json.loads(
            self._run("packages/showdown-bridge/scripts/format-manifest.ts", "--format", format_id)
        )
        if not isinstance(found, dict):
            raise PlayError("format-manifest.ts did not print an object")
        return found

    def play(self, request: PlayRequest) -> None:
        self.resource_usage.clear()
        out = self._run(
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
            heap_mb=self.node_heap_mb,
            timeout_s=self.generation_timeout_s,
        )
        self.resource_usage.update(_resource_usage(out))


def _descendants(pid: int) -> list[int]:
    """The pids below ``pid`` (Linux ``/proc``; empty where it cannot be read)."""
    parents: dict[int, int] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    for name in names:
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text()
            parents[int(name)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    found: list[int] = []
    frontier = [pid]
    while frontier:
        below = [p for p, parent in parents.items() if parent in frontier]
        found.extend(below)
        frontier = below
    return found


def _kill_group(child: subprocess.Popen[str]) -> None:
    """SIGKILL the child and everything below it (game processes and policy workers)."""
    for pid in [*_descendants(child.pid), child.pid]:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


_USAGE_KEYS = {
    "game_process_peak_rss_bytes": "game_process",
    "policy_worker_peak_rss_bytes": "policy_worker",
}


def _resource_usage(stdout: str) -> dict[str, int]:
    """The ``resource_usage`` line of the play script's output; anything odd is ignored."""
    found: dict[str, int] = {}
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        usage = value.get("resource_usage") if isinstance(value, dict) else None
        if not isinstance(usage, dict):
            continue
        for key, name in _USAGE_KEYS.items():
            number = usage.get(key)
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                found[name] = number
    return found
