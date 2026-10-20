"""Keep the CUDA runtime out of everything Areneon ships (ADR-010 §5.2, C-1 to C-3).

GPU training runs in a separate uv project (``GPU_ENV``, with its own
``pyproject.toml`` and lock) that the operator installs locally. The rules, all
checked from the repository root:

* C-1: the root uv workspace does not list the GPU environment as a member, and
  no shipped package is named like it.
* C-2: the root ``uv.lock`` and the closure of every shipped package hold no
  ``nvidia-*``, ``cuda-toolkit``, ``cuda-bindings`` or ``triton`` package.
* C-3: no Dockerfile refers to the GPU environment.

No CUDA package is ever added to the licence allowlist; this guard is what keeps
the checked closure free of them. Run as ``uv run --locked python -m
scripts.check_cuda_separation``; it prints each violation and exits 1.
"""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
import sys
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

from scripts.python_licences import SHIPPED, ClosureError, closure, normalise

# Convention: the GPU training project lives here, beside the shipped ``ml`` package.
GPU_ENV = "ml/gpu-env"
GPU_PROJECT_NAME = "areneon-gpu-env"
FORBIDDEN_EXACT = frozenset({"cuda-toolkit", "cuda-bindings", "triton"})
FORBIDDEN_PREFIX = "nvidia-"
_SKIP_DIRS = frozenset({".git", ".venv", "node_modules", "__pycache__"})
_SKIP_ROOT_DIRS = frozenset({"vendor"})  # the pinned engine checkout, at the repo root only
_COMPOSE = re.compile(r"(docker-)?compose(\..+)?\.ya?ml", re.IGNORECASE)


def is_cuda_package(name: str) -> bool:
    normalised = normalise(name)
    return normalised in FORBIDDEN_EXACT or normalised.startswith(FORBIDDEN_PREFIX)


def lock_violations(lock_text: str) -> list[str]:
    """C-2 on a ``uv.lock``: every package in it that is a CUDA package."""
    packages = tomllib.loads(lock_text).get("package", [])
    return sorted({f"uv.lock contains {p['name']}" for p in packages if is_cuda_package(p["name"])})


def closure_violations(
    run: Callable[[list[str]], str] | None = None, shipped: tuple[str, ...] = SHIPPED
) -> list[str]:
    """C-2 on the exported closure of every shipped package."""
    found: list[str] = []
    for package in shipped:
        found += [
            f"the closure of {package} contains {name}"
            for name in closure(package, run)
            if is_cuda_package(name)
        ]
    return sorted(found)


def _covers(member: str, path: str) -> bool:
    cleaned = posixpath.normpath(member.strip())
    return fnmatch.fnmatchcase(path, cleaned)


def workspace_violations(pyproject_text: str, shipped: tuple[str, ...] = SHIPPED) -> list[str]:
    """C-1: neither the workspace members nor SHIPPED name the GPU environment.

    A member is judged after path normalisation (``./ml/gpu-env/``), and the
    environment counts as excluded (defence in depth) only if the workspace's
    ``exclude`` covers it; a member that covers it is a violation either way.
    """
    workspace = tomllib.loads(pyproject_text).get("tool", {}).get("uv", {}).get("workspace", {})
    excluded = any(_covers(e, GPU_ENV) for e in workspace.get("exclude", []))
    found = [
        f"workspace member {member!r} covers {GPU_ENV}"
        for member in workspace.get("members", [])
        if _covers(member, GPU_ENV) and not excluded
    ]
    found += [
        f"SHIPPED names the GPU environment: {name}"
        for name in shipped
        if normalise(name) == GPU_PROJECT_NAME
    ]
    return found


def _is_image_recipe(file: str) -> bool:
    name = file.lower()
    return (
        name in {"dockerfile", "containerfile"}
        or name.startswith(("dockerfile.", "containerfile."))
        or name.endswith((".dockerfile", ".containerfile"))
        or bool(_COMPOSE.fullmatch(file))
    )


def dockerfiles(root: Path) -> Iterator[Path]:
    """Every Dockerfile, Containerfile and compose file under ``root``."""
    for directory, subdirs, files in os.walk(root):
        skip = _SKIP_DIRS | (_SKIP_ROOT_DIRS if Path(directory) == root else frozenset())
        subdirs[:] = [d for d in subdirs if d not in skip]
        yield from (Path(directory) / f for f in files if _is_image_recipe(f))


def _squash(text: str) -> str:
    return re.sub(r"[-_]", "", text.lower())


def dockerfile_violations(root: Path) -> list[str]:
    """C-3: no image recipe mentions the GPU environment's path or project name.

    This is a name-based check (case-insensitive, ``-`` and ``_`` ignored). That
    an image installs from the root lock only is enforced by the build test of
    the first ``ml/`` image, not here.
    """
    needles = (_squash("gpu-env"), _squash(GPU_PROJECT_NAME))
    found = []
    for path in dockerfiles(root):
        text = _squash(path.read_text(encoding="utf-8", errors="replace"))
        if any(needle in text for needle in needles):
            found.append(f"{path.relative_to(root)} refers to the GPU environment ({GPU_ENV})")
    return sorted(found)


def violations(root: Path, run: Callable[[list[str]], str] | None = None) -> list[str]:
    return [
        *workspace_violations((root / "pyproject.toml").read_text(encoding="utf-8")),
        *lock_violations((root / "uv.lock").read_text(encoding="utf-8")),
        *closure_violations(run),
        *dockerfile_violations(root),
    ]


def main(root: Path | None = None, run: Callable[[list[str]], str] | None = None) -> int:
    try:
        bad = violations(root or Path.cwd(), run)
    except (OSError, ClosureError, tomllib.TOMLDecodeError, KeyError) as exc:
        print(f"CUDA separation guard failed closed: {exc!r}", file=sys.stderr)
        return 2
    for entry in bad:
        print(f"CUDA separation violated: {entry}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
