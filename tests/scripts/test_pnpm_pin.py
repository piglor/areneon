"""pnpm is pinned by hash and installed only through corepack (#800)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
PINNED = re.compile(r"^pnpm@\d+\.\d+\.\d+\+sha512\.[0-9a-f]{128}$")
PNPM = re.compile(r"\bpnpm\b")
COREPACK = re.compile(r"\bcorepack enable\b")
# Installers that ignore the packageManager hash; setup-node's pnpm cache needs
# a pnpm on PATH before corepack can provide it.
BYPASSES = ("pnpm/action-setup", "npm i -g pnpm", "npm install -g pnpm", "cache: pnpm")
SKIPPED_DIRS = {"node_modules", "vendor", ".git"}

WORKFLOWS = sorted(p for pattern in ("*.yml", "*.yaml") for p in (ROOT / ".github").rglob(pattern))
DOCKERFILES = sorted(
    p
    for p in ROOT.rglob("Dockerfile*")
    if p.is_file()
    and not p.name.endswith(".dockerignore")
    and not SKIPPED_DIRS.intersection(p.relative_to(ROOT).parts)
)


def _code(text: str) -> str:
    """The text without its comment lines."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _corepack_first(code: str) -> bool:
    """True when the code uses no pnpm, or enables corepack before its first pnpm."""
    pnpm = PNPM.search(code)
    if pnpm is None:
        return True
    corepack = COREPACK.search(code)
    return corepack is not None and corepack.start() < pnpm.start()


def workflow_violations(text: str) -> list[str]:
    """Jobs that run pnpm without enabling corepack first, and hash bypasses."""
    found = [f"uses {b}" for b in BYPASSES if b in text]
    document: Any = yaml.safe_load(text)
    jobs = document.get("jobs") if isinstance(document, dict) else None
    for name, job in (jobs or {}).items():
        runs = [_code(str(step.get("run", ""))) for step in job.get("steps") or []]
        if not _corepack_first("\n".join(runs)):
            found.append(f"job {name} runs pnpm before corepack enable")
    return found


def dockerfile_violations(text: str) -> list[str]:
    """pnpm used before corepack is enabled after the package.json COPY."""
    code = _code(text)
    # A file named by COPY or ADD (pnpm-lock.yaml) is not a pnpm use: blank
    # those lines, keeping offsets, before looking for one.
    used = re.sub(r"^\s*(COPY|ADD)\b[^\n]*", lambda m: " " * len(m[0]), code, flags=re.M | re.I)
    pnpm = PNPM.search(used)
    if pnpm is None:
        return []
    copy = re.search(r"^\s*COPY\b[^\n]*\bpackage\.json\b", code, re.MULTILINE | re.IGNORECASE)
    corepack = COREPACK.search(code, copy.end()) if copy else None
    if corepack is None or corepack.start() > pnpm.start():
        return ["pnpm used before corepack enable follows the package.json COPY"]
    return []


def test_package_manager_names_pnpm_with_its_sha512() -> None:
    manifest = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert PINNED.fullmatch(manifest["packageManager"])


def test_scope_covers_the_known_install_paths() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in WORKFLOWS + DOCKERFILES}
    assert {
        ".github/workflows/ci.yml",
        ".github/workflows/rules-publish.yml",
        "packages/rules-importer/Dockerfile",
    } <= names


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: str(p.relative_to(ROOT)))
def test_workflow_enables_corepack_before_pnpm(path: Path) -> None:
    assert workflow_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_dockerfile_enables_corepack_before_pnpm(path: Path) -> None:
    assert dockerfile_violations(path.read_text(encoding="utf-8")) == []


GOOD_STEPS = """    steps:
      - uses: actions/setup-node@0 # v7
      # pnpm comes only from corepack
      - run: corepack enable && pnpm --version
      - run: pnpm install --frozen-lockfile
"""


def test_workflow_flags_only_the_job_without_corepack() -> None:
    text = f"jobs:\n  good:\n{GOOD_STEPS}  bad:\n    steps:\n      - run: pnpm lint\n"
    assert workflow_violations(text) == ["job bad runs pnpm before corepack enable"]


def test_workflow_flags_corepack_after_pnpm() -> None:
    text = """jobs:
  late:
    steps:
      - run: pnpm install
      - run: corepack enable
"""
    assert workflow_violations(text) == ["job late runs pnpm before corepack enable"]


def test_workflow_ignores_pnpm_in_comment_lines() -> None:
    text = """jobs:
  commented:
    steps:
      - run: |
          # pnpm is not used here
          echo ok
"""
    assert workflow_violations(text) == []


def test_workflow_flags_a_bypassing_installer() -> None:
    text = f"jobs:\n  setup:\n{GOOD_STEPS}      - uses: pnpm/action-setup@0\n"
    assert workflow_violations(text) == ["uses pnpm/action-setup"]


def test_workflow_check_reads_a_yaml_file(tmp_path: Path) -> None:
    workflow = tmp_path / "release.yaml"
    workflow.write_text("on: push\njobs:\n  j:\n    steps:\n      - run: pnpm -r test\n")
    assert workflow_violations(workflow.read_text()) == ["job j runs pnpm before corepack enable"]


def test_non_workflow_yaml_has_no_jobs() -> None:
    assert workflow_violations("version: 2\nupdates: []\n") == []


GOOD_DOCKERFILE = """FROM node AS build
COPY package.json pnpm-lock.yaml ./
RUN corepack enable \\
  && pnpm install --frozen-lockfile
"""


@pytest.mark.parametrize(
    ("text", "violations"),
    [
        (GOOD_DOCKERFILE, 0),
        ("FROM node\n# pnpm is not installed here\nRUN node -v\n", 0),
        ("FROM node\nCOPY package.json ./\nRUN pnpm install && corepack enable\n", 1),
        ("FROM node\nRUN corepack enable\nCOPY package.json ./\nRUN pnpm install\n", 1),
        ("FROM node\nRUN corepack enable && pnpm install\n", 1),
        ("FROM node\nCOPY package.json pnpm-lock.yaml ./\nRUN node -v\n", 0),
    ],
    ids=[
        "corepack-first",
        "no-pnpm",
        "pnpm-first",
        "corepack-before-copy",
        "no-copy",
        "lockfile-copy-only",
    ],
)
def test_dockerfile_order(text: str, violations: int) -> None:
    assert len(dockerfile_violations(text)) == violations
