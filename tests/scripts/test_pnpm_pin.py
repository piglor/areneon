"""pnpm is pinned by hash and installed only through corepack (#800)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
DOCKERFILES = sorted(ROOT.glob("packages/*/Dockerfile"))
PINNED = re.compile(r"^pnpm@\d+\.\d+\.\d+\+sha512\.[0-9a-f]{128}$")


def test_package_manager_names_pnpm_with_its_sha512() -> None:
    manifest = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert PINNED.fullmatch(manifest["packageManager"])


@pytest.mark.parametrize(
    "path",
    WORKFLOWS + DOCKERFILES,
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_pnpm_is_installed_only_through_corepack(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    # Installers that ignore the packageManager hash; setup-node's pnpm cache
    # needs a pnpm on PATH before corepack can provide it.
    for bypass in ("pnpm/action-setup", "npm i -g pnpm", "npm install -g pnpm", "cache: pnpm"):
        assert bypass not in text
    if "pnpm install" in text:
        assert "corepack enable" in text
