"""Shared pieces of the inspector tests: the committed fixture run and helpers to edit it."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.inspector.model import InspectModel, build_model, read_run_dir_config
from areneon_ml.inspector.pool import Pool, read_pool_text
from areneon_ml.loop import layout

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "fixtures" / "loop" / "inspector-run"


def fixture_pool() -> Pool:
    return read_pool_text((FIXTURE / "pool.json").read_text(encoding="utf-8"))


def fixture_model() -> InspectModel:
    return build_model(FIXTURE, fixture_pool(), read_run_dir_config(FIXTURE))


def copy_run(tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(FIXTURE, target, ignore=shutil.ignore_patterns("pool.json", "README.md"))
    return target


def reseal(run: Path, generation: int) -> None:
    """Rewrite a generation's manifest to match its files: a consistent edit, not a corrupt one."""
    path = layout.generation_dir(run, generation)
    manifest = layout.read_json(path / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(path)
    layout.write_json(path / layout.MANIFEST, manifest)


def edit(run: Path, generation: int, name: str, change: Callable[[Any], Any]) -> None:
    """Change one canonical JSON file of a generation and reseal the manifest."""
    path = layout.generation_dir(run, generation) / name
    value = layout.read_json(path)
    result = change(value)
    layout.write_json(path, value if result is None else result)
    reseal(run, generation)


def edit_episode(
    run: Path, generation: int, index: int, change: Callable[[dict[str, Any]], None]
) -> None:
    """Change the ``index``-th Episode (by file order) and keep every hash in step: the Episode is
    renamed to its new hash, generation.json lists it, and the manifest is resealed."""
    gen = layout.generation_dir(run, generation)
    files = sorted((gen / "episodes").glob("*.json"))
    episode = json.loads(files[index].read_text(encoding="utf-8"))
    change(episode)
    data = jcs(episode)
    new = sha256_hex(data)
    files[index].unlink()
    (gen / "episodes" / f"{new}.json").write_bytes(data + b"\n")
    listing = layout.read_json(gen / "generation.json")
    listing["episode_hashes"] = sorted(
        [h for h in listing["episode_hashes"] if h != files[index].stem] + [new]
    )
    layout.write_json(gen / "generation.json", listing)
    reseal(run, generation)
