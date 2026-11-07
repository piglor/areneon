"""Shared pieces of the population inspector tests: the committed population fixture and edits."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.inspector.pool import Pool, read_pool_text
from areneon_ml.inspector.population_model import (
    PopulationModel,
    build_population_model,
    read_population_config,
)
from areneon_ml.loop import layout

from tests.ml.inspector.support import ROOT

POP_FIXTURE = ROOT / "fixtures" / "loop" / "population-run"


def pop_pool() -> Pool:
    return read_pool_text((POP_FIXTURE / "pool.json").read_text(encoding="utf-8"))


def pop_model() -> PopulationModel:
    return build_population_model(POP_FIXTURE, pop_pool(), read_population_config(POP_FIXTURE))


def copy_pop_run(tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(POP_FIXTURE, target, ignore=shutil.ignore_patterns("pool.json", "README.md"))
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


def gen_dir(run: Path, generation: int) -> Path:
    return layout.generation_dir(run, generation)


def rename_episode(
    run: Path, generation: int, member: str, index: int, change: Callable[[dict[str, Any]], None]
) -> None:
    """Change the ``index``-th Episode of a member (by file order), keep the hashes in step."""
    gen = gen_dir(run, generation)
    folder = gen / "members" / member / "episodes"
    files = sorted(folder.glob("*.json"))
    episode = json.loads(files[index].read_text(encoding="utf-8"))
    change(episode)
    data = jcs(episode)
    new = sha256_hex(data)
    old = files[index].stem
    files[index].unlink()
    (folder / f"{new}.json").write_bytes(data + b"\n")

    def fix(listing: dict[str, Any]) -> None:
        for entry in listing["members"]:
            if entry["member"] == member:
                entry["episode_hashes"] = sorted(
                    [h for h in entry["episode_hashes"] if h != old] + [new]
                )

    edit(run, generation, "generation.json", fix)


def set_ladder(run: Path, generation: int, change: Callable[[dict[str, Any]], None]) -> None:
    """Edit a generation's ladder.json and keep the manifest's ladder_hash in step."""
    edit(run, generation, "ladder.json", change)
    ladder = layout.read_json(gen_dir(run, generation) / "ladder.json")
    edit(
        run,
        generation,
        "manifest.json",
        lambda m: m.update(ladder_hash=sha256_hex(jcs(ladder))),
    )


def quarantine(
    run: Path, generation: int, member: str, *, held: bool, cause: str = "approx_kl"
) -> None:
    """Make the ladder quarantine ``member`` at ``generation`` as the loop writes it: a ``health``
    selection, a null sealed training set and, when ``held``, the records of the generation before
    byte for byte (a discarded update freezes the pair the member already had)."""
    set_ladder(
        run,
        generation,
        lambda ladder: ladder["selections"].append(
            {"cause": cause, "kind": "health", "member": member}
        ),
    )
    before = gen_dir(run, generation - 1)
    if held:
        for role in ("preview", "battle"):
            name = f"members/{member}/records/{role}.json"
            shutil.copyfile(before / name, gen_dir(run, generation) / name)
        sealed = layout.read_json(before / layout.MANIFEST)["members"][member]["checkpoints"]
    else:
        sealed = layout.read_json(gen_dir(run, generation) / layout.MANIFEST)["members"][member][
            "checkpoints"
        ]

    def seal(manifest: dict[str, Any]) -> None:
        manifest["members"][member].update(checkpoints=sealed, training_set_hash=None)

    edit(run, generation, "manifest.json", seal)
