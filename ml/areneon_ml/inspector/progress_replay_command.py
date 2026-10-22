"""``python -m areneon_ml.loop replay --progress-dir``: progress-game replay pages (ADR-017 4a).

Read-only on the progress directory (a committed generation is never touched: its manifest would
no longer match); writes only under ``<out>/progress-replays/``. ``--out`` defaults to the
inspector's output directory for the same run (``<run>.inspect/`` for ``<run>.progress/``), so a
page can link back to the inspector's index by a relative path when that exists, and never
inside the progress directory. Each page is written to a temporary name and renamed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from areneon_ml.inspector.command import PROGRESS_REPLAYS
from areneon_ml.inspector.progress_replay import read_progress_games
from areneon_ml.inspector.progress_replay_page import (
    GAME_PAGE,
    dex_for,
    index_page,
    page_path,
    replay_page,
)
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay import GEN_DIR
from areneon_ml.inspector.replay_command import _write
from areneon_ml.inspector.replay_page import read_engine_data
from areneon_ml.loop import layout
from areneon_ml.progress.looks import Look, append_look, looks_path
from areneon_ml.progress.records import SCHEMA_V2


def default_out(progress_dir: Path) -> Path:
    """``<run>.inspect`` for ``<run>.progress``, else ``<name>.inspect``, beside it."""
    stem = progress_dir.name.removesuffix(".progress") or progress_dir.name
    return progress_dir.with_name(stem + ".inspect")


def sibling_run(progress_dir: Path) -> Path | None:
    """The run directory a ``<run>.progress`` root sits beside, when that directory exists."""
    name = progress_dir.name.removesuffix(".progress")
    if not name or name == progress_dir.name:
        return None
    run = progress_dir.with_name(name)
    return run if run.is_dir() else None


def both_seat_generations(progress_dir: Path, numbers: Iterable[int]) -> set[int]:
    """The generations among ``numbers`` whose committed record is version 2 (both seats).

    Display only (how the index groups games): a record that cannot be read counts as version 1.
    """
    found: set[int] = set()
    for number in set(numbers):
        try:
            raw = layout.read_json(progress_dir / layout.generation_name(number) / "progress.json")
        except (layout.RunDirError, OSError, ValueError):
            continue
        if isinstance(raw, dict) and raw.get("schema_version") == SCHEMA_V2:
            found.add(number)
    return found


def existing_pages(out: Path) -> list[tuple[int, str, int]]:
    """The (generation, match, game) triples that have a page under ``out/progress-replays``."""
    found: list[tuple[int, str, int]] = []
    base = out / PROGRESS_REPLAYS
    if not base.is_dir() or base.is_symlink():
        return found
    for folder in base.iterdir():
        gen = GEN_DIR.fullmatch(folder.name)
        if gen is None or not folder.is_dir() or folder.is_symlink():
            continue
        for file in folder.iterdir():
            hit = GAME_PAGE.fullmatch(file.name)
            if hit is not None and file.is_file() and not file.is_symlink():
                found.append((int(gen[1]), hit[1], int(hit[2])))
    return sorted(found)


def replay_progress(  # noqa: PLR0913
    progress_dir: Path,
    generation: int,
    out: Path | None,
    *,
    repo: Path,
    match: str | None = None,
    games: list[int] | None = None,
    run_dir: Path | None = None,
    engine_data: Path | None = None,
) -> dict[str, Any]:
    """Build the pages of the selected stored games and the index of the pages that exist."""
    progress_dir = progress_dir.resolve()
    target = (out if out is not None else default_out(progress_dir)).resolve()
    if target == progress_dir or progress_dir in target.parents:
        raise InspectError("the output directory must not be inside the progress directory")
    for run in {r.resolve() for r in (run_dir, sibling_run(progress_dir)) if r is not None}:
        if target == run or run in target.parents:
            raise InspectError("the output directory must not be inside the run directory")
    selected = read_progress_games(progress_dir, generation, match, games, run_dir)
    # The stored games are read and consistent, so a validation result is about to be shown.
    shown = Counter((g.generation, g.match_id) for g in selected)
    append_look(
        looks_path(progress_dir, run_dir),
        "replay",
        selected[0].run_id,
        [Look(g, m, n) for (g, m), n in shown.items()],
    )
    manifest = selected[0].record["version_manifest"]
    table = read_engine_data(
        engine_data or repo / "fixtures" / "showdown" / manifest["format_id"] / "engine-data.json",
        manifest["format_id"],
    )
    inspector = (target / "index.html").is_file()
    pages = {
        page_path(g.generation, g.match_id, g.game): replay_page(
            g, dex_for(table, g), inspector=inspector
        )
        for g in selected
    }
    total = sum(_write(target / name, text) for name, text in sorted(pages.items()))
    layout.sync_dir(target)
    listing = existing_pages(target)
    total += _write(
        target / PROGRESS_REPLAYS / "index.html",
        index_page(
            listing,
            selected[0].run_id,
            inspector=inspector,
            both_seats=both_seat_generations(progress_dir, (g for g, _, _ in listing)),
        ),
    )
    return {"out": str(target), "pages": len(pages), "bytes": total, "replays": len(listing)}
