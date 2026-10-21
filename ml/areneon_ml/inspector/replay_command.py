"""``python -m areneon_ml.loop replay``: write the training-game replay pages (ADR-017 4a).

Read-only on the run directory; writes only under ``<out>/replays/`` (``--out`` defaults to the
inspector's output directory, ``<run-dir>.inspect/``, beside the run and never inside it). Replay
pages are built per game or per generation, never for a whole run by default. Each page file is
written to a temporary name and renamed, and ``replays/index.html`` lists what exists.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_ml.inspector.command import default_out, existing_replays
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay import read_games
from areneon_ml.inspector.replay_page import (
    dex_for,
    index_page,
    page_path,
    read_engine_data,
    replay_page,
)
from areneon_ml.loop import layout

REPLAYS = "replays"


def _write(path: Path, text: str) -> int:
    """Write one file atomically (temporary name, flush, rename) and return its size."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise InspectError("a replay output path is not a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return len(data)


def parse_games(spec: str | None) -> list[int] | None:
    """``--game 3,5`` or ``--game 2-4``: game numbers; None is every played game."""
    if spec is None:
        return None
    games: list[int] = []
    for part in spec.split(","):
        found = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
        if found is None or (found[2] is not None and int(found[2]) < int(found[1])):
            raise InspectError(f"--game is a list of numbers or ranges like 3,5-7, not {spec!r}")
        games.extend(range(int(found[1]), int(found[2] or found[1]) + 1))
    return sorted(set(games))


@dataclass(frozen=True)
class Selection:
    """Which games: one generation's ``games`` (None: every played game of it)."""

    generation: int
    games: list[int] | None = None


def replay_games(
    run_dir: Path,
    out: Path | None,
    selection: Selection,
    *,
    repo: Path,
    engine_data: Path | None = None,
) -> dict[str, Any]:
    """Build the pages of ``games`` (None: every played game of the generation) and the index."""
    run_dir = run_dir.resolve()
    target = (out if out is not None else default_out(run_dir)).resolve()
    if target == run_dir or run_dir in target.parents:
        raise InspectError("the output directory must not be inside the run directory")
    config, replays = read_games(run_dir, selection.generation, selection.games)
    table = read_engine_data(
        engine_data or repo / "fixtures" / "showdown" / config.format_id / "engine-data.json",
        config.format_id,
    )
    pages: dict[str, str] = {}
    for game in replays:
        pages[page_path(game.generation, game.row.game)] = replay_page(game, dex_for(table, game))
    total = sum(_write(target / name, text) for name, text in sorted(pages.items()))
    layout.sync_dir(target)
    listing = existing_replays(target)
    total += _write(target / REPLAYS / "index.html", index_page(listing, config.run_key))
    return {"out": str(target), "pages": len(pages), "bytes": total, "replays": len(listing)}
