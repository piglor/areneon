"""``inspect --resimulate G:K``: one engine re-simulation page (ADR-017 4a).

Read-only on the run directory; writes one page under ``<out>/resimulations/`` (``--out`` defaults
to the inspector's output directory, ``<run-dir>.inspect/``, beside the run and never inside it).
The page is written only after the re-simulation was accepted; a refusal writes nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from areneon_ml.inspector.command import default_out
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_command import _write
from areneon_ml.inspector.replay_page import read_engine_data
from areneon_ml.inspector.resimulate import Runner, resimulate
from areneon_ml.inspector.resimulate_page import dex_for, page_path, resimulation_page
from areneon_ml.loop import layout


def parse_target(spec: str) -> tuple[int, int]:
    """``--resimulate 3:5``: generation 3, game 5."""
    found = re.fullmatch(r"([0-9]+):([0-9]+)", spec)
    if found is None:
        raise InspectError(f"--resimulate is GENERATION:GAME, like 3:5, not {spec!r}")
    return int(found[1]), int(found[2])


def resimulate_game(  # noqa: PLR0913
    run_dir: Path,
    out: Path | None,
    target: tuple[int, int],
    *,
    repo: Path,
    runner: Runner,
    engine_data: Path | None = None,
) -> dict[str, Any]:
    """Re-run, check and write one played game's page; any difference refuses and writes nothing."""
    run_dir = run_dir.resolve()
    where = (out if out is not None else default_out(run_dir)).resolve()
    if where == run_dir or run_dir in where.parents:
        raise InspectError("the output directory must not be inside the run directory")
    generation, game = target
    done, format_id = resimulate(run_dir, generation, game, runner=runner)
    table = read_engine_data(
        engine_data or repo / "fixtures" / "showdown" / format_id / "engine-data.json", format_id
    )
    inspector = (where / "index.html").is_file()
    path = page_path(generation, game)
    size = _write(where / path, resimulation_page(done, dex_for(table, done), inspector=inspector))
    layout.sync_dir(where)
    return {"out": str(where), "page": path, "bytes": size}
