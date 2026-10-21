"""``python -m areneon_ml.loop inspect``: write the training-data inspector's static pages.

Read-only on the run directory and on the dataset: it takes no lock the loop needs, writes only to
``--out`` (default ``<run-dir>.inspect/``, beside the run directory and never inside it), and feeds
nothing back. The pool is rebuilt by the loop's own TypeScript function through Node.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areneon_ml.inspector.model import build_model, read_run_dir_config
from areneon_ml.inspector.pool import Pool, load_pool
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.render import build_pages
from areneon_ml.loop import layout

_OURS = re.compile(r"(index|distributions|games-g\d{6}|drawn-g\d{6}|teams-\d{4})\.html")

PoolLoader = Callable[[Path, str, Path], Pool]


def default_out(run_dir: Path) -> Path:
    return run_dir.with_name(run_dir.name + ".inspect")


def inspect_run(
    run_dir: Path,
    out: Path | None,
    *,
    repo: Path,
    node: str,
    pool_loader: PoolLoader = load_pool,
) -> dict[str, Any]:
    """Build and write the pages; refuses on any mismatch before writing anything."""
    run_dir = run_dir.resolve()
    target = (out if out is not None else default_out(run_dir)).resolve()
    if target == run_dir or run_dir in target.parents:
        raise InspectError("the output directory must not be inside the run directory")
    config = read_run_dir_config(run_dir)
    pool = pool_loader(repo, node, run_dir / layout.CONFIG)
    pages = build_pages(build_model(run_dir, pool, config))
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.iterdir():
        if stale.is_file() and _OURS.fullmatch(stale.name) and stale.name not in pages:
            stale.unlink()
    total = 0
    for name, text in sorted(pages.items()):
        data = text.encode("utf-8")
        (target / name).write_bytes(data)
        total += len(data)
    return {"out": str(target), "pages": len(pages), "bytes": total}
