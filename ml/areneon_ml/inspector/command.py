"""``python -m areneon_ml.loop inspect``: write the training-data inspector's static pages.

Read-only on the run directory and on the dataset: it takes no lock the loop needs, writes only to
``--out`` (default ``<run-dir>.inspect/``, beside the run directory and never inside it), and feeds
nothing back. The pool is rebuilt by the loop's own TypeScript function through Node.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
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
    check_target(target)
    config = read_run_dir_config(run_dir)
    pool = pool_loader(repo, node, run_dir / layout.CONFIG)
    pages = build_pages(build_model(run_dir, pool, config))
    total = write_pages(target, pages)
    return {"out": str(target), "pages": len(pages), "bytes": total}


def check_target(target: Path) -> None:
    """The output may be new or an earlier inspector output; any other directory is not touched."""
    if not target.exists():
        return
    if not target.is_dir() or any(
        not (path.is_file() and not path.is_symlink() and _OURS.fullmatch(path.name))
        for path in target.iterdir()
    ):
        raise InspectError(
            "the output exists and holds something other than inspector pages: "
            "give an empty or new directory"
        )


def write_pages(target: Path, pages: dict[str, str]) -> int:
    """Write the pages into a temporary sibling, flush them, and rename it over ``target``.

    A reader of ``target`` sees the old set or the new one, never a mixture, and a failed build
    leaves the old set. Returns the bytes written.
    """
    check_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent))
    backup = work.with_name(work.name + ".old")
    try:
        total = 0
        for name, text in sorted(pages.items()):
            data = text.encode("utf-8")
            layout.write_file(work / name, data)
            total += len(data)
        layout.sync_dir(work)
        if target.exists():
            os.replace(target, backup)
        os.replace(work, target)
        layout.sync_dir(target.parent)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return total
