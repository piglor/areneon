"""``python -m areneon_ml.loop report``: write the run dashboard, one offline HTML page.

Read-only on the run directory, the progress root and the inspector output; the one write is the
page (``--out`` is a ``.html`` file, or a directory that then gets ``index.html``). The page is
written to a temporary sibling and renamed, so a reader sees the old page or the new one.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from areneon_ml.loop import layout
from areneon_ml.report.model import ReportError, read_report_model
from areneon_ml.report.render import Links, build_report

_NAME = re.compile(r"[a-z0-9_-]+")
_REPLAY_LIMIT = 40


def default_out(run_dir: Path) -> Path:
    return run_dir.resolve().with_name(run_dir.resolve().name + ".report.html")


_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~")


def _quote(part: str) -> str:
    """Percent-encode one path segment (everything but the unreserved characters)."""
    return "".join(chr(b) if b in _SAFE else f"%{b:02X}" for b in part.encode("utf-8"))


def _href(target: Path, base: Path) -> str:
    return "/".join(_quote(part) for part in Path(os.path.relpath(target, base)).parts)


def inspector_links(inspect_dir: Path | None, base: Path, generations: list[int]) -> Links:
    """Links to the games and replay pages that exist in ``inspect_dir``, relative to ``base``."""
    if inspect_dir is None:
        return Links()
    inspect_dir = inspect_dir.resolve()
    if not inspect_dir.is_dir():
        raise ReportError("the inspector directory does not exist")
    games: dict[int, str] = {}
    replays: dict[int, tuple[tuple[str, str], ...]] = {}
    for g in generations:
        name = f"games-{layout.generation_name(g)}.html"
        page = inspect_dir / name
        if page.is_file() and not page.is_symlink():
            games[g] = _href(page, base)
        replays_dir = inspect_dir / "replays"
        folder = replays_dir / layout.generation_name(g)
        if not replays_dir.is_symlink() and folder.is_dir() and not folder.is_symlink():
            found = sorted(
                p
                for p in folder.iterdir()
                if p.is_file()
                and not p.is_symlink()
                and p.suffix == ".html"
                and _NAME.fullmatch(p.stem)
            )
            if found:
                replays[g] = tuple(
                    (f"replay {p.stem}", _href(p, base)) for p in found[:_REPLAY_LIMIT]
                )
    return Links(games, replays)


def report_run(
    run_dir: Path, progress_dir: Path | None, out: Path | None, inspect_dir: Path | None = None
) -> dict[str, Any]:
    """Build and write the page; refuses on any inconsistency before writing anything."""
    run_dir = run_dir.resolve()
    target = (out if out is not None else default_out(run_dir)).resolve()
    if target.is_dir() or (out is not None and out.suffix != ".html"):
        target = target / "index.html"
    if run_dir in target.parents or (
        progress_dir is not None and progress_dir.resolve() in target.parents
    ):
        raise ReportError("the report must not be written inside the run or progress directory")
    model = read_report_model(run_dir, progress_dir.resolve() if progress_dir else None)
    links = inspector_links(inspect_dir, target.parent, [g.number for g in model.generations])
    page = build_report(model, links).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(page)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return {
        "out": str(target),
        "bytes": len(page),
        "generations": len(model.generations),
        "progress_generations": len(model.progress),
    }
