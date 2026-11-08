"""The situational move-use diagnostic run by ``progress --situational`` (#1286, after #1283).

Opt-in: after the evaluation, run ``situational-move-use.ts`` (engine re-simulation of every
stored game, minutes) at ``nice -n 19`` on the committed generation, summarise the raw counts
with :mod:`areneon_ml.report.situational` and write the summary to
``<progress root>.situational/g<NNNNNN>.json``, a sibling of the root (outside its manifest and
every hash, like the look log). The printed result carries the Protect and switch rates in ppm.
A rerun with the same counts leaves the file's bytes as they were.

LABEL: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength
measure.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs

from areneon_ml.report.situational import LABEL, SituationalError, summarise

SUFFIX = ".situational"
SCRIPT = "packages/showdown-bridge/scripts/situational-move-use.ts"
#: The statistics the printed result carries (the Protect and switch rates of #1283).
PRINTED = (
    "protect_threatened",
    "protect_safe",
    "protect_threatened_minus_safe",
    "repeat_protect",
    "switch_bad_matchup",
    "switch_not_threatened",
    "switch_bad_matchup_minus_not_threatened",
)

Runner = Callable[[Sequence[str]], int]


def summary_dir(root: Path) -> Path:
    root = root.resolve()
    return root.with_name(root.name + SUFFIX)


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def situational_generation(  # noqa: PLR0913
    root: Path,
    generation: int,
    run_dir: Path,
    *,
    repo: Path,
    node: str,
    max_games: int | None = None,
    run: Runner = _run,
) -> dict[str, Any]:
    """Run the Node script on one committed generation; write and return the printed block."""
    out_dir = summary_dir(root)
    if out_dir == run_dir.resolve() or run_dir.resolve() in out_dir.parents:
        raise SituationalError("the situational summary must not be inside the run directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f".g{generation:06d}.raw.json"
    command = [
        "nice",
        "-n",
        "19",
        node,
        str(repo / SCRIPT),
        "--progress-dir",
        str(root),
        "--generation",
        str(generation),
        "--run-dir",
        str(run_dir),
        "--out",
        str(raw_path),
    ]
    if max_games is not None:
        command += ["--max-games", str(max_games)]
    try:
        if run(command) != 0:
            raise SituationalError("the situational move-use script failed")
        raw = json.loads(raw_path.read_text(encoding="utf8"))
    finally:
        raw_path.unlink(missing_ok=True)
    summary = summarise(raw)
    target = out_dir / f"g{generation:06d}.json"
    data = jcs(summary) + b"\n"
    if not target.exists() or target.read_bytes() != data:
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(data)
        partial.replace(target)
    return printed(summary, target)


def printed(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    """The rates of every match and policy, in ppm, for the printed ``progress`` result."""
    return {
        "label": LABEL,
        "summary": str(path),
        "unreplayable_games": summary["unreplayable_games"],
        "matches": {
            match_id: {
                policy: {
                    "pairs": body["pairs"],
                    **{name: body["statistics"][name]["mean_ppm"] for name in PRINTED},
                }
                for policy, body in policies.items()
            }
            for match_id, policies in summary["matches"].items()
        },
    }
