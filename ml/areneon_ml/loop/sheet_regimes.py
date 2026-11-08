"""Sheet regimes of a mixed-format run (ADR-027 sections 5 and 5a, #1153).

The twin of ``packages/showdown-bridge/src/sheet-regimes.ts``: which format each game of a
generation is played in is a pure function of the run config, the generation and the game
(purpose ``sheet-regime/1``). The TS side writes the plan; this module derives the formats again,
independently, so the loop refuses a generation whose plan names other formats than the quota
gives, and checks the manifests a generation reports.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.learner.config import RunConfig, SheetRegimes

PURPOSE = "sheet-regime/1"


class SheetRegimeError(ValueError):
    """A plan, a record or a manifest list breaks a sheet-regime rule."""


def regime_digest(run_key: str, generation: int, game: int) -> str:
    """SHA-256 (hex) over the JCS of ``{run_key, generation, game, purpose}``."""
    body = {"run_key": run_key, "generation": generation, "game": game, "purpose": PURPOSE}
    return str(sha256_hex(jcs(body)))


def closed_game_count(games: int, closed_share_ppm: int) -> int:
    """``round(games * share / 10^6)``, half up, in integers."""
    return (games * closed_share_ppm + 500_000) // 1_000_000


def self_play_regimes(sheet: SheetRegimes, run_key: str, generation: int, games: int) -> list[str]:
    """The regime (``open`` or ``closed``) of each self-play game, by game index."""
    regimes = ["open"] * games
    if generation <= sheet.open_only_generations:
        return regimes
    order = sorted(range(games), key=lambda g: (regime_digest(run_key, generation, g), g))
    for game in order[: closed_game_count(games, sheet.closed_share_ppm)]:
        regimes[game] = "closed"
    return regimes


def plan_formats(config: RunConfig, generation: int) -> list[str]:
    """The ``format_id`` each plan entry must name: self-play games by regime, baselines closed."""
    sheet = config.sheet_regimes
    if sheet is None:
        raise SheetRegimeError("the run has no sheet_regimes")
    gen = config.generation
    self_play = self_play_regimes(sheet, config.run_key, generation, gen.games)
    formats = [sheet.open_format if r == "open" else config.format_id for r in self_play]
    return formats + [config.format_id] * sum(b.games for b in gen.baselines)


def check_plan_formats(plan: Sequence[Any], config: RunConfig, generation: int) -> dict[str, int]:
    """The plan names the quota's formats (none at all without ``sheet_regimes``).

    Returns the regime counts ``{open, closed}`` of a mixed-format plan (``{}`` otherwise).
    """
    if config.sheet_regimes is None:
        if any(isinstance(e, dict) and "format_id" in e for e in plan):
            raise SheetRegimeError(
                "a plan entry names a format, but the config has no sheet_regimes"
            )
        return {}
    expected = plan_formats(config, generation)
    actual = [e.get("format_id") if isinstance(e, dict) else None for e in plan]
    if actual != expected:
        raise SheetRegimeError("the plan's formats are not the sheet-regime/1 quota of the config")
    opened = sum(1 for f in actual if f == config.sheet_regimes.open_format)
    return {"open": opened, "closed": len(actual) - opened}


def manifests_problem(
    manifests: Sequence[dict[str, Any]], config: RunConfig, closed: dict[str, Any] | None = None
) -> str | None:
    """Why ``manifests`` is not a run's ``game_manifests``, or None.

    Sorted by JCS bytes and unique; the closed format's manifest (``closed``, when given) is in it;
    exactly one manifest for each of the run's two formats, and they share the pinned engine
    commit and the regulation (the engine-side sheet-variant check is the Node side's).
    """
    sheet = config.sheet_regimes
    if sheet is None:
        return "the run has no sheet_regimes"
    keys = [jcs(m) for m in manifests]
    if any(a >= b for a, b in itertools.pairwise(keys)):
        return "game_manifests must be sorted by their JCS bytes, each once"
    if closed is not None and closed not in manifests:
        return "the manifest must be one of game_manifests"
    formats = {config.format_id, sheet.open_format}
    if sorted(m["format_id"] for m in manifests) != sorted(formats):
        return "game_manifests must hold one manifest for each of the run's two formats"
    if len({(m["showdown_commit"], m["regulation"]) for m in manifests}) > 1:
        return "the game manifests do not share one engine commit and regulation"
    return None
