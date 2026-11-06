"""Rates and pair-level intervals of the situational move-use diagnostic (#1283, gate #1274).

``packages/showdown-bridge/scripts/situational-move-use.ts`` re-simulates the stored games of a
progress generation and writes raw integer counts per game and seat (see its module comment for
the situations: Threatened, Safe, Exposed slower, Bad matchup, Repeat). This module sums them per
match and policy and derives the rates and their descriptive intervals:

* Protect rate when Threatened against when Safe (offered and chosen, any streak), and their
  difference;
* the repeat-Protect rate (a stalling move chosen again at ``protect_streak >= 1``, any situation);
* the switch rate in Bad-matchup decisions against when not threatened (Safe or Exposed slower),
  and their difference.

LABEL: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength
measure.

The interval is the pair-level percentile bootstrap of ``progress.stats`` (same SHA-256 counter
stream and tail ranks; one stream per match and policy, shared by that policy's statistics):
resample the PAIRS (both games of a pair, both seats, summed), recompute each statistic from the
resampled sums as an exact rational, drop a resample in which a rate has no offered decision, and
take the 2.5% and 97.5% order statistics of the rest, in parts per million rounded outward. If
fewer than half the resamples are defined the interval is ``null``. Everything is integers and
rationals; the output is byte-stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs

from areneon_ml.progress.stats import B_BOOT, PPM, TAIL_PERMILLE, _draws, bootstrap_stream_key

LABEL = (
    "situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure"
)
FORMAT = "situational-move-use/1"
SITUATIONS = ("threatened", "safe", "exposed_slower")

Counts = dict[str, int]


class SituationalError(ValueError):
    pass


def _sum(counts: Counts, keys: Sequence[str]) -> int:
    return sum(counts.get(k, 0) for k in keys)


def _protect(situations: Sequence[str], leaf: str) -> list[str]:
    return [f"protect.{s}.{streak}.{leaf}" for s in situations for streak in ("fresh", "repeat")]


def _repeat(leaf: str) -> list[str]:
    return [f"protect.{s}.repeat.{leaf}" for s in SITUATIONS]


def _switch(classes: Sequence[str], leaf: str) -> list[str]:
    return [f"switch.{c}.{leaf}" for c in classes]


#: Each rate: (numerator keys, denominator keys).
RATES: dict[str, tuple[list[str], list[str]]] = {
    "protect_threatened": (_protect(["threatened"], "chosen"), _protect(["threatened"], "offered")),
    "protect_safe": (_protect(["safe"], "chosen"), _protect(["safe"], "offered")),
    "repeat_protect": (_repeat("chosen"), _repeat("offered")),
    "switch_bad_matchup": (_switch(["bad_matchup"], "chosen"), _switch(["bad_matchup"], "offered")),
    "switch_not_threatened": (
        _switch(["safe", "exposed_slower"], "chosen"),
        _switch(["safe", "exposed_slower"], "offered"),
    ),
}
#: Differences of two rates: (name, minuend, subtrahend).
DIFFERENCES = (
    ("protect_threatened_minus_safe", "protect_threatened", "protect_safe"),
    ("switch_bad_matchup_minus_not_threatened", "switch_bad_matchup", "switch_not_threatened"),
)


def _rate(counts: Counts, name: str) -> Fraction | None:
    numerator, denominator = RATES[name]
    offered = _sum(counts, denominator)
    return None if offered == 0 else Fraction(_sum(counts, numerator), offered)


def _statistics(counts: Counts) -> dict[str, Fraction | None]:
    out: dict[str, Fraction | None] = {name: _rate(counts, name) for name in RATES}
    for name, a, b in DIFFERENCES:
        x, y = out[a], out[b]
        out[name] = None if x is None or y is None else x - y
    return out


def _floor_ppm(x: Fraction) -> int:
    return (x.numerator * PPM) // x.denominator


def _ceil_ppm(x: Fraction) -> int:
    return -((-x.numerator * PPM) // x.denominator)


STATISTIC_NAMES = (*RATES, *(name for name, _, _ in DIFFERENCES))
_KEYS = sorted({k for numerator, denominator in RATES.values() for k in (*numerator, *denominator)})


def _intervals(pairs: list[Counts], key: bytes, b_boot: int) -> dict[str, dict[str, int] | None]:
    """Every statistic's interval from ONE set of draws and one merge per resample."""
    n = len(pairs)
    vectors = [[p.get(k, 0) for k in _KEYS] for p in pairs]
    draws = _draws(key, n, n * b_boot)
    values: dict[str, list[Fraction]] = {name: [] for name in STATISTIC_NAMES}
    for b in range(b_boot):
        chosen = (vectors[j] for j in draws[b * n : (b + 1) * n])
        total = [sum(column) for column in zip(*chosen, strict=True)]
        for name, value in _statistics(dict(zip(_KEYS, total, strict=True))).items():
            if value is not None:
                values[name].append(value)
    out: dict[str, dict[str, int] | None] = {}
    for name, found in values.items():
        if len(found) * 2 < b_boot:
            out[name] = None
            continue
        found.sort()
        tail = len(found) * TAIL_PERMILLE // 1000
        out[name] = {
            "low_ppm": _floor_ppm(found[tail]),
            "high_ppm": _ceil_ppm(found[len(found) - 1 - tail]),
            "resamples": len(found),
        }
    return out


def summarise(raw: dict[str, Any], *, b_boot: int = B_BOOT) -> dict[str, Any]:
    """Per match and policy: summed counts, rates and differences in ppm, and intervals."""
    if raw.get("format") != FORMAT:
        raise SituationalError("not a situational move-use raw file")
    seed = str(raw.get("evaluation_key") or raw.get("run_id") or "")
    generation = int(raw["generation"])
    grouped: dict[tuple[str, str], dict[tuple[str, int], Counts]] = defaultdict(
        lambda: defaultdict(dict)
    )
    games: dict[tuple[str, str], int] = defaultdict(int)
    for game in raw["games"]:
        for seat in ("p1", "p2"):
            who = game["seats"][seat]
            slot = (game["match_id"], who["policy"])
            # A game without a pair is its own unit, keyed apart from the pair numbers.
            unit = ("p", game["pair"]) if game["pair"] is not None else ("g", game["game"])
            into = grouped[slot][unit]
            for k, v in who["counts"].items():
                into[k] = into.get(k, 0) + v
            games[slot] += 1
    matches: dict[str, dict[str, Any]] = {}
    for (match_id, policy), by_pair in sorted(grouped.items()):
        pairs = [by_pair[p] for p in sorted(by_pair)]
        total: Counts = defaultdict(int)
        for counts in pairs:
            for k, v in counts.items():
                total[k] += v
        key = bootstrap_stream_key(seed, generation, f"situational:{match_id}:{policy}")
        stats = _statistics(total)
        found = _intervals(pairs, key, b_boot)
        out: dict[str, Any] = {}
        for name, value in stats.items():
            out[name] = {
                "mean_ppm": None if value is None else _floor_ppm(value),
                "interval": None if value is None else found[name],
            }
        matches.setdefault(match_id, {})[policy] = {
            "games": games[(match_id, policy)],
            "pairs": len(pairs),
            "counts": {k: total[k] for k in sorted(total)},
            "statistics": out,
        }
    return {
        "format": FORMAT,
        "label": LABEL,
        "generation": generation,
        "pin": raw.get("pin"),
        "unreplayable_games": raw.get("unreplayable_games", 0),
        "interval": (
            "pair-level percentile bootstrap, descriptive, conditional on the fixed validation plan"
        ),
        "matches": matches,
    }


def read_summary(value: object) -> dict[str, Any]:
    """A summary as ``summarise`` writes it (the dashboard's input): the shape it renders."""
    if not isinstance(value, dict) or value.get("format") != FORMAT or value.get("label") != LABEL:
        raise SituationalError("not a situational move-use summary")
    matches = value.get("matches")
    if not isinstance(matches, dict) or not isinstance(value.get("generation"), int):
        raise SituationalError("the summary has no matches or generation")
    for policies in matches.values():
        for body in policies.values():
            stats = body["statistics"]
            if set(stats) != set(STATISTIC_NAMES) or not isinstance(body["pairs"], int):
                raise SituationalError("a summary entry lacks a statistic")
    return value


def render(summary: dict[str, Any]) -> str:
    """A plain-text table: one line per match, policy and statistic (integers, ppm)."""
    lines = [summary["label"], f"generation {summary['generation']}"]
    for match_id, policies in summary["matches"].items():
        for policy, body in policies.items():
            lines.append(f"{match_id} {policy} pairs={body['pairs']}")
            for name, stat in body["statistics"].items():
                interval = stat["interval"]
                span = (
                    "n/a"
                    if interval is None
                    else f"[{interval['low_ppm']}, {interval['high_ppm']}]"
                )
                lines.append(f"  {name}: {stat['mean_ppm']} ppm {span}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, write: Callable[[str], Any] = sys.stdout.write) -> int:
    parser = argparse.ArgumentParser(prog="python -m areneon_ml.report.situational")
    parser.add_argument("raw", type=Path, help="the file situational-move-use.ts wrote")
    parser.add_argument(
        "--out", type=Path, help="write the JCS summary here (default: print the table)"
    )
    args = parser.parse_args(argv)
    summary = summarise(json.loads(args.raw.read_text(encoding="utf8")))
    if args.out is None:
        write(render(summary) + "\n")
    else:
        args.out.write_bytes(jcs(summary) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
