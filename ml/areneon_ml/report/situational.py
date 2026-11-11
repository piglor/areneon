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

History rows (#1345, ADR-022 erratum E-22.5 slice H7): when the raw file carries a ``history``
count block per seat (``situational-move-use.ts --history``), each match and policy also gets a
``history`` entry with the rates below, each with its pair-level interval (the same stream and
ranks, one stream per match and policy under another label); a raw file without them gives the
summary it always gave, byte for byte.

* stalling moves chosen at ``stall_counter > 1`` (repeat) against fresh, and the mean engine
  success chance of the chosen ones;
* history-scaled power: chosen when the history-declared damage is above (up) against below (down)
  the default-history damage, and their difference;
* parts that fail in every opposing-action branch, chosen: per offered part, split by the actor's
  facts (it has acted, it has not, an item-history fact), and per decision;
* conditional parts (the outcome differs by branch): chosen per offered, and the share of the
  chosen ones' branches that do not fail.

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


#: Each rate: (numerator keys, denominator keys, scale): numerator / (denominator * scale).
Rates = dict[str, tuple[list[str], list[str], int]]
Differences = tuple[tuple[str, str, str], ...]

RATES: Rates = {
    "protect_threatened": (
        _protect(["threatened"], "chosen"),
        _protect(["threatened"], "offered"),
        1,
    ),
    "protect_safe": (_protect(["safe"], "chosen"), _protect(["safe"], "offered"), 1),
    "repeat_protect": (_repeat("chosen"), _repeat("offered"), 1),
    "switch_bad_matchup": (
        _switch(["bad_matchup"], "chosen"),
        _switch(["bad_matchup"], "offered"),
        1,
    ),
    "switch_not_threatened": (
        _switch(["safe", "exposed_slower"], "chosen"),
        _switch(["safe", "exposed_slower"], "offered"),
        1,
    ),
}
#: Differences of two rates: (name, minuend, subtrahend).
DIFFERENCES: Differences = (
    ("protect_threatened_minus_safe", "protect_threatened", "protect_safe"),
    ("switch_bad_matchup_minus_not_threatened", "switch_bad_matchup", "switch_not_threatened"),
)

PPM_PER_SHARE = 1_000_000


def _history(group: str) -> tuple[list[str], list[str]]:
    return ([f"history.{group}.chosen"], [f"history.{group}.offered"])


def _history_rate(group: str) -> tuple[list[str], list[str], int]:
    chosen, offered = _history(group)
    return (chosen, offered, 1)


#: The history rows (#1345), over the ``history`` count block of each seat.
HISTORY_RATES: Rates = {
    "stall_repeat": _history_rate("stall.repeat"),
    "stall_fresh": _history_rate("stall.fresh"),
    "stall_repeat_success": (
        ["history.stall.repeat.chosen_success_ppm"],
        ["history.stall.repeat.chosen_success_n"],
        PPM_PER_SHARE,
    ),
    "stall_fresh_success": (
        ["history.stall.fresh.chosen_success_ppm"],
        ["history.stall.fresh.chosen_success_n"],
        PPM_PER_SHARE,
    ),
    "power_up": _history_rate("power.up"),
    "power_down": _history_rate("power.down"),
    "fails_all": _history_rate("fails.all"),
    "fails_all_acted": _history_rate("fails.acted"),
    "fails_all_fresh": _history_rate("fails.fresh"),
    "fails_all_item": _history_rate("fails.item"),
    "fails_all_per_decision": (["history.fails.all.chosen"], ["history.decisions"], 1),
    "conditional": _history_rate("conditional"),
    "conditional_chosen_pass_share": (
        ["history.conditional.chosen_passing"],
        ["history.conditional.chosen_branches"],
        1,
    ),
}
HISTORY_DIFFERENCES: Differences = (
    ("stall_repeat_minus_fresh", "stall_repeat", "stall_fresh"),
    ("power_up_minus_down", "power_up", "power_down"),
)


def _rate(counts: Counts, name: str, rates: Rates = RATES) -> Fraction | None:
    numerator, denominator, scale = rates[name]
    offered = _sum(counts, denominator)
    return None if offered == 0 else Fraction(_sum(counts, numerator), offered * scale)


def _statistics(
    counts: Counts, rates: Rates = RATES, differences: Differences = DIFFERENCES
) -> dict[str, Fraction | None]:
    out: dict[str, Fraction | None] = {name: _rate(counts, name, rates) for name in rates}
    for name, a, b in differences:
        x, y = out[a], out[b]
        out[name] = None if x is None or y is None else x - y
    return out


def _floor_ppm(x: Fraction) -> int:
    return (x.numerator * PPM) // x.denominator


def _ceil_ppm(x: Fraction) -> int:
    return -((-x.numerator * PPM) // x.denominator)


STATISTIC_NAMES = (*RATES, *(name for name, _, _ in DIFFERENCES))
HISTORY_STATISTIC_NAMES = (*HISTORY_RATES, *(name for name, _, _ in HISTORY_DIFFERENCES))


def _keys(rates: Rates) -> list[str]:
    return sorted(
        {k for numerator, denominator, _ in rates.values() for k in (*numerator, *denominator)}
    )


def _intervals(
    pairs: list[Counts],
    key: bytes,
    b_boot: int,
    rates: Rates = RATES,
    differences: Differences = DIFFERENCES,
) -> dict[str, dict[str, int] | None]:
    """Every statistic's interval from ONE set of draws and one merge per resample."""
    n = len(pairs)
    keys = _keys(rates)
    names = (*rates, *(name for name, _, _ in differences))
    vectors = [[p.get(k, 0) for k in keys] for p in pairs]
    draws = _draws(key, n, n * b_boot)
    values: dict[str, list[Fraction]] = {name: [] for name in names}
    for b in range(b_boot):
        chosen = (vectors[j] for j in draws[b * n : (b + 1) * n])
        total = [sum(column) for column in zip(*chosen, strict=True)]
        for name, value in _statistics(
            dict(zip(keys, total, strict=True)), rates, differences
        ).items():
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


def _history_entry(pairs: list[Counts], key: bytes, b_boot: int) -> dict[str, Any]:
    """The history rows (#1345) of one match and policy: summed counts, rates in ppm, intervals."""
    total: Counts = defaultdict(int)
    for counts in pairs:
        for k, v in counts.items():
            total[k] += v
    stats = _statistics(total, HISTORY_RATES, HISTORY_DIFFERENCES)
    found = _intervals(pairs, key, b_boot, HISTORY_RATES, HISTORY_DIFFERENCES)
    return {
        "counts": {k: total[k] for k in sorted(total)},
        "statistics": {
            name: {
                "mean_ppm": None if value is None else _floor_ppm(value),
                "interval": None if value is None else found[name],
            }
            for name, value in stats.items()
        },
    }


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
    # The history rows (#1345): present only when the raw file carries a `history` block per seat.
    history: dict[tuple[str, str], dict[tuple[str, int], Counts]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for game in raw["games"]:
        for seat in ("p1", "p2"):
            who = game["seats"][seat]
            slot = (game["match_id"], who["policy"])
            # A game without a pair is its own unit, keyed apart from the pair numbers.
            unit = ("p", game["pair"]) if game["pair"] is not None else ("g", game["game"])
            into = grouped[slot][unit]
            for k, v in who["counts"].items():
                into[k] = into.get(k, 0) + v
            if "history" in who:
                at = history[slot][unit]
                for k, v in who["history"].items():
                    at[k] = at.get(k, 0) + v
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
        entry: dict[str, Any] = {
            "games": games[(match_id, policy)],
            "pairs": len(pairs),
            "counts": {k: total[k] for k in sorted(total)},
            "statistics": out,
        }
        if (match_id, policy) in history:
            entry["history"] = _history_entry(
                [history[(match_id, policy)][p] for p in sorted(history[(match_id, policy)])],
                bootstrap_stream_key(seed, generation, f"situational-history:{match_id}:{policy}"),
                b_boot,
            )
        matches.setdefault(match_id, {})[policy] = entry
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
            if "history" in body and set(body["history"]["statistics"]) != set(
                HISTORY_STATISTIC_NAMES
            ):
                raise SituationalError("a summary entry lacks a history statistic")
    return value


def _span(interval: dict[str, int] | None) -> str:
    return "n/a" if interval is None else f"[{interval['low_ppm']}, {interval['high_ppm']}]"


def render(summary: dict[str, Any]) -> str:
    """A plain-text table: one line per match, policy and statistic (integers, ppm)."""
    lines = [summary["label"], f"generation {summary['generation']}"]
    for match_id, policies in summary["matches"].items():
        for policy, body in policies.items():
            lines.append(f"{match_id} {policy} pairs={body['pairs']}")
            for name, stat in body["statistics"].items():
                lines.append(f"  {name}: {stat['mean_ppm']} ppm {_span(stat['interval'])}")
            for name, stat in body.get("history", {}).get("statistics", {}).items():
                lines.append(f"  history.{name}: {stat['mean_ppm']} ppm {_span(stat['interval'])}")
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
