"""The score and its Wilson interval, derived on display and never stored (ADR-017 section 2).

A record stores integers (wins, ties, losses). The score counts a tie as half a win, so it is
``(2 * wins + ties) / (2 * games)``. The interval is the Wilson 95% interval for a proportion
with that score and ``games`` trials, which is approximate when there are ties (protocol 6.2).
It is computed with integer arithmetic only (a fixed ``z**2`` and an integer square root), so
it is the same on every machine, and returned in parts per million, rounded outward.

The interval is CONDITIONAL ON THE FIXED VALIDATION PLAN: its estimand is the frozen pair's
expected score on these validation teams under this pairing and these seeds, not over the
generator's distribution of teams. That is a labelled deviation from protocol 6.2 (which
allows Wilson only for fresh team pairs from the generator). The label travels with the
numbers: ``CONDITIONAL_LABEL``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import isqrt
from typing import Any

from areneon_contracts.contracts import jcs

PPM = 1_000_000
# z for a two-sided 95% interval is 1.959964: z**2 = 3.841459 (the usual 1.96**2 = 3.8416).
Z2_PPM = 3_841_459
CONDITIONAL_LABEL = "conditional on the fixed validation plan"
#: What a version 2 record's interval is, in words (ADR-017 erratum E2, items 2 and 6).
PAIR_INTERVAL_LABEL = (
    "pair-level percentile bootstrap, descriptive, conditional on the fixed validation plan"
)


@dataclass(frozen=True)
class Interval:
    """A score and its Wilson interval, in parts per million."""

    score_ppm: int
    low_ppm: int
    high_ppm: int
    games: int


def wilson(wins: int, ties: int, games: int) -> Interval:
    """The score ``(wins + ties / 2) / games`` and its Wilson 95% interval, in integers."""
    if games < 1 or wins < 0 or ties < 0 or wins + ties > games:
        raise ValueError("a match needs at least one game, and wins + ties must not exceed it")
    n = games
    a = 2 * wins + ties  # the score is p = a / (2 n)
    # With z2 = Z2_PPM / PPM the bounds are
    #   (p + z2 / (2 n) -+ sqrt(z2 (p (1 - p) / n + z2 / (4 n^2)))) / (1 + z2 / n).
    # In units of 1 / (2 n PPM) the centre is a PPM + Z2_PPM and the root is
    # sqrt(Z2_PPM (a (2 n - a) PPM + Z2_PPM n) / n); the divisor 1 + z2 / n is
    # (n PPM + Z2_PPM) / (n PPM).
    centre = a * PPM + Z2_PPM
    inner = Z2_PPM * (a * (2 * n - a) * PPM + Z2_PPM * n)
    root = isqrt(inner // n) + 1  # an integer root, rounded up: the interval errs outward
    denominator = 2 * (n * PPM + Z2_PPM)
    low = max(0, (PPM * (centre - root)) // denominator)
    high = min(PPM, -((-PPM * (centre + root)) // denominator))
    score = (a * PPM) // (2 * n)
    return Interval(score_ppm=score, low_ppm=low, high_ppm=high, games=n)


RESULT_HALVES = {"win": 2, "tie": 1, "loss": 0}


def pair_scores_of(results: list[str]) -> list[int]:
    """Per pair, the candidate's score in half-games (0 to 4); games 2k and 2k + 1 are pair k."""
    if not results or len(results) % 2 or any(r not in RESULT_HALVES for r in results):
        raise ValueError("a match needs an even number of win, tie or loss results")
    return [
        RESULT_HALVES[results[i]] + RESULT_HALVES[results[i + 1]] for i in range(0, len(results), 2)
    ]


# --- The pair-level interval (ADR-017 erratum E2, items 2 and 6; ticket #948) ----------------
#
# A version 2 record plays every validation pair from both seats, so the two games of a pair
# share teams and a battle seed and are not independent: the PAIR is the unit. A pair's score is
# the mean of its two game scores, kept as an integer in half-games (0 to 4, so 4 is two wins).
# The interval is a percentile bootstrap over pairs (the protocol's 6.1 team-clustered bootstrap
# specialised to disjoint pairs; not the 6.3 block plan), in integers only:
#
# * ``B_BOOT`` resamples of ``n`` pairs drawn with replacement from the ``n`` pairs, each
#   resample summarised by the integer sum of its values (all resamples have the same ``n``, so
#   sums order exactly like means);
# * the ``n`` indices of a resample come from a SHA-256 counter stream (below), by rejection so
#   that no index is biased, and are the same on every machine and Python version;
# * the interval is the sorted resample sums at ranks ``floor(B * 25 / 1000)`` and
#   ``B - 1 - floor(B * 25 / 1000)`` (the 2.5% and 97.5% order statistics, 95%), converted to
#   parts per million and rounded OUTWARD (low down, high up), like the Wilson bounds.
#
# The stream is seeded from the run key, the generation and a label naming the statistic, so a
# page shows the same interval every time and two statistics do not share draws. Intervals are
# DESCRIPTIVE: one evaluation shows many (every baseline, anchor and Preview variant) with no
# multiplicity control, and none is a test or a claim. They are derived on display, never stored.

BOOTSTRAP_PURPOSE = "progress-bootstrap/1"
#: Resamples per interval. Fixed in code, not recorded: 4,000 give a 95% interval whose bounds
#: move by a few thousandths of a score between seeds at 100 pairs, enough for a progress view.
B_BOOT = 4000
#: Tail rank of the 95% percentile interval, in thousandths of B_BOOT (2.5%).
TAIL_PERMILLE = 25
MIN_RESAMPLES = 2
HALVES_PER_PAIR = 4  # a pair score of 4 half-games is two wins (score 1)


@dataclass(frozen=True)
class PairInterval:
    """A mean over pairs and its percentile interval, in parts per million."""

    mean_ppm: int
    low_ppm: int
    high_ppm: int
    pairs: int


def bootstrap_stream_key(run_key: str, generation: int, label: str) -> bytes:
    """The 32-byte key of one statistic's resampling stream: SHA-256 over the JCS of
    ``{run_key, generation, label, purpose: "progress-bootstrap/1"}``."""
    return hashlib.sha256(
        jcs(
            {
                "run_key": run_key,
                "generation": generation,
                "label": label,
                "purpose": BOOTSTRAP_PURPOSE,
            }
        )
    ).digest()


def _draws(key: bytes, n: int, count: int) -> list[int]:
    """``count`` indices below ``n`` from the stream of ``key``: SHA-256(key || counter) read as
    big-endian uint32 words, a word at or above the largest multiple of ``n`` skipped."""
    limit = (1 << 32) // n * n
    out: list[int] = []
    counter = 0
    while len(out) < count:
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        counter += 1
        for i in range(0, 32, 4):
            word = int.from_bytes(block[i : i + 4], "big")
            if word < limit:
                out.append(word % n)
                if len(out) == count:
                    break
    return out


def pair_interval(values: Sequence[int], key: bytes, *, b_boot: int = B_BOOT) -> PairInterval:
    """Mean and percentile bootstrap interval of per-pair integer values in half-games.

    ``values`` are pair scores (0 to 4) or per-pair differences (-4 to 4). The mean is
    ``sum / (4 n)`` in parts per million, rounded down; the bounds round outward.
    """
    n = len(values)
    if n < 1 or b_boot < MIN_RESAMPLES:
        raise ValueError("an interval needs at least one pair and two resamples")
    draws = _draws(key, n, n * b_boot)
    sums = sorted(sum(values[j] for j in draws[b * n : (b + 1) * n]) for b in range(b_boot))
    tail = b_boot * TAIL_PERMILLE // 1000
    denominator = HALVES_PER_PAIR * n
    return PairInterval(
        mean_ppm=sum(values) * PPM // denominator,
        low_ppm=sums[tail] * PPM // denominator,
        high_ppm=-((-sums[b_boot - 1 - tail] * PPM) // denominator),
        pairs=n,
    )


def paired_differences(mine: Sequence[int], other: Sequence[int]) -> list[int]:
    """``d_k``: this policy's pair score minus the other's on the same pair, in half-games."""
    if len(mine) != len(other):
        raise ValueError("a paired difference needs the same pairs on both sides")
    return [a - b for a, b in zip(mine, other, strict=True)]


def match_pair_scores(record: dict[str, Any], match_id: str) -> list[int]:
    """Per pair of one match of a version 2 record, the candidate's score in half-games.

    A forfeit is a scored loss for the side that forfeited; a tie counts half.
    """
    results = []
    for row in record["games"]:
        if row["match_id"] != match_id:
            continue
        outcome = row["outcome"]
        if outcome["kind"] == "forfeit":
            results.append("loss" if outcome["loser"] == "candidate" else "win")
        else:
            results.append(
                {"candidate": "win", "opponent": "loss", "tie": "tie"}[outcome["winner"]]
            )
    return pair_scores_of(results)
