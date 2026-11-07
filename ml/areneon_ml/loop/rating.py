"""Integer Bradley-Terry rating (ADR-021 section 4): strengths, bootstrap bounds, cyclic triples.

A pure module: no I/O, no torch, no floating point. Its outputs are the integers of the
``LadderRecord`` (``Strength`` ``strength`` / ``lower`` / ``upper`` at scale 10^9, ``Cell``
``scores`` / ``pairs`` / ``score_sum_ppm``, ``cyclic_triples``), so the ladder writer (slice 7)
copies them.

Input. A :class:`ResultCell` is one player's pair results against one opponent: ``scores`` counts
the pairs at 0, 250000, 500000, 750000 and 1000000 ppm (the ``Cell.scores`` of the record). The
reverse direction is implied (``scores`` reversed); cells given in both directions are
added. Exploiter seats (``x<k>``) are not rated and are refused.

Fit (``fit_strengths``). Bradley-Terry: ``P(i over j) = p_i / (p_i + p_j)``, a pair's score is a
fractional win. Ford/Hunter MM, Jacobi form, in exact integers, at scale 10^9::

    T_i  = sum_j (n_ij * 10^18) // (P_i + P_j)          (n_ij pairs, one floored quotient per term)
    P_i' = (W_i * 10^12) // T_i                         (W_i = summed pair scores of i in ppm)

derived from ``p_i' = W_i / sum_j n_ij / (p_i + p_j)``. Every quotient is floored, results are
clamped to the record's range 1..10^15, there are exactly ``MM_ITERATIONS`` (64) iterations and no
tolerance, non-anchor players start at 10^9 and anchors never change. Each rated player gets one
virtual drawn pair (500000 ppm) against the reference anchor ``actor-rule@v1`` so the estimate stays
finite. A rated player with no chain of real results to an anchor is refused.

Convergence limit of the fixed 64 iterations (ADR-mandated). For a player that wins every pair
against the anchor the estimate is monotone in the number of pairs but compressive at the top: 10
pairs give 1.998e10 (exact MLE 2.1e10), 100 give 5.5e10 (2.0e11), 400 give 6.2e10 (8.0e11), 1000
give 6.4e10 (2.0e12). A near-saturated player is under-rated, never over-rated, and the ordering
is preserved. A player scoring above about 0.97 over 100 pairs is under-estimated; the copy gate
uses ``head_to_head_lower``, which does not go through the fit.

Percentile bounds need not bracket the point estimate: consumers may not assume
``lower <= strength <= upper``.

Bootstrap (``bootstrap_strengths``, ``head_to_head_bounds``). Percentile bootstrap over pairs: each
resample redraws, within every cell, ``n`` pairs with replacement from that cell's ``n`` pairs and
refits. Randomness is ``population_draw(run_key, generation, member, "bootstrap", index)`` with
``index = lane << 32 | counter`` (the lane separates uses of one seed: ``LANE_FIT`` 0 for the fit,
``LANE_COPY`` (2^20) for the copy test, ``1 + 64 i + j`` for the edge of members ``i < j`` in
``cyclic_triples``, and ``ladder_write.LANE_SNAPSHOT + 16 g + k`` (``LANE_SNAPSHOT`` 2^16, in
``2^16 .. 2^20 - 1``) for the snapshot of member ``k`` frozen in generation ``g``; the counter
counts draws from 0). One 64-bit draw is two 32-bit pieces, high
first. ``below(n)`` for ``1 <= n <= 2^32`` reads nothing when ``n == 1``, else takes the next piece
and accepts it when it is below ``floor(2^32 / n) * n``, returning it modulo ``n`` (rejection
sampling, no bias); a rejected piece is skipped. A drawn index ``d`` in a cell with counts ``c``
picks the first score bucket whose running total exceeds ``d``. Cells are visited in sorted
(lower id, higher id) order, resamples one after the other on one stream. Sorted resample values
``v[0..B-1]``: the lower bound at ``alpha`` ppm is ``v[alpha * B // 10^6]`` and the upper bound is
``v[B - 1 - alpha * B // 10^6]`` (one-sided each). The virtual pairs are not resampled.
"""

from __future__ import annotations

import bisect
import itertools
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from areneon_ml.loop.population_seed import population_draw

MM_ITERATIONS = 64
SCALE = 1_000_000_000
REFERENCE_ANCHOR = "actor-rule@v1"
MIN_STRENGTH = 1
MAX_STRENGTH = 10**15
SCORE_STEP_PPM = 250_000
FULL_PPM = 1_000_000
EVEN_PPM = 500_000
BUCKETS = 5
LANE_FIT = 0
LANE_COPY = 1 << 20
# The draw index is lane << 32 | counter and JCS refuses integers above 2^53 - 1.
LANE_LIMIT = 1 << 21
EDGE_WIDTH = 64
_PIECE_RANGE = 1 << 32
_COUNTER_LIMIT = 1 << 32


class RatingError(ValueError):
    """The rating input or a request breaks a rule."""


@dataclass(frozen=True)
class ResultCell:
    """One player's pair results against one opponent: pairs at 0/250k/500k/750k/1M ppm."""

    player: str
    opponent: str
    scores: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class BootstrapSeed:
    """The ``population-seed/1`` label a bootstrap draws under."""

    run_key: str
    generation: int
    member: str


@dataclass(frozen=True)
class BootstrapPlan:
    """The seed, number of resamples and one-sided level (ppm) of a bootstrap."""

    seed: BootstrapSeed
    resamples: int
    alpha_ppm: int


@dataclass(frozen=True)
class Strength:
    """A ``LadderRecord`` ``Strength``: point estimate and percentile bounds, scale 10^9."""

    strength: int
    lower: int
    upper: int


def cell_totals(scores: Sequence[int]) -> tuple[int, int]:
    """``(pairs, score_sum_ppm)`` of a score table, the record's ``Cell`` fields."""
    if len(scores) != BUCKETS or any(count < 0 for count in scores):
        raise RatingError("scores must be five non-negative counts")
    return sum(scores), SCORE_STEP_PPM * sum(i * c for i, c in enumerate(scores))


class _Stream:
    """The integer draws of one bootstrap lane."""

    def __init__(self, seed: BootstrapSeed, lane: int) -> None:
        self._seed = seed
        self._lane = lane
        self._counter = 0
        self._pieces: deque[int] = deque()

    def _piece(self) -> int:
        if not self._pieces:
            if self._counter >= _COUNTER_LIMIT:
                raise RatingError("the bootstrap stream is exhausted")
            seed = self._seed
            value = population_draw(
                seed.run_key,
                seed.generation,
                seed.member,
                "bootstrap",
                self._lane << 32 | self._counter,
            )
            self._counter += 1
            self._pieces.extend((value >> 32, value & 0xFFFFFFFF))
        return self._pieces.popleft()

    def below(self, n: int) -> int:
        """A uniform integer in ``[0, n)`` by rejection sampling over 32-bit pieces."""
        if n < 1 or n > _PIECE_RANGE:
            raise RatingError("below needs 1 <= n <= 2^32")
        if n == 1:
            return 0
        limit = _PIECE_RANGE // n * n
        while True:
            piece = self._piece()
            if piece < limit:
                return piece % n


def _resample(stream: _Stream, scores: Sequence[int]) -> list[int]:
    """Redraw the cell's pairs with replacement; the new score counts."""
    cumulative = list(itertools.accumulate(scores))
    total = cumulative[-1]
    out = [0] * BUCKETS
    for _ in range(total):
        # the first bucket whose running total exceeds the drawn index
        out[bisect.bisect_right(cumulative, stream.below(total))] += 1
    return out


Scores = tuple[int, int, int, int, int]
Cells = dict[tuple[str, str], Scores]


def _resample4(stream: _Stream, scores: Sequence[int]) -> Scores:
    a, b, c, d, e = _resample(stream, scores)
    return a, b, c, d, e


def _merge(cells: Iterable[ResultCell]) -> Cells:
    """Cells keyed by (lower id, higher id), scores from the lower id's side."""
    merged: Cells = {}
    for cell in cells:
        if len(cell.scores) != BUCKETS or any(c < 0 for c in cell.scores):
            raise RatingError("scores must be five non-negative counts")
        if cell.player == cell.opponent:
            raise RatingError("a player cannot play itself")
        scores = cell.scores
        key = (cell.player, cell.opponent)
        if cell.player > cell.opponent:
            key = (cell.opponent, cell.player)
            scores = (scores[4], scores[3], scores[2], scores[1], scores[0])
        old = merged.get(key, (0, 0, 0, 0, 0))
        merged[key] = (
            old[0] + scores[0],
            old[1] + scores[1],
            old[2] + scores[2],
            old[3] + scores[3],
            old[4] + scores[4],
        )
    return {key: value for key, value in sorted(merged.items()) if sum(value) > 0}


def _is_exploiter(player: str) -> bool:
    return len(player) > 1 and player[0] == "x" and player[1:].isdigit()


def _check_anchors(anchors: Mapping[str, int]) -> None:
    if anchors.get(REFERENCE_ANCHOR) != SCALE:
        raise RatingError("the reference anchor must be present at exactly 10^9")
    for strength in anchors.values():
        if not MIN_STRENGTH <= strength <= MAX_STRENGTH:
            raise RatingError("anchor strengths must be in 1..10^15")


def _check_connected(cells: Cells, rated: Sequence[str], anchors: Mapping[str, int]) -> None:
    neighbours: dict[str, list[str]] = {}
    for a, b in cells:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    seen = {a for a in anchors}
    queue = deque(seen)
    while queue:
        for other in neighbours.get(queue.popleft(), ()):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    if any(player not in seen for player in rated):
        raise RatingError("a rated player is not connected to an anchor")


def _prepare(
    cells: Iterable[ResultCell], rated: Sequence[str], anchors: Mapping[str, int]
) -> tuple[Cells, list[str]]:
    _check_anchors(anchors)
    if len(set(rated)) != len(rated) or any(p in anchors for p in rated):
        raise RatingError("rated players must be unique and not anchors")
    if any(_is_exploiter(p) for p in rated):
        raise RatingError("exploiter seats are not rated")
    merged = _merge(cells)
    known = set(rated) | set(anchors)
    if any(a not in known or b not in known for a, b in merged):
        raise RatingError("a result names a player that is neither rated nor an anchor")
    _check_connected(merged, rated, anchors)
    return merged, sorted(rated)


def _solve(players: Sequence[str], anchors: Mapping[str, int], cells: Cells) -> dict[str, int]:
    """The 64 MM iterations on merged cells; returns the rated players' strengths."""
    # edges: (index a, index b, pairs, summed score of a in ppm); indices into ``names``
    names = list(players) + [a for a in sorted(anchors) if a not in players]
    index = {name: i for i, name in enumerate(names)}
    rated_count = len(players)
    edges: list[tuple[int, int, int, int]] = []
    for (a, b), counts in cells.items():
        pairs, total = cell_totals(counts)
        edges.append((index[a], index[b], pairs, total))
    ref = index[REFERENCE_ANCHOR]
    for i in range(rated_count):  # one virtual drawn pair against the reference per player
        edges.append((i, ref, 1, EVEN_PPM))
    wins = [0] * rated_count
    adjacent: list[list[tuple[int, int]]] = [[] for _ in range(rated_count)]
    for left, right, pairs, total in edges:
        if left < rated_count:
            wins[left] += total
            adjacent[left].append((right, pairs))
        if right < rated_count:
            wins[right] += pairs * FULL_PPM - total
            adjacent[right].append((left, pairs))
    strength = [SCALE] * rated_count + [anchors[name] for name in names[rated_count:]]
    for _ in range(MM_ITERATIONS):
        updated = list(strength)
        for i in range(rated_count):
            own = strength[i]
            denominator = sum(n * 10**18 // (own + strength[j]) for j, n in adjacent[i])
            updated[i] = min(MAX_STRENGTH, max(MIN_STRENGTH, wins[i] * 10**12 // denominator))
        strength = updated
    return {name: strength[i] for i, name in enumerate(players)}


def fit_strengths(
    cells: Iterable[ResultCell], rated: Sequence[str], anchors: Mapping[str, int]
) -> dict[str, int]:
    """Point strengths (scale 10^9) of the ``rated`` players; anchors stay as given."""
    merged, players = _prepare(cells, rated, anchors)
    return _solve(players, anchors, merged)


def _rank(resamples: int, alpha_ppm: int) -> int:
    if resamples < 1:
        raise RatingError("bootstrap_resamples must be at least 1")
    if not 1 <= alpha_ppm <= EVEN_PPM:
        raise RatingError("alpha_ppm must be in 1..500000")
    return alpha_ppm * resamples // FULL_PPM


def bootstrap_strengths(
    cells: Iterable[ResultCell],
    rated: Sequence[str],
    anchors: Mapping[str, int],
    plan: BootstrapPlan,
    lane: int = LANE_FIT,
) -> dict[str, Strength]:
    """Strengths with percentile bounds at ``alpha_ppm`` per tail (``lane``: default the fit's)."""
    if lane < 0 or lane >= LANE_LIMIT:
        raise RatingError("lane must be in 0..2^21-1")
    resamples = plan.resamples
    rank = _rank(resamples, plan.alpha_ppm)
    merged, players = _prepare(cells, rated, anchors)
    point = _solve(players, anchors, merged)
    stream = _Stream(plan.seed, lane)
    draws: dict[str, list[int]] = {player: [] for player in players}
    for _ in range(resamples):
        redrawn = {key: _resample4(stream, counts) for key, counts in merged.items()}
        fitted = _solve(players, anchors, redrawn)
        for player, value in fitted.items():
            draws[player].append(value)
    result: dict[str, Strength] = {}
    for player in players:
        ordered = sorted(draws[player])
        result[player] = Strength(point[player], ordered[rank], ordered[resamples - 1 - rank])
    return result


def _mean_ppm(scores: Sequence[int]) -> int:
    pairs, total = cell_totals(scores)
    return total // pairs


def head_to_head_bounds(
    scores: Sequence[int],
    plan: BootstrapPlan,
    lane: int = LANE_COPY,
) -> tuple[int, int]:
    """One-sided ``(lower, upper)`` bounds at ``alpha_ppm`` of the mean pair score, in ppm.

    ``lane`` must be in ``0 <= lane < 2^21`` (the draw index stays below 2^53).
    ``scores`` is the player's pair-score table against the opponent (at least one pair); the mean
    is ``score_sum_ppm // pairs`` and each resample redraws the pairs with replacement.
    """
    resamples = plan.resamples
    rank = _rank(resamples, plan.alpha_ppm)
    if lane < 0 or lane >= LANE_LIMIT:
        raise RatingError("lane must be in 0..2^21-1")
    pairs, _ = cell_totals(scores)
    if pairs < 1:
        raise RatingError("a head-to-head bound needs at least one pair")
    stream = _Stream(plan.seed, lane)
    means = sorted(_mean_ppm(_resample(stream, scores)) for _ in range(resamples))
    return means[rank], means[resamples - 1 - rank]


def head_to_head_lower(
    scores: Sequence[int],
    plan: BootstrapPlan,
    lane: int = LANE_COPY,
) -> int:
    """The one-sided lower bound of the mean pair score at ``alpha_ppm`` (the copy test)."""
    return head_to_head_bounds(scores, plan, lane)[0]


def cyclic_triples(
    cells: Iterable[ResultCell],
    members: Sequence[str],
    plan: BootstrapPlan,
) -> int:
    """Triples of ``members`` with A over B, B over C and C over A, each edge one-sided significant.

    ``a`` is over ``b`` when the lower bound of a's mean pair score against b exceeds 500000 ppm.
    For members ``i < j`` (sorted ids) one bootstrap on lane ``1 + 64 i + j`` gives both
    directions (``j`` over ``i`` when the upper bound of i's mean is below 500000), so no pair of
    members is over each other both ways. A pair of members without results has no edge.
    """
    ordered = sorted(members)
    if len(set(ordered)) != len(ordered) or len(ordered) > EDGE_WIDTH:
        raise RatingError("members must be unique and at most 64")
    if any(_is_exploiter(m) for m in ordered):
        raise RatingError("exploiter seats are not rated")
    merged = _merge(cells)
    over: set[tuple[int, int]] = set()
    for i, j in itertools.combinations(range(len(ordered)), 2):
        counts = merged.get((ordered[i], ordered[j]))
        if counts is None:
            continue
        lower, upper = head_to_head_bounds(counts, plan, 1 + EDGE_WIDTH * i + j)
        if lower > EVEN_PPM:
            over.add((i, j))
        if upper < EVEN_PPM:
            over.add((j, i))
    return sum(
        ((i, j) in over and (j, k) in over and (k, i) in over)
        or ((i, k) in over and (k, j) in over and (j, i) in over)
        for i, j, k in itertools.combinations(range(len(ordered)), 3)
    )
