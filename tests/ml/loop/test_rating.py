"""Integer Bradley-Terry rating, bootstrap and cyclic triples (ADR-021 section 4)."""

from __future__ import annotations

import hashlib
import json

import pytest
from areneon_ml.loop import rating
from areneon_ml.loop.population_seed import population_draw
from areneon_ml.loop.rating import (
    LANE_COPY,
    REFERENCE_ANCHOR,
    BootstrapPlan,
    BootstrapSeed,
    RatingError,
    ResultCell,
    Strength,
    bootstrap_strengths,
    cell_totals,
    cyclic_triples,
    fit_strengths,
    head_to_head_bounds,
    head_to_head_lower,
)

ANCHORS = {REFERENCE_ANCHOR: 10**9}
SEED = BootstrapSeed("tiny-run", 3, "m0")


def plan(resamples: int, seed: BootstrapSeed = SEED) -> BootstrapPlan:
    return BootstrapPlan(seed, resamples, 50_000)


def cell(a: str, b: str, scores: tuple[int, int, int, int, int]) -> ResultCell:
    return ResultCell(a, b, scores)


def test_cell_totals() -> None:
    assert cell_totals((2, 0, 1, 0, 4)) == (7, 4_500_000)
    with pytest.raises(RatingError):
        cell_totals((1, 2, 3))
    with pytest.raises(RatingError):
        cell_totals((1, 2, 3, 4, -1))


def test_two_player_closed_form() -> None:
    # 7 real pairs worth 4.5 wins plus the virtual half pair: w = 5 of n = 8, MLE x = w/(n-w).
    got = fit_strengths([cell("m0", REFERENCE_ANCHOR, (2, 0, 1, 0, 4))], ["m0"], ANCHORS)
    assert abs(got["m0"] - 5 * 10**9 // 3) <= 4  # floor error of 64 floored iterations


def test_three_player_closed_form_and_reversed_cells() -> None:
    # m0 and m1 are mirror images (equal against the anchor, an even head to head): the same x.
    cells = [
        cell("m0", REFERENCE_ANCHOR, (2, 0, 1, 0, 4)),
        cell(REFERENCE_ANCHOR, "m1", (4, 0, 1, 0, 2)),  # given from the anchor's side
        cell("m1", "m0", (0, 0, 4, 0, 0)),
    ]
    got = fit_strengths(cells, ["m1", "m0"], ANCHORS)
    assert set(got) == {"m0", "m1"}
    for value in got.values():
        assert abs(value - 5 * 10**9 // 3) <= 4


def test_duplicate_directions_add() -> None:
    both = [
        cell("m0", REFERENCE_ANCHOR, (1, 0, 0, 0, 1)),
        cell(REFERENCE_ANCHOR, "m0", (1, 0, 0, 0, 1)),
    ]
    one = [cell("m0", REFERENCE_ANCHOR, (2, 0, 0, 0, 2))]
    assert fit_strengths(both, ["m0"], ANCHORS) == fit_strengths(one, ["m0"], ANCHORS)


def test_anchors_fixed_and_not_in_output() -> None:
    anchors = {REFERENCE_ANCHOR: 10**9, "damage-aware@v1": 3 * 10**9}
    cells = [
        cell("m0", REFERENCE_ANCHOR, (0, 0, 5, 0, 5)),
        cell("m0", "damage-aware@v1", (5, 0, 5, 0, 0)),
        cell(REFERENCE_ANCHOR, "damage-aware@v1", (6, 2, 2, 0, 0)),
    ]
    got = fit_strengths(cells, ["m0"], anchors)
    assert list(got) == ["m0"]
    assert anchors == {REFERENCE_ANCHOR: 10**9, "damage-aware@v1": 3 * 10**9}
    assert 10**9 < got["m0"] < 3 * 10**9


def test_all_wins_stays_finite() -> None:
    won = fit_strengths([cell("m0", REFERENCE_ANCHOR, (0, 0, 0, 0, 10))], ["m0"], ANCHORS)
    lost = fit_strengths([cell("m0", REFERENCE_ANCHOR, (10, 0, 0, 0, 0))], ["m0"], ANCHORS)
    assert 10**9 < won["m0"] < rating.MAX_STRENGTH
    assert 1 <= lost["m0"] < 10**9


def test_clamped_to_the_record_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rating, "MAX_STRENGTH", 5 * 10**9)
    got = fit_strengths([cell("m0", REFERENCE_ANCHOR, (0, 0, 0, 0, 100000))], ["m0"], ANCHORS)
    assert got["m0"] == 5 * 10**9


def test_chain_to_an_anchor_through_another_player_is_connected() -> None:
    cells = [cell("m0", "m1", (1, 1, 1, 1, 1)), cell("m1", REFERENCE_ANCHOR, (1, 1, 1, 1, 1))]
    assert set(fit_strengths(cells, ["m0", "m1"], ANCHORS)) == {"m0", "m1"}


def test_refusals() -> None:
    base = [cell("m0", REFERENCE_ANCHOR, (1, 1, 1, 1, 1))]
    # a player cut off from every anchor (m1 and m2 only play each other)
    cut = [*base, cell("m1", "m2", (1, 1, 1, 1, 1))]
    with pytest.raises(RatingError, match="not connected"):
        fit_strengths(cut, ["m0", "m1", "m2"], ANCHORS)
    # a rated player with no results at all
    with pytest.raises(RatingError, match="not connected"):
        fit_strengths(base, ["m0", "m1"], ANCHORS)
    with pytest.raises(RatingError, match="reference anchor"):
        fit_strengths(base, ["m0"], {"other@v1": 10**9})
    with pytest.raises(RatingError, match="reference anchor"):
        fit_strengths(base, ["m0"], {REFERENCE_ANCHOR: 5})
    with pytest.raises(RatingError, match="anchor strengths"):
        fit_strengths(base, ["m0"], {**ANCHORS, "a@v1": 0})
    with pytest.raises(RatingError, match="unique"):
        fit_strengths(base, ["m0", "m0"], ANCHORS)
    with pytest.raises(RatingError, match="unique"):
        fit_strengths(base, [REFERENCE_ANCHOR], ANCHORS)
    with pytest.raises(RatingError, match="exploiter"):
        fit_strengths([cell("x0", REFERENCE_ANCHOR, (1, 1, 1, 1, 1))], ["x0"], ANCHORS)
    with pytest.raises(RatingError, match="neither rated"):
        fit_strengths([*base, cell("m9", "m0", (1, 1, 1, 1, 1))], ["m0"], ANCHORS)
    with pytest.raises(RatingError, match="itself"):
        fit_strengths([cell("m0", "m0", (1, 1, 1, 1, 1))], ["m0"], ANCHORS)
    with pytest.raises(RatingError, match="five"):
        fit_strengths([cell("m0", REFERENCE_ANCHOR, (1, 1, 1, 1))], ["m0"], ANCHORS)  # type: ignore[arg-type]
    with pytest.raises(RatingError, match="five"):
        fit_strengths([cell("m0", REFERENCE_ANCHOR, (1, 1, 1, 1, -1))], ["m0"], ANCHORS)


def test_a_cell_without_pairs_is_no_connection() -> None:
    with pytest.raises(RatingError, match="not connected"):
        fit_strengths([cell("m0", REFERENCE_ANCHOR, (0, 0, 0, 0, 0))], ["m0"], ANCHORS)


CELLS = [
    cell("m0", REFERENCE_ANCHOR, (2, 3, 6, 5, 4)),
    cell("m1", REFERENCE_ANCHOR, (6, 4, 6, 2, 2)),
    cell("m0", "m1", (1, 3, 8, 5, 3)),
]
ORDER = ["m0", "m1"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def test_bootstrap_deterministic_changes_with_seed_and_is_pinned() -> None:
    first = bootstrap_strengths(CELLS, ORDER, ANCHORS, plan(40))
    again = bootstrap_strengths(CELLS, ORDER, ANCHORS, plan(40))
    other = bootstrap_strengths(CELLS, ORDER, ANCHORS, plan(40, BootstrapSeed("tiny-run", 4, "m0")))
    assert first == again
    assert first != other
    point = fit_strengths(CELLS, ORDER, ANCHORS)
    for name, value in first.items():
        assert isinstance(value, Strength)
        assert value.strength == point[name]
        assert 1 <= value.lower <= value.upper
    assert digest({k: [v.strength, v.lower, v.upper] for k, v in first.items()}) == PINNED_FIT


def test_head_to_head_bounds_pinned_and_seed_dependent() -> None:
    scores = (5, 15, 40, 25, 15)  # 100 pairs
    lower, upper = head_to_head_bounds(scores, plan(200))
    assert lower <= cell_totals(scores)[1] // 100 <= upper
    assert head_to_head_lower(scores, plan(200)) == lower
    assert head_to_head_bounds(scores, plan(200)) == (lower, upper)
    assert head_to_head_bounds(scores, plan(200, BootstrapSeed("tiny-run", 3, "m1"))) != (
        lower,
        upper,
    )
    assert head_to_head_bounds(scores, plan(200), lane=7) != (lower, upper)
    assert digest([lower, upper]) == PINNED_H2H


def test_head_to_head_one_pair_and_dominant_cell() -> None:
    assert head_to_head_bounds((0, 0, 0, 0, 1), plan(10)) == (FULL, FULL)
    lower, upper = head_to_head_bounds((0, 0, 0, 0, 96), plan(50))
    assert (lower, upper) == (FULL, FULL)


FULL = 1_000_000


def test_bound_refusals() -> None:
    with pytest.raises(RatingError, match="at least 1"):
        head_to_head_bounds((1, 1, 1, 1, 1), plan(0))
    with pytest.raises(RatingError, match="alpha_ppm"):
        head_to_head_bounds((1, 1, 1, 1, 1), BootstrapPlan(SEED, 10, 0))
    with pytest.raises(RatingError, match="alpha_ppm"):
        head_to_head_bounds((1, 1, 1, 1, 1), BootstrapPlan(SEED, 10, 500_001))
    with pytest.raises(RatingError, match="at least one pair"):
        head_to_head_bounds((0, 0, 0, 0, 0), plan(10))
    with pytest.raises(RatingError, match="lane"):
        head_to_head_bounds((1, 1, 1, 1, 1), plan(10), lane=1 << 21)
    with pytest.raises(RatingError, match="lane"):
        head_to_head_bounds((1, 1, 1, 1, 1), plan(10), lane=-1)


def test_lane_constants_do_not_overlap() -> None:
    assert LANE_COPY > 1 + rating.EDGE_WIDTH * rating.EDGE_WIDTH


def edge(a: str, b: str, wins: int) -> ResultCell:
    """``a`` wins ``wins`` of 30 pairs outright and loses the rest."""
    return ResultCell(a, b, (30 - wins, 0, 0, 0, wins))


def test_rock_paper_scissors_has_one_cycle() -> None:
    cells = [edge("m0", "m1", 30), edge("m1", "m2", 30), edge("m2", "m0", 30)]
    assert cyclic_triples(cells, ["m2", "m0", "m1"], plan(100)) == 1
    reverse = [edge("m1", "m0", 30), edge("m2", "m1", 30), edge("m0", "m2", 30)]
    assert cyclic_triples(reverse, ["m0", "m1", "m2"], plan(100)) == 1


def test_transitive_has_none_and_weak_edges_and_missing_cells_do_not_count() -> None:
    cells = [edge("m0", "m1", 30), edge("m1", "m2", 30), edge("m0", "m2", 30)]
    assert cyclic_triples(cells, ["m0", "m1", "m2"], plan(100)) == 0
    even = [ResultCell("m0", "m1", (0, 0, 30, 0, 0)), edge("m1", "m2", 30), edge("m2", "m0", 30)]
    assert cyclic_triples(even, ["m0", "m1", "m2"], plan(100)) == 0
    assert cyclic_triples([edge("m0", "m1", 30)], ["m0", "m1", "m2"], plan(100)) == 0
    assert cyclic_triples([], ["m0"], plan(100)) == 0


def test_cycle_among_four_members_counts_each_triple_once() -> None:
    cells = [
        edge("m0", "m1", 30),
        edge("m1", "m2", 30),
        edge("m2", "m0", 30),
        edge("m3", "m0", 30),
        edge("m3", "m1", 30),
        edge("m3", "m2", 30),
    ]
    assert cyclic_triples(cells, ["m0", "m1", "m2", "m3"], plan(100)) == 1


def test_cycle_refusals() -> None:
    with pytest.raises(RatingError, match="unique"):
        cyclic_triples([], ["m0", "m0"], plan(10))
    with pytest.raises(RatingError, match="unique"):
        cyclic_triples([], [f"m{i}" for i in range(65)], plan(10))
    with pytest.raises(RatingError, match="exploiter"):
        cyclic_triples([], ["x0", "m0"], plan(10))


def test_below_rejection_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 3 * 10**9  # limit = n, so a piece at or above 3e9 is rejected
    values = iter([(0xFFFFFFFF << 32) | 5, 7])
    monkeypatch.setattr(rating, "population_draw", lambda *a: next(values))
    stream = rating._Stream(SEED, 0)
    assert stream.below(n) == 5  # the first piece 0xFFFFFFFF is rejected, then the low piece 5
    assert stream.below(1) == 0
    with pytest.raises(RatingError, match="n <="):
        stream.below(2**32 + 1)
    with pytest.raises(RatingError, match="n <="):
        stream.below(0)
    stream._counter = 1 << 32
    stream._pieces.clear()
    with pytest.raises(RatingError, match="exhausted"):
        stream.below(2)


PINNED_FIT = "fbfd0721225f7ddd57c314e3e6689fda3a110c6f0367a5137f75ff25d92b974f"
PINNED_H2H = "59410cfc59ac1283956f8d5d0cd49511f99dd2411c20a208392bb2e505a4f96c"


def test_all_wins_compression_is_pinned() -> None:
    # The 64 fixed iterations under-rate a saturated player; exact MLE is 2 n + 1 times 10^9.
    expected = {10: 19_981_378_078, 100: 55_423_918_815, 400: 62_396_355_816, 1000: 63_940_383_153}
    previous = 0
    for n, value in expected.items():
        got = fit_strengths([cell("m0", REFERENCE_ANCHOR, (0, 0, 0, 0, n))], ["m0"], ANCHORS)
        assert got["m0"] == value
        assert previous < got["m0"] < (2 * n + 1) * 10**9
        previous = got["m0"]


def test_bounds_bracket_the_point_estimate_in_the_common_case() -> None:
    for name, value in bootstrap_strengths(CELLS, ORDER, ANCHORS, plan(40)).items():
        assert value.lower <= value.strength <= value.upper, name


def test_stream_layout_independent_of_population_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    def raw(lane: int, counter: int) -> int:
        body = {
            "generation": 3,
            "index": lane << 32 | counter,
            "member": "m0",
            "purpose": "bootstrap",
            "run_key": "tiny-run",
        }
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    lane = 5
    first, second = raw(lane, 0), raw(lane, 1)
    assert raw(0, 0) == population_draw("tiny-run", 3, "m0", "bootstrap", 0)
    stream = rating._Stream(SEED, lane)
    pieces = [first >> 32, first & 0xFFFFFFFF, second >> 32]
    for n, piece in zip((7, 1000, 3), pieces, strict=True):
        limit = 2**32 // n * n
        if piece < limit:
            assert stream.below(n) == piece % n
        else:  # pragma: no cover - the fixed vector decides
            raise AssertionError("fixture piece rejected")
