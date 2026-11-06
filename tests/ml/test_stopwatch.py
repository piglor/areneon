"""The stop-watch behind the generation profile (#1191): free when off, integers when on."""

from __future__ import annotations

from areneon_ml import stopwatch


def test_off_reads_no_clock_and_records_nothing() -> None:
    stopwatch.disable()
    mark = stopwatch.begin()
    assert mark == (0, 0)
    stopwatch.end("x", mark)
    stopwatch.absorb({"x": {"wall_ns": 5, "cpu_ns": 5, "count": 1}})
    assert stopwatch.snapshot() == {}
    assert not stopwatch.enabled()


def test_on_adds_wall_cpu_and_counts_per_name_and_take_forgets() -> None:
    stopwatch.enable()
    try:
        for _ in range(3):
            stopwatch.end("a", stopwatch.begin())
        stopwatch.end("b", stopwatch.begin())
        found = stopwatch.snapshot()
        assert sorted(found) == ["a", "b"]
        assert found["a"]["count"] == 3
        assert found["a"]["wall_ns"] >= 0
        assert found["a"]["cpu_ns"] >= 0
        stopwatch.absorb(
            {
                "a": {"wall_ns": 7, "cpu_ns": 3, "count": 2},
                "c": {"wall_ns": 1, "cpu_ns": 1, "count": 1},
            }
        )
        after = stopwatch.take()
        assert after["a"]["count"] == 5
        assert after["a"]["wall_ns"] >= found["a"]["wall_ns"] + 7
        assert after["c"] == {"wall_ns": 1, "cpu_ns": 1, "count": 1}
        assert stopwatch.snapshot() == {}
    finally:
        stopwatch.disable()


def test_ms_rounds_down() -> None:
    assert stopwatch.ms(1_999_999) == 1
    assert stopwatch.ms(2_000_000) == 2
