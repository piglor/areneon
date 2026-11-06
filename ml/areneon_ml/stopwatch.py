"""Wall and CPU timers for the profile of a generation (ADR-028 section 1, #1191).

Off unless ``enable()`` was called, so a process that never asks pays one boolean test per
call and every value it computes is untouched: a timer only reads clocks. The totals are
integers (nanoseconds) per name, plus how often each name was timed. They are diagnostics:
nothing in the loop, the loader, the learner or the worker reads them to decide anything, and
they reach no file the generation manifest hashes.

``begin()`` returns a mark, ``end(name, mark)`` adds the time since it. CPU time is the
calling thread's (``time.thread_time_ns``), which is the process's CPU while the code is
single threaded, as the worker and each load piece are.
"""

from __future__ import annotations

from time import perf_counter_ns, thread_time_ns

Mark = tuple[int, int]

_ON = False
_WALL: dict[str, int] = {}
_CPU: dict[str, int] = {}
_COUNT: dict[str, int] = {}


def enable() -> None:
    """Start timing and forget what was timed before."""
    global _ON  # noqa: PLW0603
    _ON = True
    reset()


def disable() -> None:
    global _ON  # noqa: PLW0603
    _ON = False
    reset()


def enabled() -> bool:
    return _ON


def reset() -> None:
    _WALL.clear()
    _CPU.clear()
    _COUNT.clear()


def begin() -> Mark:
    """A mark for ``end``; (0, 0) and no clock read while the timers are off."""
    return (perf_counter_ns(), thread_time_ns()) if _ON else (0, 0)


def end(name: str, mark: Mark) -> None:
    """Add the wall and CPU time since ``mark`` to ``name`` (nothing while off)."""
    if _ON:
        _WALL[name] = _WALL.get(name, 0) + perf_counter_ns() - mark[0]
        _CPU[name] = _CPU.get(name, 0) + thread_time_ns() - mark[1]
        _COUNT[name] = _COUNT.get(name, 0) + 1


def absorb(totals: dict[str, dict[str, int]]) -> None:
    """Add the totals a forked piece took (``take``) to this process's."""
    if _ON:
        for name, entry in totals.items():
            _WALL[name] = _WALL.get(name, 0) + entry["wall_ns"]
            _CPU[name] = _CPU.get(name, 0) + entry["cpu_ns"]
            _COUNT[name] = _COUNT.get(name, 0) + entry["count"]


def snapshot() -> dict[str, dict[str, int]]:
    """The totals so far, per name: ``wall_ns``, ``cpu_ns`` and ``count``."""
    return {
        name: {"wall_ns": _WALL[name], "cpu_ns": _CPU[name], "count": _COUNT[name]}
        for name in sorted(_WALL)
    }


def take() -> dict[str, dict[str, int]]:
    """``snapshot()``, then forget it."""
    found = snapshot()
    reset()
    return found


def ms(ns: int) -> int:
    """Whole milliseconds, rounded down."""
    return ns // 1_000_000
