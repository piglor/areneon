"""The load phase's validation, reduction and encoding split (#1191): timers only read clocks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from areneon_ml import stopwatch
from areneon_ml.learner import parallel
from areneon_ml.learner.data import trajectories

from tests.ml.learner.support import POLICY, stream_of
from tests.ml.learner.test_parallel import encoders, examples_of, same, write


@pytest.fixture(autouse=True)
def small_pieces(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1)
    yield
    stopwatch.disable()


def test_serial_and_parallel_loads_time_every_example_once_and_change_nothing(
    tmp_path: Path,
) -> None:
    text, manifest = stream_of(examples_of())
    path, manifest = write(tmp_path, text, manifest)
    plain = trajectories(path, manifest, encoders(), POLICY)
    steps = sum(len(t.steps) for t in plain)

    stopwatch.enable()
    serial = trajectories(path, manifest, encoders(), POLICY)
    serial_totals = stopwatch.take()
    assert sorted(serial_totals) == ["encode", "reduce", "validate"]
    assert serial_totals["encode"]["count"] == steps
    assert serial_totals["reduce"]["count"] == 2 * steps  # the view's clean and the reducer feed
    assert serial_totals["validate"]["count"] == 1  # the file check, once

    stopwatch.enable()
    forked = trajectories(path, manifest, encoders(), POLICY, processes=3)
    forked_totals = stopwatch.take()
    assert forked_totals["encode"]["count"] == steps
    assert forked_totals["reduce"]["count"] == 2 * steps
    assert forked_totals["validate"]["count"] == steps + 1  # every line, and the digest pass
    stopwatch.disable()

    same(plain, serial)
    same(plain, forked)
