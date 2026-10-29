"""Value-head diagnostics against hand-computed values (ADR-010 erratum A4 item 1)."""

from __future__ import annotations

import pytest
from areneon_ml.learner.diagnostics import (
    CalibrationBin,
    bin_of,
    calibration,
    explained_variance,
)


def test_explained_variance_on_a_fixed_table() -> None:
    # returns 0,1,0,1 (mean .5, var .25); residuals 0,0,.5,-.5 (mean 0, var .125) -> 1 - .5
    assert explained_variance([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, -0.5, 1.5]) == 0.5
    # a perfect predictor explains everything
    assert explained_variance([0.0, 1.0, 0.5], [0.0, 1.0, 0.5]) == 1.0


def test_a_constant_predictor_explains_nothing() -> None:
    # a constant shifts the residuals and leaves their variance: EV is exactly 0
    assert explained_variance([0.0, 1.0, 1.0, 0.0], [0.25] * 4) == 0.0
    # the mean's constant too
    assert explained_variance([0.0, 1.0, 1.0, 0.0], [0.5] * 4) == 0.0


def test_a_return_with_no_variance_gives_the_fixed_value() -> None:
    assert explained_variance([1.0, 1.0, 1.0], [0.2, 0.9, 0.4]) == 0.0
    assert explained_variance([0.5], [0.5]) == 0.0


def test_explained_variance_refuses_bad_shapes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        explained_variance([], [])
    with pytest.raises(ValueError, match="one length"):
        explained_variance([1.0], [1.0, 2.0])


def test_bins_are_tenths_clamped_at_the_ends() -> None:
    assert [bin_of(v) for v in (-3.0, 0.0, 0.05, 0.1, 0.55, 0.95, 1.0, 7.0)] == [
        0, 0, 0, 1, 5, 9, 9, 9,
    ]  # fmt: skip


def test_calibration_on_a_fixed_table() -> None:
    values = [0.05, 0.02, 0.55, 0.52, 1.4]
    scores = [0.0, 1.0, 1.0, 0.5, 1.0]
    table = calibration(values, scores)
    assert len(table) == 10
    assert table[0] == CalibrationBin(2, 0.035, 0.5)
    assert table[5] == CalibrationBin(2, 0.535, 0.75)
    assert table[9] == CalibrationBin(1, 1.4, 1.0)
    assert [b.count for b in table] == [2, 0, 0, 0, 0, 2, 0, 0, 0, 1]
    assert table[1] == CalibrationBin(0, 0.0, 0.0)


def test_a_constant_predictor_fills_one_bin() -> None:
    table = calibration([0.4] * 4, [1.0, 0.0, 1.0, 0.5])
    assert [b.count for b in table] == [0, 0, 0, 0, 4, 0, 0, 0, 0, 0]
    assert table[4] == CalibrationBin(4, 0.4, 0.625)


def test_calibration_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="one length"):
        calibration([0.1], [])
