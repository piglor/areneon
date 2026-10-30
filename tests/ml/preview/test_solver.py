import json
from pathlib import Path

import pytest
from areneon_ml.preview.solver import (
    MAX_DIMENSION,
    MAX_ITERATIONS,
    PPM,
    SolverError,
    normalise,
    solve,
)

from tests.ml.preview.make_vectors import build

VECTORS = Path(__file__).resolve().parents[3] / "fixtures" / "preview-solver" / "vectors.json"


def test_committed_vectors_are_fresh() -> None:
    assert json.loads(VECTORS.read_text()) == json.loads(json.dumps(build()))


def test_exploitability_falls_with_iterations() -> None:
    for c in json.loads(VECTORS.read_text())["curves"]:
        a, b, d = c["exploitability"]
        assert d < b < a
        if c["id"].startswith("random"):
            assert d <= 30_000 * PPM  # at most 3 points at 4,096 iterations


def test_hand_worked_games() -> None:
    assert solve([[PPM, 0], [0, PPM]], 64).row == (500000, 500000)
    assert solve([[PPM, 0], [0, PPM], [100000, 200000]]).row[2] < 200
    assert solve([[600000, 700000], [400000, 300000]]).row[0] > 990000
    assert solve([[0, 0], [0, 0]], 5).row == (500000, 500000)
    assert solve([[1, 2, 3]], 3).row == (PPM,)
    assert solve([[1], [2]], 3).col == (PPM,)


def test_largest_remainder() -> None:
    assert normalise([1, 1, 1]) == [333334, 333333, 333333]
    assert normalise([0, 0]) == [500000, 500000]
    assert sum(normalise([3, 1, 7])) == PPM
    assert solve([[0, 0], [0, 0]], 1).exploitability_ppm() == 0
    assert solve([[PPM, 0], [0, PPM]], 1).exploitability_ppm() == 0


@pytest.mark.parametrize(
    "args",
    [
        ([[1]], 0),
        ([[1]], MAX_ITERATIONS + 1),
        ([[1]], True),
        ([], 1),
        ([[]], 1),
        ([[1]] * (MAX_DIMENSION + 1), 1),
        ([[1] * (MAX_DIMENSION + 1)], 1),
        ([[1, 2], [3]], 1),
        ([[1], 5], 1),
        (5, 1),
        ([5], 1),
        ([[-1]], 1),
        ([[PPM + 1]], 1),
        ([[0.5]], 1),
        ([[True]], 1),
    ],
)
def test_refusals(args: tuple[object, int]) -> None:
    with pytest.raises(SolverError):
        solve(*args)  # type: ignore[arg-type]


def test_unknown_kind() -> None:
    with pytest.raises(SolverError):
        solve([[1]], 1, "x")
