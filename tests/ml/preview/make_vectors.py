"""Golden vector generator for the Team Preview solver (ADR-029 section 3a, O5).

Python is the reference: run `python tests/ml/preview/make_vectors.py` to rewrite
`fixtures/preview-solver/vectors.json`; the TypeScript module is pinned to the same file.
Matrices are hand-written or drawn from a local linear congruential generator, never from
`random`, so the file does not depend on the interpreter's version.
"""

from __future__ import annotations

import json
from pathlib import Path

from areneon_ml.preview.solver import SOLVER_FORMAT, solve

OUT = Path(__file__).resolve().parents[3] / "fixtures" / "preview-solver" / "vectors.json"
M = 1_000_000


def lcg_matrix(seed: int, n: int, m: int) -> list[list[int]]:
    state = seed
    out = []
    for _ in range(n):
        row = []
        for _ in range(m):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            row.append((state >> 33) % (M + 1))
        out.append(row)
    return out


MATRICES: list[tuple[str, list[list[int]]]] = [
    ("matching-pennies", [[M, 0], [0, M]]),
    ("rock-paper-scissors", [[500_000, 0, M], [M, 500_000, 0], [0, M, 500_000]]),
    ("biased-rps", [[500_000, 0, 800_000], [M, 500_000, 0], [200_000, M, 500_000]]),
    ("pure-saddle", [[600_000, 700_000], [400_000, 300_000]]),
    ("dominated-row", [[M, 0], [0, M], [100_000, 200_000]]),
    ("all-equal", [[500_000] * 3] * 3),
    ("all-zero", [[0, 0], [0, 0]]),
    ("all-max", [[M, M], [M, M]]),
    ("one-by-n", [[100_000, 900_000, 400_000, 900_000]]),
    ("n-by-one", [[300_000], [800_000], [800_000], [0]]),
    ("one-by-one", [[M]]),
    ("tied-best-rows", [[700_000, 300_000], [700_000, 300_000], [300_000, 700_000]]),
    ("extremes", [[0, M, 0], [M, 0, M], [0, 0, M]]),
    ("random-4x4-a", lcg_matrix(1, 4, 4)),
    ("random-4x4-b", lcg_matrix(2, 4, 4)),
    ("random-3x5", lcg_matrix(3, 3, 5)),
    ("random-6x6", lcg_matrix(4, 6, 6)),
]
ITERATIONS = [1, 2, 16, 256]
CURVE = [16, 256, 4096]


def build() -> dict[str, object]:
    cases = []
    for name, matrix in MATRICES:
        for kind in ("regret-matching", "regret-matching-plus"):
            for t in ITERATIONS:
                s = solve(matrix, t, kind)
                cases.append(
                    {
                        "id": f"{name}/{kind}/{t}",
                        "kind": kind,
                        "iterations": t,
                        "matrix": matrix,
                        "row": list(s.row),
                        "col": list(s.col),
                        "exploitability": s.exploitability,
                    }
                )
    curves = []
    for name, matrix in MATRICES:
        if name.startswith("random") or name in ("biased-rps", "rock-paper-scissors"):
            curves.append(
                {
                    "id": name,
                    "matrix": matrix,
                    "iterations": CURVE,
                    "exploitability": [solve(matrix, t).exploitability for t in CURVE],
                }
            )
    return {"format": SOLVER_FORMAT, "cases": cases, "curves": curves}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
