"""Integer regret-matching solver for the Team Preview payoff table (ADR-029 section 3a).

Pure and float-free: the matrix is integers in ppm (0 to 10^6, rows ours, columns the
opponent's), the game is constant-sum (the column player's payoff is 10^6 minus the entry), and
both sides update simultaneously. Strategies are ppm vectors made by largest remainder, so the
output is exact and identical to the TypeScript module
(`packages/opponent-model/src/preview-solver.ts`),
which uses BigInt where this uses int. No game rule is written here.
"""

from __future__ import annotations

from dataclasses import dataclass

PPM = 1_000_000
SOLVER_FORMAT = "preview-solver/1"
SOLVER_KINDS = ("regret-matching", "regret-matching-plus")
DEFAULT_ITERATIONS = 4096
MAX_ITERATIONS = 1_000_000
MAX_DIMENSION = 4096


class SolverError(ValueError):
    """Malformed input; the message is fixed and quotes no value."""


@dataclass(frozen=True)
class Solution:
    row: tuple[int, ...]  # our mixture, ppm, sums to 10^6
    col: tuple[int, ...]  # the opponent's mixture, ppm, sums to 10^6
    exploitability: int  # best pure responses against the other's average, summed, in ppm squared

    def exploitability_ppm(self) -> int:
        return -(-self.exploitability // PPM)


def normalise(weights: list[int]) -> list[int]:
    """Shares of 10^6 by largest remainder: floors, then one ppm per remainder in descending
    order, ties by lower index. All-zero weights give the uniform strategy by the same rule."""
    n = len(weights)
    total = sum(weights)
    if total == 0:
        weights, total = [1] * n, n
    shares = [w * PPM // total for w in weights]
    rems = [w * PPM % total for w in weights]
    order = sorted(range(n), key=lambda i: (-rems[i], i))
    for i in order[: PPM - sum(shares)]:
        shares[i] += 1
    return shares


def check_matrix(matrix: list[list[int]], iterations: int, kind: str) -> None:
    if kind not in SOLVER_KINDS:
        raise SolverError("unknown solver kind")
    if type(iterations) is not int or not 1 <= iterations <= MAX_ITERATIONS:
        raise SolverError("iterations must be an integer from 1 to 1000000")
    if not isinstance(matrix, list) or not 1 <= len(matrix) <= MAX_DIMENSION:
        raise SolverError("matrix must have 1 to 4096 rows")
    if not isinstance(matrix[0], list) or not 1 <= len(matrix[0]) <= MAX_DIMENSION:
        raise SolverError("matrix must have 1 to 4096 columns")
    width = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != width:
            raise SolverError("matrix must be rectangular")
        for v in row:
            if type(v) is not int or not 0 <= v <= PPM:
                raise SolverError("entries must be integers from 0 to 1000000")


def exploitability(matrix: list[list[int]], row: list[int], col: list[int]) -> int:
    """Our best pure response against `col` minus the opponent's best pure response against
    `row` (in the opponent's favour), exact in ppm squared."""
    n, m = len(matrix), len(matrix[0])
    best_ours = max(sum(matrix[s][c] * col[c] for c in range(m)) for s in range(n))
    worst_theirs = min(sum(matrix[s][c] * row[s] for s in range(n)) for c in range(m))
    return best_ours - worst_theirs


def solve(
    matrix: list[list[int]],
    iterations: int = DEFAULT_ITERATIONS,
    kind: str = "regret-matching",
) -> Solution:
    check_matrix(matrix, iterations, kind)
    n, m = len(matrix), len(matrix[0])
    plus = kind == "regret-matching-plus"
    reg_row = [0] * n
    reg_col = [0] * m
    sum_row = [0] * n
    sum_col = [0] * m
    for _ in range(iterations):
        x = normalise([max(r, 0) for r in reg_row])
        y = normalise([max(r, 0) for r in reg_col])
        for s in range(n):
            sum_row[s] += x[s]
        for c in range(m):
            sum_col[c] += y[c]
        u = [sum(matrix[s][c] * y[c] for c in range(m)) for s in range(n)]  # row payoffs
        z = [sum(matrix[s][c] * x[s] for s in range(n)) for c in range(m)]  # what x pays against c
        v = sum(x[s] * u[s] for s in range(n))  # ppm cubed; u and z are scaled up to match
        for s in range(n):
            reg_row[s] += u[s] * PPM - v
            if plus and reg_row[s] < 0:
                reg_row[s] = 0
        for c in range(m):  # the opponent's payoff is 10^6 minus ours
            reg_col[c] += v - z[c] * PPM
            if plus and reg_col[c] < 0:
                reg_col[c] = 0
    row = normalise(sum_row)
    col = normalise(sum_col)
    return Solution(tuple(row), tuple(col), exploitability(matrix, row, col))
