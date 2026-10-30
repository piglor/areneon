// The integer regret-matching solver for the Team Preview payoff table (ADR-029 section 3a).
// Pure, engine-free and float-free: the matrix is integers in ppm (0 to 10^6, rows ours, columns
// the opponent's), the game is constant-sum and both sides update simultaneously. Strategies are
// ppm vectors made by largest remainder. Cumulative regrets exceed 2^53, so every intermediate is a
// BigInt. The Python twin is `ml/areneon_ml/preview/solver.py`; the committed vectors
// (`fixtures/preview-solver/vectors.json`) are computed there and both sides are pinned to them.

export const PPM = 1_000_000;
export const SOLVER_FORMAT = "preview-solver/1";
export const SOLVER_KINDS = ["regret-matching", "regret-matching-plus"] as const;
export type SolverKind = (typeof SOLVER_KINDS)[number];
export const DEFAULT_ITERATIONS = 4096;
export const MAX_ITERATIONS = 1_000_000;
export const MAX_DIMENSION = 4096;

const at = <T>(a: readonly T[], i: number): T => a[i] as T;

/** Malformed input; the message is fixed and quotes no value. */
export class SolverError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SolverError";
  }
}

export interface Solution {
  /** Our mixture in ppm, summing to 10^6. */
  readonly row: readonly number[];
  /** The opponent's mixture in ppm, summing to 10^6. */
  readonly col: readonly number[];
  /** Best pure responses against the other side's average, summed, exact, in ppm squared. */
  readonly exploitability: bigint;
}

/** Shares of 10^6 by largest remainder: floors, then one ppm per remainder in descending order,
 * ties by lower index. All-zero weights give the uniform strategy by the same rule. */
export function normalise(weights: readonly bigint[]): bigint[] {
  const n = weights.length;
  let w = weights;
  let total = w.reduce((a, b) => a + b, 0n);
  if (total === 0n) {
    w = weights.map(() => 1n);
    total = BigInt(n);
  }
  const big = BigInt(PPM);
  const shares = w.map((x) => (x * big) / total);
  const rems = w.map((x) => (x * big) % total);
  const order = [...w.keys()].sort((a, b) => (at(rems, a) === at(rems, b) ? a - b : at(rems, a) > at(rems, b) ? -1 : 1));
  let left = Number(big - shares.reduce((a, b) => a + b, 0n));
  for (const i of order) {
    if (left-- <= 0) break;
    shares[i] = at(shares, i) + 1n;
  }
  return shares;
}

export function checkMatrix(matrix: readonly (readonly number[])[], iterations: number, kind: string): void {
  if (!(SOLVER_KINDS as readonly string[]).includes(kind)) throw new SolverError("unknown solver kind");
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > MAX_ITERATIONS) {
    throw new SolverError("iterations must be an integer from 1 to 1000000");
  }
  if (!Array.isArray(matrix) || matrix.length < 1 || matrix.length > MAX_DIMENSION) {
    throw new SolverError("matrix must have 1 to 4096 rows");
  }
  const first: unknown = matrix[0];
  if (!Array.isArray(first) || first.length < 1 || first.length > MAX_DIMENSION) {
    throw new SolverError("matrix must have 1 to 4096 columns");
  }
  for (const row of matrix as readonly unknown[]) {
    if (!Array.isArray(row) || row.length !== first.length) throw new SolverError("matrix must be rectangular");
    for (const v of row as unknown[]) {
      if (typeof v !== "number" || !Number.isInteger(v) || v < 0 || v > PPM) {
        throw new SolverError("entries must be integers from 0 to 1000000");
      }
    }
  }
}

/** Our best pure response against `col` minus the opponent's best pure response against `row`. */
export function exploitability(
  matrix: readonly (readonly bigint[])[],
  row: readonly bigint[],
  col: readonly bigint[],
): bigint {
  const n = matrix.length;
  const m = at(matrix, 0).length;
  let bestOurs = -1n;
  for (let s = 0; s < n; s++) {
    let u = 0n;
    for (let c = 0; c < m; c++) u += at(at(matrix, s), c) * at(col, c);
    if (u > bestOurs) bestOurs = u;
  }
  let worstTheirs = -1n;
  for (let c = 0; c < m; c++) {
    let z = 0n;
    for (let s = 0; s < n; s++) z += at(at(matrix, s), c) * at(row, s);
    if (worstTheirs < 0n || z < worstTheirs) worstTheirs = z;
  }
  return bestOurs - worstTheirs;
}

export function solve(
  matrix: readonly (readonly number[])[],
  iterations: number = DEFAULT_ITERATIONS,
  kind: SolverKind = "regret-matching",
): Solution {
  checkMatrix(matrix, iterations, kind);
  const a = matrix.map((r) => r.map((v) => BigInt(v)));
  const n = a.length;
  const m = at(a, 0).length;
  const plus = kind === "regret-matching-plus";
  const big = BigInt(PPM);
  const regRow = new Array<bigint>(n).fill(0n);
  const regCol = new Array<bigint>(m).fill(0n);
  const sumRow = new Array<bigint>(n).fill(0n);
  const sumCol = new Array<bigint>(m).fill(0n);
  for (let t = 0; t < iterations; t++) {
    const x = normalise(regRow.map((r) => (r > 0n ? r : 0n)));
    const y = normalise(regCol.map((r) => (r > 0n ? r : 0n)));
    for (let s = 0; s < n; s++) sumRow[s] = at(sumRow, s) + at(x, s);
    for (let c = 0; c < m; c++) sumCol[c] = at(sumCol, c) + at(y, c);
    const u = new Array<bigint>(n).fill(0n); // row payoffs, ppm squared
    const z = new Array<bigint>(m).fill(0n); // what x pays against each column
    for (let s = 0; s < n; s++) {
      for (let c = 0; c < m; c++) {
        const e = at(at(a, s), c);
        u[s] = at(u, s) + e * at(y, c);
        z[c] = at(z, c) + e * at(x, s);
      }
    }
    let v = 0n; // ppm cubed; u and z are scaled up to match
    for (let s = 0; s < n; s++) v += at(x, s) * at(u, s);
    for (let s = 0; s < n; s++) {
      const r = at(regRow, s) + at(u, s) * big - v;
      regRow[s] = plus && r < 0n ? 0n : r;
    }
    for (let c = 0; c < m; c++) {
      const r = at(regCol, c) + v - at(z, c) * big; // the opponent's payoff is 10^6 minus ours
      regCol[c] = plus && r < 0n ? 0n : r;
    }
  }
  const row = normalise(sumRow);
  const col = normalise(sumCol);
  return { row: row.map(Number), col: col.map(Number), exploitability: exploitability(a, row, col) };
}
