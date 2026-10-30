import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MAX_DIMENSION, MAX_ITERATIONS, PPM, SOLVER_FORMAT, SolverError, normalise, solve, type SolverKind } from "../src/preview-solver.ts";

interface Case {
  id: string;
  kind: SolverKind;
  iterations: number;
  matrix: number[][];
  row: number[];
  col: number[];
  exploitability: number;
}
interface Curve {
  id: string;
  matrix: number[][];
  iterations: number[];
  exploitability: number[];
}
const vectors = JSON.parse(readFileSync(new URL("../../../fixtures/preview-solver/vectors.json", import.meta.url), "utf8")) as {
  format: string;
  cases: Case[];
  curves: Curve[];
};

describe("preview solver vectors (computed in Python)", () => {
  it("is the same format", () => expect(vectors.format).toBe(SOLVER_FORMAT));
  it("gives byte-identical outputs on every vector", () => {
    expect(vectors.cases.length).toBeGreaterThan(100);
    for (const c of vectors.cases) {
      const s = solve(c.matrix, c.iterations, c.kind);
      const got = JSON.stringify({ row: s.row, col: s.col, exploitability: s.exploitability.toString() });
      const want = JSON.stringify({ row: c.row, col: c.col, exploitability: String(c.exploitability) });
      expect(got, c.id).toBe(want);
    }
  });
  it("falls in matrix exploitability as iterations grow", () => {
    for (const c of vectors.curves) {
      const got = c.iterations.map((t) => Number(solve(c.matrix, t).exploitability));
      expect(got, c.id).toEqual(c.exploitability);
      expect(got[2] as number, c.id).toBeLessThan(got[1] as number);
      expect(got[1] as number, c.id).toBeLessThan(got[0] as number);
    }
  });
});

describe("hand-worked games", () => {
  it("matching pennies is uniform", () => {
    const s = solve([[PPM, 0], [0, PPM]], 64);
    expect(s.row).toEqual([500000, 500000]);
    expect(s.exploitability).toBe(0n);
  });
  it("puts no weight on a strictly dominated row far into the run", () => {
    expect(solve([[PPM, 0], [0, PPM], [100000, 200000]], 4096).row[2] as number).toBeLessThan(200);
  });
  it("finds a pure saddle", () => {
    const s = solve([[600000, 700000], [400000, 300000]], 4096);
    expect(s.row[0] as number).toBeGreaterThan(990000);
  });
  it("rock-paper-scissors is near uniform", () => {
    const s = solve([[500000, 0, PPM], [PPM, 500000, 0], [0, PPM, 500000]], 4096);
    for (const p of s.row) expect(Math.abs(p - 333333)).toBeLessThan(10000);
  });
  it("is uniform with no positive regret and handles 1 x n and n x 1", () => {
    expect(solve([[0, 0], [0, 0]], 5).row).toEqual([500000, 500000]);
    expect(solve([[1, 2, 3]], 3).row).toEqual([PPM]);
    expect(solve([[1], [2]], 3).col).toEqual([PPM]);
  });
  it("rounds by largest remainder with ties to the lower index", () => {
    expect(normalise([1n, 1n, 1n])).toEqual([333334n, 333333n, 333333n]);
    expect(normalise([0n, 0n])).toEqual([500000n, 500000n]);
    expect(normalise([3n, 1n]).reduce((a, b) => a + b, 0n)).toBe(1000000n);
  });
  it("keeps ppm sums exact and never overflows at the bounds", () => {
    const s = solve(Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => ((i + j) % 2) * PPM)), 2000);
    expect(s.row.reduce((a, b) => a + b, 0)).toBe(PPM);
    expect(s.col.reduce((a, b) => a + b, 0)).toBe(PPM);
  });
  it("is deterministic", () => {
    const m = [[1, 999999], [999999, 1]];
    expect(solve(m, 100)).toEqual(solve(m, 100));
  });
});

describe("refusals", () => {
  const bad: [string, () => unknown][] = [
    ["kind", () => solve([[1]], 1, "x" as SolverKind)],
    ["zero iterations", () => solve([[1]], 0)],
    ["fractional iterations", () => solve([[1]], 1.5)],
    ["too many iterations", () => solve([[1]], MAX_ITERATIONS + 1)],
    ["empty", () => solve([])],
    ["no columns", () => solve([[]])],
    ["too many rows", () => solve(new Array(MAX_DIMENSION + 1).fill([1]))],
    ["too many columns", () => solve([new Array(MAX_DIMENSION + 1).fill(1)])],
    ["ragged", () => solve([[1, 2], [3]])],
    ["not a row", () => solve([[1], 5 as unknown as number[]])],
    ["not a matrix", () => solve(5 as unknown as number[][])],
    ["not a first row", () => solve([5 as unknown as number[]])],
    ["negative", () => solve([[-1]])],
    ["above 10^6", () => solve([[PPM + 1]])],
    ["float", () => solve([[0.5]])],
    ["string", () => solve([["1" as unknown as number]])],
  ];
  for (const [name, f] of bad) it(`refuses ${name}`, () => expect(f).toThrow(SolverError));
});
