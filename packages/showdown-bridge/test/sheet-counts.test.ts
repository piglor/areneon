import { describe, expect, it } from "vitest";
import { MAX_REFUSED_PPM, refusedSharePpm, sheetCountsProblem, sheetCountsValid, sortedByCode, sumSheetCounts } from "../src/sheet-counts.ts";

const counts = (requests: number, refused: number, ...refusals: [string, number][]) => ({ requests, refused, refusals: refusals.map(([code, count]) => ({ code, count })) });

describe("sheet counts (ADR-025 erratum E4 point 5)", () => {
  it("are summed with the refusals merged and sorted by code", () => {
    expect(sumSheetCounts([counts(10, 2, ["b", 1], ["a", 1]), counts(5, 1, ["b", 1])])).toEqual(counts(15, 3, ["a", 1], ["b", 2]));
    expect(sumSheetCounts([])).toEqual(counts(0, 0));
    expect(sortedByCode(new Map([["z", 1], ["a", 2], ["m", 3]])).map((r) => r.code)).toEqual(["a", "m", "z"]);
  });

  it("follow their rules: sorted unique refusals, adding up to refused, at most requests", () => {
    expect(sheetCountsProblem(counts(10, 2, ["a", 1], ["b", 1]))).toBeNull();
    expect(sheetCountsProblem(counts(10, 2, ["b", 1], ["a", 1]))).toMatch(/sorted by code/);
    expect(sheetCountsProblem(counts(10, 2, ["a", 1], ["a", 1]))).toMatch(/sorted by code and unique/);
    expect(sheetCountsProblem(counts(1, 2, ["a", 2]))).toMatch(/more blocks refused than built/);
    expect(sheetCountsProblem(counts(10, 3, ["a", 1]))).toMatch(/do not add up/);
  });

  it("the refused share is rounded down in parts per million, and 1% is the validity limit", () => {
    expect(refusedSharePpm(counts(0, 0))).toBe(0);
    expect(refusedSharePpm(counts(3, 1))).toBe(333333);
    expect(refusedSharePpm(counts(100, 1))).toBe(MAX_REFUSED_PPM);
    expect(refusedSharePpm(counts(100, 2))).toBeGreaterThan(MAX_REFUSED_PPM);
  });

  it("validity is exact: refused * 100 above requests is invalid even when the floored ppm still reads 10,000", () => {
    expect(sheetCountsValid(counts(100, 1))).toBe(true);
    expect(sheetCountsValid(counts(0, 0))).toBe(true);
    expect(sheetCountsValid(counts(100, 2))).toBe(false);
    const edge = counts(9_999_999, 100_000);
    expect(refusedSharePpm(edge)).toBe(MAX_REFUSED_PPM); // the display floors to the limit
    expect(sheetCountsValid(edge)).toBe(false); // the rule does not
  });
});
