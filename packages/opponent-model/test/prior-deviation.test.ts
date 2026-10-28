import { describe, expect, it } from "vitest";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { designDeviation } from "../src/prior-deviation.ts";

// Ten sets of four slots. `ma` is in every set; mb is included with 9/10 and mc with 1/10; exactly two moves per
// set (20 empty slots over 10 sets), so the design draws one move of {mb, mc} with odds 9 : 1/9 = 81 : 1.
// The design's inclusion is 81/82 for mb and 1/82 for mc; the stated ones are 9/10 and 1/10; both differ by
// 72/820 = 87804.87... ppm, rounded down.
const list = { denominator: 10, slots: 4, entries: [{ id: "ma", inclusion: 10 }, { id: "mb", inclusion: 9 }, { id: "mc", inclusion: 1 }], empty_slots: 20, residual: 0, unmapped: 0 };
const prior = (moves: typeof list): UsagePrior =>
  ({ species: [{ id: "sp", denominator: 1, formes: [{ id: "fo", weight: 1, moves }] }] }) as unknown as UsagePrior;

describe("designDeviation", () => {
  it("is the gap between the conditional Poisson design and the stated marginals, in ppm rounded down", () => {
    expect(designDeviation(prior(list), new Map([["fo", ["ma", "mb", "mc"]]]))).toEqual([{ species: "sp", forme: "fo", max_ppm: 87804, move: "mb" }]);
  });

  it("is zero when the design reproduces the marginals", () => {
    // Odds all 1 and a mean of two moves: the design includes each of mb, mc with 1/2, as stated.
    const even = { ...list, entries: [{ id: "ma", inclusion: 10 }, { id: "mb", inclusion: 5 }, { id: "mc", inclusion: 5 }] };
    expect(designDeviation(prior(even), new Map([["fo", ["ma", "mb", "mc"]]]))[0]?.max_ppm).toBe(0);
  });

  it("reports a forme with no named move as zero and refuses a forme with no pool", () => {
    expect(designDeviation(prior({ ...list, entries: [], residual: 20 }), new Map([["fo", ["ma", "mb"]]]))[0]).toEqual({ species: "sp", forme: "fo", max_ppm: 0, move: null });
    expect(() => designDeviation(prior(list), new Map())).toThrow(RangeError);
  });

  it("refuses a forme whose model cannot hold the empty evidence", () => {
    // A move named with positive inclusion but absent from the pool is not a candidate; five forced moves overflow the four slots.
    const forced = { denominator: 1, slots: 4, entries: ["a", "b", "c", "d", "e"].map((id) => ({ id, inclusion: 1 })), empty_slots: 0, residual: 0, unmapped: 0 };
    expect(() => designDeviation(prior(forced as unknown as typeof list), new Map([["fo", ["a", "b", "c", "d", "e"]]]))).toThrow(RangeError);
  });
});
