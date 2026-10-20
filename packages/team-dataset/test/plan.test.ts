import { describe, expect, it } from "vitest";
import { groupKey, manifestPlan, planQuotas, PlanError, type PlanParameters } from "../src/plan.ts";

// Five labels of 4, 2, 2, 2 and 2 values: the 64 cells of O6. A cell number is
// its label digits, first label most significant; dropping the last label groups
// cells 2c and 2c+1, dropping two groups four, and so on.
const RADICES = [4, 2, 2, 2, 2];
const evenly = (hits: number): number[] => new Array<number>(64).fill(hits);
const edit = (hits: number[], changes: Record<number, number>): number[] => hits.map((h, c) => changes[c] ?? h);

const params = (over: Partial<PlanParameters> = {}): PlanParameters => ({
  cellHits: evenly(600),
  radices: RADICES,
  pilotDraws: 40_000,
  pilotMinHits: 20,
  dMax: 4_000_000,
  nStratumMin: 2,
  sizes: { dev: 100 },
  ...over,
});

describe("the cell merge (D20, ADR-013 section 2)", () => {
  it("merges nothing when every cell is affordable", () => {
    const plan = planQuotas(params());
    expect(plan.strata).toHaveLength(64);
    expect(plan.strata.every((s) => s.level === 0 && s.cells === 1)).toBe(true);
    expect(Array.from(plan.stratumOf)).toEqual(Array.from({ length: 64 }, (_, c) => c));
    expect(manifestPlan(plan).merged_cells).toBeUndefined();
  });

  it("merges a cell the count rule rejects with its neighbours in the last label, siblings included", () => {
    const plan = planQuotas(params({ cellHits: edit(evenly(600), { 5: 3 }) }));
    // cell 5 has 3 < 20 hits: the last label is dropped, cells 4 and 5 become stratum 4
    expect(plan.stratumOf[4]).toBe(4);
    expect(plan.stratumOf[5]).toBe(4);
    expect(plan.strata).toHaveLength(63);
    expect(plan.strata.find((s) => s.stratum === 4)).toMatchObject({ level: 1, cells: 2, pilot_hits: 603 });
    expect(plan.stratumOf[6]).toBe(6);
    expect(manifestPlan(plan).merged_cells).toEqual([{ stratum: 4, cells: [4, 5] }]);
  });

  it("applies the cost rule at its exact boundary: quota * pilot_draws <= d_max * hits", () => {
    // one split, floor 2: a cell's bin is 2, so it needs 2 * 40000 <= d_max * hits
    const at = (hits: number): number => planQuotas(params({ pilotMinHits: 1, dMax: 4000, sizes: { dev: 1 }, cellHits: edit(evenly(600), { 9: hits }) })).stratumOf[9] as number;
    expect(at(20)).toBe(9); // 80000 <= 4000 * 20: affordable
    expect(at(19)).toBe(8); // 80000 > 4000 * 19: merged with cell 8
  });

  it("drops the last label first, then the one before, until the group is affordable", () => {
    // cells 0..3 have no hits: {0,1} and {2,3} are empty at level 1, {0..3} at level 2,
    // and {0..7} (with the hits of 4..7) is the first group that can be filled
    const plan = planQuotas(params({ cellHits: edit(evenly(600), { 0: 0, 1: 0, 2: 0, 3: 0 }) }));
    expect(plan.strata[0]).toMatchObject({ stratum: 0, level: 3, cells: 8, pilot_hits: 2400 });
    expect(Array.from(plan.stratumOf.slice(0, 9))).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 8]);
  });

  it("coarsens along the label order: a cell differing in the first label is never merged before the last labels are dropped", () => {
    const plan = planQuotas(params({ cellHits: edit(evenly(600), { 0: 0 }) }));
    expect(plan.stratumOf[0]).toBe(0);
    expect(plan.stratumOf[1]).toBe(0);
    expect(plan.stratumOf[16]).toBe(16);
  });

  it("is independent of the order of cells that need merging", () => {
    const a = planQuotas(params({ cellHits: edit(evenly(600), { 3: 1, 40: 2 }) }));
    const b = planQuotas(params({ cellHits: edit(evenly(600), { 40: 2, 3: 1 }) }));
    expect(a).toEqual(b);
    expect(a.strata.map((s) => s.stratum)).toContain(2);
    expect(a.strata.map((s) => s.stratum)).toContain(40);
  });

  it("merges further when a re-plan forces a group, and names it by level and stratum", () => {
    const plan = planQuotas(params({ forced: new Set([groupKey(0, 5)]) }));
    expect(plan.stratumOf[5]).toBe(4);
    const again = planQuotas(params({ forced: new Set([groupKey(0, 5), groupKey(1, 4)]) }));
    expect(again.stratumOf[5]).toBe(4);
    expect(again.strata.find((s) => s.stratum === 4)).toMatchObject({ level: 2, cells: 4 });
    expect(groupKey(2, 4)).toBe("2:4");
  });

  it("merges everything into one stratum when that is all that is affordable", () => {
    const plan = planQuotas(params({ cellHits: evenly(10), pilotMinHits: 400 }));
    // 640 hits in all, 320 in half: affordable only as the single group of 64 cells
    expect(plan.strata).toHaveLength(1);
    expect(plan.strata[0]).toMatchObject({ stratum: 0, level: 5, cells: 64, pilot_hits: 640 });
  });

  it("refuses a plan that is infeasible even with every cell merged", () => {
    expect(() => planQuotas(params({ cellHits: evenly(0) }))).toThrow(/even all 64 cells together are not affordable/);
    expect(() => planQuotas(params({ cellHits: evenly(10), dMax: 100, sizes: { dev: 1000 } }))).toThrow(PlanError);
  });
});

describe("the quotas", () => {
  it("are proportional to the pilot share, rounded half up, with a floor of n_stratum_min in every split", () => {
    const plan = planQuotas(params({ radices: [4], cellHits: [30_000, 100, 20, 9_880], sizes: { dev: 1000, validation: 100 } }));
    // 1000 * 30000 / 40000 = 750; 1000 * 100 / 40000 = 2.5 rounds to 3; 1000 * 20 / 40000 = 0.5 rounds to 1, floored to 2
    const quota = (cell: number, split: "dev" | "validation"): number => (plan.strata.find((s) => s.stratum === cell) as { quotas: Record<string, number> }).quotas[split] as number;
    expect(quota(0, "dev")).toBe(750);
    expect(quota(1, "dev")).toBe(3);
    expect(quota(2, "dev")).toBe(2);
    expect(quota(0, "validation")).toBe(75);
    expect(quota(2, "validation")).toBe(2);
    expect(plan.strata.find((s) => s.stratum === 0)?.total).toBe(825);
    expect(plan.splits).toEqual(["dev", "validation"]);
  });

  it("leave out a split with no size, and keep the split order", () => {
    const plan = planQuotas(params({ sizes: { training: 50, dev: 10, validation: 0 } }));
    expect(plan.splits).toEqual(["dev", "training"]);
    expect(manifestPlan(plan).splits.map((s) => s.split)).toEqual(["dev", "training"]);
  });

  it("add up to the manifest's team count per split", () => {
    const plan = planQuotas(params({ cellHits: edit(evenly(600), { 5: 3 }), sizes: { dev: 200, validation: 50 } }));
    for (const split of manifestPlan(plan).splits) {
      expect(split.teams).toBe(split.cells.reduce((n, c) => n + c.quota, 0));
      expect(split.cells.every((c) => c.quota >= 2)).toBe(true);
    }
  });

  it("stay exact for numbers near 2^53", () => {
    const plan = planQuotas(params({ dMax: Number.MAX_SAFE_INTEGER, sizes: { dev: 2 ** 40 }, pilotDraws: 40_000, cellHits: edit(evenly(0), { 0: 40_000 }) }));
    expect(plan.strata[0]?.total).toBe(2 ** 40);
  });
});

describe("planQuotas refuses bad input", () => {
  it.each([
    ["cell counts of the wrong length", { cellHits: [1, 2, 3] }, /3 cell counts for 64 cells/],
    ["more hits than draws", { cellHits: evenly(700) }, /44800 hits from 40000 draws/],
    ["no split", { sizes: {} }, /no split has a team count/],
    ["a fractional size", { sizes: { dev: 1.5 } }, /the size of dev must be an integer/],
    ["a zero pilot", { pilotDraws: 0 }, /pilot_draws must be an integer of at least 1/],
    ["a zero minimum", { nStratumMin: 0 }, /n_stratum_min/],
    ["no minimum hits", { pilotMinHits: 0 }, /pilot_min_hits/],
    ["no budget", { dMax: 0 }, /d_max/],
  ] as const)("%s", (_name, over, message) => {
    expect(() => planQuotas(params(over as Partial<PlanParameters>))).toThrow(message);
  });
});
