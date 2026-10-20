import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { AssignError, assignSplits, type PoolEntry } from "../src/assign-splits.ts";
import { planQuotas, SPLITS } from "../src/plan.ts";

const KEY = "ab".repeat(32);
const OTHER_KEY = "cd".repeat(32);
const hash = (n: number): string => createHash("sha256").update(`team-${n}`).digest("hex");

const plan = planQuotas({
  cellHits: [600, 250, 100, 40],
  radices: [2, 2],
  pilotDraws: 1100,
  pilotMinHits: 5,
  dMax: 1_000_000,
  nStratumMin: 2,
  sizes: { dev: 20, validation: 20, training: 100 },
});

/** A full pool: every stratum holds exactly its bin size of distinct teams. */
function poolOf(p = plan): PoolEntry[] {
  let n = 0;
  return p.strata.flatMap((s) => Array.from({ length: s.total }, () => ({ canonical_team_hash: hash(n++), stratum: s.stratum })));
}

describe("assignSplits", () => {
  it("fills every split's quota in every stratum, and uses each team once", () => {
    const pool = poolOf();
    const out = assignSplits(pool, plan, KEY);
    for (const split of SPLITS) {
      for (const s of plan.strata) expect(out[split].filter((e) => e.stratum === s.stratum)).toHaveLength(s.quotas[split] as number);
    }
    const all = SPLITS.flatMap((split) => out[split].map((e) => e.canonical_team_hash));
    expect(all).toHaveLength(pool.length);
    expect(new Set(all)).toEqual(new Set(pool.map((e) => e.canonical_team_hash)));
  });

  it("keeps the splits disjoint (L1) and lists strata ascending", () => {
    const out = assignSplits(poolOf(), plan, KEY);
    const sets = SPLITS.map((split) => new Set(out[split].map((e) => e.canonical_team_hash)));
    const [dev, validation, training] = sets as [Set<string>, Set<string>, Set<string>];
    expect([...dev].some((h) => validation.has(h) || training.has(h))).toBe(false);
    expect([...validation].some((h) => training.has(h))).toBe(false);
    for (const split of SPLITS) expect(out[split].map((e) => e.stratum)).toEqual(out[split].map((e) => e.stratum).sort((a, b) => a - b));
  });

  it("gives every stratum at least n_stratum_min teams in each split (E1)", () => {
    const out = assignSplits(poolOf(), plan, KEY, 2);
    for (const split of SPLITS) for (const s of plan.strata) expect(out[split].filter((e) => e.stratum === s.stratum).length).toBeGreaterThanOrEqual(2);
  });

  it("is deterministic and does not depend on the order of the pool", () => {
    const pool = poolOf();
    const first = assignSplits(pool, plan, KEY);
    expect(assignSplits(pool, plan, KEY)).toEqual(first);
    expect(assignSplits([...pool].reverse(), plan, KEY)).toEqual(first);
    const shuffled = [...pool].sort((a, b) => (hash(a.stratum) + a.canonical_team_hash < hash(b.stratum) + b.canonical_team_hash ? 1 : -1));
    expect(assignSplits(shuffled, plan, KEY)).toEqual(first);
  });

  it("orders a stratum by the HMAC of its hash: another key deals differently", () => {
    const pool = poolOf();
    const a = assignSplits(pool, plan, KEY);
    const b = assignSplits(pool, plan, OTHER_KEY);
    expect(b.dev.map((e) => e.canonical_team_hash)).not.toEqual(a.dev.map((e) => e.canonical_team_hash));
  });

  it("carries the caller's fields through untouched", () => {
    const pool = poolOf().map((e) => ({ ...e, extra: e.canonical_team_hash.slice(0, 4) }));
    const out = assignSplits(pool, plan, KEY);
    expect(out.dev.every((e) => e.extra === e.canonical_team_hash.slice(0, 4))).toBe(true);
  });

  it("deals only the splits the plan has", () => {
    const two = planQuotas({ cellHits: [600, 250, 100, 40], radices: [2, 2], pilotDraws: 1100, pilotMinHits: 5, dMax: 1_000_000, nStratumMin: 2, sizes: { dev: 20, validation: 20 } });
    const out = assignSplits(poolOf(two), two, KEY);
    expect(out.training).toEqual([]);
    expect(out.dev.length + out.validation.length).toBe(poolOf(two).length);
  });

  describe("refusals", () => {
    it("a key that is not 64 lowercase hex digits", () => {
      for (const bad of ["", "AB".repeat(32), "ab".repeat(31), "zz".repeat(32)]) expect(() => assignSplits(poolOf(), plan, bad)).toThrow(AssignError);
    });
    it("a team in a stratum the plan does not have", () => {
      expect(() => assignSplits([...poolOf(), { canonical_team_hash: hash(9999), stratum: 77 }], plan, KEY)).toThrow(/stratum 77/);
    });
    it("a short or an over-full stratum", () => {
      expect(() => assignSplits(poolOf().slice(1), plan, KEY)).toThrow(/holds/);
      expect(() => assignSplits([...poolOf(), { canonical_team_hash: hash(9999), stratum: 0 }], plan, KEY)).toThrow(/holds/);
    });
    it("a canonical hash twice", () => {
      const pool = poolOf();
      pool[1] = { ...(pool[0] as PoolEntry) };
      expect(() => assignSplits(pool, plan, KEY)).toThrow(/twice/);
    });
    it("a quota below the floor", () => {
      expect(() => assignSplits(poolOf(), plan, KEY, 1000)).toThrow(/n_stratum_min/);
    });
    it("quotas that do not sum to the bin", () => {
      const broken = { ...plan, strata: plan.strata.map((s, k) => (k === 0 ? { ...s, total: s.total + 1 } : s)) };
      expect(() => assignSplits(poolOf(broken), broken, KEY)).toThrow(/sum/);
    });
  });
});
