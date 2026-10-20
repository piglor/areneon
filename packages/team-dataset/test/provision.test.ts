import { describe, expect, it } from "vitest";
import { planQuotas } from "../src/plan.ts";
import { betaFraction, fillProbability, fillQuantile, provision, ProvisionError, regularisedBeta } from "../src/provision.ts";

/** P(T = t) summed from the negative binomial mass function, for a cross-check of the beta form. */
function exactQuantile(q: number, p: number, quantile: number): number {
  const lnFactorial = (n: number): number => {
    let s = 0;
    for (let i = 2; i <= n; i++) s += Math.log(i);
    return s;
  };
  let cumulative = 0;
  for (let t = q; ; t++) {
    cumulative += Math.exp(lnFactorial(t - 1) - lnFactorial(q - 1) - lnFactorial(t - q) + q * Math.log(p) + (t - q) * Math.log(1 - p));
    if (cumulative >= quantile) return t;
  }
}

describe("fill-time quantiles (range provisioning)", () => {
  it("equal the geometric quantile for a bin of one", () => {
    for (const [p, quantile] of [[0.1, 0.99], [0.001, 0.99], [0.2, 0.9]] as const) {
      expect(fillQuantile(1, p * 1e6, 1e6, quantile * 1000)).toBe(Math.ceil(Math.log(1 - quantile) / Math.log(1 - p)));
    }
  });

  it("equal the negative-binomial quantile summed term by term", () => {
    for (const [q, p] of [[5, 0.2], [20, 0.01], [100, 0.05], [300, 0.3]] as const) {
      expect(fillQuantile(q, p * 1e6, 1e6, 990)).toBe(exactQuantile(q, p, 0.99));
    }
  });

  it("is a high quantile, above the mean q / p, and rises with the quantile", () => {
    const mean = 1000 / 0.01;
    expect(fillQuantile(1000, 400, 40_000, 990)).toBeGreaterThan(mean);
    expect(fillQuantile(1000, 400, 40_000, 999)).toBeGreaterThan(fillQuantile(1000, 400, 40_000, 990));
    expect(fillQuantile(1000, 400, 40_000, 500)).toBeLessThan(fillQuantile(1000, 400, 40_000, 990));
  });

  it("is a probability: zero before the quota, rising to one", () => {
    expect(fillProbability(5, 0.2, 4)).toBe(0);
    expect(fillProbability(5, 0.2, 5)).toBeCloseTo(0.2 ** 5, 12);
    expect(fillProbability(5, 0.2, 500)).toBeCloseTo(1, 12);
    expect(regularisedBeta(0, 2, 3)).toBe(0);
    expect(regularisedBeta(1, 2, 3)).toBe(1);
    expect(regularisedBeta(0.9, 2, 30)).toBeCloseTo(1, 9);
  });

  it("survives an exact zero in the continued fraction", () => {
    // 1 - (a + b) x / (a + 1) is exactly 0 for a = 1, b = 3, x = 0.5
    expect(Number.isFinite(betaFraction(1, 3, 0.5))).toBe(true);
  });

  it("refuses a rate that cannot fill or is not a rate", () => {
    expect(() => fillQuantile(2, 0, 100, 990)).toThrow(ProvisionError);
    expect(() => fillQuantile(0, 1, 100, 990)).toThrow(/not a rate/);
    expect(() => fillQuantile(2, 101, 100, 990)).toThrow(/not a rate/);
  });
});

describe("provision", () => {
  it("takes the largest stratum quantile, adds the margin and counts whole ranges", () => {
    const plan = planQuotas({ cellHits: [30_000, 9_000, 1_000], radices: [3], pilotDraws: 40_000, pilotMinHits: 5, dMax: 4_000_000, nStratumMin: 2, sizes: { dev: 3000 } });
    const result = provision(plan, { pilotDraws: 40_000, fillQuantilePermille: 990, rangeMarginPermille: 100, rangeDrawsLog2: 10, dMax: 4_000_000 });
    const rarest = plan.strata[2] as { total: number; pilot_hits: number };
    expect(result.planned_draws).toBe(fillQuantile(rarest.total, rarest.pilot_hits, 40_000, 990));
    expect(result.provisioned_draws).toBe(Math.ceil(result.planned_draws * 1.1));
    expect(result.range_draws).toBe(1024);
    expect(result.exceeds_d_max).toBe(false);
    expect(provision(plan, { pilotDraws: 40_000, fillQuantilePermille: 990, rangeMarginPermille: 100, rangeDrawsLog2: 10, dMax: 100 }).exceeds_d_max).toBe(true);
    expect(result.ranges).toBe(Math.ceil(result.provisioned_draws / 1024));
  });
});
