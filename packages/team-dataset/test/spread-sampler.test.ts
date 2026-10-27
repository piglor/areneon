import { describe, expect, it } from "vitest";
import { drawStatPoints, ProposalError } from "../src/proposal.ts";
import { generatorSpreadSampler } from "../src/spread-sampler.ts";
import { config, vocabulary } from "./support.ts";

/** A seeded uniform source: `below(n)` from a Mulberry32 stream. */
function source(seed: number): (n: bigint) => bigint {
  let a = seed >>> 0;
  return (n) => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return BigInt(((t ^ (t >>> 14)) >>> 0) % Number(n));
  };
}

describe("the generator's Stat Point routine as the SpreadSampler port (ADR-025 section 4)", () => {
  const sampler = generatorSpreadSampler(vocabulary, config);

  it("draws a nature of the vocabulary and Stat Points that add up to the rule table's total within the config's maximum", () => {
    const below = source(7);
    const natures = new Set<string>();
    for (let k = 0; k < 500; k++) {
      const { nature, stat_points } = sampler.other("garchomp", "garchomp", below);
      natures.add(nature);
      const values = Object.values(stat_points);
      expect(values).toHaveLength(6);
      expect(values.reduce((a, b) => a + b, 0)).toBe(vocabulary.rules.ev_limit);
      expect(Math.max(...values)).toBeLessThanOrEqual(config.stat_point_max_per_stat);
      expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    }
    expect([...natures].every((n) => vocabulary.natures.some((x) => x.id === n))).toBe(true);
    expect(natures.size).toBe(vocabulary.natures.length);
  });

  it("is a function of the stream alone", () => {
    const run = (seed: number) => Array.from({ length: 20 }, ((below) => () => sampler.other("a", "b", below))(source(seed)));
    expect(run(3)).toEqual(run(3));
    expect(run(3)).not.toEqual(run(4));
  });

  it("is the routine the proposal calls: the same stream of picks gives the same Stat Points", () => {
    const picks = (seed: number) => {
      const below = source(seed);
      return (n: number): number => Number(below(BigInt(n)));
    };
    const direct = drawStatPoints(picks(11), vocabulary.rules.ev_limit, config.stat_point_max_per_stat);
    const below = source(11);
    // The sampler draws the nature first (one pick), then the Stat Points.
    const viaPort = sampler.other("a", "b", below).stat_points;
    const skip = picks(11);
    skip(vocabulary.natures.length);
    expect(viaPort).toEqual(drawStatPoints(skip, vocabulary.rules.ev_limit, config.stat_point_max_per_stat));
    expect(Object.values(direct).reduce((a, b) => a + b, 0)).toBe(vocabulary.rules.ev_limit);
  });

  it("refuses what the proposal refuses", () => {
    expect(() => generatorSpreadSampler({ ...vocabulary, natures: [] }, config)).toThrow(/no nature/);
    expect(() => generatorSpreadSampler(vocabulary, { stat_point_max_per_stat: 10 })).toThrow(/cannot hold 66 Stat Points/);
    expect(() => drawStatPoints(() => 0, 10, 0)).toThrow(ProposalError);
  });
});
