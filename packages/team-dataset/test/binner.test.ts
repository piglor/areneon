import { describe, expect, it } from "vitest";
import { Binner, BinError } from "../src/binner.ts";
import type { Accepted } from "../src/draw.ts";

const hash = (n: number): string => n.toString(16).padStart(64, "0");
const draw = (k: number, h: number = k): Accepted => ({ status: "accepted", k, canonical_team_hash: hash(h), team_hash: hash(h), paste: "", labels: [], cell: 0 });
const none = { has: () => false };

describe("binning in counter order (ADR-013 section 2)", () => {
  it("fills each stratum's bin with its first candidates, and calls the rest overflow", () => {
    const binner = new Binner(new Map([[0, 2], [1, 1]]), none);
    expect([binner.offer(draw(0), 0), binner.offer(draw(1), 1), binner.offer(draw(2), 1), binner.offer(draw(3), 0), binner.offer(draw(4), 0)]).toEqual(["binned", "binned", "overflow", "binned", "overflow"]);
    expect(binner.bin(0).map((d) => d.k)).toEqual([0, 3]);
    expect(binner.bin(1).map((d) => d.k)).toEqual([1]);
    expect(binner.full).toBe(true);
    expect(binner.short).toEqual([]);
  });

  it("skips a duplicate of an earlier binned candidate and takes the next one instead", () => {
    const binner = new Binner(new Map([[0, 2]]), none);
    expect(binner.offer(draw(0, 7), 0)).toBe("binned");
    expect(binner.offer(draw(1, 7), 0)).toBe("duplicate");
    expect(binner.offer(draw(2, 8), 0)).toBe("binned");
    expect(binner.bin(0).map((d) => d.k)).toEqual([0, 2]);
  });

  it("skips a team of an earlier released dataset", () => {
    const binner = new Binner(new Map([[0, 1]]), { has: (h) => h === hash(5) });
    expect(binner.offer(draw(0, 5), 0)).toBe("duplicate");
    expect(binner.offer(draw(1, 6), 0)).toBe("binned");
  });

  it("does not count a duplicate of an overflowed candidate: a full bin ignores what comes after", () => {
    const binner = new Binner(new Map([[0, 1]]), none);
    binner.offer(draw(0, 1), 0);
    expect(binner.offer(draw(1, 1), 0)).toBe("overflow");
  });

  it("reports the strata still short, and refuses a stratum it has no bin for", () => {
    const binner = new Binner(new Map([[3, 2], [9, 1]]), none);
    binner.offer(draw(0), 3);
    expect(binner.short).toEqual([3, 9]);
    expect(binner.full).toBe(false);
    binner.offer(draw(1), 9);
    expect(binner.short).toEqual([3]);
    expect(() => binner.offer(draw(2), 4)).toThrow(BinError);
    expect(() => binner.bin(4)).toThrow(/has no bin/);
  });
});

/** mulberry32: a small seeded generator, for a simulation that must repeat. */
function mulberry(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

describe("exactness: a bin is a sample from the proposal conditioned on its stratum", () => {
  // Three strata with very different rates, and a feature X whose mean depends on the stratum.
  // The binner keeps the first `quota` stream members of each stratum, so the mean of X in a
  // bin must be the stratum's own mean, however long the stream ran to fill the other bins.
  const rates = [0.7, 0.25, 0.05];
  const means = [10, 20, 30];
  const replications = 6000;

  /** Mean, standard error and size of X in each bin over `replications` independent streams, and the mean stream length. */
  function simulate(quotas: number[], seed: number): { strata: { mean: number; se: number; n: number }[]; length: number } {
    const rng = mulberry(seed);
    const sums = [0, 0, 0];
    const squares = [0, 0, 0];
    const members = [0, 0, 0];
    let draws = 0;
    for (let r = 0; r < replications; r++) {
      const binner = new Binner(new Map(quotas.map((q, c) => [c, q])), none);
      const x: number[] = [];
      let k = 0;
      for (; !binner.full; k++) {
        const u = rng();
        const stratum = u < (rates[0] as number) ? 0 : u < (rates[0] as number) + (rates[1] as number) ? 1 : 2;
        // X is the stratum's mean plus noise from the same stream
        x.push((means[stratum] as number) + (rng() - 0.5) * 8);
        binner.offer(draw(k), stratum);
      }
      draws += k;
      for (let c = 0; c < 3; c++) {
        for (const d of binner.bin(c)) {
          const v = x[d.k] as number;
          sums[c] = (sums[c] as number) + v;
          squares[c] = (squares[c] as number) + v * v;
          members[c] = (members[c] as number) + 1;
        }
      }
    }
    const rows = [0, 1, 2].map((c) => {
      const n = members[c] as number;
      const mean = (sums[c] as number) / n;
      return { mean, se: Math.sqrt(((squares[c] as number) / n - mean * mean) / n), n };
    });
    return { strata: rows, length: draws / replications };
  }

  it("gives every stratum's bin the stratum's conditional mean, although the rare stratum sets the stream length", () => {
    const result = simulate([3, 3, 3], 20260607);
    result.strata.forEach((row, c) => {
      expect(row.n).toBe(3 * replications);
      expect(Math.abs((row.mean - (means[c] as number)) / row.se), `stratum ${c}: mean ${row.mean}`).toBeLessThan(4);
    });
    // the stream waits for the rare stratum: about 3 / 0.05 = 60 draws, far more than the common strata need
    expect(result.length).toBeGreaterThan(50);
  });

  it("does not change a bin's mean when another bin's size changes the stream length", () => {
    const short = simulate([3, 3, 1], 11);
    const long = simulate([3, 3, 9], 12);
    expect(long.length).toBeGreaterThan(2 * short.length);
    for (const c of [0, 1]) {
      const a = short.strata[c] as { mean: number; se: number };
      const b = long.strata[c] as { mean: number; se: number };
      expect(Math.abs(a.mean - b.mean) / Math.hypot(a.se, b.se)).toBeLessThan(4);
    }
  });
});
