import { describe, expect, it } from "vitest";
import { checkKey, drawSeed, engineRng, pilotSeed, StreamError } from "../src/stream.ts";
import { engine, KEY } from "./support.ts";

const stream = { config_hash: "a".repeat(64), plan_id: "plan-1" };

describe("the draw seeds (ADR-013 section 2)", () => {
  it("pins the seed derivation: HMAC-SHA256 of the JCS preimage, 32 hex digits", () => {
    // The value was computed independently in Python (hmac.new(bytes.fromhex(key), jcs, sha256)). If this changes, every dataset regenerates differently: it is a format change.
    expect(drawSeed(KEY, stream, 0)).toBe("sodium,42fcd7774d1bf389c30452dfb9900eb0");
  });

  it("differs by counter, plan, config, key and purpose, and is deterministic", () => {
    const seeds = new Set([
      drawSeed(KEY, stream, 0),
      drawSeed(KEY, stream, 1),
      drawSeed(KEY, { ...stream, plan_id: "plan-2" }, 0),
      drawSeed(KEY, { ...stream, config_hash: "b".repeat(64) }, 0),
      drawSeed("22".repeat(32), stream, 0),
      pilotSeed(KEY, stream.config_hash, 0),
    ]);
    expect(seeds.size).toBe(6);
    expect(drawSeed(KEY, stream, 7)).toBe(drawSeed(KEY, stream, 7));
    for (const seed of seeds) expect(seed).toMatch(/^sodium,[0-9a-f]{32}$/);
  });

  it("keeps the pilot stream disjoint from every plan's draw stream and independent of the plan", () => {
    expect(pilotSeed(KEY, stream.config_hash, 5)).not.toBe(drawSeed(KEY, stream, 5));
    expect(pilotSeed(KEY, stream.config_hash, 5)).toBe(pilotSeed(KEY, stream.config_hash, 5));
  });

  it("refuses a bad counter and a bad key", () => {
    for (const k of [-1, 1.5, Number.NaN, 2 ** 53]) expect(() => drawSeed(KEY, stream, k)).toThrow(StreamError);
    expect(() => pilotSeed(KEY, stream.config_hash, -1)).toThrow(/non-negative safe integer/);
    expect(() => checkKey("abc", "the key")).toThrow(/64 lowercase hex digits/);
    expect(() => checkKey("AB".repeat(32), "the key")).toThrow(StreamError);
    expect(checkKey(KEY, "the key")).toBe(KEY);
  });
});

describe("engineRng", () => {
  it("is the engine's generator: the same seed gives the same picks", () => {
    const pick = (): number[] => {
      const rng = engineRng(engine, drawSeed(KEY, stream, 3));
      return Array.from({ length: 20 }, () => rng.below(1000));
    };
    expect(pick()).toEqual(pick());
    expect(new Set(pick()).size).toBeGreaterThan(10);
  });

  it("picks every outcome about equally often, for a bound that does not divide 2^32", () => {
    const rng = engineRng(engine, "sodium,0f0e0d0c0b0a09080706050403020100");
    const n = 6;
    const draws = 60_000;
    const seen = new Array<number>(n).fill(0);
    for (let k = 0; k < draws; k++) {
      const pick = rng.below(n);
      seen[pick] = (seen[pick] as number) + 1;
    }
    // chi-square with 5 degrees of freedom: 20.5 is the 0.001 point
    const expected = draws / n;
    const chi = seen.reduce((sum, c) => sum + (c - expected) ** 2 / expected, 0);
    expect(chi).toBeLessThan(20.5);
  });

  it("rejects the words that would favour small outcomes, for a bound near 2^32", () => {
    const rng = engineRng(engine, "sodium,000102030405060708090a0b0c0d0e0f");
    const n = 3_000_000_000;
    const picks = Array.from({ length: 400 }, () => rng.below(n));
    expect(Math.max(...picks)).toBeLessThan(n);
    // a modulo pick of a 32-bit word would put twice the weight on the first 1.29e9 outcomes
    const low = picks.filter((p) => p < 1_294_967_296).length / picks.length;
    expect(low).toBeGreaterThan(0.3);
    expect(low).toBeLessThan(0.56);
  });

  it("refuses a bound it cannot pick below, and handles the largest", () => {
    const rng = engineRng(engine, "sodium,000102030405060708090a0b0c0d0e0f");
    for (const n of [0, -1, 1.5, 2 ** 32 + 1]) expect(() => rng.below(n)).toThrow(StreamError);
    expect(rng.below(1)).toBe(0);
    expect(rng.below(2 ** 32)).toBeLessThan(2 ** 32);
  });
});
