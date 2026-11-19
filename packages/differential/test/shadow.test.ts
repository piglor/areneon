// The shadow sampler (#1195's bottom-k with the size rule as a parameter) and the shadow check's typed divergence.
import { describe, expect, it } from "vitest";
import { key64 } from "../src/hash.ts";
import { LEVEL_1_RULE, LEVEL_2_RULE, ShadowDivergence, shadowCheck, shadowKey, shadowSample, shadowSize } from "../src/shadow.ts";
import { backendOf, oracle, type ToyIn, toyLevel, truth } from "./support.ts";

describe("shadowSize", () => {
  it("is max(1% of the answers, 256) at level 1 and max(5% of the games, 64) at level 2, never above n", () => {
    expect([0, 100, 256, 25_600, 25_601, 30_000].map((n) => shadowSize(n, LEVEL_1_RULE))).toEqual([0, 100, 256, 256, 257, 300]);
    expect([10, 64, 1280, 1281, 2000].map((n) => shadowSize(n, LEVEL_2_RULE))).toEqual([10, 64, 64, 65, 100]);
  });
});

describe("shadowSample", () => {
  const keys = Array.from({ length: 1000 }, (_, i) => `q${i}`);
  it("takes the smallest keys, whatever the order the answers came in", () => {
    const sample = shadowSample(keys, { share_ppm: 10_000, min: 5 });
    expect(sample).toHaveLength(10);
    const expected = keys
      .map((k, i) => ({ key: shadowKey(k), i }))
      .sort((a, b) => (a.key < b.key ? -1 : 1))
      .slice(0, 10)
      .map((x) => x.i)
      .sort((a, b) => a - b);
    expect(sample).toEqual(expected);
    const reversed = shadowSample([...keys].reverse(), { share_ppm: 10_000, min: 5 }).map((p) => keys[999 - p]);
    expect(reversed.sort()).toEqual(sample.map((p) => keys[p]).sort());
  });
  it("breaks equal keys by position and keys by SHA-256 of the JCS of the key and the purpose", () => {
    expect(shadowSample(["a", "a", "a"], { share_ppm: 0, min: 2 })).toEqual([0, 1]);
    expect(shadowKey("q1")).toBe(key64({ key: "q1", purpose: "differential-shadow/1" }));
  });
});

describe("shadowCheck", () => {
  const inputs: ToyIn[] = Array.from({ length: 50 }, (_, n) => ({ n, entities: [`move:m${n}`], reads: [] }));
  const key = (input: ToyIn) => input.n;
  it("passes when the backend equals the oracle on the sample", () => {
    expect(shadowCheck(toyLevel, oracle, oracle, inputs, key, { share_ppm: 0, min: 8 })).toHaveLength(8);
  });
  it("throws a typed divergence that names the entities to revoke and the corpus positions", () => {
    const sample = shadowSample(inputs.map(key), { share_ppm: 0, min: 8 });
    const bad = sample[3] as number;
    const backend = backendOf("bad", (batch) => batch.map((input) => (input.n === bad ? { ok: true as const, result: { for: input.n, value: 0 } } : truth(input))));
    let caught: unknown;
    try {
      shadowCheck(toyLevel, oracle, backend, inputs, key, { share_ppm: 0, min: 8 });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ShadowDivergence);
    expect((caught as ShadowDivergence).entities).toEqual([`move:m${bad}`]);
    expect((caught as ShadowDivergence).differences.map((d) => d.position)).toEqual([bad]);
  });
  it("throws on an extra result too", () => {
    const extra = backendOf("extra", (batch) => [...batch.map(truth), { ok: true, result: { for: -1, value: 0 } }]);
    expect(() => shadowCheck(toyLevel, oracle, extra, inputs, key, LEVEL_1_RULE)).toThrow(ShadowDivergence);
  });
});
