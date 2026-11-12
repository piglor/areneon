import { describe, expect, it } from "vitest";
import { MechanicsProbeError } from "../src/mechanics-error.ts";
import { draw, firstTarget, need, RandomDraw, refusingSource } from "../src/mechanics-guard.ts";

// The random-source guards the scenario probes share (ADR-016 section 2), without an engine.

describe("the random-source guards", () => {
  it("a draw is a RandomDraw", () => {
    expect(() => draw()).toThrow(RandomDraw);
  });

  it("answers target resolution from the first candidate and refuses every other draw", () => {
    expect(firstTarget.sample(["a", "b"])).toBe("a");
    expect(firstTarget.random()).toBe(0);
    for (const refused of [firstTarget.randomChance, firstTarget.shuffle, firstTarget.getSeed, firstTarget.setSeed, firstTarget.clone]) expect(() => refused()).toThrow(RandomDraw);
  });

  it("replaces every method of a source with a refusal and leaves its data", () => {
    const source = { seed: [1, 2], random: () => 7 };
    const refusing = refusingSource(source) as typeof source;
    expect(refusing.seed).toEqual([1, 2]);
    expect(() => refusing.random()).toThrow(RandomDraw);
  });

  it("a declared state the engine refuses is a probe error", () => {
    expect(() => need(true, "fine")).not.toThrow();
    expect(() => need(false, "refused")).toThrow(MechanicsProbeError);
  });
});
