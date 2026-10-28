import { describe, expect, it } from "vitest";
import { acceptedDraws, critDraw, type Draw, recordingRandom } from "../src/mechanics-random.ts";

// The recording random source of the damage probe (ADR-016 section 2): it
// answers the damage roll and the crit chance, records everything, and a run is
// accepted only if its draws are exactly the expected ones.

describe("recordingRandom", () => {
  it("answers random(16) with the roll index and records the call", () => {
    const source = recordingRandom({ roll: 7, chance: false });
    expect(source.random(16)).toBe(7);
    expect(source.draws).toEqual([{ kind: "random", args: [16] }]);
  });

  it("answers a crit chance with the fixed outcome and records the exact pair", () => {
    const yes = recordingRandom({ roll: 0, chance: true });
    expect(yes.randomChance(1, 24)).toBe(true);
    expect(recordingRandom({ roll: 0, chance: false }).randomChance(1, 24)).toBe(false);
    expect(yes.draws).toEqual([{ kind: "randomChance", numerator: 1, denominator: 24 }]);
  });

  it("changes its answers on request, for the next calls only", () => {
    const source = recordingRandom({ roll: 1, chance: true });
    expect(source.random(16)).toBe(1);
    source.answer(9, false);
    expect(source.random(16)).toBe(9);
    expect(source.randomChance(1, 24)).toBe(false);
    expect(source.draws).toHaveLength(3);
    expect(() => source.answer(16, true)).toThrow(RangeError);
  });

  it("records any other draw as unexpected instead of answering it with a guess", () => {
    const source = recordingRandom({ roll: 3, chance: true });
    expect(source.random(100)).toBe(0);
    expect(source.random(2, 5)).toBe(0);
    expect(source.random()).toBe(0);
    const engine = source as unknown as { sample(items: number[]): number; shuffle(items: number[]): void };
    expect(engine.sample([9, 8])).toBe(0);
    engine.shuffle([1, 2]);
    expect(source.draws.map((draw) => draw.kind)).toEqual(["random", "random", "random", "other", "other"]);
    expect(source.draws[3]).toEqual({ kind: "other", name: "sample" });
    expect(acceptedDraws(source.draws, true)).toBe(false);
  });

  it("refuses a roll outside 0 to 15", () => {
    for (const roll of [-1, 16, 1.5]) expect(() => recordingRandom({ roll, chance: false })).toThrow(RangeError);
  });
});

describe("acceptedDraws", () => {
  const roll: Draw = { kind: "random", args: [16] };
  const chance: Draw = { kind: "randomChance", numerator: 1, denominator: 24 };

  it("accepts one roll, or nothing (a fixed-damage move, an immunity)", () => {
    expect(acceptedDraws([roll], false)).toBe(true);
    expect(acceptedDraws([], false)).toBe(true);
    expect(acceptedDraws([], true)).toBe(true);
  });

  it("accepts a crit draw before the roll only in the run that observes the chance", () => {
    expect(acceptedDraws([chance, roll], true)).toBe(true);
    expect(acceptedDraws([chance], true)).toBe(true);
    expect(acceptedDraws([chance, roll], false)).toBe(false);
    expect(acceptedDraws([roll, chance], true)).toBe(false);
  });

  it("refuses a second roll, another range, a second chance and any other member", () => {
    expect(acceptedDraws([roll, roll], false)).toBe(false);
    expect(acceptedDraws([{ kind: "random", args: [100] }], false)).toBe(false);
    expect(acceptedDraws([{ kind: "random", args: [] }], false)).toBe(false);
    expect(acceptedDraws([{ kind: "random", args: [16, 1] }], false)).toBe(false);
    expect(acceptedDraws([chance, chance, roll], true)).toBe(false);
    expect(acceptedDraws([{ kind: "other", name: "sample" }], false)).toBe(false);
  });
});

describe("critDraw", () => {
  it("is the chance pair the engine passed, or null when it made no crit draw", () => {
    expect(critDraw([{ kind: "randomChance", numerator: 1, denominator: 8 }, { kind: "random", args: [16] }])).toEqual({ numerator: 1, denominator: 8 });
    expect(critDraw([{ kind: "random", args: [16] }])).toBeNull();
    expect(critDraw([])).toBeNull();
  });
});
