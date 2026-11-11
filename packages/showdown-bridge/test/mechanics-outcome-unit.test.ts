import { describe, expect, it } from "vitest";
import { chancePpm, HitStepReached, hitChance, reachedOf } from "../src/mechanics-outcome.ts";
import type { Draw } from "../src/mechanics-random.ts";
import { MechanicsResultError, plainReason } from "../src/mechanics-result.ts";

// The arithmetic and the control flow of the outcome probe that do not need an engine (ADR-022 E-22.5.3).

const chance = (numerator: number, denominator: number): Draw => ({ kind: "randomChance", numerator, denominator });

describe("a chance is the exact product of the engine's own draws, rounded down", () => {
  it("is a million with no draw", () => {
    expect(chancePpm([])).toBe(1_000_000);
  });

  it("multiplies the pairs: 1/3 is 333,333 and 1/9 is 111,111", () => {
    expect(chancePpm([chance(1, 3)])).toBe(333_333);
    expect(chancePpm([chance(1, 3), chance(1, 3)])).toBe(111_111);
    expect(chancePpm([chance(90, 100)])).toBe(900_000);
    expect(chancePpm([chance(70, 100), chance(1, 2)])).toBe(350_000);
  });

  it("caps a pair at certain and ignores a draw that is no chance", () => {
    expect(chancePpm([chance(5, 3)])).toBe(1_000_000);
    expect(chancePpm([{ kind: "random", args: [16] }, chance(1, 4)])).toBe(250_000);
  });
});

describe("the hit chance of the asked target", () => {
  it("is 0 for a target the accuracy step never met or refused, and its own draws' chance otherwise", () => {
    expect(hitChance(undefined)).toBe(0);
    expect(hitChance({ draws: [chance(70, 100)], hit: false })).toBe(0);
    expect(hitChance({ draws: [chance(70, 100)], hit: true })).toBe(700_000);
    expect(hitChance({ draws: [], hit: true })).toBe(1_000_000);
  });
});

describe("how the move step ended", () => {
  it("hands back what the hit step had seen and rethrows anything else", () => {
    const reached = { pre: [chance(1, 3)], targets: null };
    expect(reachedOf(new HitStepReached(reached))).toBe(reached);
    const other = new Error("the engine failed");
    expect(() => reachedOf(other)).toThrow(other);
  });
});

describe("the reasons of a query with no history", () => {
  it("are the probe's own; history_unknown answers only a query that holds an unknown fact", () => {
    expect(plainReason("random_effect")).toBe("random_effect");
    expect(() => plainReason("history_unknown")).toThrow(MechanicsResultError);
  });
});
