import { describe, expect, it } from "vitest";
import { assumedHp } from "../src/damage-aware-assumed-set.ts";
import { type ShownRounding, hpRange, inverseRounding, rollsReach, shownPercent } from "../src/hp-display.ts";

const RULES: ShownRounding[] = ["floor", "ceil", "nearest"];

describe("the shown hit-point display (ticket #1132)", () => {
  it("applies the rule to every hit-point value, and a Pokemon with hit points left never shows 0", () => {
    expect(shownPercent(0, 200, "floor")).toBe(0);
    expect(shownPercent(1, 400, "floor")).toBe(1);
    expect(shownPercent(1, 400, "nearest")).toBe(1);
    expect([shownPercent(115, 200, "floor"), shownPercent(115, 200, "ceil"), shownPercent(115, 200, "nearest")]).toEqual([57, 58, 58]);
    expect(shownPercent(200, 200, "ceil")).toBe(100);
    expect(() => shownPercent(201, 200, "floor")).toThrow(RangeError);
    expect(() => shownPercent(1, 0, "floor")).toThrow(RangeError);
  });

  it("the range of a percentage is every hit-point value that shows it", () => {
    expect(hpRange(100, 158, "floor")).toEqual({ lo: 158, hi: 158 });
    expect(hpRange(0, 158, "floor")).toEqual({ lo: 0, hi: 0 });
    const r = hpRange(57, 200, "floor");
    expect(r).toEqual({ lo: 114, hi: 115 });
    expect(hpRange(57, 100, "floor")).toEqual({ lo: 57, hi: 57 });
    // 300 hit points show every percentage at least once; a maximum of 7 does not show 50.
    expect(hpRange(50, 7, "floor")).toBeNull();
  });

  it.each(RULES)("%s: ADR-018's assumedHp rounding inverts the display for every percentage and every maximum from 100 to 800", (rule) => {
    for (let max = 100; max <= 800; max += 1) {
      for (let percent = 1; percent <= 100; percent += 1) {
        const hp = assumedHp(percent, max, inverseRounding(rule));
        expect(shownPercent(hp, max, rule), `${rule} ${percent}/${max}`).toBe(percent);
      }
    }
  });
});

describe("a roll vector against two shown values", () => {
  const rolls = [30, 31, 32, 33];
  it("needs some roll that takes some hit-point value of the first range to some value of the second", () => {
    expect(rollsReach(rolls, { lo: 100, hi: 100 }, { lo: 68, hi: 68 })).toBe(true);
    expect(rollsReach(rolls, { lo: 100, hi: 100 }, { lo: 60, hi: 60 })).toBe(false);
    expect(rollsReach(rolls, { lo: 100, hi: 101 }, { lo: 70, hi: 70 })).toBe(true);
    expect(rollsReach(rolls, { lo: 100, hi: 101 }, { lo: 66, hi: 66 })).toBe(false);
  });

  it("a fainted Pokemon needs a roll that reaches the lowest value of the first range", () => {
    expect(rollsReach(rolls, { lo: 33, hi: 40 }, { lo: 0, hi: 0 })).toBe(true);
    expect(rollsReach(rolls, { lo: 34, hi: 40 }, { lo: 0, hi: 0 })).toBe(false);
  });
});
