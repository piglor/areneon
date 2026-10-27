import { describe, expect, it } from "vitest";
import { OTHER_SPREAD, abilityDist, itemDist, only, spreadDist, spreadKey, uniformDist, weightOf } from "../src/fields.ts";

describe("residual expansion", () => {
  it("spreads an ability list's residual over the options it does not name", () => {
    // a1 3 of 4, residual 1 over one other option: weights scale by n = 1.
    expect(abilityDist([{ id: "a1", weight: 3 }], 1, ["a1", "a2"])).toEqual({ total: 4n, options: [{ id: "a1", weight: 3n }, { id: "a2", weight: 1n }] });
    // two residual options: the named weight is multiplied by 2, each other option has the residual 1: shares 6/8, 1/8, 1/8.
    expect(abilityDist([{ id: "a1", weight: 3 }], 1, ["a1", "a2", "a3"])).toEqual({ total: 8n, options: [{ id: "a1", weight: 6n }, { id: "a2", weight: 1n }, { id: "a3", weight: 1n }] });
  });

  it("drops residual with nowhere to go and unmapped mass", () => {
    expect(abilityDist([{ id: "a1", weight: 3 }], 5, ["a1"])).toEqual({ total: 3n, options: [{ id: "a1", weight: 3n }] });
    expect(abilityDist([], 0, ["a1"]).options).toEqual([]);
  });

  it("items: no item is a named option, every other item takes the residual, left-out stones do not", () => {
    const d = itemDist([{ id: "ix", weight: 2 }], 1, 2, ["betite", "ix", "iy", "iz"], new Set(["betite"]));
    // n = 2 (iy, iz): ix 2*2 = 4, none 1*2 = 2, iy 2, iz 2.
    expect(d.options).toEqual([{ id: null, weight: 2n }, { id: "ix", weight: 4n }, { id: "iy", weight: 2n }, { id: "iz", weight: 2n }]);
    expect(d.total).toBe(10n);
    expect(weightOf(d, "betite")).toBe(0n);
    expect(weightOf(d, null)).toBe(2n);
  });

  it("a single option and a uniform list", () => {
    expect(only("x")).toEqual({ total: 1n, options: [{ id: "x", weight: 1n }] });
    expect(uniformDist(["b", "a", "a"])).toEqual({ total: 2n, options: [{ id: "a", weight: 1n }, { id: "b", weight: 1n }] });
  });

  it("spreads: named entries and one other-spread entry", () => {
    const s = { nature: "jolly", stat_points: { hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 }, weight: 7 };
    expect(spreadKey(s)).toBe("jolly:1-2-3-4-5-6");
    expect(spreadDist([s], 3)).toEqual({ total: 10n, options: [{ id: "jolly:1-2-3-4-5-6", weight: 7n }, { id: OTHER_SPREAD, weight: 3n }] });
    expect(spreadDist([s], 0).options).toHaveLength(1);
  });
});
