import { describe, expect, it } from "vitest";
import { type DamageAnswer, PPM, actsBefore, canKnockOut, critChancePpm, damageValue, discounted, rollsValue, type SpeedOrderAnswer } from "../src/damage-aware-score.ts";

const rolls = (fill: (index: number) => number): DamageAnswer["no_crit"] => Array.from({ length: 16 }, (_, index) => fill(index)) as DamageAnswer["no_crit"];
const answer = (change: Partial<DamageAnswer> = {}): DamageAnswer => ({
  kind: "damage",
  no_crit: rolls(() => 25),
  crit: rolls(() => 50),
  crit_chance: { kind: "never" },
  hp_current: 100,
  hp_max: 100,
  ko: { kind: "unsupported", reason: "ko_state_change" },
  ...change,
});
const scoring = { ko_weight_ppm: 1_000_000, turn_order_discount_ppm: 500_000, crit_weight_ppm: 1_000_000 };

describe("rollsValue (hand-worked integers)", () => {
  it("is the mean of min(roll, hp) / hp in ppm, floored", () => {
    // 16 rolls of 25 into 100 hp: 25%
    expect(rollsValue(rolls(() => 25), 100, 0)).toBe(250_000);
    // rolls 1..16 into 100 hp: sum 136 / 1600 = 85,000 ppm
    expect(rollsValue(rolls((i) => i + 1), 100, 0)).toBe(85_000);
    // floors: 1 damage in 3 hp on every roll is 333,333 ppm
    expect(rollsValue(rolls(() => 1), 3, 0)).toBe(333_333);
  });
  it("caps each roll at the target's hit points and adds the KO share of the weight", () => {
    // every roll of 150 into 100 hp: mean 100% and all 16 rolls reach it
    expect(rollsValue(rolls(() => 150), 100, 1_000_000)).toBe(1_000_000 + 1_000_000);
    // 4 of 16 rolls reach 100 hp: KO term is a quarter of the weight
    expect(rollsValue(rolls((i) => (i < 4 ? 100 : 0)), 100, 800_000)).toBe(250_000 + 200_000);
    expect(rollsValue(rolls(() => 0), 100, 1_000_000)).toBe(0);
  });
  it("refuses a vector that is not 16 rolls and a target with no hit points", () => {
    expect(() => rollsValue([1, 2], 100, 0)).toThrow(RangeError);
    expect(() => rollsValue(rolls(() => 1), 0, 0)).toThrow(RangeError);
    expect(() => rollsValue(rolls(() => 1), 1.5, 0)).toThrow(RangeError);
  });
});

describe("critChancePpm and damageValue", () => {
  it("reads the probe's exact chance", () => {
    expect(critChancePpm({ kind: "never" })).toBe(0);
    expect(critChancePpm({ kind: "always" })).toBe(PPM);
    expect(critChancePpm({ kind: "chance", numerator: 1, denominator: 24 })).toBe(41_666);
  });
  it("mixes the crit vector in by chance times the record's weight", () => {
    const sure = answer({ crit_chance: { kind: "always" } });
    expect(damageValue(sure, scoring)).toBe(500_000);
    const quarter = answer({ crit_chance: { kind: "chance", numerator: 1, denominator: 4 } });
    // 0.75 * 250,000 + 0.25 * 500,000
    expect(damageValue(quarter, scoring)).toBe(312_500);
    // half the weight halves the chance
    expect(damageValue(quarter, { ...scoring, crit_weight_ppm: 500_000 })).toBe(281_250);
    // weight 0 leaves crits out
    expect(damageValue(sure, { ...scoring, crit_weight_ppm: 0 })).toBe(250_000);
    expect(damageValue(answer(), scoring)).toBe(250_000);
  });
});

describe("canKnockOut, discounted and actsBefore", () => {
  it("knocks out when some no-crit roll reaches the current hit points", () => {
    expect(canKnockOut(answer())).toBe(false);
    expect(canKnockOut(answer({ no_crit: rolls((i) => (i === 15 ? 100 : 0)) }))).toBe(true);
    expect(canKnockOut(answer({ hp_current: 0 }))).toBe(false);
  });
  it("discounts by the record's share, floored", () => {
    expect(discounted(1_000_001, 500_000)).toBe(500_000);
    expect(discounted(777, 0)).toBe(777);
    expect(discounted(777, PPM)).toBe(0);
  });
  it("compares tie groups: a tie is not acting first", () => {
    const order = (groups: number[][]): SpeedOrderAnswer => ({ kind: "speed_order", order_basis: "turn_start", actions: [], groups }) as unknown as SpeedOrderAnswer;
    expect(actsBefore(order([[1], [0]]), 1, 0)).toBe(true);
    expect(actsBefore(order([[0], [1]]), 1, 0)).toBe(false);
    expect(actsBefore(order([[0, 1]]), 1, 0)).toBe(false);
    expect(() => actsBefore(order([[0]]), 1, 0)).toThrow(RangeError);
  });
});
