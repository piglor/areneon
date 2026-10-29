// The shown-hit-point display and the consistency of a roll vector with two shown values (ADR-025 section 7,
// ticket #1132). `BeliefConfig.damage_evidence.shown_hp_rounding` names the reviewed rule by which the pinned
// engine turns hit points into the percentage an opponent's Pokemon is shown with; the scan below applies it to every
// hit-point value of a maximum, so no inverse formula is written. ADR-018's `assumedHp` rounding is the inverse
// partner (the Champions display is `floor(100 * hp / max)`, which `ceil` inverts), and `inverseRounding` is the one
// place that pairs them; `test/hp-display.test.ts` holds `shownPercent(assumedHp(p, max, inverse(rule)), max, rule) = p`.

import type { DamageAwareConfig } from "./damage-aware-config.ts";

export type ShownRounding = "floor" | "ceil" | "nearest";

/** The `assumedHp` rounding that inverts a display rule. */
export const inverseRounding = (rule: ShownRounding): DamageAwareConfig["hp_rounding"] => (rule === "floor" ? "ceil" : rule === "ceil" ? "floor" : "nearest");

/** The percentage shown for `hp` of `max`: a Pokemon with hit points left never shows 0. */
export function shownPercent(hp: number, max: number, rule: ShownRounding): number {
  if (!Number.isInteger(hp) || !Number.isInteger(max) || max < 1 || hp < 0 || hp > max) throw new RangeError("hit points are integers from 0 to the maximum");
  if (hp === 0) return 0;
  const scaled = 100 * hp;
  const raw = rule === "floor" ? Math.floor(scaled / max) : rule === "ceil" ? Math.ceil(scaled / max) : Math.floor((scaled * 2 + max) / (max * 2));
  return Math.min(100, Math.max(1, raw));
}

/** An inclusive range of hit points. */
export interface HpRange {
  readonly lo: number;
  readonly hi: number;
}

/** Every hit-point value of `max` that shows `percent`, or null when none does. */
export function hpRange(percent: number, max: number, rule: ShownRounding): HpRange | null {
  let lo = -1;
  let hi = -1;
  for (let hp = 0; hp <= max; hp += 1) {
    if (shownPercent(hp, max, rule) !== percent) continue;
    if (lo < 0) lo = hp;
    hi = hp;
  }
  return lo < 0 ? null : { lo, hi };
}

/**
 * True when some roll of `rolls` takes the Pokemon from a value in `before` to one in `after` (hit points never
 * go below 0, so an `after` of 0 needs a roll that reaches the lowest `before`). The rolls are the probe's, unclamped.
 */
export function rollsReach(rolls: readonly number[], before: HpRange, after: HpRange): boolean {
  return rolls.some((roll) => {
    if (after.hi === 0) return before.lo <= roll;
    return Math.max(before.lo, after.lo + roll) <= Math.min(before.hi, after.hi + roll);
  });
}
