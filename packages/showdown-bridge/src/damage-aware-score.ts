// The damage-aware scorer's integer arithmetic (ADR-018 section 4). Every value is an integer in
// parts per million (ppm) and every division floors, so the same probe answer gives the same score
// on every machine. The numbers come from the probe's answer and the reviewed weights of the
// `DamageAwareConfig`; nothing here knows a game rule.

import type { DamageAwareConfig } from "./damage-aware-config.ts";
import type { MechanicsResult } from "./mechanics-result.ts";

export const PPM = 1_000_000;
const ROLLS = 16;

export type DamageAnswer = Extract<Extract<MechanicsResult, { kind: "computed" }>["answer"], { kind: "damage" }>;
export type SpeedOrderAnswer = Extract<Extract<MechanicsResult, { kind: "computed" }>["answer"], { kind: "speed_order" }>;

/**
 * One damage vector's value against `hp` current hit points: the mean over the 16 rolls of
 * `min(roll, hp) / hp`, plus the KO term, the weight times the share of rolls that reach `hp`.
 */
export function rollsValue(rolls: readonly number[], hp: number, koWeightPpm: number): number {
  if (rolls.length !== ROLLS || !Number.isInteger(hp) || hp < 1) throw new RangeError("a damage vector has 16 rolls and the target has at least 1 hit point");
  let dealt = 0;
  let reaching = 0;
  for (const roll of rolls) {
    dealt += Math.min(roll, hp);
    if (roll >= hp) reaching += 1;
  }
  return Math.floor((dealt * PPM) / (ROLLS * hp)) + Math.floor((koWeightPpm * reaching) / ROLLS);
}

/** The probe's exact crit chance in ppm (`never` 0, `always` 1,000,000, else the engine's own fraction, floored). */
export function critChancePpm(chance: DamageAnswer["crit_chance"]): number {
  if (chance.kind === "never") return 0;
  if (chance.kind === "always") return PPM;
  return Math.floor((chance.numerator * PPM) / chance.denominator);
}

/**
 * The value of one move into one target: the no-crit vector, and the crit vector in the share of
 * the probe's crit chance the record weights (`crit_weight_ppm`; 0 leaves crits out).
 */
export function damageValue(answer: DamageAnswer, scoring: DamageAwareConfig["scoring"]): number {
  const plain = rollsValue(answer.no_crit, answer.hp_current, scoring.ko_weight_ppm);
  const weight = Math.floor((critChancePpm(answer.crit_chance) * scoring.crit_weight_ppm) / PPM);
  if (weight === 0) return plain;
  const crit = rollsValue(answer.crit, answer.hp_current, scoring.ko_weight_ppm);
  return Math.floor(((PPM - weight) * plain + weight * crit) / PPM);
}

/** True when some no-crit roll reaches the target's current hit points: the move can knock it out. */
export const canKnockOut = (answer: DamageAnswer): boolean => answer.hp_current >= 1 && answer.no_crit.some((roll) => roll >= answer.hp_current);

/** `value` after the turn-order discount (`turn_order_discount_ppm`). */
export const discounted = (value: number, discountPpm: number): number => Math.floor((value * (PPM - discountPpm)) / PPM);

/** True when action `first` is in a strictly earlier tie group than action `second` (a tie is not "acts first"). */
export function actsBefore(answer: SpeedOrderAnswer, first: number, second: number): boolean {
  const group = (action: number): number => answer.groups.findIndex((members) => members.includes(action));
  const a = group(first);
  const b = group(second);
  if (a < 0 || b < 0) throw new RangeError("a speed order names every action in some group");
  return a < b;
}
