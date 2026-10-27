// KO counts of a `damage` result (ADR-016 section 2, slice 5). Integers only,
// never combined into a probability: the counts are how many of the 16 rolls
// (or 256 ordered pairs of two independent rolls) reach the defender's current
// HP. They are taken from the engine's roll vectors as `getDamage` returns
// them, before the HP clamp and before any damage-time effect, so an effect
// that acts when damage is dealt or changes the second hit makes the counts
// `unsupported` (`ko_state_change`) while the vectors are still returned.
// This module holds no game rule: the blockers are the mechanics config's list.

import type { MechanicsConfig } from "./mechanics-config.ts";
import type { MechanicsResult } from "./mechanics-result.ts";

type DamageAnswer = Extract<Extract<MechanicsResult, { kind: "computed" }>["answer"], { kind: "damage" }>;
export type KoCounts = DamageAnswer["ko"];

/** Rolls of a 16-roll vector whose damage is at least `hp`. */
export const countKos = (rolls: readonly number[], hp: number): number => rolls.filter((damage) => damage >= hp).length;

/**
 * Ordered pairs (first roll, second roll) of two independent hits that use the
 * same vector and whose summed damage is at least `hp`: 256 pairs for 16 rolls.
 */
export function countPairKos(rolls: readonly number[], hp: number): number {
  let count = 0;
  for (const first of rolls) for (const second of rolls) if (first + second >= hp) count += 1;
  return count;
}

/** True when the defender's item or ability is on the config's `ko_blockers` list. */
export function koBlocked(blockers: MechanicsConfig["ko_blockers"], defender: { item: string; ability: string }): boolean {
  return blockers.items.includes(defender.item) || blockers.abilities.includes(defender.ability);
}

/**
 * The KO field of a damage answer: the counts, or `unsupported` when the
 * defender holds a blocker. `noCrit` is used for both hits of the two-hit
 * count; `crit` only for the one-hit crit count.
 */
export function koCounts(noCrit: readonly number[], crit: readonly number[], hp: number, blocked: boolean): KoCounts {
  if (blocked) return { kind: "unsupported", reason: "ko_state_change" };
  return { kind: "counts", ohko_no_crit: countKos(noCrit, hp), ohko_crit: countKos(crit, hp), thko_no_crit: countPairKos(noCrit, hp) };
}
