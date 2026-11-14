// The situational part of `damage-aware@v4`'s score (#1329): what a switch out of a Threatened own active to a safe bench Pokemon
// is worth, and what a stalling move (Protect and its family) by a Threatened own active is worth. v1 and v3 score both 0.
//
// Every input is the probe's or the engine's: the slot's situation and the bench Pokemon's threats are the `probe-damage/2` arrays
// read by `situation.ts` (the same definition of "Threatened" the situational diagnostic counts), the stalling moves are the
// engine's own stall list, and the Protect streak is the world facts' `protect_streak`. The two weights are the record's
// `situational` fields. No game rule is written here.

import { type SituationRows, classifySlot, reaches, threatShare } from "./situation.ts";
import type { SlotPart } from "./damage-aware-choice.ts";
import { PPM, ROLLS } from "./damage-aware-score.ts";

export interface SituationalInput {
  rows: SituationRows;
  /**
   * Per own active slot, the world facts' `protect_streak`; an entry null when the facts are unavailable (then no stalling move scores).
   * The whole list null (`damage-aware@v5`) means no streak rule: a stalling move by a Threatened slot is worth the weight and the
   * engine's success chance scales it instead (`damage-aware-success.ts`).
   */
  streaks: readonly (number | null)[] | null;
  /** The engine's stalling moves. */
  stall: ReadonlySet<string>;
  /** The id of the move a `move N` part of own slot `slot` plays, or null. */
  moveId: (slot: number, index: number) => string | null;
  weights: { switch_value_ppm: number; protect_value_ppm: number };
}

/** The situational score of one slot part of a candidate (0 for everything the record does not value). Integer ppm. */
export function situationalScore(input: SituationalInput, slot: number, part: SlotPart): number {
  if (part.kind === "pass") return 0;
  const { situation } = classifySlot(input.rows, slot);
  if (situation !== "threatened") return 0;
  if (part.kind === "switch") {
    const bench = part.to - 1;
    // Only a bench Pokemon no revealed foe move reaches; its value falls with the share of its HP the foes' moves take.
    if (reaches(input.rows, bench).length > 0) return 0;
    return Math.floor((input.weights.switch_value_ppm * (PPM - threatShare(input.rows, bench))) / PPM);
  }
  const id = input.moveId(slot, part.index);
  if (id === null || !input.stall.has(id)) return 0;
  if (input.streaks === null) return input.weights.protect_value_ppm;
  const streak = input.streaks[slot] ?? null;
  return streak !== null && streak < 1 ? input.weights.protect_value_ppm : 0;
}

/**
 * The threat value of a Threatened slot (`damage-aware@v6`, ADR-022 erratum E-22.6.1): the largest, over the threat rows that make the
 * slot Threatened (`classifySlot(rows, slot).first`, own Mega flag false, computed), of the row's mean share plus `ko_weight_ppm`
 * times its reaching rolls over 16 (no KO term when the probe gave no KO counts). 0 for a slot that is not Threatened. Integer ppm.
 */
export function stallThreat(rows: SituationRows, slot: number, koWeightPpm: number): number {
  const { first } = classifySlot(rows, slot);
  let best = 0;
  for (const reach of first) {
    for (const t of rows.threats) {
      if (t.own !== slot || t.mega || t.kind !== "computed" || t.foe !== reach.foe || t.move !== reach.move) continue;
      const rolls = t.ohko_rolls === null ? 0 : t.ohko_rolls;
      best = Math.max(best, t.mean_ppm + Math.floor((koWeightPpm * rolls) / ROLLS));
    }
  }
  return best;
}

/** The failure charge of a stalling move whose engine chance is `successPpm`: floor(T * (10^6 - s) / 10^6). */
export const stallCharge = (threat: number, successPpm: number): number => Math.floor((threat * (PPM - successPpm)) / PPM);
