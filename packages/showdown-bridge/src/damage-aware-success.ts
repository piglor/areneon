// The success chance of a part in `damage-aware@v5` (ADR-022 erratum E-22.5, #1344). The chance is the engine's, read from the
// `outcomes` array of the `probe-outcome/1` builder (the `move_outcome` probe with the player's own channel history declared): the
// ONE definition of "does this move work now, and how likely", shared with the feature set. Pure; it asks nothing and writes no
// game rule.
//
// A part is read under one opposing-action branch: the branch of the targeted foe's first revealed damaging move, else the `switch`
// branch (every foe switches). A part with no row (a target location with no living foe, an unknown move) has no reading and counts
// as certain (1,000,000 ppm) and unknown; an `unsupported` row in the chosen branch counts as certain and unknown too, nothing is guessed.
// A part is `dead` when every one of its branches is computed and gives 0: it cannot work whatever the foes do.

import type { OutcomeRow } from "./candidate-features-outcome.ts";
import { PPM } from "./damage-aware-score.ts";

export interface PartSuccess {
  /** The chance (ppm) the part works under its branch; PPM when there is no reading. */
  ppm: number;
  /** True when no branch of the part can work (every row computed, every `success_ppm` 0). */
  dead: boolean;
  /** False when the reading was missing or `unsupported` (counted by the actor). */
  known: boolean;
}

/** What identifies a part among the rows: its slot, move position, target location (0 for none) and Mega flag. */
export interface PartKey {
  slot: number;
  move: number;
  target: number | null;
  mega: boolean;
}

/** The success reading of one move part. `branchMove` is the targeted foe's first revealed damaging move, or null for the `switch` branch. */
export function partSuccess(outcomes: readonly OutcomeRow[], key: PartKey, branchMove: string | null): PartSuccess {
  const rows = outcomes.filter((row) => row.slot === key.slot && row.move === key.move && row.target === (key.target ?? 0) && row.mega === key.mega);
  if (rows.length === 0) return { ppm: PPM, dead: false, known: false };
  const dead = rows.every((row) => row.kind === "computed" && row.success_ppm === 0);
  const chosen = (branchMove === null ? undefined : rows.find((row) => row.action === "move" && !row.assumed && row.foe_move === branchMove)) ?? rows.find((row) => row.action === "switch");
  if (chosen === undefined || chosen.kind !== "computed") return { ppm: PPM, dead, known: false };
  return { ppm: chosen.success_ppm, dead, known: true };
}

/** A value scaled by a chance in ppm, floored. */
export const scaled = (value: number, ppm: number): number => Math.floor((value * ppm) / PPM);
