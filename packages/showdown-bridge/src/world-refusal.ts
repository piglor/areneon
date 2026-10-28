// The refusals of the scenario world builder (ADR-026 Evidence 1): every world that is not built carries a
// typed reason, the check that failed and the state class involved, and says whether a redraw of the
// opponent's sets can fix it (slice B redraws up to R times per world and counts by check and class).

import type { StateClass, StateItemId } from "./world-writers.ts";

/** Why a world was not built. Every code is counted; none carries a set, a stat or a name from an input. */
export const WORLD_REFUSAL_CODES = [
  "not_move_request",
  "reader_flagged",
  "facts_unavailable",
  "facts_mismatch",
  "own_team_mismatch",
  "opponent_team_mismatch",
  "v1_set_invalid",
  "v1_not_canonical",
  "v1_revealed_fact",
  "v1_item_clause",
  "state_unwritten",
  "start_failed",
  "engine_refused",
  "v2_request",
  "v2_candidates",
  "v3_mismatch",
  "v4_conditions",
  "v4_duration",
  "v5_mismatch",
] as const;
export type WorldRefusalCode = (typeof WORLD_REFUSAL_CODES)[number];

/** The checks of the ADR (V1 to V5). */
export const WORLD_CHECKS = ["V1", "V2", "V3", "V4", "V5"] as const;
export type WorldCheck = (typeof WORLD_CHECKS)[number];

const CHECK_OF: Partial<Record<WorldRefusalCode, WorldCheck>> = {
  v1_set_invalid: "V1",
  v1_not_canonical: "V1",
  v1_revealed_fact: "V1",
  v1_item_clause: "V1",
  v2_request: "V2",
  v2_candidates: "V2",
  v3_mismatch: "V3",
  v4_conditions: "V4",
  v4_duration: "V4",
  v5_mismatch: "V5",
};

export const checkOf = (code: WorldRefusalCode): WorldCheck | null => CHECK_OF[code] ?? null;

export interface WorldRefusal {
  code: WorldRefusalCode;
  /** Fixed words, at most naming a field or a position; never a value from the input. */
  detail: string;
  /** The check that failed (V1 to V5), null for a refusal that is no check. */
  check: WorldCheck | null;
  /**
   * The class of the state item the refusal is about (the record's class: R for a refused item, W or X for a
   * writer that could not write), null when it is evidence about the opponent's unrevealed sets or no state item.
   */
  state_class: StateClass | null;
  /** The state item, for `state_unwritten` and the V2 and V4 refusals that name one. */
  item: StateItemId | null;
  /** Whether drawing the opponent's unrevealed sets again can make the world pass (slice B redraws, bounded). */
  redraw: boolean;
}

export class Refuse extends Error {
  readonly refusal: WorldRefusal;
  constructor(code: WorldRefusalCode, detail: string, extra: { item?: StateItemId; state_class?: StateClass; redraw?: boolean } = {}) {
    super(detail);
    this.refusal = { code, detail, check: checkOf(code), state_class: extra.state_class ?? null, item: extra.item ?? null, redraw: extra.redraw ?? false };
  }
}

export const need = (ok: boolean, code: WorldRefusalCode, detail: string, extra?: { redraw?: boolean }): void => {
  if (!ok) throw new Refuse(code, detail, extra);
};
