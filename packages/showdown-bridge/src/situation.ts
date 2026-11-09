// The situational reading of one own slot at a move request, from the `probe-damage/2` arrays (threats, speed relations and
// foe priorities, all on ADR-018's assumed opponent sets): the ONE definition of "Threatened" that the situational move-use
// diagnostic (#1283) counts and `damage-aware@v4` (#1329) scores by. Pure; it asks nothing and writes no game rule.
//
//   Threatened      some REVEALED foe damaging move reaches the slot's Pokemon's HP with at least one no-crit roll (`ohko_rolls > 0`,
//                   or `max_ppm >= 1,000,000` when the KO counts are null: a defender on `ko_blockers`) and that foe acts first: the foe
//                   move's priority is above 0, or equal to 0 and the speed relation is `foe_first` (the own action is taken at default
//                   priority 0; a tie is not "first").
//   Safe            no revealed foe move reaches its HP at all.
//   Exposed slower  reachable, but every reaching foe acts after it.
// An unsupported or null probe row never threatens; nothing is guessed.

import type { CandidateFeatures2 } from "./candidate-features-2.ts";

export const SITUATIONS = ["threatened", "safe", "exposed_slower"] as const;
export type Situation = (typeof SITUATIONS)[number];

/** The arrays of `probe-damage/2` a situation is read from. */
export type SituationRows = Pick<CandidateFeatures2, "threats" | "speeds" | "foe_priorities">;

/** One foe move that reaches an own Pokemon's HP; `fallback` when the reading rests on `max_ppm` because the probe gave no KO counts. */
export interface Reach {
  foe: number;
  move: string;
  fallback: boolean;
}

/** Called when a row the reading needs is missing or unsupported (the diagnostic counts these). */
export type Unknown = (kind: "foe_priority" | "speed_relation") => void;

/** The revealed foe moves that reach own position `position` (an active slot or a benched Pokemon a switch candidate names). */
export const reaches = (rows: SituationRows, position: number): Reach[] =>
  rows.threats
    .filter((t) => t.own === position && !t.mega && t.kind === "computed" && (t.ohko_rolls === null ? t.max_ppm >= 1_000_000 : t.ohko_rolls > 0))
    .map((t) => ({ foe: t.foe, move: t.move, fallback: (t as { ohko_rolls: number | null }).ohko_rolls === null }));

/** True when foe `foe`'s `move` acts before own position `position`'s default-priority action. */
export function foeFirst(rows: SituationRows, position: number, foe: number, move: string, unknown?: Unknown): boolean {
  const row = rows.foe_priorities.find((p) => p.foe === foe && p.move === move);
  let priority = 0;
  if (row === undefined || row.kind !== "computed") unknown?.("foe_priority");
  else priority = row.priority;
  if (priority !== 0) return priority > 0;
  const speed = rows.speeds.find((s) => s.own === position && !s.mega && s.foe === foe);
  if (speed === undefined || speed.kind !== "computed") {
    unknown?.("speed_relation");
    return false;
  }
  return speed.relation === "foe_first";
}

export interface SlotSituation {
  situation: Situation;
  /** The reaching foe moves that act first (non-empty exactly when `threatened`). */
  first: Reach[];
}

/** Classify own position `position` (see the head of this file). */
export function classifySlot(rows: SituationRows, position: number, unknown?: Unknown): SlotSituation {
  const reached = reaches(rows, position);
  const first = reached.filter((r) => foeFirst(rows, position, r.foe, r.move, unknown));
  return { situation: reached.length === 0 ? "safe" : first.length > 0 ? "threatened" : "exposed_slower", first };
}

/** The largest mean share (ppm of its HP) a revealed foe move takes of own position `position`: its threat share. Unsupported rows count 0. */
export const threatShare = (rows: SituationRows, position: number): number =>
  rows.threats.reduce((best, t) => (t.own === position && !t.mega && t.kind === "computed" ? Math.max(best, t.mean_ppm) : best), 0);
