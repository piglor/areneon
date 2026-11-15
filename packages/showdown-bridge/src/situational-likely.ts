// The Likely-Threatened tally of the situational move-use diagnostic (ADR-022 erratum E-22.7.10, option 1: report only; the diagnostic
// of #1283, beside the #1274 reading, which it never changes).
//
// LABEL: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure. No gate reads this tally.
//
// Computed from the same stored games, from one seat's OWN view at each move request (never the omniscient log): the `probe-damage/2` rows
// of the decision (the #1274 reading: revealed foe moves, speed relations, priorities) and the likely-move threat rows of
// `probe-outcome-likely/1` (`belief_threats`: moves a foe is LIKELY to hold, from the committed frequency table, each with the table's
// inclusion). The classes are the slot's #1274 situation refined by those rows:
//   likely_threatened   NOT Threatened by revealed moves (Safe or Exposed slower), and some likely row reaches the slot's HP (`ohko_rolls > 0`,
//                       or `max_ppm >= 1,000,000` when the KO counts are null) from a foe that acts first by the speed relation (`foe_first`; a
//                       likely move is priced at priority 0, as the feature set declares)
//   safe_unlikely       Safe by revealed moves AND no likely row reaches the slot at all (the comparator)
//   other               every other slot that is not revealed-Threatened (Exposed slower, or Safe with a likely row that reaches but acts after)
//   revealed_threatened Threatened in the #1274 sense: counted, never in the rates (the #1274 rows own it)
// Phase of the decision: `turn1` (no `|turn|` line above 1 yet), `fresh` (later, and some foe active switched in at the end of the previous
// turn: its world-facts `active_turns` is 1, so nothing of it was revealed by acting), `later` (the rest), `no_facts` for a decision whose
// facts were unavailable and that is not turn 1 (kept out of the rates).
// A refused likely build (`belief_threats_refused`) classifies no slot of that decision (its rows are empty, which would read as safe);
// it is counted. An unsupported or null likely row never threatens. Nothing is guessed.
//
// Flat dotted integer counts under `likely.`:
//   decisions                            move-request decisions both builders answered
//   feature_errors, feature_error.<code> decisions whose likely build failed
//   refused.<code>                       decisions the likelihood table refused
//   slots.<class>.<phase>                per class and phase
//   protect.<class>.<phase>.<offered|chosen>, switch.<class>.<phase>.<offered|chosen>   a stalling move / a `switch` candidate offered for the
//                                        slot, and the choice
//   likely_threatened.n, .inclusion_max_ppm, .inclusion_sum_ppm   over the likely_threatened slots: their count, the sum of each slot's largest
//                                        inclusion of a reaching first-acting row, and of the sum (capped at 1,000,000) of those rows' inclusions
import type { LikelyRow } from "./likely-threats.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";
import { classifySlot, foeFirst, type SituationRows } from "./situation.ts";

export const PHASES = ["turn1", "fresh", "later"] as const;
export type Phase = (typeof PHASES)[number] | "no_facts";
export const LIKELY_CLASSES = ["likely_threatened", "safe_unlikely", "other"] as const;

export type LikelyCounts = Record<string, number>;
const bump = (counts: LikelyCounts, key: string, by = 1): void => void (counts[key] = (counts[key] ?? 0) + by);
const asRecord = (value: unknown): Record<string, unknown> => (typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {});
const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

export interface LikelyInput {
  body: Record<string, unknown>;
  candidates: readonly { choice: string; pending: boolean }[];
  chosen: string;
  /** `probe-damage/2`'s rows of the decision (the #1274 reading's input). */
  base: SituationRows;
  /** The likely set's `belief_threats` and refusal of the same decision. */
  likely: { belief_threats: readonly LikelyRow[]; belief_threats_refused?: string };
  phase: Phase;
  stall: ReadonlySet<string>;
}

type Reaching = { foe: number; inclusion_ppm: number; move: string };

/** A likely row that reaches the slot's HP: some no-crit roll does (`ohko_rolls > 0`), or `max_ppm >= 1,000,000` when the KO counts are null. */
const reachesHp = (r: LikelyRow, position: number): boolean => r.own === position && !r.mega && r.kind === "computed" && (r.ohko_rolls === null ? r.max_ppm >= 1_000_000 : r.ohko_rolls > 0);

/** The likely rows that reach own position `position` and whose foe acts first (speed relation, priority 0). */
const reachingFirst = (rows: readonly LikelyRow[], base: SituationRows, position: number): Reaching[] =>
  rows.filter((r) => reachesHp(r, position)).map((r) => ({ foe: r.foe, inclusion_ppm: r.inclusion_ppm, move: r.move })).filter((r) => foeFirst(base, position, r.foe, r.move));

/** Does any likely row reach own position `position`? Keeps `safe_unlikely` clean. */
const anyReach = (rows: readonly LikelyRow[], position: number): boolean => rows.some((r) => reachesHp(r, position));

/** Add one decision's Likely-Threatened classes and choices to `counts`. */
export function tallyLikelyDecision(counts: LikelyCounts, input: LikelyInput): void {
  const { body, stall, phase } = input;
  const pokemon = asList(asRecord(body.side).pokemon).map(asRecord);
  const active = asList(body.active).map(asRecord);
  const parsed = input.candidates.map((c) => parseJointChoice(c.choice));
  const chosen = parseJointChoice(input.chosen);
  if (chosen.length !== active.length || parsed.some((p) => p.length !== active.length)) throw new Error("a choice has a different number of parts than the request has active Pokemon");
  bump(counts, "likely.decisions");
  if (input.likely.belief_threats_refused !== undefined) {
    bump(counts, `likely.refused.${input.likely.belief_threats_refused}`);
    return;
  }
  const moveId = (slot: number, part: SlotPart): string | null => (part.kind === "move" ? ((asList(asRecord(active[slot]).moves)[part.index - 1] as { id?: string } | undefined)?.id ?? null) : null);
  for (let slot = 0; slot < active.length; slot++) {
    if (/ fnt$/.test(String(asRecord(pokemon[slot]).condition))) continue;
    const { situation } = classifySlot(input.base, slot);
    let cls: (typeof LIKELY_CLASSES)[number] | "revealed_threatened" = "other";
    let first: Reaching[] = [];
    if (situation === "threatened") cls = "revealed_threatened";
    else {
      first = reachingFirst(input.likely.belief_threats, input.base, slot);
      if (first.length > 0) cls = "likely_threatened";
      else if (situation === "safe" && !anyReach(input.likely.belief_threats, slot)) cls = "safe_unlikely";
    }
    bump(counts, `likely.slots.${cls}.${phase}`);
    if (cls === "revealed_threatened" || phase === "no_facts") continue;
    // The inclusions are counted with the rates' slots (a no_facts decision is kept out of both).
    if (cls === "likely_threatened") {
      bump(counts, "likely.likely_threatened.n");
      bump(counts, "likely.likely_threatened.inclusion_max_ppm", Math.max(...first.map((r) => r.inclusion_ppm)));
      bump(counts, "likely.likely_threatened.inclusion_sum_ppm", Math.min(1_000_000, first.reduce((n, r) => n + r.inclusion_ppm, 0)));
    }

    const key = (action: "protect" | "switch"): string => `likely.${action}.${cls}.${phase}`;
    if (parsed.some((p) => stall.has(moveId(slot, p[slot] as SlotPart) ?? ""))) {
      bump(counts, `${key("protect")}.offered`);
      if (stall.has(moveId(slot, chosen[slot] as SlotPart) ?? "")) bump(counts, `${key("protect")}.chosen`);
    }
    if (parsed.some((p) => (p[slot] as SlotPart).kind === "switch")) {
      bump(counts, `${key("switch")}.offered`);
      if ((chosen[slot] as SlotPart).kind === "switch") bump(counts, `${key("switch")}.chosen`);
    }
  }
}

/** The phase of a decision from the channel's highest `|turn|` number and the opponent actives' `active_turns` (null: facts unavailable). */
export function phaseOf(turn: number, opponentActiveTurns: readonly (number | null)[] | null): Phase {
  if (turn <= 1) return "turn1";
  if (opponentActiveTurns === null) return "no_facts";
  return opponentActiveTurns.some((n) => n === 1) ? "fresh" : "later";
}
