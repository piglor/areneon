// The history rows of the situational move-use diagnostic (#1345, ADR-022 erratum E-22.5 slice H7), beside #1283's rows.
//
// LABEL: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure.
//
// Computed from the same stored progress games, from one seat's OWN view at each move request (never the omniscient log): the
// `probe-outcome/1` arrays of the seat's own channel (history facts declared in every probe query: `rows`, `outcomes`), the
// `probe-damage/2` `rows` of the same decision (no history declared: the DEFAULT-history row), and the history facts of the own
// channel (`stall_counter`, `move_actions`, the item facts). Every group below is defined by the engine's answers and those facts;
// no move id is written here (the stalling moves are the engine-derived list of the world facts' data).
//
// Per own living active, over the distinct move parts (slot, move position, target, Mega flag) of the candidates ("offered", pending
// included) and the part the seat chose ("chosen"); keys are flat dotted integer counts under `history.`:
//   decisions                      move-request decisions both builders answered
//   stall.<fresh|repeat>.<offered|chosen>   a stalling move (engine list) at `stall_counter` null or <= 1 (fresh) / > 1 (repeat);
//                                  `.chosen_success_ppm` and `.chosen_success_n`: the sum, over the chosen parts, of the mean engine
//                                  success chance (`success_ppm` over the part's branches, floored) and their number
//   stall.no_facts.offered         a stalling move whose slot has no history facts (excluded from the rates)
//   power.<up|down>.<offered|chosen>   a damaging part whose history-declared rows sum to MORE (up) or LESS (down) mean damage than
//                                  the default-history rows of `probe-damage/2`; equal rows belong to no group; `power.unmatched`
//                                  counts parts whose two row sets cannot be compared (unsupported or different row counts)
//   fails.all.<offered|chosen>     a part whose outcome `fails` in every opposing-action branch; `fails.acted`, `fails.fresh` and
//                                  `fails.item` split it by the actor's facts (it has acted since switching in, it has not, it has an
//                                  item-history fact: a berry eaten or a last item), so a first-turn-only or an item-history gate shows as
//                                  failing chosen parts with the matching fact
//   conditional.<offered|chosen>   a part whose outcome differs by branch; `conditional.chosen_branches` and `.chosen_passing` count the
//                                  chosen ones' branches and those whose precondition does not `fail`
//   outcome_unsupported            parts with an unsupported outcome row (never classified)
// An unsupported or missing row never classifies a part; nothing is guessed.
import type { CandidateFeatures2 } from "./candidate-features-2.ts";
import type { CandidateFeaturesOutcome, OutcomeRow } from "./candidate-features-outcome.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";

/** The history facts of one own slot's Pokemon the groups split by; null for the slot's facts when the module gave none. */
export interface SlotFact {
  move_actions: number;
  stall_counter: number | null;
  /** The Pokemon has an item-history fact: it ate a berry or has a last item (the facts' `ate_berry`, `last_item`). */
  item_history: boolean;
}

export interface HistoryInput {
  body: Record<string, unknown>;
  candidates: readonly { choice: string; pending: boolean }[];
  chosen: string;
  /** `probe-damage/2`'s rows of the decision: the default-history rows. */
  base: Pick<CandidateFeatures2, "rows">;
  /** `probe-outcome/1`'s rows (history declared) and outcomes of the same decision. */
  outcome: Pick<CandidateFeaturesOutcome, "rows" | "outcomes">;
  facts: readonly (SlotFact | null)[];
  stall: ReadonlySet<string>;
}

export type HistoryCounts = Record<string, number>;
const bump = (counts: HistoryCounts, key: string, by = 1): void => void (counts[key] = (counts[key] ?? 0) + by);
const asRecord = (value: unknown): Record<string, unknown> => (typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {});
const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

type Move = Extract<SlotPart, { kind: "move" }>;
type Computed = Extract<OutcomeRow, { kind: "computed" }>;
/** A damage row as far as this reads it. */
type DamageRow = { slot: number; move: number; target: number; mega: boolean; kind: string; mean_ppm?: number };

const partKey = (slot: number, part: Move): string => `${slot}/${part.index}/${part.target ?? 0}/${part.mega}`;

/** The rows of `rows` a part is about: its slot, move and Mega flag, and its target when it names one. */
const rowsOf = <R extends { slot: number; move: number; mega: boolean; target: number }>(rows: readonly R[], slot: number, part: Move): R[] => rows.filter((r) => r.slot === slot && r.move === part.index && r.mega === part.mega && (part.target === null || part.target <= 0 || r.target === part.target));

/** The sum of the mean damage of the rows, or null when any is unsupported or there is none. */
function meanSum(rows: readonly DamageRow[]): number | null {
  if (rows.length === 0 || rows.some((r) => r.kind !== "computed")) return null;
  return rows.reduce((sum, r) => sum + (r.mean_ppm as number), 0);
}

/** Add one decision's history groups to `counts`. */
export function tallyHistoryDecision(counts: HistoryCounts, input: HistoryInput): void {
  const { body, stall } = input;
  const active = asList(body.active).map(asRecord);
  const pokemon = asList(asRecord(body.side).pokemon).map(asRecord);
  const parsed = input.candidates.map((c) => parseJointChoice(c.choice));
  const chosen = parseJointChoice(input.chosen);
  if (chosen.length !== active.length || parsed.some((p) => p.length !== active.length)) throw new Error("a choice has a different number of parts than the request has active Pokemon");
  bump(counts, "history.decisions");
  const moveId = (slot: number, part: Move): string | null => (asList(asRecord(active[slot]).moves)[part.index - 1] as { id?: string } | undefined)?.id ?? null;

  for (let slot = 0; slot < active.length; slot++) {
    if (/ fnt$/.test(String(asRecord(pokemon[slot]).condition))) continue;
    const parts = new Map<string, Move>();
    for (const p of parsed) {
      const part = p[slot] as SlotPart;
      if (part.kind === "move") parts.set(partKey(slot, part), part);
    }
    const pick = chosen[slot] as SlotPart;
    const chosenKey = pick.kind === "move" ? partKey(slot, pick) : null;
    const fact = input.facts[slot] ?? null;

    for (const [key, part] of parts) {
      const isChosen = key === chosenKey;
      const count = (group: string): void => {
        bump(counts, `history.${group}.offered`);
        if (isChosen) bump(counts, `history.${group}.chosen`);
      };
      const outcomes = rowsOf(input.outcome.outcomes, slot, part);

      // Stall repeats: the engine's stall list, split by the history facts' counter.
      if (stall.has(moveId(slot, part) ?? "")) {
        if (fact === null) bump(counts, "history.stall.no_facts.offered");
        else {
          const group = `stall.${fact.stall_counter !== null && fact.stall_counter > 1 ? "repeat" : "fresh"}`;
          count(group);
          const computed = outcomes.filter((r): r is Computed => r.kind === "computed");
          if (isChosen && computed.length > 0) {
            bump(counts, `history.${group}.chosen_success_ppm`, Math.floor(computed.reduce((sum, r) => sum + r.success_ppm, 0) / computed.length));
            bump(counts, `history.${group}.chosen_success_n`);
          }
        }
      }

      // History-scaled power: the declared rows against the default-history rows of the same part.
      const declaredRows = rowsOf(input.outcome.rows, slot, part);
      const standardRows = rowsOf(input.base.rows, slot, part);
      const declared = meanSum(declaredRows);
      const standard = meanSum(standardRows);
      const declaredCount = declaredRows.length;
      const standardCount = standardRows.length;
      if (declaredCount > 0 || standardCount > 0) {
        if (declared === null || standard === null || declaredCount !== standardCount) bump(counts, "history.power.unmatched");
        else if (declared !== standard) count(declared > standard ? "power.up" : "power.down");
      }

      // Outcomes: fails in every branch, and conditional ones.
      if (outcomes.length === 0) continue;
      const computed = outcomes.filter((r): r is Computed => r.kind === "computed");
      if (computed.length !== outcomes.length) {
        bump(counts, "history.outcome_unsupported");
        continue;
      }
      if (computed.every((r) => r.precondition === "fails")) {
        count("fails.all");
        if (fact !== null) count(fact.move_actions > 0 ? "fails.acted" : "fails.fresh");
        if (fact?.item_history === true) count("fails.item");
      } else if (computed.some((r) => r.precondition !== computed[0]?.precondition || r.precondition_ppm !== computed[0]?.precondition_ppm)) {
        count("conditional");
        if (isChosen) {
          bump(counts, "history.conditional.chosen_branches", computed.length);
          bump(counts, "history.conditional.chosen_passing", computed.filter((r) => r.precondition !== "fails").length);
        }
      }
    }
  }
}
