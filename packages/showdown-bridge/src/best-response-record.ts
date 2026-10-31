// The BestResponseRecord (ADR-029 section 2, E0-P, ticket #1257): the result of an evaluation-only responder against a
// frozen pair, the blind control and the room for the table, on the ADR-017 E2 plan (validation pairs from both seats).
// The schema checks the shape; the rules the contract profile cannot say are checked here and again by the Python
// reader (`areneon_ml.progress.best_response`): rows run 0..games-1 with the pair and team A's seat implied by the game
// index, each arm's tally recounts from the rows, and every table decision is internally consistent. A message states a
// rule and never repeats a value.
//
// The responder has white-box access to the evaluated policy's weights and sees only public team information. The
// record is a conditional lower bound and a progress view on validation teams: never the Training gate, never held-out
// evidence, never an Episode and never training data.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/best-response-record.schema.json" with { type: "json" };
import type { BestResponseRecord, Decision, GameRow, PayoffCounts, Tally } from "@areneon/schema/types/v1/best-response-record.ts";
import { type Json, jcs } from "./canonical-json.ts";

export type { BestResponseRecord, Decision, GameRow, PayoffCounts, Tally };
export type BestResponseOutcome = GameRow["mirror"];

/** The arms of an evaluation, in the record's order. */
export const BEST_RESPONSE_ARMS = ["mirror", "responder", "blind", "room"] as const;
export type BestResponseArm = (typeof BEST_RESPONSE_ARMS)[number];

/** The label every display of a best-response evaluation carries. */
export const BEST_RESPONSE_LABEL = "evaluation-only responder with white-box access to the evaluated policy's weights; a conditional lower bound on validation teams; not the Training gate; never held-out evidence";

export class BestResponseRecordError extends Error {
  override name = "BestResponseRecordError";
}

const fail = (message: string): never => {
  throw new BestResponseRecordError(message);
};
const valid = ajvFor("contract").compile(schema);

/** SHA-256 of a record's JCS bytes. */
export const bestResponseRecordHash = (record: BestResponseRecord): string => createHash("sha256").update(jcs(record as unknown as Json)).digest("hex");

/** One arm's tally from its outcomes (all seen from the arm's focal side). */
export function tallyArm(outcomes: readonly BestResponseOutcome[], maxTurns: number): Tally {
  const count = { wins: 0, losses: 0, ties: 0, turn_cap_ties: 0, turns_total: 0 };
  const forfeits = new Map<string, Tally["forfeits"][number]>();
  for (const outcome of outcomes) {
    count.turns_total += outcome.turns;
    if (outcome.kind === "forfeit") {
      count[outcome.loser === "focal" ? "losses" : "wins"]++;
      const key = `${outcome.loser}\u0000${outcome.cause}`;
      const found = forfeits.get(key) ?? { side: outcome.loser, cause: outcome.cause, count: 0 };
      found.count++;
      forfeits.set(key, found);
    } else if (outcome.winner === "tie") {
      count.ties++;
      count.turn_cap_ties += Number(outcome.turns > maxTurns);
    } else count[outcome.winner === "focal" ? "wins" : "losses"]++;
  }
  return { ...count, forfeits: [...forfeits.keys()].sort().map((key) => forfeits.get(key) as Tally["forfeits"][number]) };
}

function checkDecision(decision: Decision, arms: readonly string[], what: string, plan: BestResponseRecord["plan"]): void {
  if (decision.kind === "fallback") return;
  const { counts, choices } = decision;
  if (counts.worlds_asked + decision.worlds_unstarted + decision.worlds_refused !== plan.responder.worlds) fail(`${what}: worlds asked, unstarted and refused must add up to the plan's worlds`);
  if (counts.worlds_used > counts.worlds_asked || counts.worlds_dropped > counts.worlds_used) fail(`${what}: worlds used cannot exceed worlds asked, nor worlds dropped worlds used`);
  if (counts.entries > counts.entry_attempts) fail(`${what}: entries cannot exceed entry attempts`);
  if (counts.clamped !== counts.clamped_low + counts.clamped_high) fail(`${what}: clamped must be the sum of clamped_low and clamped_high`);
  if (decision.own > plan.responder.k_own) fail(`${what}: own strategies cannot exceed k_own`);
  if (decision.complete !== (counts.missing_entries === 0)) fail(`${what}: complete must hold exactly when no entry is missing`);
  if (counts.missing_entries > decision.own * decision.columns) fail(`${what}: missing entries cannot exceed the table's cells`);
  if (choices.map((c) => c.arm).join() !== arms.join()) fail(`${what}: the choices must be exactly ${arms.join(", ")}, in that order`);
  for (const choice of choices) {
    if (choice.rank >= decision.own) fail(`${what}: a chosen rank must be below the table's own strategies`);
    if (choice.columns_used > decision.columns) fail(`${what}: columns used cannot exceed the table's columns`);
  }
}

/** Read a BestResponseRecord: the schema, then the arithmetic, order and presence rules. */
export function readBestResponseRecord(value: unknown): BestResponseRecord {
  if (!valid(value)) throw new BestResponseRecordError(`invalid BestResponseRecord: ${JSON.stringify(valid.errors)}`);
  const record = structuredClone(value) as unknown as BestResponseRecord;
  const { plan, teams, games, result } = record;
  if (plan.games % 2 !== 0) fail("plan.games must be even: every pair is played from both seats");
  if (teams.pairs_used !== plan.games / 2) fail("teams.pairs_used must be half of plan.games");
  if (teams.evaluation_dataset_hash !== undefined && teams.evaluation_dataset_hash === teams.dataset_hash) fail("the evaluation dataset cannot be the run's own dataset");
  if (games.length !== plan.games) fail("there must be one game row per game");
  games.forEach((row, index) => {
    if (row.game !== index || row.pair !== index >> 1) fail("game rows must run 0..games-1 with pair equal to the game divided by two");
    if (row.a_seat !== (index % 2 === 0 ? "p1" : "p2")) fail("team A sits on p1 for an even game and p2 for an odd one");
    checkDecision(row.b_decision, ["responder", "blind"], "b_decision", plan);
    checkDecision(row.a_decision, ["room"], "a_decision", plan);
  });
  for (const arm of BEST_RESPONSE_ARMS) {
    const tally = tallyArm(games.map((row) => row[arm]), plan.max_turns);
    const listed = result[arm].forfeits.map((f) => `${f.side}\u0000${f.cause}`);
    if (listed.some((key, index) => index > 0 && key <= (listed[index - 1] as string))) fail(`${arm}: forfeits must be sorted by side then cause, each pair once`);
    if (result[arm].wins + result[arm].losses + result[arm].ties !== plan.games) fail(`${arm}: wins + losses + ties must equal plan.games`);
    if (jcs(result[arm] as unknown as Json) !== jcs(tally as unknown as Json)) fail(`${arm}: the tally must equal the game rows`);
  }
  return record;
}
