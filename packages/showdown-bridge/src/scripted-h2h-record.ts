// The ScriptedHeadToHeadRecord (ADR-025 section 13, ticket #1130): the result of two scripted
// baselines played against each other on the ADR-017 E2 plan, every validation pair from both
// seats. The schema checks the shape; the rules the contract profile cannot say are checked
// here and again by the Python reader (`areneon_ml.progress.scripted`): rows run 0..games-1
// with the pair and arm a's seat implied by the game index, the result recounts from the rows,
// the forfeits and the counts are sorted and sum from the rows, and counts and belief appear
// exactly where the arm's policy has them. A message states a rule and never repeats a value.
//
// It is a progress view on validation teams, never the Training gate and never an Episode.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/scripted-head-to-head-record.schema.json" with { type: "json" };
import type { BaselineCounts, BeliefCounts, ScriptedHeadToHeadRecord } from "@areneon/schema/types/v1/scripted-head-to-head-record.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { isDamageAware, sumBaselineCounts } from "./progress-records.ts";

export type { BaselineCounts, BeliefCounts, ScriptedHeadToHeadRecord };
export type ScriptedGameRow = ScriptedHeadToHeadRecord["games"][number];
export type ScriptedResult = ScriptedHeadToHeadRecord["result"];

/** The label every display of a scripted head-to-head carries. */
export const SCRIPTED_H2H_LABEL = "scripted head-to-head on validation teams; not the Training gate; no learned policy";

export class ScriptedH2HRecordError extends Error {
  override name = "ScriptedH2HRecordError";
}

const fail = (message: string): never => {
  throw new ScriptedH2HRecordError(message);
};
const valid = ajvFor("contract").compile(schema);

/** SHA-256 of a record's JCS bytes. */
export const scriptedH2HRecordHash = (record: ScriptedHeadToHeadRecord): string => createHash("sha256").update(jcs(record as unknown as Json)).digest("hex");

/** Whether a policy id plays on belief worlds (`damage-aware@v2`: its record names a belief). */
export const playsOnBelief = (policyId: string): boolean => policyId === "damage-aware@v2";

const BELIEF_KEYS = ["decisions", "worlds_sampled", "redraws", "contradicted_worlds", "no_consistent_world"] as const;

/** Per-game belief counts summed, key by key; refusal codes sorted, each once. */
export function sumBeliefCounts(list: readonly BeliefCounts[]): BeliefCounts {
  const total = Object.fromEntries(BELIEF_KEYS.map((key) => [key, 0])) as unknown as BeliefCounts;
  const refusals = new Map<string, number>();
  for (const counts of list) {
    for (const key of BELIEF_KEYS) total[key] += counts[key];
    for (const { code, count } of counts.refusals) refusals.set(code, (refusals.get(code) ?? 0) + count);
  }
  total.refusals = [...refusals.keys()].sort().map((code) => ({ code, count: refusals.get(code) as number }));
  return total;
}

/** The result a set of rows must carry. */
export function tallyScripted(rows: readonly Pick<ScriptedGameRow, "outcome">[], maxTurns: number): ScriptedResult {
  const count = { wins_a: 0, wins_b: 0, ties: 0, turn_cap_ties: 0, turns_total: 0 };
  const forfeits = new Map<string, ScriptedResult["forfeits"][number]>();
  for (const { outcome } of rows) {
    count.turns_total += outcome.turns;
    if (outcome.kind === "forfeit") {
      count[outcome.loser === "a" ? "wins_b" : "wins_a"]++;
      const key = `${outcome.loser}\u0000${outcome.cause}`;
      const found = forfeits.get(key) ?? { side: outcome.loser, cause: outcome.cause, count: 0 };
      found.count++;
      forfeits.set(key, found);
    } else if (outcome.winner === "tie") {
      count.ties++;
      count.turn_cap_ties += Number(outcome.turns > maxTurns);
    } else count[outcome.winner === "a" ? "wins_a" : "wins_b"]++;
  }
  return { ...count, forfeits: [...forfeits.keys()].sort().map((key) => forfeits.get(key) as ScriptedResult["forfeits"][number]) };
}

function checkCounts(counts: BaselineCounts, what: string): void {
  if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries || (counts.probe_refusals ?? 0) > counts.queries) fail(`${what}: probe runs, unsupported queries and probe refusals cannot exceed queries`);
  if (counts.preview_decisions + counts.scored_decisions > counts.decisions) fail(`${what}: preview and scored decisions cannot exceed decisions`);
}

function checkBelief(counts: BeliefCounts, what: string): void {
  const codes = counts.refusals.map((r) => r.code);
  if (codes.some((code, index) => index > 0 && code <= (codes[index - 1] as string))) fail(`${what}: belief refusals must be sorted by code, each once`);
  const refused = counts.refusals.reduce((sum, r) => sum + r.count, 0);
  if (counts.no_consistent_world + refused > counts.decisions) fail(`${what}: refusals and requests without a consistent world cannot exceed the belief's decisions`);
}

/** Read a ScriptedHeadToHeadRecord: the schema, then the arithmetic, order and presence rules. */
export function readScriptedH2HRecord(value: unknown): ScriptedHeadToHeadRecord {
  if (!valid(value)) throw new ScriptedH2HRecordError(`invalid ScriptedHeadToHeadRecord: ${JSON.stringify(valid.errors)}`);
  const record = structuredClone(value) as unknown as ScriptedHeadToHeadRecord;
  const { plan, teams, arms, games, result } = record;
  if (plan.games % 2 !== 0) fail("plan.games must be even: every pair is played from both seats");
  if (teams.pairs_used !== plan.games / 2) fail("teams.pairs_used must be half of plan.games");
  if (arms.a.policy_id === arms.b.policy_id) fail("the two arms must be different baselines");
  if (games.length !== plan.games) fail("there must be one game row per game");
  for (const side of ["a", "b"] as const) {
    const arm = arms[side];
    const damage = isDamageAware(arm.policy_id);
    if (damage !== (arm.identity_hash !== undefined) || damage !== (arm.counts !== undefined)) fail(`arm ${side}: identity_hash and counts are present exactly for a damage-aware arm`);
    if (playsOnBelief(arm.policy_id) !== (arm.belief !== undefined)) fail(`arm ${side}: belief is present exactly for an arm that plays on belief worlds`);
    const counts = games.map((row) => row[`${side}_counts`]);
    const beliefs = games.map((row) => row[`${side}_belief`]);
    if (counts.some((c) => (c !== undefined) !== damage)) fail(`arm ${side}: every game row carries counts exactly when the arm is damage-aware`);
    if (beliefs.some((b) => (b !== undefined) !== (arm.belief !== undefined))) fail(`arm ${side}: every game row carries belief counts exactly when the arm has a belief`);
    if (arm.counts !== undefined) {
      (counts as BaselineCounts[]).forEach((c, index) => checkCounts(c, `arm ${side} game ${index}`));
      checkCounts(arm.counts, `arm ${side}`);
      if (jcs(arm.counts as unknown as Json) !== jcs(sumBaselineCounts(counts as BaselineCounts[]) as unknown as Json)) fail(`arm ${side}: counts must be the sum of the game rows' counts`);
    }
    if (arm.belief !== undefined) {
      (beliefs as BeliefCounts[]).forEach((b, index) => checkBelief(b, `arm ${side} game ${index}`));
      checkBelief(arm.belief.totals, `arm ${side}`);
      if (jcs(arm.belief.totals as unknown as Json) !== jcs(sumBeliefCounts(beliefs as BeliefCounts[]) as unknown as Json)) fail(`arm ${side}: belief totals must be the sum of the game rows' belief counts`);
    }
  }
  games.forEach((row, index) => {
    if (row.game !== index || row.pair !== index >> 1) fail("game rows must run 0..games-1 with pair equal to the game divided by two");
    if (row.a_seat !== (index % 2 === 0 ? "p1" : "p2")) fail("arm a sits on p1 for an even game and p2 for an odd one");
  });
  if (result.wins_a + result.wins_b + result.ties !== plan.games) fail("wins_a + wins_b + ties must equal plan.games");
  const listed = result.forfeits.map((f) => `${f.side}\u0000${f.cause}`);
  if (listed.some((key, index) => index > 0 && key <= (listed[index - 1] as string))) fail("forfeits must be sorted by arm then cause, each pair once");
  if (jcs(result as unknown as Json) !== jcs(tallyScripted(games, plan.max_turns) as unknown as Json)) fail("the result must equal the game rows");
  return record;
}
