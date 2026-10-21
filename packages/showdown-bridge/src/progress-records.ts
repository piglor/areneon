// The progress evaluation's two records (ADR-017 section 3): the
// ProgressEvaluationRecord of one generation and the ProgressGameRecord of one
// stored game. The schemas check the shape; the rules the contract profile
// cannot say (arithmetic, ordering, hashes) are checked here, and again by the
// Python reader (`areneon_ml.progress.records`).
//
// A progress record is a progress view only. It is never an Episode: the
// training loader refuses it by schema, and nothing here writes one.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import evaluationSchema from "@areneon/schema/contracts/v1/progress-evaluation-record.schema.json" with { type: "json" };
import gameSchema from "@areneon/schema/contracts/v1/progress-game-record.schema.json" with { type: "json" };
import type { ProgressEvaluationRecord } from "@areneon/schema/types/v1/progress-evaluation-record.ts";
import type { ProgressGameRecord } from "@areneon/schema/types/v1/progress-game-record.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { progressGameStreams } from "./progress-seed.ts";
import { teamHash } from "./team-class.ts";
import type { PokemonSet } from "./engine.ts";

export type { ProgressEvaluationRecord, ProgressGameRecord };

export class ProgressRecordError extends Error {
  override name = "ProgressRecordError";
}

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");
const validEvaluation = ajvFor("contract").compile(evaluationSchema);
const validGame = ajvFor("contract").compile(gameSchema);

/** SHA-256 of a record's JCS bytes: the hash a record is named by. */
export const progressRecordHash = (record: ProgressEvaluationRecord | ProgressGameRecord): string => sha256(jcs(record as unknown as Json));

/** SHA-256 of the JCS bytes of the omniscient log's lines, as a JSON array of strings. */
export const omniscientLogHash = (lines: readonly string[]): string => sha256(jcs(lines as unknown as Json));

type Match = ProgressEvaluationRecord["matches"][number];
type Row = ProgressEvaluationRecord["games"][number];

const fail = (message: string): never => {
  throw new ProgressRecordError(message);
};
const strictlyAscending = (values: readonly string[]): boolean => values.every((value, index) => index === 0 || value > (values[index - 1] as string));

/** What a match's game rows say, counted from the candidate's side: the numbers the match must carry. */
export function tallyRows(rows: readonly Row[], maxTurns: number): Pick<Match, "wins" | "ties" | "losses" | "turn_cap_ties" | "turns_total" | "forfeits"> {
  const count = { wins: 0, ties: 0, losses: 0, turn_cap_ties: 0, turns_total: 0 };
  const forfeits = new Map<string, Match["forfeits"][number]>();
  for (const { outcome } of rows) {
    count.turns_total += outcome.turns;
    if (outcome.kind === "forfeit") {
      // The loser's side is the forfeiting side; the other side wins.
      count[outcome.loser === "candidate" ? "losses" : "wins"]++;
      const key = `${outcome.loser}\u0000${outcome.cause}`;
      const found = forfeits.get(key) ?? { side: outcome.loser, cause: outcome.cause, count: 0 };
      found.count++;
      forfeits.set(key, found);
    } else {
      count[{ candidate: "wins", opponent: "losses", tie: "ties" }[outcome.winner] as "wins" | "losses" | "ties"]++;
      count.turn_cap_ties += Number(outcome.winner === "tie" && outcome.turns > maxTurns);
    }
  }
  return { ...count, forfeits: [...forfeits.keys()].sort().map((key) => forfeits.get(key) as Match["forfeits"][number]) };
}

/** A match's numbers against its rows. */
function recount(match: Match, rows: readonly Row[], maxTurns: number): void {
  const found = tallyRows(rows, maxTurns);
  if (match.wins !== found.wins || match.ties !== found.ties || match.losses !== found.losses) fail(`${match.match_id}: wins, ties and losses must equal the game rows`);
  if (match.turn_cap_ties !== found.turn_cap_ties) fail(`${match.match_id}: turn_cap_ties must equal the tied games that ran past max_turns`);
  if (match.turns_total !== found.turns_total) fail(`${match.match_id}: turns_total must equal the sum of the game rows' turns`);
  const listed = match.forfeits.map((f) => `${f.side}\u0000${f.cause}`);
  if (!strictlyAscending(listed)) fail(`${match.match_id}: forfeits must be sorted by side then cause, each pair once`);
  if (jcs(match.forfeits as unknown as Json) !== jcs(found.forfeits as unknown as Json)) fail(`${match.match_id}: forfeits must equal the forfeited game rows`);
}

/** Read a ProgressEvaluationRecord: the schema, then the arithmetic, order and completeness rules. */
export function readProgressEvaluationRecord(value: unknown): ProgressEvaluationRecord {
  if (!validEvaluation(value)) throw new ProgressRecordError(`invalid ProgressEvaluationRecord: ${JSON.stringify(validEvaluation.errors)}`);
  const record = structuredClone(value) as unknown as ProgressEvaluationRecord;
  const { plan, matches, games } = record;
  if (record.teams.pairs_used !== plan.games_per_match) fail("teams.pairs_used must equal plan.games_per_match");
  if (plan.recorded_games_per_match > plan.games_per_match) fail("recorded_games_per_match must not exceed games_per_match");
  const ids = matches.map((match) => match.match_id);
  if (!strictlyAscending(ids)) fail("matches must be sorted by match_id, each once");
  const rowsOf = new Map<string, Row[]>(ids.map((id) => [id, []]));
  for (const row of games) {
    const rows = rowsOf.get(row.match_id);
    if (rows === undefined) return fail(`a game row names the unknown match ${row.match_id}`);
    rows.push(row);
  }
  const flat = ids.flatMap((id) => rowsOf.get(id) as Row[]);
  if (flat.some((row, index) => row !== games[index])) fail("game rows must be in match then game order");
  for (const match of matches) {
    const rows = rowsOf.get(match.match_id) as Row[];
    if (match.games !== plan.games_per_match) fail(`${match.match_id}: games must equal plan.games_per_match`);
    if (match.wins + match.ties + match.losses !== match.games) fail(`${match.match_id}: wins + ties + losses must equal games`);
    if (rows.length !== match.games) fail(`${match.match_id}: there must be one game row per game`);
    rows.forEach((row, index) => {
      if (row.game !== index || row.pair !== index) fail(`${match.match_id}: game rows must run 0..games-1 with pair equal to the game`);
      if (row.candidate_seat !== (index % 2 === 0 ? "p1" : "p2")) fail(`${match.match_id}: the candidate sits on p1 for an even game and p2 for an odd one`);
    });
    recount(match, rows, plan.max_turns);
    if (!strictlyAscending(match.game_record_hashes)) fail(`${match.match_id}: game_record_hashes must be sorted, each once`);
    if (match.game_record_hashes.length !== plan.recorded_games_per_match) fail(`${match.match_id}: game_record_hashes must number recorded_games_per_match`);
  }
  return record;
}

type Outcome = ProgressGameRecord["outcome"];

/** Read a ProgressGameRecord: the schema, then the rules; `chosen` is among its candidates and every hash recomputes. */
export function readProgressGameRecord(value: unknown): ProgressGameRecord {
  if (!validGame(value)) throw new ProgressRecordError(`invalid ProgressGameRecord: ${JSON.stringify(validGame.errors)}`);
  const record = structuredClone(value) as unknown as ProgressGameRecord;
  for (const decision of record.decisions) {
    if (!decision.candidates.some((candidate) => candidate.choice === decision.chosen)) {
      fail(`the choice of ${decision.player} at request ${decision.request_seq} is not among its candidates`);
    }
  }
  for (const seat of ["p1", "p2"] as const) {
    if (teamHash(record.teams[seat].team as unknown as PokemonSet[]) !== record.teams[seat].team_hash) fail(`the team hash of ${seat} does not recompute`);
  }
  if (omniscientLogHash(record.omniscient_log) !== record.omniscient_log_sha256) fail("omniscient_log_sha256 does not recompute");
  if (progressGameStreams(record.run_id, record.game).seed !== record.seed) fail("the seed is not the one progress-seed/1 gives this game index");
  return record;
}

/** The two seats of a game record's outcome as the engine's winner field: `p1`, `p2` or `null` for a tie. */
export const outcomeWinner = (outcome: Outcome): "p1" | "p2" | null =>
  outcome.kind === "played" ? (outcome.winner === "tie" ? null : outcome.winner) : outcome.loser === "p1" ? "p2" : "p1";
