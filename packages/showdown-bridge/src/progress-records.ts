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
import evaluationSchemaV2 from "@areneon/schema/contracts/v1/progress-evaluation-record-2.schema.json" with { type: "json" };
import gameSchema from "@areneon/schema/contracts/v1/progress-game-record.schema.json" with { type: "json" };
import type { ProgressEvaluationRecord } from "@areneon/schema/types/v1/progress-evaluation-record.ts";
import type { ProgressEvaluationRecordV2 } from "@areneon/schema/types/v1/progress-evaluation-record-2.ts";
import type { BaselineCounts, ProgressGameRecord } from "@areneon/schema/types/v1/progress-game-record.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { progressGameStreams } from "./progress-seed.ts";
import { teamHash } from "./team-class.ts";
import type { PokemonSet } from "./engine.ts";

export type { BaselineCounts, ProgressEvaluationRecord, ProgressEvaluationRecordV2, ProgressGameRecord };

/** The counts a damage-aware baseline seat reports per game (ADR-018 section 4), in the record's order. */
export const BASELINE_COUNT_KEYS = ["decisions", "preview_decisions", "scored_decisions", "queries", "probe_runs", "unsupported_queries", "reader_fallbacks", "assumed_set_fallbacks", "epsilon_plays"] as const;

/** Whether a policy id names the damage-aware baseline (`damage-aware@<commit>`). */
export const isDamageAware = (policyId: string): boolean => policyId.startsWith("damage-aware@");

/** The sums of per-game baseline counts, key by key. */
export function sumBaselineCounts(list: readonly BaselineCounts[]): BaselineCounts {
  const total = Object.fromEntries(BASELINE_COUNT_KEYS.map((key) => [key, 0])) as unknown as BaselineCounts;
  for (const counts of list) for (const key of BASELINE_COUNT_KEYS) total[key] += counts[key];
  // Written only above 0, so a run without refusals keeps the bytes it always had.
  const refused = list.reduce((sum, counts) => sum + (counts.probe_refusals ?? 0), 0);
  if (refused > 0) total.probe_refusals = refused;
  return total;
}

export class ProgressRecordError extends Error {
  override name = "ProgressRecordError";
}

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");
const validEvaluation = ajvFor("contract").compile(evaluationSchema);
const validEvaluationV2 = ajvFor("contract").compile(evaluationSchemaV2);
const validGame = ajvFor("contract").compile(gameSchema);

/** SHA-256 of a record's JCS bytes: the hash a record is named by. */
export const progressRecordHash = (record: ProgressEvaluationRecord | ProgressEvaluationRecordV2 | ProgressGameRecord): string => sha256(jcs(record as unknown as Json));

/** SHA-256 of the JCS bytes of the omniscient log's lines, as a JSON array of strings. */
export const omniscientLogHash = (lines: readonly string[]): string => sha256(jcs(lines as unknown as Json));

type Match = ProgressEvaluationRecord["matches"][number];
type Row = ProgressEvaluationRecord["games"][number];

const fail = (message: string): never => {
  throw new ProgressRecordError(message);
};
const strictlyAscending = (values: readonly string[]): boolean => values.every((value, index) => index === 0 || value > (values[index - 1] as string));

/** What a match's game rows say, counted from the candidate's side: the numbers the match must carry. */
export function tallyRows(rows: readonly Pick<Row, "outcome">[], maxTurns: number): Pick<Match, "wins" | "ties" | "losses" | "turn_cap_ties" | "turns_total" | "forfeits"> {
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

/** The rules the profile cannot say about one baseline count record (a game's, or a sum of games'). */
function checkBaselineCounts(counts: BaselineCounts, what: string): void {
  if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries || (counts.probe_refusals ?? 0) > counts.queries) fail(`${what}: probe runs, unsupported queries and probe refusals cannot exceed queries`);
  if (counts.preview_decisions + counts.scored_decisions > counts.decisions) fail(`${what}: preview and scored decisions cannot exceed decisions`);
}

/** A match's numbers against its rows. */
function recount(match: Match, rows: readonly Pick<Row, "outcome">[], maxTurns: number): void {
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

type RowV2 = ProgressEvaluationRecordV2["games"][number];

/**
 * Read a ProgressEvaluationRecordV2 (ADR-017 erratum E2): every pair is played twice, so a match of n pairs has 2n
 * games, game g is pair g >> 1 and the candidate sits on p1 for an even g and p2 for an odd one. Anchor matches
 * are named anchor-g<A>-<mode> and play an earlier generation; a cross-run match is named run-<mode> and plays
 * another run's frozen pair (a different run key).
 */
export function readProgressEvaluationRecordV2(value: unknown): ProgressEvaluationRecordV2 {
  if (!validEvaluationV2(value)) throw new ProgressRecordError(`invalid ProgressEvaluationRecordV2: ${JSON.stringify(validEvaluationV2.errors)}`);
  const record = structuredClone(value) as unknown as ProgressEvaluationRecordV2;
  const { plan, matches, games } = record;
  const anchored = matches.some((match) => match.opponent.kind === "anchor");
  const anchorGames = plan.anchor_games_per_match;
  if (anchored !== (anchorGames !== undefined)) fail("plan.anchor_games_per_match is present exactly when the record has anchor matches");
  const runGames = plan.run_games_per_match;
  if (matches.some((match) => match.opponent.kind === "run") !== (runGames !== undefined)) fail("plan.run_games_per_match is present exactly when the record has cross-run matches");
  if (plan.games_per_match % 2 !== 0 || (anchorGames ?? 0) % 2 !== 0 || (runGames ?? 0) % 2 !== 0) fail("games_per_match, anchor_games_per_match and run_games_per_match must be even: every pair is played from both seats");
  if (record.teams.pairs_used !== Math.max(plan.games_per_match, anchorGames ?? 0, runGames ?? 0) / 2) fail("teams.pairs_used must be half the largest games_per_match, anchor_games_per_match or run_games_per_match");
  if (plan.recorded_games_per_match > plan.games_per_match) fail("recorded_games_per_match must not exceed games_per_match");
  const ids = matches.map((match) => match.match_id);
  if (!strictlyAscending(ids)) fail("matches must be sorted by match_id, each once");
  const rowsOf = new Map<string, RowV2[]>(ids.map((id) => [id, []]));
  for (const row of games) {
    const rows = rowsOf.get(row.match_id);
    if (rows === undefined) return fail(`a game row names the unknown match ${row.match_id}`);
    rows.push(row);
  }
  if (new Set(matches.map((match) => match.candidate_preview === undefined)).size > 1) fail("a Preview ablation record holds ablation matches only");
  const flat = ids.flatMap((id) => rowsOf.get(id) as RowV2[]);
  if (flat.some((row, index) => row !== games[index])) fail("game rows must be in match then game order");
  const identities = new Set<string>();
  for (const match of matches) {
    const rows = rowsOf.get(match.match_id) as RowV2[];
    const preview = match.candidate_preview;
    if (preview !== undefined) {
      // A Preview ablation match (ticket #940): a baseline opponent, named preview-<variant>-<baseline>-<mode>.
      if (match.mode !== "argmax") fail(`${match.match_id}: an ablation match is played in argmax`);
      if (match.opponent.kind !== "baseline") fail(`${match.match_id}: an ablation match plays a scripted baseline`);
      const variant = preview.kind === "generation" ? `g${preview.generation}` : preview.kind;
      if (preview.kind === "generation" && preview.generation >= record.generation) fail(`${match.match_id}: an ablation Preview comes from an earlier generation`);
      if (match.match_id !== `preview-${variant}-${match.opponent.policy_id.split("@")[0] as string}-${match.mode}`) fail(`${match.match_id}: an ablation match is named preview-<variant>-<baseline>-<mode>`);
    }
    const anchor = match.opponent.kind === "anchor" ? match.opponent : undefined;
    if (anchor !== undefined && (anchor.generation >= record.generation || match.match_id !== `anchor-g${anchor.generation}-${match.mode}`)) fail(`${match.match_id}: an anchor match is named anchor-g<generation>-<mode> and plays an earlier generation`);
    const cross = match.opponent.kind === "run" ? match.opponent : undefined;
    if (cross !== undefined && (cross.run_id === record.run_id || match.match_id !== `run-${match.mode}`)) fail(`${match.match_id}: a cross-run match is named run-<mode> and plays another run's pair`);
    const expectedGames = cross !== undefined ? (runGames as number) : anchor !== undefined ? (anchorGames as number) : plan.games_per_match;
    if (match.games !== expectedGames) fail(`${match.match_id}: games must equal ${cross !== undefined ? "plan.run_games_per_match" : anchor === undefined ? "plan.games_per_match" : "plan.anchor_games_per_match"}`);
    if (match.wins + match.ties + match.losses !== match.games) fail(`${match.match_id}: wins + ties + losses must equal games`);
    if (rows.length !== match.games) fail(`${match.match_id}: there must be one game row per game`);
    rows.forEach((row, index) => {
      if (row.game !== index || row.pair !== index >> 1) fail(`${match.match_id}: game rows must run 0..games-1 with pair equal to the game divided by two`);
      if (row.candidate_seat !== (index % 2 === 0 ? "p1" : "p2")) fail(`${match.match_id}: the candidate sits on p1 for an even game and p2 for an odd one`);
    });
    recount(match as unknown as Match, rows, plan.max_turns);
    checkBaselineOfMatch(match, rows, identities);
    if (!strictlyAscending(match.game_record_hashes)) fail(`${match.match_id}: game_record_hashes must be sorted, each once`);
    if (match.game_record_hashes.length !== Math.min(plan.recorded_games_per_match, match.games)) fail(`${match.match_id}: game_record_hashes must number recorded_games_per_match`);
  }
  if (identities.size > 1) fail("every damage-aware match of a record must carry one baseline identity hash");
  return record;
}

type MatchV2 = ProgressEvaluationRecordV2["matches"][number];

/**
 * ADR-018 sections 4 and 5: a match against the damage-aware baseline carries its identity hash and the sum of its
 * rows' per-game counts; no other match carries either. The counts recompute from the rows.
 */
function checkBaselineOfMatch(match: MatchV2, rows: readonly RowV2[], identities: Set<string>): void {
  const damage = match.opponent.kind === "baseline" && isDamageAware(match.opponent.policy_id);
  if (!damage) {
    if (match.baseline !== undefined || rows.some((row) => row.baseline_counts !== undefined)) fail(`${match.match_id}: only a damage-aware match carries baseline counts`);
    return;
  }
  if (match.baseline === undefined || rows.some((row) => row.baseline_counts === undefined)) fail(`${match.match_id}: a damage-aware match carries baseline counts in its summary and in every game row`);
  const found = match.baseline as NonNullable<MatchV2["baseline"]>;
  identities.add(found.identity_hash);
  const perGame = rows.map((row) => row.baseline_counts as BaselineCounts);
  perGame.forEach((counts, index) => checkBaselineCounts(counts, `${match.match_id} game ${index}`));
  if (jcs(found.counts as unknown as Json) !== jcs(sumBaselineCounts(perGame) as unknown as Json)) fail(`${match.match_id}: baseline.counts must be the sum of the game rows' counts`);
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
  const { baseline } = record;
  if (baseline !== undefined) {
    const other = baseline.seat === "p1" ? "p2" : "p1";
    if (!isDamageAware(record.policies[baseline.seat]) || isDamageAware(record.policies[other])) fail("baseline.seat must be the one seat whose policy is damage-aware");
    checkBaselineCounts(baseline.counts, "baseline");
    // The actor counts every request it answered; a forfeit can stop it on one it never answered.
    const answered = record.decisions.filter((decision) => decision.player === baseline.seat).length;
    const forfeited = record.outcome.kind === "forfeit" ? 1 : 0;
    if (baseline.counts.decisions < answered || baseline.counts.decisions > answered + forfeited) fail("baseline.counts.decisions must be the baseline seat's decisions");
  } else if (isDamageAware(record.policies.p1) || isDamageAware(record.policies.p2)) fail("a game with a damage-aware seat must carry the baseline counts");
  // A game of a both-seats evaluation names its pair: both games of a pair share the pair's seed.
  if (record.pair !== undefined && record.pair !== record.game >> 1) fail("pair must be the game index divided by two");
  if (progressGameStreams(record.run_id, record.pair ?? record.game).seed !== record.seed) fail("the seed is not the one progress-seed/1 gives this game's pair (or game index)");
  return record;
}

/** The two seats of a game record's outcome as the engine's winner field: `p1`, `p2` or `null` for a tie. */
export const outcomeWinner = (outcome: Outcome): "p1" | "p2" | null =>
  outcome.kind === "played" ? (outcome.winner === "tie" ? null : outcome.winner) : outcome.loser === "p1" ? "p2" : "p1";
