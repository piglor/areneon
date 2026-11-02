// The LeagueRunRecord (ADR-009 §3, ADR-010 §4): what one generation of a league
// run played, tied to the TrainingRunConfig it ran under and to the seed
// derivation that made its battle seeds. The schema checks the shape; the rules
// the contract profile cannot say are checked here, and again by the Python
// reader (`areneon_ml.loop.league`): the counts agree, and the opponents are
// sorted and unique, and the forfeit causes, when present, are sorted, unique
// and sum to the forfeits. `baselines` (ADR-018 §5, §7), present only for a
// generation that played a damage-aware baseline, names its identity hash and
// sums its per-game counts; each entry matches an opponent entry's games.

import { ajvFor } from "@areneon/schema/ajv";
import leagueRunRecordSchema from "@areneon/schema/contracts/v1/league-run-record.schema.json" with { type: "json" };
import leagueRunRecordSchemaV2 from "@areneon/schema/contracts/v1/league-run-record-2.schema.json" with { type: "json" };
import type { LeagueRunRecord } from "@areneon/schema/types/v1/league-run-record.ts";
import type { LeagueRunRecordV2 } from "@areneon/schema/types/v1/league-run-record-2.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { LEAGUE_SEED_DERIVATION } from "./league-seed.ts";
import { nodeSha256 } from "./player-channel.ts";
import type { ForfeitCause } from "./runner.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import { runConfigHash } from "./training-run-config.ts";

export type { LeagueRunRecord, LeagueRunRecordV2 };

/** One damage-aware baseline's entry of a record. */
type BaselineRecord = NonNullable<LeagueRunRecord["baselines"]>[number];
/** What one game of a damage-aware baseline reports (`DamageAwareActor.report()`), without the identity itself. */
export interface BaselineGameReport {
  policy_id: string;
  identity_hash: string;
  counts: BaselineRecord["counts"];
}

export class LeagueRunRecordError extends Error {
  override name = "LeagueRunRecordError";
}

const validRecord = ajvFor("contract").compile(leagueRunRecordSchema);
const validRecordV2 = ajvFor("contract").compile(leagueRunRecordSchemaV2);

/** Read a record: the schema (which fixes the derivation version), then the count and order rules. */
export function readLeagueRunRecord(value: unknown): LeagueRunRecord {
  if (!validRecord(value)) throw new LeagueRunRecordError(`invalid LeagueRunRecord: ${JSON.stringify(validRecord.errors)}`);
  const record = structuredClone(value) as unknown as LeagueRunRecord;
  if (record.played + record.forfeits !== record.games) throw new LeagueRunRecordError("played + forfeits must equal games");
  if (record.episodes !== 2 * record.played) throw new LeagueRunRecordError("episodes must be two per played game");
  const ids = record.opponents.map((opponent) => opponent.opponent_policy_id);
  if (ids.some((id, index) => index > 0 && id <= (ids[index - 1] as string))) {
    throw new LeagueRunRecordError("opponents must be sorted by opponent_policy_id, each once");
  }
  if (record.opponents.reduce((sum, opponent) => sum + opponent.games, 0) !== record.games) {
    throw new LeagueRunRecordError("the opponents' games must sum to games");
  }
  const causes = record.forfeit_causes;
  if (causes !== undefined) {
    if (causes.some((entry, index) => index > 0 && entry.cause <= (causes[index - 1]?.cause ?? ""))) {
      throw new LeagueRunRecordError("forfeit_causes must be sorted by cause, each once");
    }
    if (causes.reduce((sum, entry) => sum + entry.count, 0) !== record.forfeits) {
      throw new LeagueRunRecordError("the forfeit causes' counts must sum to forfeits");
    }
  }
  const baselines = record.baselines;
  if (baselines !== undefined) {
    if (baselines.some((entry, index) => index > 0 && entry.policy_id <= (baselines[index - 1]?.policy_id ?? ""))) {
      throw new LeagueRunRecordError("baselines must be sorted by policy_id, each once");
    }
    for (const entry of baselines) {
      const opponent = record.opponents.find((candidate) => candidate.opponent_policy_id === entry.policy_id);
      if (opponent === undefined || opponent.games !== entry.games) throw new LeagueRunRecordError("a baseline's games must equal its opponent entry's games");
      const { counts } = entry;
      if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries || (counts.probe_refusals ?? 0) > counts.queries) {
        throw new LeagueRunRecordError("a baseline's probe runs, unsupported queries and probe refusals cannot exceed its queries");
      }
      if (counts.preview_decisions + counts.scored_decisions > counts.decisions) {
        throw new LeagueRunRecordError("a baseline's preview and scored decisions cannot exceed its decisions");
      }
    }
  }
  return record;
}

/** The hash a later record names this one by: SHA-256 of its JCS bytes. */
export const leagueRunRecordHash = (record: LeagueRunRecord | LeagueRunRecordV2): Promise<string> =>
  nodeSha256(new TextEncoder().encode(jcs(record as unknown as Json)));

/** Check that a record belongs to `config`: its run key and its hash (version 2 also the numbers of members and exploiters). */
export async function checkLeagueRunRecord(record: LeagueRunRecord | LeagueRunRecordV2, config: TrainingRunConfig): Promise<void> {
  if (record.schema_version === 2) return checkLeagueRunRecordV2(record, config);
  if (record.run_id !== config.seeds.run_key) throw new LeagueRunRecordError(`run_id ${record.run_id} is not the config's run key`);
  if (record.run_config_hash !== (await runConfigHash(config))) throw new LeagueRunRecordError("run_config_hash is not the config's hash");
  // `baselines` exists exactly for the config's damage-aware entries, with the same policy ids and games.
  const expected = config.generation.baseline_games.filter((baseline) => baseline.actor === "damage-aware" && baseline.games > 0).map((baseline) => `${baseline.policy_id}:${baseline.games}`).sort();
  const actual = (record.baselines ?? []).map((entry) => `${entry.policy_id}:${entry.games}`).sort();
  if (expected.join() !== actual.join()) throw new LeagueRunRecordError("baselines must list exactly the config's damage-aware baselines, with their games");
}

/** The `population` object of a TrainingRunConfig (ADR-021 §8), read as data: only the two counts a record is checked against. */
interface PopulationCounts {
  members: number;
  exploiters: number;
}
const populationOf = (config: TrainingRunConfig): PopulationCounts | undefined => (config as unknown as { population?: PopulationCounts }).population;

async function checkLeagueRunRecordV2(record: LeagueRunRecordV2, config: TrainingRunConfig): Promise<void> {
  if (record.run_id !== config.seeds.run_key) throw new LeagueRunRecordError(`run_id ${record.run_id} is not the config's run key`);
  if (record.run_config_hash !== (await runConfigHash(config))) throw new LeagueRunRecordError("run_config_hash is not the config's hash");
  const population = populationOf(config);
  if (population === undefined) throw new LeagueRunRecordError("a version 2 record belongs to a run whose config has a population");
  const learning = record.members.filter((entry) => entry.member.startsWith("m")).length;
  if (learning !== population.members || record.members.length - learning !== population.exploiters) {
    throw new LeagueRunRecordError("the members and exploiters must be the config's population.members and population.exploiters");
  }
  // The ids run 0..n-1 within each kind: sorted and unique already, so each must be its position within its kind.
  record.members.forEach((entry, index) => {
    const expected = `${entry.member[0]}${index < learning ? index : index - learning}`;
    if (entry.member !== expected) throw new LeagueRunRecordError("member ids must be m0.. and x0.. without gaps");
  });
}

export interface LeagueRunFacts {
  config: TrainingRunConfig;
  generation: number;
  pair_policy_id: string;
  /** The match plan, as `plan.json` holds it. */
  plan: readonly unknown[];
  /** One entry per game: who the pair played, and whether it was played to an end. */
  games: readonly { opponent: string; outcome: "played" | "forfeit"; forfeit?: { cause: ForfeitCause } }[];
  /** One report per game played against a damage-aware baseline (none for other runs: the record then has no `baselines`). */
  baseline_reports?: readonly BaselineGameReport[];
}

const COUNT_KEYS = ["decisions", "preview_decisions", "scored_decisions", "queries", "probe_runs", "unsupported_queries", "reader_fallbacks", "assumed_set_fallbacks", "epsilon_plays"] as const;

/** The baselines entries: per policy id, the games reported and the counts summed; one identity hash per policy id or the generation is refused. */
function baselineEntries(reports: readonly BaselineGameReport[]): BaselineRecord[] {
  const byId = new Map<string, BaselineRecord>();
  const refused = new Map<string, number>();
  for (const report of reports) {
    let entry = byId.get(report.policy_id);
    if (entry === undefined) {
      entry = { policy_id: report.policy_id, identity_hash: report.identity_hash, games: 0, counts: { decisions: 0, preview_decisions: 0, scored_decisions: 0, queries: 0, probe_runs: 0, unsupported_queries: 0, reader_fallbacks: 0, assumed_set_fallbacks: 0, epsilon_plays: 0 } };
      byId.set(report.policy_id, entry);
    }
    if (entry.identity_hash !== report.identity_hash) throw new LeagueRunRecordError("one baseline policy id played with two identity hashes");
    entry.games += 1;
    for (const key of COUNT_KEYS) entry.counts[key] += report.counts[key];
    refused.set(report.policy_id, (refused.get(report.policy_id) ?? 0) + (report.counts.probe_refusals ?? 0));
  }
  // Written only above 0, so a run without refusals keeps the bytes it always had.
  for (const [id, total] of refused) if (total > 0) (byId.get(id) as BaselineRecord).counts.probe_refusals = total;
  return [...byId.values()].sort((a, b) => (a.policy_id < b.policy_id ? -1 : 1));
}

/** Build the record of a generation from what it played, and read it back so a bad one cannot leave. */
export async function buildLeagueRunRecord(facts: LeagueRunFacts): Promise<LeagueRunRecord> {
  const { config, games } = facts;
  const counts = new Map<string, number>();
  for (const game of games) counts.set(game.opponent, (counts.get(game.opponent) ?? 0) + 1);
  const played = games.filter((game) => game.outcome === "played").length;
  // Per-cause totals, sorted by cause; absent when no game forfeited, so a run without forfeits writes the bytes it always did.
  const byCause = new Map<string, number>();
  for (const game of games) if (game.forfeit !== undefined) byCause.set(game.forfeit.cause, (byCause.get(game.forfeit.cause) ?? 0) + 1);
  const forfeit_causes = [...byCause.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([cause, count]) => ({ cause, count }));
  return readLeagueRunRecord({
    schema_version: 1,
    run_id: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    generation: facts.generation,
    seed_derivation: LEAGUE_SEED_DERIVATION,
    plan_hash: await nodeSha256(new TextEncoder().encode(jcs(facts.plan as unknown as Json))),
    pair_policy_id: facts.pair_policy_id,
    games: games.length,
    played,
    forfeits: games.length - played,
    ...(games.length > played ? { forfeit_causes } : {}),
    episodes: 2 * played,
    opponents: [...counts.entries()]
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([opponent_policy_id, count]) => ({ opponent_policy_id, games: count })),
    // Present only for a damage-aware baseline, so every other run writes the bytes it always did.
    ...(facts.baseline_reports !== undefined && facts.baseline_reports.length > 0 ? { baselines: baselineEntries(facts.baseline_reports) } : {}),
  });
}

const memberKey = (id: string): [number, number] => [id.startsWith("m") ? 0 : 1, Number(id.slice(1))];
type MemberEntry = LeagueRunRecordV2["members"][number];

function checkCauses(causes: { cause: string; count: number }[] | undefined, forfeits: number, what: string): void {
  if (forfeits === 0) {
    if (causes !== undefined) throw new LeagueRunRecordError(`${what} forfeit_causes must be absent without forfeits`);
    return;
  }
  if (causes === undefined) throw new LeagueRunRecordError(`${what} forfeit_causes must be present with forfeits`);
  if (causes.some((entry, index) => index > 0 && entry.cause <= (causes[index - 1]?.cause ?? ""))) {
    throw new LeagueRunRecordError(`${what} forfeit_causes must be sorted by cause, each once`);
  }
  if (causes.reduce((sum, entry) => sum + entry.count, 0) !== forfeits) throw new LeagueRunRecordError(`${what} forfeit causes' counts must sum to forfeits`);
}

function checkMember(entry: MemberEntry): void {
  if (entry.played + entry.forfeits !== entry.games) throw new LeagueRunRecordError("a member's played + forfeits must equal games");
  if (entry.episodes !== 2 * entry.played) throw new LeagueRunRecordError("a member's episodes must be two per played game");
  const ids = entry.opponents.map((opponent) => opponent.opponent_policy_id);
  if (ids.some((id, index) => index > 0 && id <= (ids[index - 1] as string))) {
    throw new LeagueRunRecordError("a member's opponents must be sorted by opponent_policy_id, each once");
  }
  if (entry.opponents.reduce((sum, opponent) => sum + opponent.games, 0) !== entry.games) {
    throw new LeagueRunRecordError("a member's opponents' games must sum to its games");
  }
  checkCauses(entry.forfeit_causes, entry.forfeits, "a member's");
  const baselines = entry.baselines;
  if (baselines === undefined) return;
  if (baselines.some((b, index) => index > 0 && b.policy_id <= (baselines[index - 1]?.policy_id ?? ""))) {
    throw new LeagueRunRecordError("a member's baselines must be sorted by policy_id, each once");
  }
  for (const baseline of baselines) {
    const opponent = entry.opponents.find((candidate) => candidate.opponent_policy_id === baseline.policy_id);
    if (opponent === undefined || opponent.games !== baseline.games) throw new LeagueRunRecordError("a member's baseline games must equal its opponent entry's games");
    const { counts } = baseline;
    if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries || (counts.probe_refusals ?? 0) > counts.queries) {
      throw new LeagueRunRecordError("a baseline's probe runs, unsupported queries and probe refusals cannot exceed its queries");
    }
    if (counts.preview_decisions + counts.scored_decisions > counts.decisions) {
      throw new LeagueRunRecordError("a baseline's preview and scored decisions cannot exceed its decisions");
    }
  }
}

/** Read a version 2 record: the schema, then the member, total and cause rules the profile cannot say. */
export function readLeagueRunRecordV2(value: unknown): LeagueRunRecordV2 {
  if (!validRecordV2(value)) throw new LeagueRunRecordError(`invalid LeagueRunRecordV2: ${JSON.stringify(validRecordV2.errors)}`);
  const record = structuredClone(value) as unknown as LeagueRunRecordV2;
  const keys = record.members.map((entry) => memberKey(entry.member));
  const unsorted = keys.some((key, index) => {
    const before = keys[index - 1];
    return before !== undefined && (before[0] > key[0] || (before[0] === key[0] && before[1] >= key[1]));
  });
  if (unsorted) throw new LeagueRunRecordError("members must be sorted (m<k> by k, then x<k> by k), each once");
  for (const entry of record.members) checkMember(entry);
  const sum = (pick: (entry: MemberEntry) => number): number => record.members.reduce((total, entry) => total + pick(entry), 0);
  if (record.games !== sum((e) => e.games) || record.played !== sum((e) => e.played) || record.forfeits !== sum((e) => e.forfeits) || record.episodes !== sum((e) => e.episodes)) {
    throw new LeagueRunRecordError("the totals must equal the sums over the members");
  }
  checkCauses(record.forfeit_causes, record.forfeits, "the");
  const byCause = new Map<string, number>();
  for (const entry of record.members) for (const c of entry.forfeit_causes ?? []) byCause.set(c.cause, (byCause.get(c.cause) ?? 0) + c.count);
  const listed = record.forfeit_causes ?? [];
  if (listed.length !== byCause.size || listed.some((c) => byCause.get(c.cause) !== c.count)) {
    throw new LeagueRunRecordError("each forfeit cause must equal the sum over the members");
  }
  return record;
}

/** Read a record of either version, by its `schema_version`; another version is refused. */
export function readAnyLeagueRunRecord(value: unknown): LeagueRunRecord | LeagueRunRecordV2 {
  const version = typeof value === "object" && value !== null ? (value as { schema_version?: unknown }).schema_version : undefined;
  if (version === 1) return readLeagueRunRecord(value);
  if (version === 2) return readLeagueRunRecordV2(value);
  throw new LeagueRunRecordError("a LeagueRunRecord has schema_version 1 or 2");
}
