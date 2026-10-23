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
import type { LeagueRunRecord } from "@areneon/schema/types/v1/league-run-record.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { LEAGUE_SEED_DERIVATION } from "./league-seed.ts";
import { nodeSha256 } from "./player-channel.ts";
import type { ForfeitCause } from "./runner.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import { runConfigHash } from "./training-run-config.ts";

export type { LeagueRunRecord };

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
      if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries) {
        throw new LeagueRunRecordError("a baseline's probe runs and unsupported queries cannot exceed its queries");
      }
      if (counts.preview_decisions + counts.scored_decisions > counts.decisions) {
        throw new LeagueRunRecordError("a baseline's preview and scored decisions cannot exceed its decisions");
      }
    }
  }
  return record;
}

/** The hash a later record names this one by: SHA-256 of its JCS bytes. */
export const leagueRunRecordHash = (record: LeagueRunRecord): Promise<string> =>
  nodeSha256(new TextEncoder().encode(jcs(record as unknown as Json)));

/** Check that a record belongs to `config`: its run key and its hash. */
export async function checkLeagueRunRecord(record: LeagueRunRecord, config: TrainingRunConfig): Promise<void> {
  if (record.run_id !== config.seeds.run_key) throw new LeagueRunRecordError(`run_id ${record.run_id} is not the config's run key`);
  if (record.run_config_hash !== (await runConfigHash(config))) throw new LeagueRunRecordError("run_config_hash is not the config's hash");
  // `baselines` exists exactly for the config's damage-aware entries, with the same policy ids and games.
  const expected = config.generation.baseline_games.filter((baseline) => baseline.actor === "damage-aware" && baseline.games > 0).map((baseline) => `${baseline.policy_id}:${baseline.games}`).sort();
  const actual = (record.baselines ?? []).map((entry) => `${entry.policy_id}:${entry.games}`).sort();
  if (expected.join() !== actual.join()) throw new LeagueRunRecordError("baselines must list exactly the config's damage-aware baselines, with their games");
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
  for (const report of reports) {
    let entry = byId.get(report.policy_id);
    if (entry === undefined) {
      entry = { policy_id: report.policy_id, identity_hash: report.identity_hash, games: 0, counts: { decisions: 0, preview_decisions: 0, scored_decisions: 0, queries: 0, probe_runs: 0, unsupported_queries: 0, reader_fallbacks: 0, assumed_set_fallbacks: 0, epsilon_plays: 0 } };
      byId.set(report.policy_id, entry);
    }
    if (entry.identity_hash !== report.identity_hash) throw new LeagueRunRecordError("one baseline policy id played with two identity hashes");
    entry.games += 1;
    for (const key of COUNT_KEYS) entry.counts[key] += report.counts[key];
  }
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
