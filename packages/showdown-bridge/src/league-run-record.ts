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

type BaselineEntry = NonNullable<LeagueRunRecord["baselines"]>[number];

/** The `baselines` rules of both versions (ADR-018): sorted, each id once, games and counts agree with the opponents' (`what` is "a member's " for a version 2 member). */
function checkBaselines(baselines: BaselineEntry[] | undefined, opponents: readonly { opponent_policy_id: string; games: number; member?: string }[], what: string): void {
  if (baselines === undefined) return;
  const label = what === "" ? "a baseline" : `${what}baseline`;
  if (baselines.some((entry, index) => index > 0 && entry.policy_id <= (baselines[index - 1]?.policy_id ?? ""))) {
    throw new LeagueRunRecordError(`${what}baselines must be sorted by policy_id, each once`);
  }
  for (const entry of baselines) {
    const opponent = opponents.find((candidate) => candidate.opponent_policy_id === entry.policy_id && candidate.member === undefined);
    if (opponent === undefined || opponent.games !== entry.games) throw new LeagueRunRecordError(`${label}'s games must equal its opponent entry's games`);
    const { counts } = entry;
    if (counts.probe_runs > counts.queries || counts.unsupported_queries > counts.queries || (counts.probe_refusals ?? 0) > counts.queries) {
      throw new LeagueRunRecordError(`${label}'s probe runs, unsupported queries and probe refusals cannot exceed its queries`);
    }
    if (counts.preview_decisions + counts.scored_decisions > counts.decisions) {
      throw new LeagueRunRecordError(`${label}'s preview and scored decisions cannot exceed its decisions`);
    }
  }
}

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
  checkBaselines(record.baselines, record.opponents, "");
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

async function checkLeagueRunRecordV2(record: LeagueRunRecordV2, config: TrainingRunConfig): Promise<void> {
  if (record.run_id !== config.seeds.run_key) throw new LeagueRunRecordError(`run_id ${record.run_id} is not the config's run key`);
  if (record.run_config_hash !== (await runConfigHash(config))) throw new LeagueRunRecordError("run_config_hash is not the config's hash");
  const population = config.population;
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
const memberOrder = (a: string, b: string): number => {
  const [x, y] = [memberKey(a), memberKey(b)];
  return x[0] !== y[0] ? x[0] - y[0] : x[1] - y[1];
};
type MemberEntry = LeagueRunRecordV2["members"][number];
type MemberOpponent = MemberEntry["opponents"][number];

/** Forfeit causes: present exactly with forfeits, sorted, unique, summing to them; `what` names the owner in the messages. */
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

const opponentOrder = (a: MemberOpponent, b: MemberOpponent): number => {
  if (a.opponent_policy_id !== b.opponent_policy_id) return a.opponent_policy_id < b.opponent_policy_id ? -1 : 1;
  const [x, y] = [a.member ?? "", b.member ?? ""];
  return x === y ? 0 : x < y ? -1 : 1;
};

/** One member's own rules: its counts, its opponents, its Episodes (own seats), its causes and baselines. */
function checkMember(entry: MemberEntry): void {
  if (entry.played + entry.forfeits !== entry.games) throw new LeagueRunRecordError("a member's played + forfeits must equal games");
  const ops = entry.opponents;
  if (ops.some((o, index) => index > 0 && opponentOrder(ops[index - 1] as MemberOpponent, o) >= 0)) {
    throw new LeagueRunRecordError("a member's opponents must be sorted by opponent_policy_id and member, each once");
  }
  if (ops.some((o) => o.played > o.games)) throw new LeagueRunRecordError("an opponent's played cannot exceed its games");
  if (ops.reduce((sum, o) => sum + o.games, 0) !== entry.games) throw new LeagueRunRecordError("a member's opponents' games must sum to its games");
  if (ops.reduce((sum, o) => sum + o.played, 0) !== entry.played) throw new LeagueRunRecordError("a member's opponents' played counts must sum to its played");
  const selfPlayed = ops.filter((o) => o.member === entry.member).reduce((sum, o) => sum + o.played, 0);
  if (entry.episodes !== entry.played + selfPlayed) throw new LeagueRunRecordError("a member's episodes must be its played games plus its played self-play games");
  checkCauses(entry.forfeit_causes, entry.forfeits, "a member's");
  checkBaselines(entry.baselines, ops, "a member's ");
}

/** Members name each other consistently: the pair id of a named member, and equal counts from both sides. */
function checkMemberLinks(members: MemberEntry[]): void {
  const byId = new Map(members.map((entry) => [entry.member, entry]));
  for (const entry of members) {
    for (const o of entry.opponents) {
      if (o.member === undefined) continue;
      const other = byId.get(o.member);
      if (other === undefined) throw new LeagueRunRecordError("an opponent names a member the record does not hold");
      if (o.opponent_policy_id !== other.pair_policy_id) throw new LeagueRunRecordError("an opponent's policy id must be the named member's pair_policy_id");
      if (other === entry) continue;
      const back = other.opponents.find((p) => p.member === entry.member);
      if (back === undefined || back.games !== o.games || back.played !== o.played) {
        throw new LeagueRunRecordError("two members' entries about each other must agree in games and played");
      }
    }
  }
}

/** The root totals count distinct games: the members' sums less each member-vs-member game counted twice. */
function checkTotals(record: LeagueRunRecordV2): void {
  const { members } = record;
  let games = 0;
  let played = 0;
  for (const entry of members) {
    games += entry.games;
    played += entry.played;
    for (const o of entry.opponents) {
      // Count a pair of members once: from the side of the lower-ranked member.
      if (o.member !== undefined && o.member !== entry.member && memberOrder(o.member, entry.member) > 0) {
        games -= o.games;
        played -= o.played;
      }
    }
  }
  if (record.games !== games || record.played !== played) throw new LeagueRunRecordError("the root totals must count distinct games: the members' sums less the games between two members");
  if (record.played + record.forfeits !== record.games) throw new LeagueRunRecordError("the record's played + forfeits must equal games");
  if (record.episodes !== 2 * record.played) throw new LeagueRunRecordError("the record's episodes must be two per played game");
  checkCauses(record.forfeit_causes, record.forfeits, "the record's");
  // A cause of the root is one some member has; a distinct game counts once at the root and once or twice among the members.
  const bounds = new Map<string, { max: number; sum: number }>();
  for (const entry of members) {
    for (const c of entry.forfeit_causes ?? []) {
      const seen = bounds.get(c.cause) ?? { max: 0, sum: 0 };
      bounds.set(c.cause, { max: Math.max(seen.max, c.count), sum: seen.sum + c.count });
    }
  }
  const listed = record.forfeit_causes ?? [];
  if (listed.length !== bounds.size || listed.some((c) => c.count < (bounds.get(c.cause)?.max ?? Infinity) || c.count > (bounds.get(c.cause)?.sum ?? -1))) {
    throw new LeagueRunRecordError("each root forfeit cause must be one a member has, between the largest member count and the sum over the members");
  }
}

/** Read a version 2 record: the schema, then the member, link and total rules the profile cannot say. */
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
  checkMemberLinks(record.members);
  checkTotals(record);
  return record;
}

/** Read a record of either version, by its `schema_version`; another version is refused. */
export function readAnyLeagueRunRecord(value: unknown): LeagueRunRecord | LeagueRunRecordV2 {
  const version = typeof value === "object" && value !== null ? (value as { schema_version?: unknown }).schema_version : undefined;
  if (version === 1) return readLeagueRunRecord(value);
  if (version === 2) return readLeagueRunRecordV2(value);
  throw new LeagueRunRecordError("a LeagueRunRecord has schema_version 1 or 2");
}

/** What a seat of a population game was: a member (by its label) or a scripted baseline (by its policy id). */
export type FactSeat = { member: string } | { baseline: string };

export interface LeagueRunFactsV2 {
  config: TrainingRunConfig;
  generation: number;
  /** The match plan, as `plan.json` holds it. */
  plan: readonly unknown[];
  /** The learning members, in order m0.., with the frozen pair each plays this generation. */
  members: readonly { member: string; pair_policy_id: string; checkpoints: { preview: string; battle: string } }[];
  /** One entry per game: its seats, whether it was played to an end, and the baseline's report when a baseline took a seat. */
  games: readonly {
    seats: readonly [FactSeat, FactSeat];
    outcome: "played" | "forfeit";
    forfeit?: { cause: ForfeitCause };
    /** The baseline seat's report, attached to the member that played it. */
    baseline_report?: BaselineGameReport;
  }[];
}

const countCauses = (causes: Map<string, number>): { cause: string; count: number }[] =>
  [...causes.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([cause, count]) => ({ cause, count }));

/**
 * Build the version 2 record of a population generation from what it played (the counting rule is the
 * schema's): a member's entry holds every game in which it has a seat, a game between two members is in both
 * entries, a self-play game is in one with the member itself as opponent, and the root counts distinct games.
 * It is read back by its own reader, so a bad record cannot leave.
 */
export async function buildLeagueRunRecordV2(facts: LeagueRunFactsV2): Promise<LeagueRunRecordV2> {
  const { config, games } = facts;
  const byMember = new Map(facts.members.map((entry) => [entry.member, entry]));
  const tallies = new Map(
    facts.members.map((entry) => [
      entry.member,
      { games: 0, played: 0, forfeits: 0, episodes: 0, causes: new Map<string, number>(), opponents: new Map<string, MemberOpponent>(), reports: [] as BaselineGameReport[] },
    ]),
  );
  const rootCauses = new Map<string, number>();
  let played = 0;
  for (const game of games) {
    const isPlayed = game.outcome === "played";
    if (isPlayed) played += 1;
    else rootCauses.set((game.forfeit as { cause: ForfeitCause }).cause, (rootCauses.get((game.forfeit as { cause: ForfeitCause }).cause) ?? 0) + 1);
    const members = [...new Set(game.seats.flatMap((seat) => ("member" in seat ? [seat.member] : [])))];
    for (const label of members) {
      const tally = tallies.get(label);
      if (tally === undefined) throw new LeagueRunRecordError("a game names a member the record does not hold");
      tally.games += 1;
      if (isPlayed) {
        tally.played += 1;
        tally.episodes += game.seats.filter((seat) => "member" in seat && seat.member === label).length;
      } else {
        tally.forfeits += 1;
        const cause = (game.forfeit as { cause: ForfeitCause }).cause;
        tally.causes.set(cause, (tally.causes.get(cause) ?? 0) + 1);
      }
      // The opponent is the other seat; in self-play it is this member itself.
      const [first, second] = game.seats;
      const other = "member" in first && first.member === label ? second : first;
      const key = "member" in other ? `m:${other.member}` : `b:${other.baseline}`;
      const entry: MemberOpponent =
        tally.opponents.get(key) ??
        ("member" in other
          ? { opponent_policy_id: (byMember.get(other.member) as { pair_policy_id: string }).pair_policy_id, member: other.member, games: 0, played: 0 }
          : { opponent_policy_id: other.baseline, games: 0, played: 0 });
      entry.games += 1;
      if (isPlayed) entry.played += 1;
      tally.opponents.set(key, entry);
      if (game.baseline_report !== undefined && "baseline" in other) tally.reports.push(game.baseline_report);
    }
  }
  return readLeagueRunRecordV2({
    schema_version: 2,
    run_id: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    generation: facts.generation,
    seed_derivation: LEAGUE_SEED_DERIVATION,
    plan_hash: await nodeSha256(new TextEncoder().encode(jcs(facts.plan as unknown as Json))),
    games: games.length,
    played,
    forfeits: games.length - played,
    ...(games.length > played ? { forfeit_causes: countCauses(rootCauses) } : {}),
    episodes: 2 * played,
    copy_decision_hash: null,
    members: facts.members.map((entry) => {
      const tally = tallies.get(entry.member) as NonNullable<ReturnType<typeof tallies.get>>;
      return {
        member: entry.member,
        pair_policy_id: entry.pair_policy_id,
        checkpoints: entry.checkpoints,
        games: tally.games,
        played: tally.played,
        forfeits: tally.forfeits,
        ...(tally.forfeits > 0 ? { forfeit_causes: countCauses(tally.causes) } : {}),
        episodes: tally.episodes,
        opponents: [...tally.opponents.values()].sort(opponentOrder),
        ...(tally.reports.length > 0 ? { baselines: baselineEntries(tally.reports) } : {}),
      };
    }),
  });
}
