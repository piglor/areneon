// One population generation of A1 play (ADR-021 sections 2, 3, 5 and 8, slice 4): every learning member's
// frozen pair has its own policy worker, the plan (`planPopulation`) says who sits where, and each seat of a
// game queries its own member's client, so a game's two seats may be two members. Every played game's
// Episodes are written at once under `members/m<k>/episodes/` of the generation's directory, under the member
// that holds the seat (a scripted baseline's seat under the member it was scheduled for), and each member gets
// its own stream and TrainingSet through the unchanged loader, which accepts an Episode only for the policy
// that played it. The v2 league run record, `plan.json` and (last) `generation.json` follow.
//
// A past-pool pair (slice 10) puts a frozen snapshot of a member's pair on one seat: it has its own worker session,
// started when the first wave that needs it begins and closed after the last, so at most the snapshots in use are
// resident. Its Episodes are kept under the member the pair was scheduled for and are never loaded into a TrainingSet
// (the loader accepts an Episode only for the policy that played it), as a scripted baseline's are.
//
// What this does not do (later slices): train, freeze or copy anything (the Python driver, slice 5), exploiter
// seats (the plan refuses them), candidate features or an opponent sheet (a population run
// that names either is refused), or split a generation over game processes.

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type BaselineOpponent, baselineActor, type DamageAwareSetup, baselineRecords, forfeitCauses, GenerationError, seatEpisodes, type TrainingPool, writeDurable } from "./league-generation.ts";
import { buildLeagueRunRecordV2, type FactSeat, type LeagueRunRecordV2 } from "./league-run-record.ts";
import { type PastPlanInput, type PopulationGamePlan, type PopulationSeat, planPopulation } from "./population-plan.ts";
import type { PastPoolFile, PastSnapshot } from "./past-pool.ts";
import { episodeHash } from "./episode.ts";
import type { PlayedGame } from "./evaluation-harness.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { nodeSha256 } from "./player-channel.ts";
import { type PolicyClient } from "./policy-worker.ts";
import { type PolicyGameOptions, policyGame, type PolicySeat } from "./policy-actor.ts";
import type { BattleRecord, ForfeitCause, Player } from "./runner.ts";
import { type StoredEpisode, type TrainingSet, writeStoredTrainingStream } from "./training-loader.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import type { ChunkLimits } from "./chunked-file.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import type { BaselineGameReport } from "./league-run-record.ts";

export const POPULATION_FEATURES_REFUSAL = "population: a run that names candidate features or an opponent sheet is refused until the driver builds them per member";

/** One learning member's frozen pair and the worker that serves it. */
export interface PopulationMember {
  /** `m<k>`; the members are given in order m0.. */
  member: string;
  /** The pair's `behaviour_policy_id`; each member's must differ from every other's. */
  policy_id: string;
  checkpoints: { preview: string; battle: string };
  client: PolicyClient & { beginGame?: () => Promise<void> };
}

/** The frozen worker session of a snapshot, and how to end it. */
export interface SnapshotSession {
  client: PopulationMember["client"];
  close: () => Promise<void>;
}

export interface PastOptions {
  pool: PastPoolFile;
  /** Start the frozen session of a snapshot; the generation closes it after the last wave that uses it. */
  open: (snapshot: PastSnapshot) => Promise<SnapshotSession>;
}

export interface PopulationOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  members: readonly PopulationMember[];
  pool: TrainingPool;
  /** The generation's directory (a partial one): the members' directories and streams are written below it as the games finish. */
  out_dir: string;
  /** The committed past pool of the generation before; required exactly when the config has a past-pool share. */
  past?: PastOptions;
  damage_aware?: DamageAwareSetup;
  damage_aware_v3?: DamageAwareSetup;
  damage_aware_v2?: DamageAwareSetup;
  belief_worlds?: BeliefWorldsFactory | undefined;
  stream_limits?: ChunkLimits;
  /** Ends this process's games between waves. */
  signal?: AbortSignal;
}

export interface PopulationGameRow {
  game: number;
  pair: number;
  half: 0 | 1;
  seed: string;
  kind: PopulationGamePlan["kind"];
  /** The member the game was scheduled for. */
  member: string;
  /** The other seat: a member label (this member's own, for self-play) or a baseline's policy id. */
  opponent: string;
  seats: Record<Player, string>;
  outcome: "played" | "forfeit";
  winner: Player | null;
  turns: number;
  forfeit?: { loser: Player; cause: ForfeitCause; request_seq: number; detail: string };
}

export interface PopulationMemberResult {
  member: string;
  policy_id: string;
  set: TrainingSet;
  /** SHA-256 of the TrainingSet's JCS bytes. */
  training_set_hash: string;
  /** Hashes of every Episode file under this member's directory, sorted. */
  episode_hashes: string[];
}

export interface PopulationResult {
  generation: number;
  plan: PopulationGamePlan[];
  games: PopulationGameRow[];
  members: PopulationMemberResult[];
  run_record: LeagueRunRecordV2;
  manifest: VersionManifest;
  /** Hashes of the teams the plan drew, sorted and unique. */
  team_hashes: string[];
}

/** The directory of one member's files below a generation's directory. */
export const memberDir = (dir: string, member: string): string => join(dir, "members", member);

const seatName = (seat: PopulationSeat): string => ("member" in seat ? seat.member : "snapshot" in seat ? seat.snapshot : seat.policy_id);

/** The worker-served seat a game uses, by label: a member's label or a snapshot's id (a scripted baseline has none). */
const clientLabel = (seat: PopulationSeat): string[] => ("member" in seat ? [seat.member] : "snapshot" in seat ? [seat.snapshot] : []);

/**
 * Everything that can be refused before a worker starts: the plan's own refusals (past-pool share, exploiter seats, odd
 * baseline counts), the members (one per `population.members`, labelled m0.., with distinct pairs) and a run that names
 * candidate features or an opponent sheet. The CLI calls it before it creates a client (each worker session costs memory).
 * `generation` and `poolSize` only make the plan well-formed; no refusal depends on them.
 */
export function checkPopulationConfig(config: TrainingRunConfig, members: readonly { member: string; policy_id: string }[], generation = 1, poolSize = 1, past?: PastPoolFile): void {
  const wantsPast = (config.population?.shares.past_pool_ppm ?? 0) !== 0;
  if (wantsPast !== (past !== undefined)) throw new GenerationError(wantsPast ? "population: a past-pool share needs the committed past pool (options.past)" : "population: a past pool was given to a config without a past-pool share");
  planPopulation(config, generation, poolSize, past === undefined ? undefined : { pool: past, current: Object.fromEntries(members.map((entry) => [entry.member, entry.policy_id])) });
  checkMembers(config, members);
}

function checkMembers(config: TrainingRunConfig, members: readonly { member: string; policy_id: string }[]): void {
  const population = config.population;
  if (population === undefined) throw new GenerationError("population: the config has no population section");
  if (members.length !== population.members || members.some((entry, index) => entry.member !== `m${index}`)) {
    throw new GenerationError("population: the members must be m0.. in order, as many as population.members");
  }
  if (new Set(members.map((entry) => entry.policy_id)).size !== members.length) {
    throw new GenerationError("population: two members have the same frozen pair, so their Episodes could not be told apart");
  }
  if (config.model.encoder.options?.candidate_features !== undefined || config.model.encoder.options?.opponent_sheet !== undefined) {
    throw new GenerationError(POPULATION_FEATURES_REFUSAL);
  }
}

/** Waves of games: no client serves more than `n_parallel` games of a wave (a game counts once for each member it uses). */
export function populationWaves(plan: readonly PopulationGamePlan[], n_parallel: number): PopulationGamePlan[][] {
  const waves: PopulationGamePlan[][] = [];
  let rest = [...plan];
  while (rest.length > 0) {
    const load = new Map<string, number>();
    const wave: PopulationGamePlan[] = [];
    const left: PopulationGamePlan[] = [];
    for (const entry of rest) {
      const used = [...new Set([entry.seats.p1, entry.seats.p2].flatMap(clientLabel))];
      if (used.every((label) => (load.get(label) ?? 0) < n_parallel)) {
        for (const label of used) load.set(label, (load.get(label) ?? 0) + 1);
        wave.push(entry);
      } else {
        left.push(entry);
      }
    }
    waves.push(wave);
    rest = left;
  }
  return waves;
}

interface PlayedPopulationGame extends PlayedGame {
  baseline?: BaselineGameReport;
}

/**
 * Play one game: member seats query their own member's client, a baseline seat is a scripted actor behind the
 * latency guard (a fresh one per game, seeded by the pair's baseline stream).
 */
async function playOne(options: PopulationOptions, entry: PopulationGamePlan, byLabel: ReadonlyMap<string, PopulationMember>): Promise<PlayedPopulationGame> {
  const { engine, config, pool } = options;
  const { generation: settings } = config;
  const budget = { preview_ms: settings.latency_budget_ms.preview_ms, battle_ms: settings.latency_budget_ms.battle_ms };
  const play = settings.play?.temperature_ppm === undefined ? undefined : { temperature_ppm: settings.play.temperature_ppm };
  let report: (() => BaselineGameReport) | undefined;
  const seatFor = (seat: PopulationSeat): PolicySeat | ReturnType<typeof baselineActor>["actor"] => {
    if ("member" in seat || "snapshot" in seat) {
      const member = byLabel.get("member" in seat ? seat.member : seat.snapshot) as PopulationMember;
      return { client: member.client, policy_id: member.policy_id, sampling: "sample", ...(play === undefined ? {} : { play }) };
    }
    const opponent: BaselineOpponent = { actor: seat.actor, policy_id: seat.policy_id };
    const baseline = baselineActor(options, opponent, entry, budget);
    report = baseline.report;
    return baseline.actor;
  };
  const played = await policyGame({
    engine,
    formatId: config.format_id,
    seed: entry.seed,
    game: entry.game,
    teams: { p1: pool.teams[entry.teams.p1] as PokemonSet[], p2: pool.teams[entry.teams.p2] as PokemonSet[] },
    known_synthetic: pool.known_synthetic,
    seats: { p1: seatFor(entry.seats.p1), p2: seatFor(entry.seats.p2) } as PolicyGameOptions["seats"],
    latency_budget: budget,
    maxTurns: settings.max_turns,
  });
  return { ...played, ...(report === undefined ? {} : { baseline: report() }) };
}

// A snapshot is an opponent that is neither a member nor a scripted baseline with a report: the record names it by its pair's policy id, as it names a baseline.
const factSeat = (seat: PopulationSeat): FactSeat => ("member" in seat ? { member: seat.member } : { baseline: seat.policy_id });

/**
 * Play the generation's population plan against the members' workers, writing each member's Episodes under
 * `out_dir/members/<member>/episodes/` as the games finish, then each member's `stream.jsonl` and
 * `training-set.json`. Refused: a harness incident in any game (nothing is retried), a member with no Episode of
 * its own pair, and everything `planPopulation` and the member checks refuse.
 */
export async function playPopulationGeneration(options: PopulationOptions): Promise<PopulationResult> {
  const { engine, config, generation, members, pool, out_dir } = options;
  checkPopulationConfig(config, members, generation, pool.teams.length, options.past?.pool);
  const pastInput: PastPlanInput | undefined = options.past === undefined ? undefined : { pool: options.past.pool, current: Object.fromEntries(members.map((entry) => [entry.member, entry.policy_id])) };
  const plan = planPopulation(config, generation, pool.teams.length, pastInput);
  const withRecords: PopulationOptions = { ...options, ...baselineRecords(options, config) };
  const byLabel = new Map<string, PopulationMember>(members.map((entry) => [entry.member, entry]));
  // A snapshot's session lives from the first wave that uses it to the last (`last` is that wave's index).
  const waves = populationWaves(plan, config.generation.n_parallel);
  const snapshots = new Map(options.past?.pool.snapshots.map((entry) => [entry.id, entry]));
  const last = new Map<string, number>();
  waves.forEach((wave, index) => {
    for (const entry of wave) for (const seat of [entry.seats.p1, entry.seats.p2]) if ("snapshot" in seat) last.set(seat.snapshot, index);
  });
  const sessions = new Map<string, SnapshotSession>();
  const closeSession = async (id: string): Promise<void> => {
    const session = sessions.get(id);
    sessions.delete(id);
    byLabel.delete(id);
    await session?.close();
  };
  for (const entry of members) mkdirSync(join(memberDir(out_dir, entry.member), "episodes"), { recursive: true });
  const rows = new Map<number, PopulationGameRow>();
  const facts = new Map<number, { outcome: "played" | "forfeit"; forfeit?: { cause: ForfeitCause }; baseline_report?: BaselineGameReport }>();
  // The Episode files each member's directory holds: the hash and the policy that played the seat.
  const stored = new Map<string, { hash: string; behaviour_policy_id: string }[]>(members.map((entry) => [entry.member, []]));
  const stopped = (): void => {
    if (options.signal?.aborted === true) throw new GenerationError("the generation was stopped: this process plays no further game");
  };
  try {
    for (const [waveIndex, wave] of waves.entries()) {
      stopped();
      const used = [...new Set(wave.flatMap((entry) => [entry.seats.p1, entry.seats.p2].flatMap(clientLabel)))];
      for (const id of used) {
        if (byLabel.has(id)) continue;
        const snapshot = snapshots.get(id) as PastSnapshot;
        const session = await (options.past as PastOptions).open(snapshot);
        sessions.set(id, session);
        byLabel.set(id, { member: id, policy_id: snapshot.policy_id, checkpoints: { preview: snapshot.preview, battle: snapshot.battle }, client: session.client });
      }
      // The client's overrun rule: call `beginGame` while no game runs on it, so each wave starts every client it uses afresh (a query that overran its budget may still be outstanding).
      await Promise.all(used.map((label) => (byLabel.get(label) as PopulationMember).client.beginGame?.()));
      stopped();
      const played = await Promise.all(wave.map((entry) => playOne(withRecords, entry, byLabel)));
      stopped();
      for (const [index, game] of played.entries()) {
        const entry = wave[index] as PopulationGamePlan;
        const { outcome, record, baseline } = game;
        if (outcome.kind === "incident") throw new GenerationError(`game ${entry.game} (${entry.seed}): ${outcome.incident}: ${outcome.message}`);
        const base = {
          game: entry.game,
          pair: entry.pair,
          half: entry.half,
          seed: entry.seed,
          kind: entry.kind,
          member: entry.member,
          opponent: entry.kind === "self" ? entry.member : seatName(entry.seats[entry.half === 0 ? "p2" : "p1"]),
          seats: { p1: seatName(entry.seats.p1), p2: seatName(entry.seats.p2) },
        };
        if (outcome.kind === "forfeit") {
          const { loser, cause, request_seq, detail } = outcome;
          rows.set(entry.game, { ...base, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser, cause, request_seq, detail } });
          facts.set(entry.game, { outcome: "forfeit", forfeit: { cause }, ...(baseline === undefined ? {} : { baseline_report: baseline }) });
          continue;
        }
        if (record === null) throw new GenerationError(`game ${entry.game} (${entry.seed}): a played game has no record`);
        rows.set(entry.game, { ...base, outcome: "played", winner: record.winner, turns: record.turns });
        facts.set(entry.game, { outcome: "played", ...(baseline === undefined ? {} : { baseline_report: baseline }) });
        await writeEpisodes(withRecords, entry, record, stored);
      }
      for (const id of used) if (last.get(id) === waveIndex && sessions.has(id)) await closeSession(id);
    }
  } finally {
    await Promise.all([...sessions.keys()].map(closeSession));
  }
  const results: PopulationMemberResult[] = [];
  for (const entry of members) {
    const dir = memberDir(out_dir, entry.member);
    const refs = stored.get(entry.member) as { hash: string; behaviour_policy_id: string }[];
    const own = refs.filter((ref) => ref.behaviour_policy_id === entry.policy_id);
    if (own.length === 0) throw new GenerationError(`no Episode of ${entry.member}'s frozen pair: every game it played was forfeited`);
    const episodes: StoredEpisode[] = own.map(({ hash }) => ({ hash, path: join(dir, "episodes", `${hash}.json`) }));
    const set = await writeStoredTrainingStream(
      { episodes, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: entry.policy_id },
      { ...(options.stream_limits ?? {}), path: join(dir, "stream.jsonl") },
    );
    const training_set_hash = await nodeSha256(new TextEncoder().encode(jcs(set as unknown as Json)));
    writeDurable(join(dir, "training-set.json"), `${jcs(set as unknown as Json)}\n`);
    results.push({ member: entry.member, policy_id: entry.policy_id, set, training_set_hash, episode_hashes: refs.map((ref) => ref.hash).sort() });
  }
  const games = plan.map((entry) => rows.get(entry.game) as PopulationGameRow);
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) drawn.add(pool.hashes[entry.teams[player]] as string);
  return {
    generation,
    plan,
    games,
    members: results,
    run_record: await buildLeagueRunRecordV2({
      config,
      generation,
      plan,
      members: members.map(({ member, policy_id, checkpoints }) => ({ member, pair_policy_id: policy_id, checkpoints })),
      games: plan.map((entry) => ({ seats: [factSeat(entry.seats.p1), factSeat(entry.seats.p2)] as const, ...(facts.get(entry.game) as NonNullable<ReturnType<typeof facts.get>>) })),
    }),
    manifest: resolveFormat(engine, config.format_id).manifest,
    team_hashes: [...drawn].sort(),
  };
}

/**
 * Write the Episodes of a played game: a member's seat under that member, a baseline's seat under the member
 * the game was scheduled for. Each is flushed before the next wave starts (memory does not grow with games).
 */
async function writeEpisodes(options: PopulationOptions, entry: PopulationGamePlan, record: BattleRecord, stored: Map<string, { hash: string; behaviour_policy_id: string }[]>): Promise<void> {
  const policyOf = (seat: PopulationSeat): string => ("member" in seat ? (options.members.find((m) => m.member === seat.member) as PopulationMember).policy_id : seat.policy_id);
  const sides: Record<Player, string> = { p1: policyOf(entry.seats.p1), p2: policyOf(entry.seats.p2) };
  const episodes = await seatEpisodes(record, entry.teams, options.pool, options.config.seeds.run_key, sides);
  for (const [seat, episode] of episodes.entries()) {
    const holder = entry.seats[seat === 0 ? "p1" : "p2"];
    const owner = "member" in holder ? holder.member : entry.member;
    const hash = await episodeHash(episode, nodeSha256);
    writeDurable(join(memberDir(options.out_dir, owner), "episodes", `${hash}.json`), `${jcs(episode as unknown as Json)}\n`);
    (stored.get(owner) as { hash: string; behaviour_policy_id: string }[]).push({ hash, behaviour_policy_id: episode.provenance.behaviour_policy_id });
  }
}

/**
 * Write the generation's top-level files below `dir` (the members' files are already there): `plan.json`,
 * `league-run.json` (version 2) and, last, `generation.json`, the file that says the rest is whole.
 */
export function writePopulationGeneration(dir: string, result: PopulationResult): void {
  mkdirSync(dir, { recursive: true });
  const put = (name: string, value: unknown): void => writeDurable(join(dir, name), `${jcs(value as Json)}\n`);
  put("plan.json", result.plan);
  put("league-run.json", result.run_record);
  const forfeited = result.games.some((game) => game.outcome === "forfeit");
  put("generation.json", {
    schema_version: 2,
    generation: result.generation,
    games: result.games,
    forfeits: result.games.filter((game) => game.outcome === "forfeit").length,
    ...(forfeited ? { forfeit_causes: forfeitCauses(result.games) } : {}),
    members: result.members.map((entry) => ({ member: entry.member, episode_hashes: entry.episode_hashes, training_set_hash: entry.training_set_hash, examples: entry.set.examples })),
    manifest: result.manifest,
    team_hashes: result.team_hashes,
  });
}
