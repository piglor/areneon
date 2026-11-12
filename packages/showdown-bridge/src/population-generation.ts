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
// Candidate features (#1333): a run that names a feature set gives every seat that reads a worker (a member's, a snapshot's)
// the rows of that set, built at play with the same `FeatureConfigs` the single-lineage generation uses; the loader
// recomputes them for each member's examples or reads the play-time sidecar the game processes left (`features_dir`).
//
// Game processes (#1333): the games are the plan's, and a game is played by the process whose slice holds it
// (`populationShards`, #1355: consecutive runs of the games ordered by the members that hold their seats, so a process needs
// the workers of few members); a further process leaves its entries (row, facts, the Episode files it wrote) and the process that
// started it merges them in plan order, so the generation's files are the same for any number of processes and workers.
//
// What this does not do (later slices): train, freeze or copy anything (the Python driver, slice 5), exploiter
// seats (the plan refuses them), or an opponent sheet (a population run that names one is refused).

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type BaselineOpponent, baselineActor, type DamageAwareSetup, baselineRecords, checkShard, forfeitCauses, GenerationError, IDLE_GRACE_MS, seatEpisodes, type Shard, type TrainingPool, watchIdle, writeDurable } from "./league-generation.ts";
import { createFeatureMemo, type FeatureConfigs } from "./candidate-features.ts";
import type { AnyCandidateFeatures } from "./feature-build.ts";
import { readFeatureRows, writeFeatureRows, writePlayRows } from "./feature-recompute.ts";
import { buildLeagueRunRecordV2, type FactSeat, type LeagueRunRecordV2 } from "./league-run-record.ts";
import { type PopulationGamePlan, type PopulationSeat, planPopulation, populationShards } from "./population-plan.ts";
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

export const POPULATION_SHEET_REFUSAL = "population: a run that names an opponent sheet is refused until the driver builds it per member";

/** One learning member's frozen pair and the worker that serves it. */
export interface PopulationMember {
  /** `m<k>`; the members are given in order m0.. */
  member: string;
  /** The pair's `behaviour_policy_id`; each member's must differ from every other's. */
  policy_id: string;
  checkpoints: { preview: string; battle: string };
  client: PolicyClient & { beginGame?: () => Promise<void> };
  /**
   * Further workers of the same frozen pair (`--policy-workers`): a game stays on one worker, and every worker answers a
   * query the same way (it is deterministic and the sampling draw is the harness's), so the generation's files do not
   * depend on how many there are. Waves grow with them so each keeps `n_parallel` games in flight.
   */
  peers?: readonly (PolicyClient & { beginGame?: () => Promise<void> })[];
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
  damage_aware_v4?: DamageAwareSetup;
  damage_aware_v5?: DamageAwareSetup;
  damage_aware_v2?: DamageAwareSetup;
  belief_worlds?: BeliefWorldsFactory | undefined;
  stream_limits?: ChunkLimits;
  /** Ends this process's games between waves. */
  signal?: AbortSignal;
  /**
   * The candidate features the run's pair reads (`featureSetup`): given to every seat that reads a worker (the workers state
   * the set at the handshake), and to the training loader, which recomputes the rows of each member's examples.
   */
  features?: FeatureConfigs;
  /**
   * With `features`: the scratch directory the game processes leave the rows of their Episodes in (the play-time sidecar and the
   * post-game pass of #1185, #1271), so the loader reads them instead of recomputing. Never part of a generation: the stream's
   * bytes do not depend on whether the files exist.
   */
  features_dir?: string | undefined;
  /** The generation's directory the Episodes are written below (a further game process writes into its starter's); default `out_dir`. */
  shared_dir?: string;
  /** The games this process plays (default all); the rest come from `others`. */
  shard?: Shard;
  /** The entries of the games of the other shards, played by further game processes. */
  others?: Promise<readonly PopulationEntry[]>;
  /** The number of waves in progress, for the idle watchdog (#1100). */
  activity?: { busy: number };
  /** How many further game processes are still running; the idle watchdog reads it. */
  live_children?: () => number;
  /** How long a generation may have no live game process and no game in flight before it fails (default 60 s). */
  idle_grace_ms?: number;
}

/** An Episode file a played game wrote: the member directory it is under, its hash and the policy that played the seat. */
export interface PopulationStored {
  owner: string;
  hash: string;
  behaviour_policy_id: string;
}

/** One played game as a game process hands it back: its row, what the run record needs of it, and the Episode files it wrote. */
export interface PopulationEntry {
  game: number;
  row: PopulationGameRow;
  fact: { outcome: "played" | "forfeit"; forfeit?: { cause: ForfeitCause }; baseline_report?: BaselineGameReport };
  stored: PopulationStored[];
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
 * an opponent sheet. The CLI calls it before it creates a client (each worker session costs memory).
 * `generation` and `poolSize` only make the plan well-formed; no refusal depends on them.
 */
export function checkPopulationConfig(config: TrainingRunConfig, members: readonly { member: string; policy_id: string }[], generation = 1, poolSize = 1, past?: PastPoolFile): void {
  const wantsPast = (config.population?.shares.past_pool_ppm ?? 0) !== 0;
  if (wantsPast !== (past !== undefined)) throw new GenerationError(wantsPast ? "population: a past-pool share needs the committed past pool (options.past)" : "population: a past pool was given to a config without a past-pool share");
  populationPlanOf(config, generation, poolSize, members, past);
  checkMembers(config, members);
}

/** The plan of a generation: `planPopulation` with the members' current pairs as the past pool's "current" policies. */
export function populationPlanOf(config: TrainingRunConfig, generation: number, poolSize: number, members: readonly { member: string; policy_id: string }[], past?: PastPoolFile): PopulationGamePlan[] {
  return planPopulation(config, generation, poolSize, past === undefined ? undefined : { pool: past, current: Object.fromEntries(members.map((entry) => [entry.member, entry.policy_id])) });
}

/** The members that hold a seat in a game of `shard` (all of them without one): the only members whose workers this process needs. */
export function membersPlaying(plan: readonly PopulationGamePlan[], shard: Shard = { index: 0, of: 1 }): Set<string> {
  const found = new Set<string>();
  const owner = populationShards(plan, shard.of);
  for (const [at, entry] of plan.entries()) if (owner[at] === shard.index) for (const seat of [entry.seats.p1, entry.seats.p2]) if ("member" in seat) found.add(seat.member);
  return found;
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
  if (config.model.encoder.options?.opponent_sheet !== undefined) {
    throw new GenerationError(POPULATION_SHEET_REFUSAL);
  }
}

/**
 * Waves of games: no worker serves more than `n_parallel` games of a wave (a game counts once for each member it uses).
 * `workers` says how many workers a label has (default 1: a snapshot has one), so a member with peers takes more games.
 */
export function populationWaves(plan: readonly PopulationGamePlan[], n_parallel: number, workers: (label: string) => number = () => 1): PopulationGamePlan[][] {
  const waves: PopulationGamePlan[][] = [];
  let rest = [...plan];
  while (rest.length > 0) {
    const load = new Map<string, number>();
    const wave: PopulationGamePlan[] = [];
    const left: PopulationGamePlan[] = [];
    for (const entry of rest) {
      const used = [...new Set([entry.seats.p1, entry.seats.p2].flatMap(clientLabel))];
      if (used.every((label) => (load.get(label) ?? 0) < n_parallel * workers(label))) {
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
  /** With a rows directory: the rows each member seat attached to its queries, by request number (the play-time sidecar). */
  play_rows?: Partial<Record<Player, Map<number, AnyCandidateFeatures>>>;
}

/**
 * Play one game: member seats query the worker chosen for their member, a baseline seat is a scripted actor behind the
 * latency guard (a fresh one per game, seeded by the pair's baseline stream).
 */
async function playOne(options: PopulationOptions, entry: PopulationGamePlan, clients: ReadonlyMap<string, PopulationMember["client"]>, byLabel: ReadonlyMap<string, PopulationMember>): Promise<PlayedPopulationGame> {
  const { engine, config, pool } = options;
  const { generation: settings } = config;
  const budget = { preview_ms: settings.latency_budget_ms.preview_ms, battle_ms: settings.latency_budget_ms.battle_ms };
  const play = settings.play?.temperature_ppm === undefined ? undefined : { temperature_ppm: settings.play.temperature_ppm };
  const keepRows = options.features !== undefined && options.features_dir !== undefined;
  const rows: Partial<Record<Player, Map<number, AnyCandidateFeatures>>> = {};
  let report: (() => BaselineGameReport) | undefined;
  const seatFor = (seat: PopulationSeat, player: Player): PolicySeat | ReturnType<typeof baselineActor>["actor"] => {
    if ("member" in seat || "snapshot" in seat) {
      const label = "member" in seat ? seat.member : seat.snapshot;
      const member = byLabel.get(label) as PopulationMember;
      const base = { client: clients.get(label) as PopulationMember["client"], policy_id: member.policy_id, sampling: "sample" as const, ...(play === undefined ? {} : { play }) };
      if (options.features === undefined) return base;
      // Every worker states the run's feature set, so every seat that reads one attaches the rows (a snapshot's seat too); only a member's own seat leaves them as a sidecar.
      const kept = new Map<number, AnyCandidateFeatures>();
      const record = keepRows && "member" in seat ? { record: (seq: number, found: AnyCandidateFeatures) => void kept.set(seq, found) } : {};
      if (Object.keys(record).length > 0) rows[player] = kept;
      return { ...base, features: { configs: options.features, memo: createFeatureMemo(), ...record } };
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
    seats: { p1: seatFor(entry.seats.p1, "p1"), p2: seatFor(entry.seats.p2, "p2") } as PolicyGameOptions["seats"],
    latency_budget: budget,
    maxTurns: settings.max_turns,
  });
  return { ...played, ...(report === undefined ? {} : { baseline: report() }), ...(keepRows ? { play_rows: rows } : {}) };
}

// A snapshot is an opponent that is neither a member nor a scripted baseline with a report: the record names it by its pair's policy id, as it names a baseline.
const factSeat = (seat: PopulationSeat): FactSeat => ("member" in seat ? { member: seat.member } : { baseline: seat.policy_id });

/**
 * Play the games of `shard` (all of them without one) against the members' workers in waves, writing each Episode under
 * `<shared_dir>/members/<member>/episodes/` as its game finishes, and return each played game's entry. With a rows directory
 * and a feature set, the rows of the members' own Episodes are left in it afterwards (the play-time sidecar while the game
 * runs, then the post-game pass for any that are missing). A game's result depends on its plan entry alone, never on its
 * worker, its wave or the other games, so any split of the plan gives the same entries.
 *
 * Refused: a harness incident in any game (nothing is retried), and everything `planPopulation` and the member checks refuse.
 */
export async function playPopulationEntries(options: PopulationOptions, shard: Shard = { index: 0, of: 1 }): Promise<PopulationEntry[]> {
  checkShard(shard);
  const { engine, config, generation, members, pool } = options;
  checkPopulationConfig(config, members, generation, pool.teams.length, options.past?.pool);
  const root = options.shared_dir ?? options.out_dir;
  const plan = populationPlanOf(config, generation, pool.teams.length, members, options.past?.pool);
  const owner = populationShards(plan, shard.of);
  const mine = plan.filter((_, at) => owner[at] === shard.index);
  const withRecords: PopulationOptions = { ...options, ...(baselineRecords(options, config) as Pick<PopulationOptions, "damage_aware" | "damage_aware_v2" | "damage_aware_v3" | "damage_aware_v4">) };
  const byLabel = new Map<string, PopulationMember>(members.map((entry) => [entry.member, entry]));
  const workersOf = (label: string): number => 1 + (byLabel.get(label)?.peers?.length ?? 0);
  // A snapshot's session lives from the first wave that uses it to the last (`last` is that wave's index).
  const waves = populationWaves(mine, config.generation.n_parallel, workersOf);
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
  for (const entry of members) mkdirSync(join(memberDir(root, entry.member), "episodes"), { recursive: true });
  const entries: PopulationEntry[] = [];
  const activity = options.activity ?? { busy: 0 };
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
      const workers = (label: string): PopulationMember["client"][] => {
        const member = byLabel.get(label) as PopulationMember;
        return [member.client, ...(member.peers ?? [])];
      };
      activity.busy++;
      let played: PlayedPopulationGame[];
      try {
        // The client's overrun rule: call `beginGame` while no game runs on it, so each wave starts every worker it uses afresh (a query that overran its budget may still be outstanding).
        await Promise.all(used.flatMap((label) => workers(label).map((client) => client.beginGame?.())));
        stopped();
        // A game stays on one worker of each member it uses, chosen by its place among the wave's games of that member.
        const places = new Map<string, number>();
        const chosen = wave.map((entry) => {
          const labels = [...new Set([entry.seats.p1, entry.seats.p2].flatMap(clientLabel))];
          return new Map(
            labels.map((label) => {
              const place = places.get(label) ?? 0;
              places.set(label, place + 1);
              const all = workers(label);
              return [label, all[place % all.length] as PopulationMember["client"]] as const;
            }),
          );
        });
        played = await Promise.all(wave.map((entry, index) => playOne(withRecords, entry, chosen[index] as Map<string, PopulationMember["client"]>, byLabel)));
      } finally {
        activity.busy--;
      }
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
          entries.push({
            game: entry.game,
            row: { ...base, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser, cause, request_seq, detail } },
            fact: { outcome: "forfeit", forfeit: { cause }, ...(baseline === undefined ? {} : { baseline_report: baseline }) },
            stored: [],
          });
          continue;
        }
        if (record === null) throw new GenerationError(`game ${entry.game} (${entry.seed}): a played game has no record`);
        const stored = await writeEpisodes(withRecords, root, entry, record, game.play_rows);
        entries.push({ game: entry.game, row: { ...base, outcome: "played", winner: record.winner, turns: record.turns }, fact: { outcome: "played", ...(baseline === undefined ? {} : { baseline_report: baseline }) }, stored });
      }
      for (const id of used) if (last.get(id) === waveIndex && sessions.has(id)) await closeSession(id);
    }
  } finally {
    await Promise.all([...sessions.keys()].map(closeSession));
  }
  if (options.features !== undefined && options.features_dir !== undefined) {
    // The post-game pass: the rows of each member's own Episodes that the play-time sidecar did not leave (a gap, a failure).
    const policyOf = new Map(members.map((entry) => [entry.member, entry.policy_id]));
    const own = entries.flatMap((entry) => entry.stored).filter((ref) => ref.behaviour_policy_id === policyOf.get(ref.owner));
    await writeFeatureRows(
      { engine, configs: options.features, known_synthetic: pool.known_synthetic },
      own.map(({ owner, hash }) => ({ hash, path: join(memberDir(root, owner), "episodes", `${hash}.json`) })),
      options.features_dir,
      () => options.signal?.aborted === true,
    );
  }
  return entries;
}

/**
 * Play the generation's population plan against the members' workers (this process's shard here, the rest through
 * `options.others`: entries played by further game processes), writing each member's Episodes under
 * `out_dir/members/<member>/episodes/` as the games finish, then each member's `stream.jsonl` and `training-set.json`.
 * The entries are put in plan order, so the result is the one a single process gives. Refused: a harness incident in any
 * game (nothing is retried), a game missing or played twice, a member with no Episode of its own pair, and everything
 * `planPopulation` and the member checks refuse.
 */
export async function playPopulationGeneration(options: PopulationOptions): Promise<PopulationResult> {
  const { engine, config, generation, members, pool, out_dir } = options;
  checkPopulationConfig(config, members, generation, pool.teams.length, options.past?.pool);
  const plan = populationPlanOf(config, generation, pool.teams.length, members, options.past?.pool);
  const stop = new AbortController();
  if (options.signal?.aborted === true) stop.abort();
  else options.signal?.addEventListener("abort", () => stop.abort(), { once: true });
  const activity = options.activity ?? { busy: 0 };
  const here = playPopulationEntries({ ...options, signal: stop.signal, activity }, options.shard);
  const there = options.others ?? Promise.resolve([]);
  here.catch(() => stop.abort()); // a failure of either side ends the other's games at once (#1100)
  there.catch(() => stop.abort());
  const idle = watchIdle(options.live_children, activity, options.idle_grace_ms ?? IDLE_GRACE_MS, there);
  let entries: PopulationEntry[];
  try {
    entries = (await Promise.race([Promise.all([here, there]), idle.tripped])).flat().sort((a, b) => a.game - b.game);
  } catch (error) {
    stop.abort();
    throw error;
  } finally {
    idle.stop();
  }
  if (entries.length !== plan.length || entries.some((entry, index) => entry.game !== index)) {
    throw new GenerationError("the played games are not the plan's games, each once");
  }
  const rows = new Map<number, PopulationGameRow>(entries.map((entry) => [entry.game, entry.row]));
  const facts = new Map(entries.map((entry) => [entry.game, entry.fact]));
  // The Episode files each member's directory holds: the hash and the policy that played the seat.
  const stored = new Map<string, { hash: string; behaviour_policy_id: string }[]>(members.map((entry) => [entry.member, []]));
  for (const entry of entries) for (const ref of entry.stored) stored.get(ref.owner)?.push({ hash: ref.hash, behaviour_policy_id: ref.behaviour_policy_id });
  const results: PopulationMemberResult[] = [];
  for (const entry of members) {
    const dir = memberDir(out_dir, entry.member);
    const refs = stored.get(entry.member) as { hash: string; behaviour_policy_id: string }[];
    const own = refs.filter((ref) => ref.behaviour_policy_id === entry.policy_id);
    if (own.length === 0) throw new GenerationError(`no Episode of ${entry.member}'s frozen pair: every game it played was forfeited`);
    const episodes: StoredEpisode[] = own.map(({ hash }) => ({ hash, path: join(dir, "episodes", `${hash}.json`) }));
    const set = await writeStoredTrainingStream(
      {
        episodes,
        evaluation_manifests: [],
        known_synthetic: pool.known_synthetic,
        behaviour_policy_id: entry.policy_id,
        ...(options.features === undefined ? {} : { features: { engine, configs: options.features, ...(options.features_dir === undefined ? {} : { precomputed: readFeatureRows(engine, options.features, options.features_dir) }) } }),
      },
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
 * the game was scheduled for. Each is flushed before the next wave starts (memory does not grow with games). A member's
 * own seat also leaves its play-time rows as the Episode's rows file (a gap or a failure leaves nothing: the post-game pass builds them).
 */
async function writeEpisodes(options: PopulationOptions, root: string, entry: PopulationGamePlan, record: BattleRecord, playedRows: PlayedPopulationGame["play_rows"]): Promise<PopulationStored[]> {
  const policyOf = (seat: PopulationSeat): string => ("member" in seat ? (options.members.find((m) => m.member === seat.member) as PopulationMember).policy_id : seat.policy_id);
  const sides: Record<Player, string> = { p1: policyOf(entry.seats.p1), p2: policyOf(entry.seats.p2) };
  const episodes = await seatEpisodes(record, entry.teams, options.pool, options.config.seeds.run_key, sides);
  const stored: PopulationStored[] = [];
  for (const [seat, episode] of episodes.entries()) {
    const player: Player = seat === 0 ? "p1" : "p2";
    const holder = entry.seats[player];
    const owner = "member" in holder ? holder.member : entry.member;
    const hash = await episodeHash(episode, nodeSha256);
    writeDurable(join(memberDir(root, owner), "episodes", `${hash}.json`), `${jcs(episode as unknown as Json)}\n`);
    stored.push({ owner, hash, behaviour_policy_id: episode.provenance.behaviour_policy_id });
    const kept = playedRows?.[player];
    if (kept !== undefined && options.features !== undefined && options.features_dir !== undefined) {
      // A failure here skips the sidecar (the post-game pass builds the rows) and never fails the generation; it is logged without any content.
      try {
        await writePlayRows({ engine: options.engine, configs: options.features }, episode, hash, kept, options.features_dir);
      } catch (error) {
        process.stderr.write(`rows sidecar skipped for game ${entry.game}: ${(error as Error).name}\n`);
      }
    }
  }
  return stored;
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
