// One generation of the training loop's A1 play (ADR-010 §1): the frozen pair
// plays its match plan, every seat's Episode is written (a league collection
// record, the seed the battle ran with), and only the Episodes of the pair's
// own seats go through the training loader, which refuses any other policy.
// The plan is a pure function of the run config, the generation and the team
// pool, so a generation that was killed is replayed from the same plan.
//
// Teams are drawn from the committed synthetic fixtures (the golden team, the
// legal paste fixtures and the random-play corpus): the ADR-013 team
// generator is not built yet, and this pool is the interim. Every team drawn
// is reported by its hash, which becomes the team-set manifest of the run.

import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type DamageAwareActor, type DamageAwareOptions, damageAwareActor } from "./damage-aware-actor.ts";
import type { FeatureConfigs } from "./candidate-features.ts";
import { configuredSetHash } from "./feature-build.ts";
import { loadDamageAwareConfig, loadDamageAwareConfigV2 } from "./damage-aware-config.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { damageAwarePolicyId } from "./damage-aware-identity.ts";
import { type BaselineGameReport, buildLeagueRunRecord, type LeagueRunRecord } from "./league-run-record.ts";
import { loadMechanicsConfig } from "./mechanics-config.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { leagueGameStreams } from "./league-seed.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { episodeHash } from "./episode.ts";
import { guardActor, type LatencyBudget, type PlayedGame } from "./evaluation-harness.ts";
import { firstLegal, GOLDEN_FORMAT, goldenTeamVocabulary, syntheticTeam, TEAMS_DIR, teamFixtureManifest } from "./golden.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { nodeSha256 } from "./player-channel.ts";
import { type PolicyClient } from "./policy-worker.ts";
import { type PolicyGameOptions, policyGame, type PolicySeat } from "./policy-actor.ts";
import { readerSha256 } from "@areneon/public-state";
import { loadRandomCorpus } from "./random-corpus.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, BattleRecord, ForfeitCause, Player } from "./runner.ts";
import { importPaste } from "./teams.ts";
import { stampTeam, type SyntheticTeamHashes, teamHash } from "./team-class.ts";
import { loadTrainingSet, type StoredEpisode, type TrainingSet, writeStoredTrainingStream, writeTrainingStream } from "./training-loader.ts";
import { type ChunkLimits } from "./chunked-file.ts";
import { readFeatureRows, writeFeatureRows } from "./feature-recompute.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import type { Episode } from "./episode.ts";

/** The teams a run may draw: synthetic by source, in a fixed order. */
export interface TrainingPool {
  /** Sorted by team hash, so an index names the same team on every machine. */
  teams: readonly PokemonSet[][];
  hashes: readonly string[];
  /** Every team of the pool, proven synthetic by where it was read from. */
  known_synthetic: SyntheticTeamHashes;
}

/**
 * The interim team pool (the team generator is ADR-013, not built): the golden
 * team, every legal paste fixture and every team of the random-play corpus.
 * All three are committed under `fixtures/` as CC0 synthetic data. A team with
 * a set whose ability is in `excludedAbilities` (the run config's list) is left out.
 */
export function trainingTeamPool(engine: Engine, excludedAbilities: readonly string[] = []): TrainingPool {
  const byHash = new Map<string, PokemonSet[]>();
  const add = (team: PokemonSet[]): void => void byHash.set(teamHash(team), team);
  add(syntheticTeam());
  const manifest = teamFixtureManifest();
  for (const [file, fixture] of Object.entries(manifest.fixtures)) {
    if (fixture.expect === "legal") add(importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8")));
  }
  for (const battle of loadRandomCorpus()) {
    add(battle.teams.p1);
    add(battle.teams.p2);
  }
  const excluded = new Set(excludedAbilities);
  const hashes = [...byHash.keys()].filter((hash) => !(byHash.get(hash) as PokemonSet[]).some((set) => excluded.has(set.ability))).sort();
  if (hashes.length === 0) throw new GenerationError("the team pool is empty after the excluded abilities");
  return { teams: hashes.map((hash) => byHash.get(hash) as PokemonSet[]), hashes, known_synthetic: new Set(hashes) };
}

/** What a team pool is built from: the part of a run config (or a teacher run) that names the format, the dataset and the abilities left out. */
export interface PoolRequest {
  format_id: string;
  team_dataset?: { path: string; dataset_hash: string };
  generation: { excluded_abilities: readonly string[] };
}

/** The scripted opponent of a baseline game; the policy takes the other seat. */
export interface BaselineOpponent {
  actor: "actor-rule" | "random-legal" | "damage-aware";
  policy_id: string;
}

/** What a damage-aware baseline seat is built from (everything of `DamageAwareOptions` but the engine and the per-game seed). */
export type DamageAwareSetup = Omit<DamageAwareOptions, "engine" | "seed" | "commit" | "data_class">;

/**
 * The reviewed records a damage-aware baseline reads, against the golden team vocabulary (the committed
 * export of the one supported format; another format is refused, never given another vocabulary).
 * `version` "v3" is the v1 scorer with the engine's choice-location convention (#1180).
 */
export function damageAwareSetup(engine: Engine, formatId: string, version: "v1" | "v3" = "v1"): DamageAwareSetup {
  if (formatId !== GOLDEN_FORMAT) throw new GenerationError(`the damage-aware baseline has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  return {
    vocabulary,
    config: loadDamageAwareConfig(version, vocabulary),
    mechanics: loadMechanicsConfig(formatId),
    snapshot: loadRulesSnapshot(engine.commit),
  };
}

/** The feature set a run reads and the records its rows are built from (ADR-022 section 3). */
export interface FeatureSetup {
  configs: FeatureConfigs;
  /** What the policy worker is expected to state at the handshake (and the rows must be of). */
  statement: { feature_set: string; feature_set_hash: string };
}

/**
 * The candidate feature set a run config names (`model.encoder.options.candidate_features`), set up from the
 * reviewed records of this checkout, or undefined when the config names none (then nothing about the run changes).
 * Fails closed: the hash this checkout computes (pin, mechanics and assumed-set configs, probe adapter, reader) must be
 * the one the config names, so a run never plays or trains on rows of another set than its checkpoints were made for.
 */
export function featureSetup(engine: Engine, config: TrainingRunConfig): FeatureSetup | undefined {
  const named = config.model.encoder.options?.candidate_features;
  if (named === undefined) return undefined;
  if (config.format_id !== GOLDEN_FORMAT) throw new GenerationError(`the candidate features have a vocabulary for ${GOLDEN_FORMAT} only, not ${config.format_id}`);
  const vocabulary = goldenTeamVocabulary();
  const configs: FeatureConfigs = {
    vocabulary,
    config: loadDamageAwareConfig("v1", vocabulary),
    mechanics: loadMechanicsConfig(config.format_id),
    snapshot: loadRulesSnapshot(engine.commit),
    data_class: "synthetic",
    feature_set: named.feature_set,
  };
  const found = configuredSetHash(engine, configs);
  if (found !== named.feature_set_hash) {
    throw new GenerationError(`the run config names the candidate feature set ${named.feature_set} ${named.feature_set_hash}, which this checkout computes as ${found}`);
  }
  return { configs, statement: { feature_set: named.feature_set, feature_set_hash: found } };
}

/** The policy id of `damage-aware@v2`, the baseline that plays on sampled belief worlds (ADR-025 section 6). */
export const DAMAGE_AWARE_V2_POLICY_ID = damageAwarePolicyId("v2");

/** The policy id of `damage-aware@v3`, the v1 scorer with the engine's choice-location convention (#1180, #1183). */
export const DAMAGE_AWARE_V3_POLICY_ID = damageAwarePolicyId("v3");

/**
 * The records `damage-aware@v2` reads: its config (which names the usage prior and the belief config by hash) and
 * the belief port the composition root builds from them. The bridge does not import the opponent model, so the
 * root supplies `beliefWorlds`; the actor refuses a port that is not the one the record names, or whose prior is
 * for another format.
 */
export function damageAwareSetupV2(engine: Engine, formatId: string, beliefWorlds: BeliefWorldsFactory): DamageAwareSetup {
  if (formatId !== GOLDEN_FORMAT) throw new GenerationError(`the damage-aware baseline has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  const config = loadDamageAwareConfigV2(vocabulary);
  const snapshot = loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, formatId, snapshot).manifest;
  return {
    vocabulary,
    config,
    mechanics: loadMechanicsConfig(formatId),
    snapshot,
    worlds: beliefWorlds({ engine, vocabulary, config, reader_sha256: readerSha256(), manifest }),
  };
}

export interface GamePlan {
  /** The game's index within the generation, the sampler's `game`. */
  game: number;
  seed: string;
  /** Indices into the pool. */
  teams: Record<Player, number>;
  /** `self`: the pair plays both seats. Otherwise the pair takes `policy_seat` and a baseline the other. */
  opponent: "self" | BaselineOpponent;
  policy_seat: Player | "both";
  /** Seeds a baseline's stream: random-legal's, or damage-aware's epsilon draws (an unsigned 32-bit integer). */
  baseline_seed: number;
}

/**
 * The generation's match plan: `generation.games` self-play games, then each
 * listed baseline's games. Every number comes from `leagueGameStreams` (league-seed/1: SHA-256
 * of the JCS of the run key, the generation and the game index), so the plan
 * is reproducible from the config alone. A baseline game alternates the policy's seat.
 */
export function planGeneration(config: TrainingRunConfig, generation: number, poolSize: number): GamePlan[] {
  const { games, baseline_games } = config.generation;
  const opponents: GamePlan["opponent"][] = [];
  for (let i = 0; i < games; i++) opponents.push("self");
  for (const baseline of baseline_games) {
    for (let i = 0; i < baseline.games; i++) opponents.push({ actor: baseline.actor, policy_id: baseline.policy_id });
  }
  return planGames(config.seeds.run_key, generation, poolSize, opponents);
}

/**
 * The plan of one run of games from its run key, a generation number and the opponent of each
 * game, in game order (`league-seed/1`; a game's streams depend on those three alone). The pair
 * takes seat p1 of an even game and p2 of an odd one when a baseline is the other seat.
 */
export function planGames(run_key: string, generation: number, poolSize: number, opponents: readonly GamePlan["opponent"][]): GamePlan[] {
  if (!Number.isInteger(generation) || generation < 1) throw new RangeError("a generation is numbered from 1");
  if (!Number.isInteger(poolSize) || poolSize < 1) throw new RangeError("the team pool is empty");
  return opponents.map((opponent, game) => {
    const streams = leagueGameStreams(run_key, generation, game);
    return {
      game,
      seed: streams.seed,
      teams: { p1: streams.team_draws.p1 % poolSize, p2: streams.team_draws.p2 % poolSize },
      opponent,
      policy_seat: opponent === "self" ? "both" : game % 2 === 0 ? "p1" : "p2",
      baseline_seed: streams.baseline_seed,
    };
  });
}

export interface GenerationOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** The frozen pair's `behaviour_policy_id` (`policy:` + SHA-256). */
  policy_id: string;
  /** A started worker client; `beginGame` (when it has one) runs before each wave, as the client's overrun rule asks. */
  client: PolicyClient & { beginGame?: () => Promise<void> };
  /**
   * Further workers of the same frozen pair (#957). A game stays on one worker, and
   * every worker answers a query the same way (it is deterministic and the sampling
   * draw is the harness's), so the Episodes do not depend on how many workers there
   * are or which serves a game. Waves grow with the workers so each keeps
   * `n_parallel` games in flight.
   */
  peers?: readonly (PolicyClient & { beginGame?: () => Promise<void> })[];
  /** The games this process plays (default all). The rest come from `others`. */
  shard?: Shard;
  /** The entries of the games of the other shards, played by other game processes. */
  others?: Promise<readonly PlayedEntry[]>;
  pool: TrainingPool;
  /** The damage-aware baseline's records; read from `config/` when a plan game needs them and this is absent (a test seam). */
  damage_aware?: DamageAwareSetup;
  /** `damage-aware@v3`'s records, likewise (#1183). */
  damage_aware_v3?: DamageAwareSetup;
  /** `damage-aware@v2`'s records, likewise (built from `belief_worlds` when a plan game needs them and this is absent). */
  damage_aware_v2?: DamageAwareSetup;
  /**
   * The candidate features the run's pair reads (`featureSetup`): given to the pair's seats, which attach rows to a query
   * only for a worker that stated the set, and to the training loader, which recomputes them for every example.
   */
  features?: FeatureConfigs;
  /** The belief port factory of the composition root (`packages/team-dataset`); without it a plan that names `damage-aware@v2` is refused. */
  belief_worlds?: BeliefWorldsFactory | undefined;
  /**
   * Write the training stream to this file as the examples are made (in chunks; the stream
   * is never one string: a generation of 800 games is over Node's string limit, #998), and
   * keep none of it in the result. Without it the stream is built in memory, which is only
   * for streams under the limit (tests). `writeGeneration` expects it at `<dir>/stream.jsonl`.
   */
  stream_path?: string;
  /** Chunk size and the string-size limit (a test injects a small one). */
  stream_limits?: ChunkLimits;
  /**
   * Write each Episode to `<episodes_dir>/<hash>.json` (flushed) as its game finishes and keep none of
   * them, so a generation's memory does not grow with its games (#1069). The training loader then
   * reads them back one at a time in hash order. The files and the stream are the bytes of the
   * in-memory path; `writeGeneration` expects the directory to be `<dir>/episodes` and writes nothing
   * more there. Needs `stream_path`.
   */
  episodes_dir?: string;
  /**
   * With `episodes_dir` and `features`: a scratch directory the game processes leave the recomputed candidate feature
   * rows of their Episodes in once their games are done (`feature-recompute.ts`, #1185), so the rows of the stream are
   * recomputed by every process in parallel instead of serially by the one that writes the stream. The files are
   * never part of the generation, and the stream's bytes do not depend on whether they exist.
   */
  features_dir?: string | undefined;
  /**
   * Ends this process's own games (#1100): checked before each wave and after each worker
   * restart, so a process whose generation has failed finishes its current wave and plays no further one and starts no
   * further worker. `playGeneration` aborts it when a game process fails.
   */
  signal?: AbortSignal;
  /** The number of waves (and worker restarts) in progress, for the idle watchdog (#1100). */
  activity?: { busy: number };
  /** How many further game processes are still running; `playGeneration`'s idle watchdog reads it. */
  live_children?: () => number;
  /** How long a generation may have no live game process and no game in flight before it fails (default 60 s; a test injects a short one). */
  idle_grace_ms?: number;
}

export interface GameRow {
  game: number;
  seed: string;
  opponent: string;
  outcome: "played" | "forfeit";
  winner: Player | null;
  turns: number;
  /** Only on a forfeited game: the seat that lost, why, at which request of its channel, and the harness's detail (#912). */
  forfeit?: { loser: Player; cause: ForfeitCause; request_seq: number; detail: string };
}

/** The forfeits of a generation by cause, sorted by cause; empty when none. */
export function forfeitCauses(games: readonly GameRow[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const game of games) if (game.forfeit !== undefined) counts[game.forfeit.cause] = (counts[game.forfeit.cause] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => (a < b ? -1 : 1)));
}

export interface GenerationResult {
  generation: number;
  plan: GamePlan[];
  games: GameRow[];
  /** The Episodes of every seat that was played (the league record), as stored. Empty when they were written to `episodes_dir`. */
  episodes: Episode[];
  /** The directory the Episodes were written to as their games finished (`GenerationOptions.episodes_dir`), else null. */
  episodes_dir: string | null;
  /** Episode hashes, in `episodes` order. */
  episode_hashes: string[];
  /** The stream's bytes, only when it was built in memory (no `stream_path`); else null. */
  stream: string | null;
  /** The file the stream was written to (`GenerationOptions.stream_path`), else null. */
  stream_path: string | null;
  set: TrainingSet;
  /** What the generation played, tied to the run config and the seed derivation (`league-run.json`). */
  run_record: LeagueRunRecord;
  /** SHA-256 of the TrainingSet's JCS bytes: what `CheckpointRecord.training_set_hashes` holds. */
  training_set_hash: string;
  manifest: VersionManifest;
  /** Hashes of the teams the plan drew, sorted and unique. */
  team_hashes: string[];
}

/** A game that could not be played: a harness incident, which fails the generation. */
export class GenerationError extends Error {
  override name = "GenerationError";
}

type StampSets = Parameters<typeof stampTeam>[0];

const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");

interface BaselineSeat {
  actor: Actor;
  /** The damage-aware baseline's report once the game is over; absent for the other baselines. */
  report?: () => BaselineGameReport;
}

function baselineActor(options: GenerationOptions, opponent: BaselineOpponent, plan: GamePlan, budget: LatencyBudget): BaselineSeat {
  const guarded = (actor: Actor): Actor => guardActor(actor, { role: "scripted", latency_budget: budget });
  if (opponent.actor !== "damage-aware") return { actor: guarded(opponent.actor === "actor-rule" ? firstLegal : randomLegalActor(plan.baseline_seed)) };
  const setup = opponent.policy_id === DAMAGE_AWARE_V2_POLICY_ID ? options.damage_aware_v2 : opponent.policy_id === DAMAGE_AWARE_V3_POLICY_ID ? options.damage_aware_v3 : options.damage_aware;
  if (setup === undefined) throw new GenerationError("a damage-aware baseline game needs the baseline's records");
  // A fresh actor per game: it keeps the reader's state, the game's memo and counts. The seed is drawn from only when epsilon_ppm is above 0.
  const damage: DamageAwareActor = damageAwareActor({
    ...setup,
    engine: options.engine,
    data_class: "synthetic",
    commit: opponent.policy_id.slice(opponent.policy_id.indexOf("@") + 1),
    seed: plan.baseline_seed,
  });
  return {
    actor: guarded(damage),
    report: () => {
      const { policy_id, identity_hash, counts } = damage.report();
      if (policy_id !== opponent.policy_id) throw new GenerationError(`the config names the baseline ${opponent.policy_id}, which plays as ${policy_id}`);
      return { policy_id, identity_hash, counts };
    },
  };
}

interface PlayedOne extends PlayedGame {
  baseline?: BaselineGameReport;
}

async function playOne(options: GenerationOptions, plan: GamePlan, client: PolicyClient): Promise<PlayedOne> {
  const { engine, config, policy_id, pool } = options;
  const { generation: settings } = config;
  const budget = { preview_ms: settings.latency_budget_ms.preview_ms, battle_ms: settings.latency_budget_ms.battle_ms };
  const seat: PolicySeat = { client, policy_id, sampling: "sample", ...(options.features === undefined ? {} : { features: { configs: options.features } }) };
  const baseline = plan.opponent === "self" ? undefined : baselineActor(options, plan.opponent, plan, budget);
  const seats: PolicyGameOptions["seats"] =
    baseline === undefined ? { p1: seat, p2: seat } : plan.policy_seat === "p1" ? { p1: seat, p2: baseline.actor } : { p1: baseline.actor, p2: seat };
  const played = await policyGame({
    engine,
    formatId: config.format_id,
    seed: plan.seed,
    game: plan.game,
    teams: { p1: pool.teams[plan.teams.p1] as PokemonSet[], p2: pool.teams[plan.teams.p2] as PokemonSet[] },
    known_synthetic: pool.known_synthetic,
    seats,
    latency_budget: budget,
    maxTurns: settings.max_turns,
  });
  return baseline?.report === undefined ? played : { ...played, baseline: baseline.report() };
}

const idOf = (opponent: GamePlan["opponent"], policy_id: string): string => (opponent === "self" ? policy_id : opponent.policy_id);

/** A slice of the plan for one game process: games `g` with `g mod of = index`. */
export interface Shard {
  index: number;
  of: number;
}

/** One played game: its row and the Episodes of its seats (none for a forfeit). */
export interface PlayedEntry {
  game: number;
  row: GameRow;
  episodes: Episode[];
  /** With `episodes_dir`: the Episodes already written to it, in seat order, instead of `episodes`. */
  stored?: StoredRef[];
  /** Only a game against the damage-aware baseline: its identity hash and this game's counts (ADR-018 §5). */
  baseline?: BaselineGameReport;
}

/** An Episode written to disk: the hash its file is named by and the policy that played the seat. */
export interface StoredRef {
  hash: string;
  behaviour_policy_id: string;
}

function checkShard(shard: Shard): void {
  if (!Number.isInteger(shard.of) || shard.of < 1 || !Number.isInteger(shard.index) || shard.index < 0 || shard.index >= shard.of) {
    throw new RangeError("a shard is an index below a positive count");
  }
}

/**
 * One Episode per seat of a played game (a league collection record: `run_id`, the seed the battle ran
 * with and the other seat's policy id), each classified by both teams. `sides` names each seat's policy.
 */
export async function seatEpisodes(record: BattleRecord, teams: Record<Player, number>, pool: TrainingPool, run_id: string, sides: Record<Player, string>): Promise<Episode[]> {
  const episodes: Episode[] = [];
  for (const player of ["p1", "p2"] as const) {
    episodes.push(
      await episodeFromRecord(record, player, {
        team: stampTeam(pool.teams[teams[player]] as StampSets, pool.known_synthetic),
        opponent_team: stampTeam(pool.teams[teams[rival(player)]] as StampSets, pool.known_synthetic),
        known_synthetic: pool.known_synthetic,
        behaviour_policy_id: sides[player],
        collection: { kind: "league", run_id, seed: record.seed, opponent_policy_id: sides[rival(player)] },
      }),
    );
  }
  return episodes;
}

/**
 * Play the games of `shard` (all of them without one) against `options.client` and
 * `options.peers` (started, with the frozen pair loaded) in waves, and write an
 * Episode for every played seat. A wave holds `n_parallel` games for each worker.
 * A game's result depends on its plan entry alone, never on its worker, its wave
 * or the other games, so any split of the plan gives the same entries.
 *
 * Refused: a harness incident in any game (nothing is retried).
 */
export async function playEntries(options: GenerationOptions, shard: Shard = { index: 0, of: 1 }): Promise<PlayedEntry[]> {
  checkShard(shard);
  const { config, generation, policy_id, client } = options;
  const plan = planGeneration(config, generation, options.pool.teams.length);
  // The damage-aware records are read once, and only when a game of this process needs them.
  const named = config.generation.baseline_games.filter((baseline) => baseline.actor === "damage-aware");
  const needsV1 = named.some((baseline) => baseline.policy_id !== DAMAGE_AWARE_V2_POLICY_ID && baseline.policy_id !== DAMAGE_AWARE_V3_POLICY_ID);
  const needsV3 = named.some((baseline) => baseline.policy_id === DAMAGE_AWARE_V3_POLICY_ID);
  const needsV2 = named.some((baseline) => baseline.policy_id === DAMAGE_AWARE_V2_POLICY_ID);
  if (needsV2 && options.damage_aware_v2 === undefined && options.belief_worlds === undefined) throw new GenerationError(`the plan names ${DAMAGE_AWARE_V2_POLICY_ID}: its script needs the belief port of the composition root (packages/team-dataset)`);
  const withRecords: GenerationOptions = {
    ...options,
    ...(needsV1 && options.damage_aware === undefined ? { damage_aware: damageAwareSetup(options.engine, config.format_id) } : {}),
    ...(needsV3 && options.damage_aware_v3 === undefined ? { damage_aware_v3: damageAwareSetup(options.engine, config.format_id, "v3") } : {}),
    ...(needsV2 && options.damage_aware_v2 === undefined ? { damage_aware_v2: damageAwareSetupV2(options.engine, config.format_id, options.belief_worlds as BeliefWorldsFactory) } : {}),
  };
  const { pool } = withRecords;
  const workers = [client, ...(options.peers ?? [])];
  const mine = plan.filter((entry) => entry.game % shard.of === shard.index);
  const entries: PlayedEntry[] = [];
  const waveSize = config.generation.n_parallel * workers.length;
  const activity = options.activity ?? { busy: 0 };
  const stopped = (): void => {
    if (options.signal?.aborted === true) throw new GenerationError("the generation was stopped: this process plays no further game");
  };
  for (let start = 0; start < mine.length; start += waveSize) {
    stopped();
    activity.busy++;
    let played: PlayedOne[];
    const wave = mine.slice(start, start + waveSize);
    try {
      await Promise.all(workers.map((worker) => worker.beginGame?.()));
      stopped(); // a restart that began before the stop must not start games
      // A game stays on one worker, chosen by its place in the wave.
      played = await Promise.all(wave.map((entry, place) => playOne(withRecords, entry, workers[place % workers.length] as PolicyClient)));
    } finally {
      activity.busy--;
    }
    stopped();
    for (const [index, game] of played.entries()) {
      const entry = wave[index] as GamePlan;
      const { outcome, record, baseline } = game;
      const reported = baseline === undefined ? {} : { baseline };
      if (outcome.kind === "incident") throw new GenerationError(`game ${entry.game} (${entry.seed}): ${outcome.incident}: ${outcome.message}`);
      const row = { game: entry.game, seed: entry.seed, opponent: idOf(entry.opponent, policy_id) };
      if (outcome.kind === "forfeit" || record === null) {
        // A forfeit has no Episode, but it is kept the way its generation keeps them: an empty list of references on the spilled path (#1184).
        const kept = options.episodes_dir === undefined ? {} : { stored: [] };
        if (outcome.kind !== "forfeit") {
          entries.push({ game: entry.game, row: { ...row, outcome: "forfeit", winner: null, turns: 0 }, episodes: [], ...kept, ...reported });
        } else {
          const { loser, cause, request_seq, detail } = outcome;
          entries.push({ game: entry.game, row: { ...row, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser, cause, request_seq, detail } }, episodes: [], ...kept, ...reported });
        }
        continue;
      }
      const sides: Record<Player, string> = {
        p1: entry.opponent !== "self" && entry.policy_seat !== "p1" ? entry.opponent.policy_id : policy_id,
        p2: entry.opponent !== "self" && entry.policy_seat !== "p2" ? entry.opponent.policy_id : policy_id,
      };
      const episodes = await seatEpisodes(record, entry.teams, pool, config.seeds.run_key, sides);
      const base = { game: entry.game, row: { ...row, outcome: "played" as const, winner: record.winner, turns: record.turns }, ...reported };
      if (options.episodes_dir === undefined) {
        entries.push({ ...base, episodes });
      } else {
        // Written now and dropped: the next waves do not carry this game's Episodes (#1069).
        const stored: StoredRef[] = [];
        for (const episode of episodes) {
          const hash = await episodeHash(episode, nodeSha256);
          // Not renamed into place: the directory is the generation's partial one, discarded whole on a restart, and the loader checks each file against its name.
          writeDurable(join(options.episodes_dir, `${hash}.json`), `${jcs(episode as unknown as Json)}\n`);
          stored.push({ hash, behaviour_policy_id: episode.provenance.behaviour_policy_id });
        }
        entries.push({ ...base, episodes: [], stored });
      }
    }
  }
  if (options.features_dir !== undefined && options.features !== undefined && options.episodes_dir !== undefined) {
    const dir = options.episodes_dir;
    const own = entries.flatMap((entry) => entry.stored ?? []).filter((ref) => ref.behaviour_policy_id === policy_id);
    await writeFeatureRows({ engine: options.engine, configs: options.features, known_synthetic: pool.known_synthetic }, own.map(({ hash }) => ({ hash, path: join(dir, `${hash}.json`) })), options.features_dir, () => options.signal?.aborted === true);
  }
  return entries;
}

/** How long a generation may stand with no live game process and no game in flight (#1100). */
const IDLE_GRACE_MS = 60_000;

/**
 * The idle watchdog (#1100): while the other game processes' entries are still awaited, a
 * generation with none of them alive and none of this process's games or worker restarts in
 * progress is not going to finish. `tripped` rejects after `graceMs` of that state; it never
 * resolves. Without a `live` probe (one game process) there is nothing to watch.
 */
function watchIdle(live: (() => number) | undefined, activity: { busy: number }, graceMs: number, awaited: Promise<unknown>): { tripped: Promise<never>; stop: () => void } {
  let timer: ReturnType<typeof setInterval> | undefined;
  let settled = false;
  awaited.then(() => (settled = true), () => (settled = true));
  const tripped = new Promise<never>((_, reject) => {
    if (live === undefined) return;
    let since: number | undefined;
    timer = setInterval(() => {
      const idle = !settled && live() === 0 && activity.busy === 0;
      if (!idle) {
        since = undefined;
        return;
      }
      since ??= Date.now();
      if (Date.now() - since >= graceMs) reject(new GenerationError(`the generation is idle: no game process is running and no game is in flight for ${graceMs} ms, yet the other game processes' games have not been reported`));
    }, Math.max(1, Math.min(1000, Math.floor(graceMs / 4))));
    timer.unref();
  });
  tripped.catch(() => undefined);
  return { tripped, stop: () => clearInterval(timer) };
}

/**
 * Play the generation's plan: `options.shard`'s games here, the rest through
 * `options.others` (entries played by other game processes), then write nothing
 * yet and load the pair's own seats through the training loader. The entries are
 * put in plan order, so the result is the one a single process gives.
 *
 * Refused: a harness incident in any game (the generation fails, nothing is
 * retried); a game missing or played twice; no Episode of the pair at all (every
 * game forfeited). A game that a policy forfeited is not an Episode (it is not
 * training data) and is reported in `games`.
 */
export async function playGeneration(options: GenerationOptions): Promise<GenerationResult> {
  const { engine, config, generation, policy_id, pool } = options;
  const plan = planGeneration(config, generation, pool.teams.length);
  const stop = new AbortController();
  const activity = options.activity ?? { busy: 0 };
  const here = playEntries({ ...options, signal: stop.signal, activity }, options.shard);
  const there = options.others ?? Promise.resolve([]);
  here.catch(() => stop.abort()); // a failure of either side ends the other's games at once (#1100)
  there.catch(() => stop.abort());
  let entries: PlayedEntry[];
  const idle = watchIdle(options.live_children, activity, options.idle_grace_ms ?? IDLE_GRACE_MS, there);
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
  const rows = entries.map((entry) => entry.row);
  const spilled = options.episodes_dir !== undefined;
  if (entries.some((entry) => (entry.stored !== undefined) !== spilled || (spilled && entry.episodes.length > 0))) {
    throw new GenerationError("the played games are not all kept the way this generation keeps its Episodes");
  }
  const episodes = entries.flatMap((entry) => entry.episodes);
  const refs: StoredRef[] = entries.flatMap((entry) => entry.stored ?? []);
  const limits = options.stream_limits ?? {};
  let stream: string | null = null;
  let set: TrainingSet;
  let episodeHashes: string[];
  if (options.episodes_dir !== undefined) {
    if (options.stream_path === undefined) throw new GenerationError("episodes_dir needs stream_path");
    const dir = options.episodes_dir;
    const ownRefs = refs.filter((ref) => ref.behaviour_policy_id === policy_id);
    if (ownRefs.length === 0) throw new GenerationError("no Episode of the frozen pair: every game was forfeited");
    const stored: StoredEpisode[] = ownRefs.map(({ hash }) => ({ hash, path: join(dir, `${hash}.json`) }));
    set = await writeStoredTrainingStream({ episodes: stored, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: policy_id, ...(options.features === undefined ? {} : { features: { engine, configs: options.features, ...(options.features_dir === undefined ? {} : { precomputed: readFeatureRows(engine, options.features, options.features_dir) }) } }) }, { ...limits, path: options.stream_path });
    episodeHashes = refs.map((ref) => ref.hash);
  } else {
    const own = episodes.filter((episode) => episode.provenance.behaviour_policy_id === policy_id);
    if (own.length === 0) throw new GenerationError("no Episode of the frozen pair: every game was forfeited");
    const input = { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: policy_id, ...(options.features === undefined ? {} : { features: { engine, configs: options.features } }) };
    if (options.stream_path === undefined) {
      ({ stream, set } = await loadTrainingSet(input, limits));
    } else {
      set = await writeTrainingStream(input, { ...limits, path: options.stream_path });
    }
    episodeHashes = await Promise.all(episodes.map((episode) => episodeHash(episode, nodeSha256)));
  }
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) drawn.add(pool.hashes[entry.teams[player]] as string);
  return {
    generation,
    plan,
    games: rows,
    episodes,
    episodes_dir: options.episodes_dir ?? null,
    episode_hashes: episodeHashes,
    stream,
    stream_path: options.stream_path ?? null,
    set,
    run_record: await buildLeagueRunRecord({ config, generation, pair_policy_id: policy_id, plan, games: rows, baseline_reports: entries.flatMap((entry) => (entry.baseline === undefined ? [] : [entry.baseline])) }),
    training_set_hash: await nodeSha256(new TextEncoder().encode(jcs(set as unknown as Json))),
    manifest: resolveFormat(engine, config.format_id).manifest,
    team_hashes: [...drawn].sort(),
  };
}

/**
 * Write a generation's files under `dir` (created if needed), all canonical JSON
 * except the stream, which is the loader's own bytes:
 * `episodes/<episode hash>.json` (every played seat), `stream.jsonl` (already on disk when the
 * generation was played with a `stream_path`, written in chunks),
 * `training-set.json`, `plan.json`, `league-run.json` (the LeagueRunRecord) and `generation.json` (the per-game rows and
 * the hashes the Python driver records). Nothing in them depends on the clock.
 */
export function writeGeneration(dir: string, result: GenerationResult): void {
  const episodes = join(dir, "episodes");
  mkdirSync(episodes, { recursive: true });
  const put = (name: string, value: unknown): void => writeDurable(join(dir, name), `${jcs(value as Json)}\n`);
  if (result.episodes_dir === null) {
    result.episodes.forEach((episode, index) => put(`episodes/${result.episode_hashes[index]}.json`, episode));
  } else if (result.episodes_dir !== episodes) {
    throw new GenerationError(`the Episodes were written to ${result.episodes_dir}, not to ${episodes}`);
  }
  const streamPath = join(dir, "stream.jsonl");
  if (result.stream_path === null) {
    writeDurable(streamPath, result.stream as string);
  } else if (result.stream_path !== streamPath) {
    throw new GenerationError(`the stream was written to ${result.stream_path}, not to ${streamPath}`);
  }
  put("training-set.json", result.set);
  put("plan.json", result.plan);
  put("league-run.json", result.run_record);
  // generation.json last: it is the file that says the rest is whole.
  put("generation.json", {
    schema_version: 1,
    generation: result.generation,
    games: result.games,
    forfeits: result.games.filter((game) => game.outcome === "forfeit").length,
    // Present only when a game was forfeited, so a run without forfeits writes the bytes it always did.
    ...(result.games.some((game) => game.outcome === "forfeit") ? { forfeit_causes: forfeitCauses(result.games) } : {}),
    episode_hashes: [...result.episode_hashes].sort(),
    training_set_hash: result.training_set_hash,
    examples: result.set.examples,
    manifest: result.manifest,
    team_hashes: result.team_hashes,
  });
  syncDir(episodes);
  syncDir(dir);
}

/** Write a file and flush it to disk before returning. */
export function writeDurable(path: string, data: string): void {
  const descriptor = openSync(path, "w");
  try {
    writeSync(descriptor, data);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

/** Flush a directory's entries to disk. */
export function syncDir(path: string): void {
  const descriptor = openSync(path, "r");
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}
