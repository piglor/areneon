// Teacher data for the cloning learner (ADR-019 track 1, ADR-010 erratum A2, Epic #961): seeded A1
// games of the scripted `damage-aware@v1` baseline (ADR-018) on synthetic training teams, written as
// ordinary league Episodes and loaded through the training loader into a learner-ready TrainingSet
// whose behaviour policy is the teacher. Nothing here is a new contract: an Episode is a league
// Episode (`collection.kind` "league", `run_id` the teacher run's key, `opponent_policy_id` the other
// seat's policy), the TrainingSet is the loader's own.
//
// Who plays: the default game is the teacher against itself (both seats are the teacher, so both
// seats' Episodes train). A run may add games against `random-legal@v1` or `actor-rule@v1` for
// diversity: the teacher takes one seat, the baseline the other, and only the teacher's seat loads
// (the loader refuses any other policy id). `epsilon_ppm` (ADR-018 Q5, training only) is the one
// other source of diversity; a non-zero value is another baseline identity and another policy id.
//
// Determinism: the plan is a pure function of the run key and the pool (`league-seed/1`, generation
// 1), the teacher is a pure function of its inputs, and a game depends on its plan entry alone, so any
// split of the plan across game processes gives the same bytes. The harness's seeds stay in the plan
// and the Episodes' `collection.seed` (the A1 contract requires it); a TrainingExample carries none.

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { type Json, jcs } from "./canonical-json.ts";
import { damageAwareActor, type DamageAwareActor, type DamageAwareCounts } from "./damage-aware-actor.ts";
import { type DamageAwareConfig, loadDamageAwareConfig } from "./damage-aware-config.ts";
import { DAMAGE_AWARE_NAME, damageAwarePolicyId } from "./damage-aware-identity.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { episodeHash } from "./episode.ts";
import type { Episode } from "./episode.ts";
import { type LatencyBudget, playGame, guardActor } from "./evaluation-harness.ts";
import { type VersionManifest, resolveFormat } from "./formats.ts";
import { firstLegal, GOLDEN_FORMAT, goldenTeamVocabulary } from "./golden.ts";
import { randomLegalActor } from "./random-legal.ts";
import {
  type BaselineOpponent,
  GenerationError,
  type GamePlan,
  type GameRow,
  type PlayedEntry,
  planGames,
  seatEpisodes,
  type Shard,
  syncDir,
  type TrainingPool,
  writeDurable,
} from "./league-generation.ts";
import { loadMechanicsConfig } from "./mechanics-config.ts";
import { nodeSha256 } from "./player-channel.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { sha256Hex } from "./rules.ts";
import type { Actor, Player } from "./runner.ts";
import { loadTrainingSet, type TrainingSet, writeTrainingStream } from "./training-loader.ts";
import type { ChunkLimits } from "./chunked-file.ts";

/** The teacher run's own settings (command arguments; the real run's size is set from the cost measurement). */
export interface TeacherSettings {
  /** The run's name: the league `run_id` of every Episode and the root of every seed. */
  run_key: string;
  format_id: string;
  /** Games of the teacher against itself. */
  games: number;
  /** Games of the teacher against `random-legal@v1` and `actor-rule@v1` (the teacher alternates seats). */
  random_games: number;
  actor_rule_games: number;
  max_turns: number;
  /** ADR-018 Q5: the chance, in parts per million, that the teacher plays a random candidate. 0 is the committed baseline. */
  epsilon_ppm: number;
  /** The committed `damage-aware` record the teacher plays (#1182): absent is `v1`, the byte-compatible default. */
  teacher_version?: TeacherVersion;
  /** The per-decision limit; an overrun of the teacher is a harness incident that fails the run. */
  latency_budget_ms: LatencyBudget;
}

/** The generation number of every teacher game: `league-seed/1` counts generations from 1. */
export const TEACHER_GENERATION = 1;

/** The committed `damage-aware` config record the teacher plays (`config/baselines/damage-aware.<version>.json`). */
export type TeacherVersion = "v1" | "v3";
export const TEACHER_VERSIONS: readonly TeacherVersion[] = ["v1", "v3"];
export const TEACHER_CONFIG_VERSION: TeacherVersion = "v1";

const RANDOM_OPPONENT: BaselineOpponent = { actor: "random-legal", policy_id: "random-legal@v1" };
const ACTOR_RULE_OPPONENT: BaselineOpponent = { actor: "actor-rule", policy_id: "actor-rule@v1" };

const PPM = 1_000_000;

/** `damage-aware@v1` (or `@v3`), or `damage-aware@v1-eps<ppm>` when the teacher explores (another baseline identity). */
export const teacherPolicyId = (epsilon_ppm: number, version: TeacherVersion = TEACHER_CONFIG_VERSION): string =>
  damageAwarePolicyId(epsilon_ppm === 0 ? version : `${version}-eps${epsilon_ppm}`);

/** Refuse a settings record that would play nothing or play out of range. */
export function checkTeacherSettings(settings: TeacherSettings): void {
  const count = (name: string, value: number, least: number): void => {
    if (!Number.isSafeInteger(value) || value < least) throw new RangeError(`${name} must be an integer of at least ${least}`);
  };
  count("games", settings.games, 0);
  count("random_games", settings.random_games, 0);
  count("actor_rule_games", settings.actor_rule_games, 0);
  count("max_turns", settings.max_turns, 1);
  count("epsilon_ppm", settings.epsilon_ppm, 0);
  if (settings.teacher_version !== undefined && !TEACHER_VERSIONS.includes(settings.teacher_version)) throw new RangeError("teacher_version is v1 or v3");
  if (settings.epsilon_ppm > PPM) throw new RangeError("epsilon_ppm is at most 1,000,000");
  if (settings.games + settings.random_games + settings.actor_rule_games < 1) throw new RangeError("a teacher run plays at least one game");
  if (settings.run_key === "" || settings.run_key.startsWith("CHANGE-ME")) throw new RangeError("run_key must name the run");
  for (const ms of [settings.latency_budget_ms.preview_ms, settings.latency_budget_ms.battle_ms]) {
    if (!Number.isFinite(ms) || ms <= 0) throw new RangeError("latency budget must be a positive number of ms");
  }
}

/** The plan: mirror games first, then the `random-legal` games, then the `actor-rule` games (game order is the plan order). */
export function planTeacher(settings: TeacherSettings, poolSize: number): GamePlan[] {
  const opponents: GamePlan["opponent"][] = [];
  for (let i = 0; i < settings.games; i++) opponents.push("self");
  for (let i = 0; i < settings.random_games; i++) opponents.push(RANDOM_OPPONENT);
  for (let i = 0; i < settings.actor_rule_games; i++) opponents.push(ACTOR_RULE_OPPONENT);
  return planGames(settings.run_key, TEACHER_GENERATION, poolSize, opponents);
}

/**
 * The seed of a teacher seat's exploration stream (a uint32): the game's baseline stream, split per seat so
 * the two teacher seats of a mirror game never share one. Drawn from only when `epsilon_ppm` is above 0.
 */
export const teacherSeatSeed = (baseline_seed: number, player: Player): number =>
  parseInt(sha256Hex(jcs({ baseline_seed, player, purpose: "teacher-epsilon" } as Json)).slice(0, 8), 16);

/** What a teacher seat reports for one game: its identity hash and its counts (ADR-018 section 4). */
export interface TeacherSeatReport {
  identity_hash: string;
  counts: DamageAwareCounts;
}

/** One played game of a teacher run: the league entry and the teacher seats' reports. */
export interface TeacherEntry extends PlayedEntry {
  teacher: Partial<Record<Player, TeacherSeatReport>>;
}

export interface TeacherOptions {
  engine: Engine;
  settings: TeacherSettings;
  pool: TrainingPool;
  /** The games this process plays (default all). The rest come from `others`. */
  shard?: Shard;
  /** The entries of the games of the other shards, played by other game processes. */
  others?: Promise<readonly TeacherEntry[]>;
  /** Write the training stream to this file as the examples are made (in chunks, #998). Without it the stream is built in memory (tests). */
  stream_path?: string;
  stream_limits?: ChunkLimits;
}

/** What the teacher needs beyond the pool: the committed config records and the pinned engine's vocabulary. Read once per process. */
export interface TeacherContext {
  config: DamageAwareConfig;
  makeActor(player: Player, baseline_seed: number): DamageAwareActor;
  policy_id: string;
}

export function teacherContext(engine: Engine, settings: TeacherSettings): TeacherContext {
  // The committed team vocabulary is the golden format's; another format needs its own exported vocabulary first.
  if (settings.format_id !== GOLDEN_FORMAT) throw new GenerationError(`the teacher plays ${GOLDEN_FORMAT} only (no vocabulary for ${settings.format_id})`);
  const vocabulary = goldenTeamVocabulary();
  const base = loadDamageAwareConfig(settings.teacher_version ?? TEACHER_CONFIG_VERSION, vocabulary);
  const config: DamageAwareConfig = { ...base, epsilon_ppm: settings.epsilon_ppm };
  const mechanics = loadMechanicsConfig(settings.format_id);
  const snapshot = loadRulesSnapshot(engine.commit);
  const policy_id = teacherPolicyId(settings.epsilon_ppm, settings.teacher_version);
  const commit = policy_id.slice(DAMAGE_AWARE_NAME.length + 1);
  return {
    config,
    policy_id,
    makeActor: (player, baseline_seed) =>
      damageAwareActor({
        engine,
        vocabulary,
        config,
        mechanics,
        snapshot,
        data_class: "synthetic",
        commit,
        ...(settings.epsilon_ppm > 0 ? { seed: teacherSeatSeed(baseline_seed, player) } : {}),
      }),
  };
}

/**
 * Play the teacher games of `shard` (all of them without one), one after the other: the teacher is
 * synchronous CPU work, so a further core means a further game process, never a wave. A game's
 * result depends on its plan entry alone. Refused: a harness incident or a forfeit in any game (the
 * teacher is scripted, so a failure, a limit overrun included, is an incident; nothing is retried).
 */
export async function playTeacherEntries(options: TeacherOptions, context: TeacherContext, shard: Shard = { index: 0, of: 1 }): Promise<TeacherEntry[]> {
  const { engine, settings, pool } = options;
  if (!Number.isInteger(shard.of) || shard.of < 1 || !Number.isInteger(shard.index) || shard.index < 0 || shard.index >= shard.of) {
    throw new RangeError("a shard is an index below a positive count");
  }
  const mine = planTeacher(settings, pool.teams.length).filter((entry) => entry.game % shard.of === shard.index);
  const budget = settings.latency_budget_ms;
  const entries: TeacherEntry[] = [];
  for (const entry of mine) {
    const teachers: Partial<Record<Player, DamageAwareActor>> = {};
    const seatOf = (player: Player): Actor => {
      if (entry.opponent !== "self" && entry.policy_seat !== player) return guardActor(entry.opponent.actor === "actor-rule" ? firstLegal : randomLegalActor(entry.baseline_seed), { role: "scripted", latency_budget: budget });
      const teacher = context.makeActor(player, entry.baseline_seed);
      teachers[player] = teacher;
      return guardActor(teacher, { role: "scripted", latency_budget: budget });
    };
    const { outcome, record } = await playGame({
      engine,
      formatId: settings.format_id,
      seed: entry.seed,
      teams: { p1: pool.teams[entry.teams.p1] as PokemonSet[], p2: pool.teams[entry.teams.p2] as PokemonSet[] },
      actors: { p1: seatOf("p1"), p2: seatOf("p2") },
      maxTurns: settings.max_turns,
    });
    if (outcome.kind !== "played" || record === null) {
      throw new GenerationError(`game ${entry.game} (${entry.seed}): ${outcome.kind === "incident" ? `${outcome.incident}: ${outcome.message}` : `forfeit: ${outcome.kind === "forfeit" ? outcome.detail : ""}`}`);
    }
    const sides = Object.fromEntries(
      (["p1", "p2"] as const).map((player) => [player, teachers[player] === undefined ? (entry.opponent as BaselineOpponent).policy_id : context.policy_id]),
    ) as Record<Player, string>;
    const row: GameRow = {
      game: entry.game,
      seed: entry.seed,
      opponent: entry.opponent === "self" ? context.policy_id : entry.opponent.policy_id,
      outcome: "played",
      winner: record.winner,
      turns: record.turns,
    };
    const teacher: TeacherEntry["teacher"] = {};
    for (const player of ["p1", "p2"] as const) {
      const report = teachers[player]?.report();
      if (report !== undefined) teacher[player] = { identity_hash: report.identity_hash, counts: report.counts };
    }
    entries.push({ game: entry.game, row, episodes: await seatEpisodes(record, entry.teams, pool, settings.run_key, sides), teacher });
  }
  return entries;
}

/** What the teacher did across the run, by integer count (ADR-018 section 4: a baseline drifting towards `actor-rule` must be visible). */
export interface TeacherReport {
  policy_id: string;
  /** SHA-256 of the baseline identity (config, mechanics config, adapter format, reader, version manifest): the one every teacher seat played with. */
  identity_hash: string;
  epsilon_ppm: number;
  latency_budget_ms: LatencyBudget;
  /** Teacher seats played (two per mirror game, one per baseline game). */
  seats: number;
  /** The counts summed over every teacher seat. */
  totals: DamageAwareCounts;
  /** Each game's teacher counts by seat, in game order. */
  games: { game: number; p1?: DamageAwareCounts; p2?: DamageAwareCounts }[];
}

const COUNT_KEYS: (keyof DamageAwareCounts)[] = [
  "decisions", "preview_decisions", "scored_decisions", "queries", "probe_runs", "unsupported_queries", "reader_fallbacks", "assumed_set_fallbacks", "epsilon_plays", "probe_refusals",
];

/** Sum the entries' teacher reports. Refused: seats of more than one baseline identity (they are never pooled) or none at all. */
export function teacherReport(entries: readonly TeacherEntry[], context: TeacherContext, settings: TeacherSettings): TeacherReport {
  const totals = Object.fromEntries(COUNT_KEYS.map((key) => [key, 0])) as unknown as DamageAwareCounts;
  const hashes = new Set<string>();
  let seats = 0;
  const games: TeacherReport["games"] = [];
  for (const entry of entries) {
    const row: TeacherReport["games"][number] = { game: entry.game };
    for (const player of ["p1", "p2"] as const) {
      const seat = entry.teacher[player];
      if (seat === undefined) continue;
      seats += 1;
      hashes.add(seat.identity_hash);
      for (const key of COUNT_KEYS) totals[key] += seat.counts[key];
      row[player] = seat.counts;
    }
    games.push(row);
  }
  if (hashes.size !== 1) throw new GenerationError(`the teacher seats hold ${hashes.size} baseline identities, not one`);
  return {
    policy_id: context.policy_id,
    identity_hash: [...hashes][0] as string,
    epsilon_ppm: settings.epsilon_ppm,
    latency_budget_ms: settings.latency_budget_ms,
    seats,
    totals,
    games,
  };
}

export interface TeacherResult {
  settings: TeacherSettings;
  plan: GamePlan[];
  games: GameRow[];
  /** Every played seat's Episode, in game order (p1 then p2). */
  episodes: Episode[];
  episode_hashes: string[];
  /** The stream's bytes, only when it was built in memory (no `stream_path`); else null. */
  stream: string | null;
  stream_path: string | null;
  /** The TrainingSet of the teacher's own seats: `learner_ready`, `synthetic`, behaviour policy the teacher. */
  set: TrainingSet;
  /** SHA-256 of the TrainingSet's JCS bytes: `CloneRunConfig.teacher.training_set_hash`. */
  training_set_hash: string;
  teacher: TeacherReport;
  manifest: VersionManifest;
  /** Hashes of the teams the plan drew, sorted and unique. */
  team_hashes: string[];
}

/**
 * Play the teacher run: `options.shard`'s games here, the rest through `options.others`, then load
 * the teacher's own seats through the training loader (which refuses any other policy id) and report.
 */
export async function playTeacherRun(options: TeacherOptions): Promise<TeacherResult> {
  const { engine, settings, pool } = options;
  checkTeacherSettings(settings);
  const context = teacherContext(engine, settings);
  const plan = planTeacher(settings, pool.teams.length);
  const here = playTeacherEntries(options, context, options.shard);
  const there = options.others ?? Promise.resolve([]);
  there.catch(() => undefined); // a failure here is reported through the Promise.all below
  const entries = (await Promise.all([here, there])).flat().sort((a, b) => a.game - b.game);
  if (entries.length !== plan.length || entries.some((entry, index) => entry.game !== index)) {
    throw new GenerationError("the played games are not the plan's games, each once");
  }
  const episodes = entries.flatMap((entry) => entry.episodes);
  const own = episodes.filter((episode) => episode.provenance.behaviour_policy_id === context.policy_id);
  const input = { episodes: own, evaluation_manifests: [], known_synthetic: pool.known_synthetic, behaviour_policy_id: context.policy_id };
  const limits = options.stream_limits ?? {};
  let stream: string | null = null;
  let set: TrainingSet;
  if (options.stream_path === undefined) {
    ({ stream, set } = await loadTrainingSet(input, limits));
  } else {
    set = await writeTrainingStream(input, { ...limits, path: options.stream_path });
  }
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) drawn.add(pool.hashes[entry.teams[player]] as string);
  return {
    settings,
    plan,
    games: entries.map((entry) => entry.row),
    episodes,
    episode_hashes: await Promise.all(episodes.map((episode) => episodeHash(episode, nodeSha256))),
    stream,
    stream_path: options.stream_path ?? null,
    set,
    training_set_hash: await nodeSha256(new TextEncoder().encode(jcs(set as unknown as Json))),
    teacher: teacherReport(entries, context, settings),
    manifest: resolveFormat(engine, settings.format_id).manifest,
    team_hashes: [...drawn].sort(),
  };
}

/**
 * Write a teacher run's files under `dir` (created if needed), all canonical JSON except the stream:
 * `episodes/<episode hash>.json` (every played seat), `stream.jsonl` (already on disk when the run was
 * played with a `stream_path`), `training-set.json`, `plan.json`, `version-manifest.json`, `teacher.json`
 * (the baseline's identity and its counts) and `run.json` last (it says the rest is whole). Nothing
 * depends on the clock. `dataset_hash` names the team dataset the pool came from, or null for the interim pool.
 */
export function writeTeacherRun(dir: string, result: TeacherResult, dataset_hash: string | null): void {
  const episodes = join(dir, "episodes");
  mkdirSync(episodes, { recursive: true });
  const put = (name: string, value: unknown): void => writeDurable(join(dir, name), `${jcs(value as Json)}\n`);
  result.episodes.forEach((episode, index) => put(`episodes/${result.episode_hashes[index]}.json`, episode));
  const streamPath = join(dir, "stream.jsonl");
  if (result.stream_path === null) {
    writeDurable(streamPath, result.stream as string);
  } else if (result.stream_path !== streamPath) {
    throw new GenerationError(`the stream was written to ${result.stream_path}, not to ${streamPath}`);
  }
  put("training-set.json", result.set);
  put("plan.json", result.plan);
  put("version-manifest.json", result.manifest);
  put("teacher.json", result.teacher);
  put("run.json", {
    schema_version: 1,
    run_key: result.settings.run_key,
    format_id: result.settings.format_id,
    settings: { games: result.settings.games, random_games: result.settings.random_games, actor_rule_games: result.settings.actor_rule_games, max_turns: result.settings.max_turns },
    seed_derivation: "league-seed/1",
    teacher_policy_id: result.teacher.policy_id,
    teacher_identity_hash: result.teacher.identity_hash,
    dataset_hash,
    games: result.games,
    episode_hashes: [...result.episode_hashes].sort(),
    training_set_hash: result.training_set_hash,
    examples: result.set.examples,
    manifest: result.manifest,
    team_hashes: result.team_hashes,
  });
  syncDir(episodes);
  syncDir(dir);
}
