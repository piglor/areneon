// The population branch of `scripts/play-generation.ts` (ADR-021 slice 4): policy workers per member, one
// generation of population play, the members' files and the top-level files written, one canonical JSON line
// printed. The Python driver (slice 5) passes `--member` for each member. With `--game-processes` above 1 the games
// are shared over further game processes (runs of games grouped by member, #1355), each with workers for the members that play in its slice (#1333).

import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Engine } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { type FeatureSetup, GenerationError, type Shard, type TrainingPool } from "./league-generation.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { playCalibration, writeCalibration } from "./population-calibration.ts";
import { calibrationAnchors } from "./population-plan.ts";
import { checkPopulationConfig, membersPlaying, type PopulationEntry, type PopulationMember, type PopulationOptions, playPopulationEntries, playPopulationGeneration, populationPlanOf, type SnapshotSession, writePopulationGeneration } from "./population-generation.ts";
import { exitWhenParentDies, vouchTo } from "./parent-link.ts";
import { obtainPool } from "./pool-share.ts";
import { featuresDirOf, optionalFlag, POOL_FILE, SHARD_FILE, type ShardProcess, startShard, writeShardFile } from "./shard-process.ts";
import type { PastPoolFile, PastSnapshot } from "./past-pool.ts";
import { createStartLimiter, DEFAULT_CONCURRENT_STARTS } from "./start-slots.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

/** One `--member m<k>,PREVIEW,BATTLE,POLICY_ID` flag: the member's frozen pair (a policy id holds a colon, so the fields are comma separated). */
export interface MemberFlag {
  member: string;
  preview: string;
  battle: string;
  policy_id: string;
}

export function parseMemberFlag(text: string): MemberFlag {
  const match = /^(m(?:0|[1-9][0-9]?)),([0-9a-f]{64}),([0-9a-f]{64}),(policy:[0-9a-f]{64})$/.exec(text);
  if (match === null) throw new GenerationError("--member is m<k>,PREVIEW,BATTLE,POLICY_ID (two 64-hex checkpoint ids and a policy: id)");
  return { member: match[1] as string, preview: match[2] as string, battle: match[3] as string, policy_id: match[4] as string };
}

/** How a game process gets the team pool: from its starter's file, or by loading it (and leaving the file for the processes it starts). */
export type PoolShare = Parameters<typeof obtainPool>[2];

export interface PopulationCliInput {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** The team pool, already loaded; or, given `load_pool`, absent (the pool is then loaded after the further game processes are started). */
  pool?: TrainingPool;
  /** Load the pool the way a game process does (`obtainPool`): the share says whether to read its starter's file or to write one for the processes this one starts. */
  load_pool?: (share: PoolShare) => Promise<TrainingPool>;
  members: readonly MemberFlag[];
  out: string;
  run_dir: string;
  worker_python: string;
  engine_data: string;
  engine_data_hash: string;
  encoder_version: string;
  belief_worlds?: BeliefWorldsFactory | undefined;
  /** The candidate feature set the run names (`featureSetup`); every worker states it and every seat that reads a worker attaches its rows. */
  features?: FeatureSetup | undefined;
  /** The committed past pool of the generation before, for a config with a past-pool share. */
  past?: PastPoolFile;
  /** Policy workers per member in this process (`--policy-workers`, default 1). Operational: the generation's files do not depend on it. */
  policy_workers?: number;
  /** Game processes, this one included (`--game-processes`, default 1; operational). */
  processes?: number;
  /** This process's slice of the plan: set when there are game processes (`index` 0 for the starter), absent for a single process. */
  shard?: Shard;
  /** This is a further game process: it plays its slice, leaves `shard.jsonl` in `out` and ends with its starter. */
  further?: boolean;
  /** The script that started this process, run again for each further game process. */
  shard_script?: string;
  /** The flags a further game process is started with besides its internal ones (the config's, the members', the past pool's, the worker counts). */
  forward?: readonly string[];
  /** Internal, given by the starter: the generation directory the Episodes are written below. */
  episodes_dir?: string;
  /** Internal, given by the starter: the scratch directory the game processes leave candidate feature rows in. */
  features_dir?: string;
  /** Internal, given by the starter: the file the pool is shared in. */
  pool_file?: string;
  /** Worker start-ups in flight across all game processes (`--max-concurrent-starts`, default 2; operational). */
  max_concurrent_starts?: number;
  /** Internal, given by the starter: the directory of the start-up slots every game process shares. */
  start_slots_dir?: string;
  /** A test seam: how a further game process is started (default `startShard`). */
  start_shard?: (script: string, args: readonly string[], out: string, index: number) => ShardProcess<PopulationEntry>;
  /** A test seam: the link to the starter of a further game process (default `exitWhenParentDies`). */
  parent_link?: () => Promise<string>;
  /** A test seam: the client of a member (default: a `PolicyWorkerClient` on the Python worker). */
  make_client?: (entry: MemberFlag) => MemberClient;
  /** A test seam: the client of a snapshot (default: a `PolicyWorkerClient` on its frozen pair). */
  make_snapshot_client?: (snapshot: PastSnapshot) => MemberClient;
}

/** What the CLI needs of a member's worker client. */
export type MemberClient = PopulationMember["client"] & { start: () => Promise<void>; close: () => Promise<void> };

/** The client of a member none of whose games this process plays: no worker starts for it, and it has nothing to ask (a game that used it would not run). */
const UNUSED = {} as PopulationMember["client"];

/** Run one population generation; returns the exit code (0 ok, 1 a failed generation) and prints the summary line. */
export async function runPopulationCli(input: PopulationCliInput, print: (line: string) => void): Promise<number> {
  const { config, generation } = input;
  const { max_line_bytes, n_parallel, handshake_timeout_ms } = config.generation;
  const processes = input.processes ?? 1;
  const workers = input.policy_workers ?? 1;
  // A refused config costs no worker: each session is about 445 MB.
  try {
    checkPopulationConfig(config, input.members, generation, 1, input.past);
  } catch (error) {
    if (!(error instanceof GenerationError)) throw error;
    console.error(`play-generation: ${error.message}`);
    return 1;
  }
  const workerFor = (pair: { preview: string; battle: string }): MemberClient =>
    new PolicyWorkerClient({
      command: input.worker_python,
      args: ["-I", "-m", "areneon_ml.worker", "--run-dir", input.run_dir, "--preview", pair.preview, "--battle", pair.battle, "--engine-data", input.engine_data, "--max-line-bytes", String(max_line_bytes)],
      env: {},
      max_line_bytes,
      n_parallel,
      handshake_timeout_ms,
      ...(input.features === undefined ? {} : { features: input.features.statement }),
      expect: {
        encoder_version: input.encoder_version,
        engine_data_hash: input.engine_data_hash,
        checkpoints: [
          { role: "battle", checkpoint_id: pair.battle },
          { role: "preview", checkpoint_id: pair.preview },
        ],
      },
    });
  // A snapshot's session starts when the generation first needs it and is closed after its last wave (the generation closes it).
  const openSnapshot = async (snapshot: PastSnapshot): Promise<SnapshotSession> => {
    const client = input.make_snapshot_client?.(snapshot) ?? workerFor(snapshot);
    await limiter().run(() => client.start());
    return { client, close: () => client.close() };
  };
  const clients: MemberClient[] = [];
  let startLimiter: ReturnType<typeof createStartLimiter> | undefined;
  const limiter = (): ReturnType<typeof createStartLimiter> => (startLimiter ??= createStartLimiter({ dir: slotsDir, max: input.max_concurrent_starts ?? DEFAULT_CONCURRENT_STARTS }));
  const children: ShardProcess<PopulationEntry>[] = [];
  const scratch = !input.further && processes > 1 ? mkdtempSync(join(tmpdir(), "areneon-shards-")) : undefined;
  const featuresDir = featuresDirOf(input.features_dir, scratch, input.features !== undefined);
  const slotsDir = input.start_slots_dir ?? (scratch === undefined ? undefined : join(scratch, "slots"));
  const poolFile = input.pool_file ?? (scratch === undefined ? undefined : join(scratch, POOL_FILE));
  try {
    mkdirSync(input.out, { recursive: true });
    const vouched = input.further === true ? (input.parent_link ?? (() => exitWhenParentDies()))() : undefined; // before the pool and the workers, which take long
    // Start the further game processes first: each loads its own pool while this one does.
    if (scratch !== undefined) {
      const start = input.start_shard ?? ((script, args, out, index) => startShard<PopulationEntry>(script, args, out, index));
      for (let index = 1; index < processes; index++) {
        const out = join(scratch, String(index));
        children.push(start(input.shard_script as string, [...(input.forward ?? []), "--out", out, "--shard", `${index}/${processes}`, "--episodes-dir", input.out, ...optionalFlag("--pool-file", poolFile), ...optionalFlag("--features-dir", featuresDir), ...optionalFlag("--start-slots-dir", slotsDir)], out, index));
      }
    }
    // A further game process has no worker running while it waits for its starter's pool (its workers start once it knows the plan), and its pipe to the starter is unreferenced: keep the process alive until the pool arrives.
    const waiting = vouched === undefined ? undefined : setInterval(() => undefined, 1_000);
    let pool: TrainingPool;
    try {
      pool = input.pool ?? (await (input.load_pool as NonNullable<PopulationCliInput["load_pool"]>)({ read: vouched === undefined || input.pool_file === undefined ? undefined : { file: input.pool_file, digest: vouched }, ...(scratch === undefined ? {} : { write: poolFile, vouch: vouchTo(children) }) }));
    } finally {
      clearInterval(waiting);
    }
    // Workers only for the members that play in this process's games (a member's workers are 445 MB each).
    const plan = populationPlanOf(config, generation, pool.teams.length, input.members, input.past);
    const needed = membersPlaying(plan, input.shard);
    const members: PopulationMember[] = input.members.map((entry) => {
      if (!needed.has(entry.member)) return { member: entry.member, policy_id: entry.policy_id, checkpoints: { preview: entry.preview, battle: entry.battle }, client: UNUSED };
      const own = Array.from({ length: workers }, () => input.make_client?.(entry) ?? workerFor(entry));
      clients.push(...own);
      const [client, ...peers] = own as [MemberClient, ...MemberClient[]];
      return { member: entry.member, policy_id: entry.policy_id, checkpoints: { preview: entry.preview, battle: entry.battle }, client, ...(peers.length === 0 ? {} : { peers }) };
    });
    // The workers this process starts (#1355): a process plays the games of few members, so this stays near the worker count.
    console.error(`play-generation: process ${input.shard?.index ?? 0} of ${processes} starts ${clients.length} policy workers for ${needed.size} of ${input.members.length} members`);
    // One start at a time in this process, and at most `--max-concurrent-starts` across all of them: a worker's handshake timer runs only while it holds a slot.
    for (const client of clients) await limiter().run(() => client.start());
    const options: PopulationOptions = {
      engine: input.engine,
      config,
      generation,
      members,
      pool,
      out_dir: input.out,
      belief_worlds: input.belief_worlds,
      ...(input.episodes_dir === undefined ? {} : { shared_dir: input.episodes_dir }),
      ...(input.features === undefined ? {} : { features: input.features.configs }),
      features_dir: featuresDir,
      ...(input.shard === undefined ? {} : { shard: input.shard }),
      ...(input.past === undefined ? {} : { past: { pool: input.past, open: openSnapshot } }),
    };
    if (input.further === true) {
      // A further game process (it ends with the process that started it). It plays its slice and leaves it for its starter.
      const entries = await playPopulationEntries(options, input.shard);
      writeShardFile(join(input.out, SHARD_FILE), entries);
      return 0;
    }
    // The first game process to fail fails this one at once: `Promise.all` rejects with its message, `playPopulationGeneration` stops this process's own games, and the `finally` below kills the others (#1100).
    const others = Promise.all(children.map((child) => child.entries)).then((all) => all.flat());
    const result = await playPopulationGeneration({ ...options, ...(scratch === undefined ? {} : { others, live_children: () => children.filter((child) => child.alive()).length }) });
    writePopulationGeneration(input.out, result);
    print(jcs({ generation, run_config_hash: await runConfigHash(config), members: result.members.map((entry) => ({ member: entry.member, training_set_hash: entry.training_set_hash, examples: entry.set.examples })) } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    for (const child of children) child.stop();
    await Promise.all(clients.map((client) => client.close()));
    if (scratch !== undefined) rmSync(scratch, { recursive: true, force: true });
  }
}

export interface CalibrationCliInput {
  engine: Engine;
  config: TrainingRunConfig;
  pool: TrainingPool;
  out: string;
  belief_worlds?: BeliefWorldsFactory | undefined;
}

/**
 * Play the anchor calibration of a population run (generation 0, ADR-021 section 4) and write `calibration.json`; no
 * worker starts. Returns the exit code (0 ok, 1 a failed calibration) and prints one canonical JSON line.
 */
export async function runCalibrationCli(input: CalibrationCliInput, print: (line: string) => void): Promise<number> {
  const { config } = input;
  try {
    if (config.population === undefined) throw new GenerationError("population: the config has no population section");
    calibrationAnchors(config);
    mkdirSync(input.out, { recursive: true });
    const result = await playCalibration({ engine: input.engine, config, pool: input.pool, belief_worlds: input.belief_worlds });
    writeCalibration(input.out, result);
    print(jcs({ generation: 0, run_config_hash: result.run_config_hash, anchors: result.anchors, games: result.games.length } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  }
}
