// `scripts/play-generation.ts`: one generation of A1 play for the Python driver
// (ml/areneon_ml/loop). It takes everything on the command line, starts the
// Python policy worker (CPU torch, stdio only) with the frozen pair, plays the
// generation, writes its files and prints one canonical JSON line, the
// generation's `generation.json`. Exit 0 ok, 1 a generation that failed, 2 a
// usage or configuration error. Nothing is retried.

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { engineDataHash, readEngineDataAny } from "./engine-data.ts";
import { loadEngine } from "./engine.ts";
import type { Engine } from "./engine.ts";
import { featureSetup, GenerationError, labelSetup, type SearchSetupFactory, sheetSetup, type PoolRequest, playEntries, playGeneration, type Shard, type TrainingPool, trainingTeamPool, writeGeneration } from "./league-generation.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { resolveFormat } from "./formats.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { type BeliefFeaturesFactory, SheetTotals } from "./belief-features.ts";
import { labelCounts } from "./search-labeller.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { PastPoolError, readPastPool } from "./past-pool.ts";
import { leaveFigures, printProfile, startProfile, stopProfile, timed } from "./profile.ts";
import { exitWhenParentDies, vouchTo } from "./parent-link.ts";
import { obtainPool } from "./pool-share.ts";
import { type MemberFlag, parseMemberFlag, runCalibrationCli, runPopulationCli } from "./population-cli.ts";
import { largest, ownPeakRss } from "./process-usage.ts";
import { combineUsage, featuresDirOf, optionalFlag, ownShard, parallelCount, POOL_FILE, readShardFile, type ResourceUsage, SHARD_FILE, type ShardProcess, startShard, USAGE_FILE, usageOf, writeShardFile } from "./shard-process.ts";
import { readTrainingRunConfig, runConfigHash } from "./training-run-config.ts";

// The further-game-process helpers live in `shard-process.ts` (the population generation uses them too); re-exported for the existing importers.
export { combineUsage, featuresDirOf, optionalFlag, readShardFile, type ResourceUsage, type ShardProcess, startShard, writeShardFile };

export class UsageError extends Error {
  override name = "UsageError";
}

/** Where a run's teams come from: the CLI asks for the pool once the config and the engine are read. */
export type PoolSource = (engine: Engine, config: PoolRequest) => TrainingPool;

/**
 * The pool of this package's own script: the interim fixture pool. A config
 * that names a team dataset is refused here, never played on the interim pool:
 * the dataset reader lives in `@areneon/team-dataset`, which depends on this
 * package, so its script supplies the pool through `runGenerationCli`'s port.
 */
export const interimPoolSource: PoolSource = (engine, config) => {
  if (config.team_dataset !== undefined) {
    throw new UsageError("this config names a team dataset: run packages/team-dataset/scripts/play-generation.ts");
  }
  return trainingTeamPool(engine, config.generation.excluded_abilities);
};

const FLAGS = {
  config: { type: "string" },
  generation: { type: "string" },
  "policy-id": { type: "string" },
  preview: { type: "string" },
  battle: { type: "string" },
  "run-dir": { type: "string" },
  "engine-data": { type: "string" },
  "encoder-version": { type: "string" },
  "worker-python": { type: "string" },
  out: { type: "string" },
  // Operational, not part of the run config: how many worker processes serve the games (default 1).
  "policy-workers": { type: "string" },
  // Operational too: game processes, this one included, each with its own workers (default 1).
  "game-processes": { type: "string" },
  // Internal: set on the game processes this one starts (`INDEX/COUNT`); such a process plays its slice and writes `shard.jsonl`.
  shard: { type: "string" },
  // Internal: the directory the Episodes of the generation are written to as their games finish (the process that started this one gives it to the game processes; by default `<out>/episodes`).
  "episodes-dir": { type: "string" },
  // Internal: the scratch directory the game processes leave the recomputed candidate feature rows of their Episodes in (#1185; only with game processes and a candidate feature set).
  "features-dir": { type: "string" },
  // Internal: the scratch directory the game processes leave their search label files in (only with `generation.search_labels`, from its start generation).
  "labels-dir": { type: "string" },
  // Internal: the file the process that started this one leaves the team pool in once it has loaded it (#1192); a game process reads it instead of loading the pool itself.
  "pool-file": { type: "string" },
  // Operational: time the phases of this generation (ADR-028 section 1, #1191) and print the figures as one more output line; no file the manifest hashes changes.
  profile: { type: "boolean" },
  // Internal: the directory the game processes and policy workers leave their profile figures in (made by the process that starts the others when --profile is given).
  "profile-dir": { type: "string" },
  // A population run (ADR-021): one `m<k>,PREVIEW,BATTLE,POLICY_ID` per member, in place of --preview, --battle and --policy-id.
  member: { type: "string", multiple: true },
  // A population run with a past-pool share (ADR-021 section 3, slice 10): the committed `past-pool.json` of the generation before.
  "past-pool": { type: "string" },
  // A population run's generation 0 (ADR-021 section 4): play the anchor calibration, no member and no worker.
  calibrate: { type: "boolean" },
} as const;

const OPTIONAL: readonly (keyof typeof FLAGS)[] = ["policy-workers", "game-processes", "shard", "episodes-dir", "features-dir", "labels-dir", "pool-file", "profile", "profile-dir", "member", "calibrate", "past-pool"];
/** The flags a population run gives per member instead. */
const PER_MEMBER: readonly (keyof typeof FLAGS)[] = ["preview", "battle", "policy-id"];

/**
 * An upper bound on game processes x policy workers x games in flight per worker: past it the machine is
 * oversubscribed (every one is an operating system process or a live battle), so the generation warns on
 * stderr. A sanity bound for the operator, not a rule: nothing is refused (#980).
 */
export const MAX_COMBINED_LOAD = 256;

/** The most games the processes of a generation can have in flight at once, and whether that is past `MAX_COMBINED_LOAD`. */
export function combinedLoad(processes: number, workers: number, nParallel: number): { games: number; over: boolean } {
  const games = processes * workers * nParallel;
  return { games, over: games > MAX_COMBINED_LOAD };
}

/** The flags a further game process is started with: the ones this process was given, but its own output, slice and internal directories. */
export function forwardedFlags(values: Record<string, string | string[] | boolean | undefined>): string[] {
  const internal = ["out", "game-processes", "shard", "episodes-dir", "features-dir", "labels-dir", "pool-file", "profile", "profile-dir"];
  return Object.entries(values).flatMap(([name, value]) => {
    if (internal.includes(name) || value === undefined) return [];
    if (Array.isArray(value)) return value.flatMap((each) => [`--${name}`, each]); // --member, once per member
    return typeof value === "string" ? [`--${name}`, value] : [];
  });
}

/** Run one generation; returns the process exit code and writes its one JSON line with `print`. */
export async function runGenerationCli(
  argv: readonly string[],
  print: (line: string) => void = console.log,
  poolSource: PoolSource = interimPoolSource,
  /** The script that started this process: `--game-processes` above 1 runs it again for each further game process. */
  shardScript?: string,
  /** The belief port factory, for a plan that names `damage-aware@v2` (the composition root supplies it: the bridge does not import the opponent model). */
  beliefWorlds?: BeliefWorldsFactory,
  /** The opponent sheet port factory, for a run config that names `opponent_sheet` (the composition root supplies it). */
  beliefFeatures?: BeliefFeaturesFactory,
  /** The search setup factory, for a run config that asks for search labels (the composition root supplies it). */
  searchSetups?: SearchSetupFactory,
): Promise<number> {
  let values: Record<Exclude<keyof typeof FLAGS, "member" | "calibrate" | "profile">, string | undefined> & { member?: string[]; calibrate?: boolean; profile?: boolean };
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: Exclude<keyof typeof FLAGS, "member" | "calibrate" | "profile">): string => {
    const value = values[name];
    if (value === undefined) throw new UsageError(`--${name} is required`);
    return value;
  };
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) {
    if (OPTIONAL.includes(name) || ((values.member !== undefined || values.calibrate === true) && PER_MEMBER.includes(name))) continue;
    need(name as Exclude<keyof typeof FLAGS, "member" | "calibrate" | "profile">);
  } // refuse a missing flag before anything starts
  const generation = Number(need("generation"));
  const calibrating = values.calibrate === true;
  if (calibrating ? generation !== 0 : !Number.isInteger(generation) || generation < 1) throw new UsageError(calibrating ? "--calibrate plays generation 0" : "--generation must be a positive integer");
  if (calibrating && (values.member !== undefined || values.shard !== undefined || values["game-processes"] !== undefined || values["policy-workers"] !== undefined)) throw new UsageError("--calibrate takes no --member and plays in one process");
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (config.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineDataAny(JSON.parse(readFileSync(dataFile, "utf8"))));
  if (calibrating) {
    if (config.population === undefined) throw new UsageError("--calibrate needs a population config");
    return runCalibrationCli({ engine, config, pool: poolSource(engine, config), out: need("out"), belief_worlds: beliefWorlds }, print);
  }
  if ((config.population !== undefined) !== (values.member !== undefined)) throw new UsageError("a population config is played with --member for each member, and --member only with a population config");
  if (config.population !== undefined) {
    if (values.profile === true || values["profile-dir"] !== undefined) throw new UsageError("--profile does not apply to a population run");
    let flags: MemberFlag[];
    try {
      flags = (values.member as string[]).map(parseMemberFlag);
    } catch (error) {
      throw new UsageError((error as Error).message);
    }
    const wantsPast = config.population.shares.past_pool_ppm !== 0;
    if (wantsPast !== (values["past-pool"] !== undefined)) throw new UsageError("--past-pool is required with a past-pool share and refused without one");
    let past: ReturnType<typeof readPastPool> | undefined;
    if (values["past-pool"] !== undefined) {
      try {
        past = readPastPool(JSON.parse(readFileSync(values["past-pool"], "utf8")), config, await runConfigHash(config));
      } catch (error) {
        if (error instanceof PastPoolError) throw new UsageError(error.message);
        throw error;
      }
    }
    // The game processes and policy workers of a population run: as in a single lineage, with each member's workers started in each process that plays one of its games (#1333).
    const refuse = (message: string): Error => new UsageError(message);
    const workerCount = parallelCount(values["policy-workers"], "policy-workers", refuse);
    const processes = parallelCount(values["game-processes"], "game-processes", refuse);
    if (values["pool-file"] !== undefined && values.shard === undefined) throw new UsageError("--pool-file is for a further game process only");
    const shard = ownShard(values.shard, processes, shardScript, refuse);
    const load = combinedLoad(processes, workerCount, config.generation.n_parallel);
    if (load.over && values.shard === undefined) console.error(`play-generation: ${processes} game processes x ${workerCount} workers x ${config.generation.n_parallel} games in flight is ${load.games} at once, over ${MAX_COMBINED_LOAD}: the machine may be oversubscribed`);
    let features: ReturnType<typeof featureSetup>;
    try {
      features = featureSetup(engine, config, beliefWorlds, beliefFeatures);
    } catch (error) {
      if (!(error instanceof GenerationError)) throw error;
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    return runPopulationCli(
      {
        engine, config, generation, members: flags, out: need("out"), run_dir: need("run-dir"), worker_python: need("worker-python"), engine_data: dataFile, engine_data_hash: dataHash, encoder_version: need("encoder-version"), belief_worlds: beliefWorlds,
        load_pool: (share) => obtainPool(config, () => poolSource(engine, config), share).then((found) => found.pool),
        features,
        policy_workers: workerCount,
        processes,
        ...(shard === undefined ? {} : { shard }),
        ...(values.shard === undefined ? {} : { further: true }),
        ...(shardScript === undefined ? {} : { shard_script: shardScript }),
        forward: forwardedFlags(values),
        ...(values["episodes-dir"] === undefined ? {} : { episodes_dir: values["episodes-dir"] }),
        ...(values["features-dir"] === undefined ? {} : { features_dir: values["features-dir"] }),
        ...(values["pool-file"] === undefined ? {} : { pool_file: values["pool-file"] }),
        ...(past === undefined ? {} : { past }),
      },
      print,
    );
  }
  const { max_line_bytes, n_parallel, handshake_timeout_ms } = config.generation;
  const preview = need("preview");
  const battle = need("battle");
  const count = (name: "policy-workers" | "game-processes"): number => parallelCount(values[name], name, (message) => new UsageError(message));
  const workerCount = count("policy-workers");
  const processes = count("game-processes");
  if (values["pool-file"] !== undefined && values.shard === undefined) throw new UsageError("--pool-file is for a further game process only");
  const load = combinedLoad(processes, workerCount, n_parallel);
  if (load.over && values.shard === undefined) console.error(`play-generation: ${processes} game processes x ${workerCount} workers x ${n_parallel} games in flight is ${load.games} at once, over ${MAX_COMBINED_LOAD}: the machine may be oversubscribed`);
  const own: Shard | undefined = ownShard(values.shard, processes, shardScript, (message) => new UsageError(message));
  // The candidate feature set the run's config names (none: nothing below changes). Built before any worker starts.
  let features: ReturnType<typeof featureSetup>;
  try {
    features = featureSetup(engine, config, beliefWorlds, beliefFeatures);
  } catch (error) {
    if (!(error instanceof GenerationError)) throw error;
    console.error(`play-generation: ${error.message}`);
    return 1;
  }
  // The opponent sheet source the config names (none: nothing below changes), set up likewise.
  let sheet: ReturnType<typeof sheetSetup>;
  try {
    sheet = sheetSetup(engine, config, beliefFeatures);
  } catch (error) {
    if (!(error instanceof GenerationError)) throw error;
    console.error(`play-generation: ${error.message}`);
    return 1;
  }
  // The sheet builders' counts of this process's games, summed and written to stderr at the end (not in the run files yet).
  const totals = new SheetTotals();
  const sheetArgs =
    sheet === undefined
      ? { client: {}, options: {}, report: () => undefined }
      : { client: { sheet: sheet.statement }, options: { sheet: sheet.setup, on_sheet_counts: totals.add }, report: totals.report };
  // The search labelling the config asks for (none: nothing below changes), at or after its start generation.
  let labels: ReturnType<typeof labelSetup>;
  try {
    labels = labelSetup(engine, config, { preview, battle }, searchSetups, features?.configs);
  } catch (error) {
    if (!(error instanceof GenerationError)) throw error;
    console.error(`play-generation: ${error.message}`);
    return 1;
  }
  const labelling = labels !== undefined && generation >= labels.start_generation;
  // The profile (#1191): on for --profile, and for a game process whose starter gave it the directory. Timers only read clocks.
  const profile = startProfile(values["profile-dir"], values.profile === true);
  const profileDir = profile?.dir;
  const makeClient = (): PolicyWorkerClient => new PolicyWorkerClient({
    command: need("worker-python"),
    args: [
      "-I", "-m", "areneon_ml.worker", "--run-dir", need("run-dir"), "--preview", preview, "--battle", battle,
      "--engine-data", dataFile, "--max-line-bytes", String(max_line_bytes),
      ...optionalFlag("--profile-dir", profileDir),
    ],
    env: {},
    max_line_bytes,
    n_parallel,
    handshake_timeout_ms,
    ...(features === undefined ? {} : { features: features.statement }),
    ...sheetArgs.client,
    expect: {
      encoder_version: need("encoder-version"),
      engine_data_hash: dataHash,
      checkpoints: [
        { role: "battle", checkpoint_id: battle },
        { role: "preview", checkpoint_id: preview },
      ],
    },
  });
  const clients = Array.from({ length: workerCount }, makeClient);
  // The Episodes go to one directory as their games finish; a further game process writes into its starter's.
  const episodesDir = values["episodes-dir"] ?? join(need("out"), "episodes");
  mkdirSync(episodesDir, { recursive: true });
  const scratch = values.shard === undefined && processes > 1 ? mkdtempSync(join(tmpdir(), "areneon-shards-")) : undefined;
  // With game processes and a feature set, every process recomputes the rows of its own Episodes when its games are done (#1185).
  const featuresDir = featuresDirOf(values["features-dir"], scratch, features !== undefined || sheet !== undefined);
  // With game processes and labels, every process labels its own Episodes when its games are done (#1290); a lone process leaves its files in a scratch directory of its own.
  const ownLabels = labelling && values["labels-dir"] === undefined && scratch === undefined ? mkdtempSync(join(tmpdir(), "areneon-labels-")) : undefined;
  const labelsDir = !labelling ? undefined : values["labels-dir"] ?? (scratch !== undefined ? join(scratch, "labels") : ownLabels);
  const labelArgs = labels === undefined || !labelling ? {} : { labels, labels_dir: labelsDir };
  const children: ShardProcess[] = [];
  let closed = false;
  // The pool the further game processes read instead of loading it themselves (#1192).
  const poolFile = values["pool-file"] ?? (scratch !== undefined ? join(scratch, POOL_FILE) : undefined);
  try {
    const vouched = values.shard !== undefined ? exitWhenParentDies() : undefined; // before the pool and the workers, which take long; it also brings the digest of the pool file
    // Start the further game processes first: each loads its own pool while this one does.
    if (scratch !== undefined) {
      for (let index = 1; index < processes; index++) {
        const out = join(scratch, String(index));
        const rest = forwardedFlags(values);
        children.push(startShard(shardScript as string, [...rest, "--out", out, "--shard", `${index}/${processes}`, "--episodes-dir", episodesDir, ...optionalFlag("--pool-file", poolFile), ...optionalFlag("--features-dir", featuresDir), ...optionalFlag("--labels-dir", labelsDir), ...optionalFlag("--profile-dir", profileDir)], out, index));
      }
    }
    // A further game process reads the pool its starter loaded, once the starter has sent the digest of the file, while its own workers start;
    // the starter sends it as soon as the file is written. A worker that fails to start ends the wait.
    const [{ pool }] = await Promise.all([
      obtainPool(config, () => poolSource(engine, config), { read: vouched === undefined || values["pool-file"] === undefined ? undefined : { file: values["pool-file"], digest: vouched }, ...(scratch === undefined ? {} : { write: poolFile, vouch: vouchTo(children) }) }),
      ...clients.map((client) => client.start()),
    ]);
    const [client, ...peers] = clients as [PolicyWorkerClient, ...PolicyWorkerClient[]];
    if (values.shard !== undefined) {
      // A further game process (it ends with the process that started it, see above). It plays the slice and leaves it for the process that started this one.
      const entries = await playEntries({ engine, config, generation, policy_id: need("policy-id"), client, peers, pool, episodes_dir: episodesDir, belief_worlds: beliefWorlds, ...(features === undefined ? {} : { features: features.configs }), ...sheetArgs.options, ...labelArgs, features_dir: featuresDir }, own);
      mkdirSync(need("out"), { recursive: true });
      writeShardFile(join(need("out"), SHARD_FILE), entries);
      writeFileSync(join(need("out"), USAGE_FILE), `${jcs(usageOf(ownPeakRss(), largest(clients.map((worker) => worker.peakRssBytes()))) as unknown as Json)}\n`);
      leaveFigures(profile, (own as Shard).index, ownPeakRss());
      return 0;
    }
    // The first game process to fail fails this one at once: `Promise.all` rejects with its message, `playGeneration` stops this process's own games, and the `finally` below kills the others (#1100).
    const others = Promise.all(children.map((child) => child.entries));
    // The stream goes straight to its file in chunks as the examples are made (#998).
    mkdirSync(need("out"), { recursive: true });
    const result = await playGeneration({ engine, config, generation, policy_id: need("policy-id"), client, peers, pool, stream_path: join(need("out"), "stream.jsonl"), episodes_dir: episodesDir, ...labelArgs, ...(labelling ? { labels_path: join(need("out"), "labels.jsonl") } : {}), belief_worlds: beliefWorlds, ...(features === undefined ? {} : { features: features.configs }), ...sheetArgs.options, features_dir: featuresDir, ...(own === undefined ? {} : { shard: own }), others: others.then((all) => all.flat()), live_children: () => children.filter((child) => child.alive()).length });
    timed("merge", () => writeGeneration(need("out"), result));
    // The labelling counts of this generation (also in generation.json): a generation with no label is told from a broken labeller by them.
    if (result.labels !== undefined) process.stderr.write(`search labels: ${jcs(labelCounts(result.labels.totals) as unknown as Json)}\n`);
    sheetArgs.report();
    print(jcs({ generation, run_config_hash: await runConfigHash(config), training_set_hash: result.training_set_hash, examples: result.set.examples }));
    // After the summary line, so a reader of the first line is unchanged: the peak memory of this process, its workers and the further game processes.
    const usage = combineUsage([usageOf(ownPeakRss(), largest(clients.map((worker) => worker.peakRssBytes()))), ...(await Promise.all(children.map((child) => child.usage)))]);
    if (Object.keys(usage).length > 0) print(jcs({ resource_usage: usage } as unknown as Json));
    // The workers write their figures when they end, so they end now; the game processes have ended (their results were read above).
    await printProfile(profile, async () => {
      await Promise.all(clients.map((worker) => worker.close()));
      closed = true;
    }, ownPeakRss(), print);
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`play-generation: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    for (const child of children) child.stop();
    if (!closed) await Promise.all(clients.map((client) => client.close()));
    if (scratch !== undefined) rmSync(scratch, { recursive: true, force: true });
    if (ownLabels !== undefined) rmSync(ownLabels, { recursive: true, force: true });
    stopProfile(profile);
  }
}

/** `scripts/format-manifest.ts --format <id>`: print the version manifest of a supported format as one canonical JSON line. */
export function runManifestCli(argv: readonly string[], print: (line: string) => void = console.log): number {
  let format: string | undefined;
  try {
    format = parseArgs({ args: [...argv], options: { format: { type: "string" } }, strict: true }).values.format;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  if (format === undefined) throw new UsageError("--format is required");
  print(jcs(resolveFormat(loadEngine(process.env.SHOWDOWN_DIR || undefined), format).manifest as unknown as Json));
  return 0;
}
