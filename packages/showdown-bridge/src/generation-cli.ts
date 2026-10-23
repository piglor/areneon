// `scripts/play-generation.ts`: one generation of A1 play for the Python driver
// (ml/areneon_ml/loop). It takes everything on the command line, starts the
// Python policy worker (CPU torch, stdio only) with the frozen pair, plays the
// generation, writes its files and prints one canonical JSON line, the
// generation's `generation.json`. Exit 0 ok, 1 a generation that failed, 2 a
// usage or configuration error. Nothing is retried.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { engineDataHash, readEngineData } from "./engine-data.ts";
import { loadEngine } from "./engine.ts";
import type { Engine } from "./engine.ts";
import { GenerationError, type PlayedEntry, type PoolRequest, playEntries, playGeneration, type Shard, type TrainingPool, trainingTeamPool, writeGeneration } from "./league-generation.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { ChunkedFileWriter, readFileLines } from "./chunked-file.ts";
import { resolveFormat } from "./formats.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { readTrainingRunConfig, runConfigHash } from "./training-run-config.ts";

/** An upper bound on `--policy-workers` and `--game-processes`: each is an operating system process. */
const MAX_PARALLEL = 64;

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
} as const;

const OPTIONAL: readonly (keyof typeof FLAGS)[] = ["policy-workers", "game-processes", "shard"];

/** Run one generation; returns the process exit code and writes its one JSON line with `print`. */
export async function runGenerationCli(
  argv: readonly string[],
  print: (line: string) => void = console.log,
  poolSource: PoolSource = interimPoolSource,
  /** The script that started this process: `--game-processes` above 1 runs it again for each further game process. */
  shardScript?: string,
): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: keyof typeof FLAGS): string => {
    const value = values[name];
    if (value === undefined) throw new UsageError(`--${name} is required`);
    return value;
  };
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) if (!OPTIONAL.includes(name)) need(name); // refuse a missing flag before anything starts
  const generation = Number(need("generation"));
  if (!Number.isInteger(generation) || generation < 1) throw new UsageError("--generation must be a positive integer");
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (config.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const { max_line_bytes, n_parallel, handshake_timeout_ms } = config.generation;
  const preview = need("preview");
  const battle = need("battle");
  const count = (name: "policy-workers" | "game-processes"): number => {
    const text = values[name] ?? "1";
    const value = /^\d+$/.test(text) ? Number(text) : Number.NaN;
    if (!Number.isInteger(value) || value < 1 || value > MAX_PARALLEL) throw new UsageError(`--${name} must be an integer from 1 to ${MAX_PARALLEL}`);
    return value;
  };
  const workerCount = count("policy-workers");
  const processes = count("game-processes");
  let own: Shard | undefined;
  if (values.shard !== undefined) {
    const match = /^(\d+)\/(\d+)$/.exec(values.shard);
    if (match === null || processes !== 1) throw new UsageError("--shard is INDEX/COUNT, and only for a process of its own");
    own = { index: Number(match[1]), of: Number(match[2]) };
    if (own.of < 2 || own.of > MAX_PARALLEL || own.index >= own.of) throw new UsageError("--shard is INDEX/COUNT, and only for a process of its own");
  } else if (processes > 1) {
    own = { index: 0, of: processes };
    if (shardScript === undefined) throw new UsageError("--game-processes needs the script to start again");
  }
  const makeClient = (): PolicyWorkerClient => new PolicyWorkerClient({
    command: need("worker-python"),
    args: [
      "-I", "-m", "areneon_ml.worker", "--run-dir", need("run-dir"), "--preview", preview, "--battle", battle,
      "--engine-data", dataFile, "--max-line-bytes", String(max_line_bytes),
    ],
    env: {},
    max_line_bytes,
    n_parallel,
    handshake_timeout_ms,
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
  const scratch = values.shard === undefined && processes > 1 ? mkdtempSync(join(tmpdir(), "areneon-shards-")) : undefined;
  const children: ShardProcess[] = [];
  try {
    // Start the further game processes first: each loads its own pool while this one does.
    if (scratch !== undefined) {
      for (let index = 1; index < processes; index++) {
        const out = join(scratch, String(index));
        const rest = Object.entries(values).flatMap(([name, value]) =>
          value === undefined || ["out", "game-processes", "shard"].includes(name) ? [] : [`--${name}`, value],
        );
        children.push(startShard(shardScript as string, [...rest, "--out", out, "--shard", `${index}/${processes}`], out, index));
      }
    }
    const pool = poolSource(engine, config);
    await Promise.all(clients.map((client) => client.start()));
    const [client, ...peers] = clients as [PolicyWorkerClient, ...PolicyWorkerClient[]];
    if (values.shard !== undefined) {
      // A further game process: it ends if the process that started it dies (its stdin closes).
      process.stdin.once("end", () => process.exit(1));
      process.stdin.resume();
      (process.stdin as unknown as { unref?: () => void }).unref?.(); // an open pipe must not keep the finished process alive
      // It plays the slice and leaves it for the process that started this one.
      const entries = await playEntries({ engine, config, generation, policy_id: need("policy-id"), client, peers, pool }, own);
      mkdirSync(need("out"), { recursive: true });
      writeShardFile(join(need("out"), SHARD_FILE), entries);
      return 0;
    }
    const others = Promise.all(children.map((child) => child.entries));
    // The stream goes straight to its file in chunks as the examples are made (#998).
    mkdirSync(need("out"), { recursive: true });
    const result = await playGeneration({ engine, config, generation, policy_id: need("policy-id"), client, peers, pool, stream_path: join(need("out"), "stream.jsonl"), ...(own === undefined ? {} : { shard: own }), others: others.then((all) => all.flat()) });
    writeGeneration(need("out"), result);
    print(jcs({ generation, run_config_hash: await runConfigHash(config), training_set_hash: result.training_set_hash, examples: result.set.examples }));
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

/** The game process's result: one canonical JSON entry per line, so no string holds them all (#998). */
const SHARD_FILE = "shard.jsonl";

/** Write a game process's entries, one line each, flushed to disk. */
export function writeShardFile(path: string, entries: readonly PlayedEntry[]): void {
  const writer = new ChunkedFileWriter(path);
  try {
    for (const entry of entries) writer.write(`${jcs(entry as unknown as Json)}\n`);
    writer.finish();
  } finally {
    writer.close();
  }
}

/** Read a shard file back, a line at a time. */
export function readShardFile(path: string): PlayedEntry[] {
  const entries: PlayedEntry[] = [];
  for (const line of readFileLines(path)) entries.push(JSON.parse(line) as PlayedEntry);
  return entries;
}

/** A further game process: the entries it will have played, and a way to end it. */
export interface ShardProcess {
  entries: Promise<PlayedEntry[]>;
  stop: () => void;
}

export function startShard(script: string, args: readonly string[], out: string, index: number, node: string = process.execPath): ShardProcess {
  const child = spawn(node, [...process.execArgv, script, ...args], { stdio: ["pipe", "ignore", "pipe"] }); // stdin stays open: the child ends when this process does
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => {
    stderr = (stderr + chunk.toString("utf8")).slice(-2000);
  });
  const entries = new Promise<PlayedEntry[]>((resolve, reject) => {
    child.once("error", (error) => reject(new GenerationError(`game process ${index} did not start: ${error.message}`)));
    child.once("close", (code, signal) => {
      if (code !== 0) {
        reject(new GenerationError(`game process ${index} failed (${signal === null ? `exit ${code}` : `signal ${signal}`}): ${stderr.trim()}`));
        return;
      }
      try {
        resolve(readShardFile(join(out, SHARD_FILE)));
      } catch (error) {
        reject(new GenerationError(`game process ${index} left no readable result: ${(error as Error).message}`));
      }
    });
  });
  entries.catch(() => undefined); // reported where it is awaited
  return { entries, stop: () => void child.kill("SIGKILL") };
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
