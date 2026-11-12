// The further game processes of a generation and their handoff (#957, #998, #1100): shared by the single-lineage
// generation (`generation-cli.ts`) and the population generation (`population-cli.ts`). A further game process plays
// its slice of the plan, leaves its entries one canonical JSON line each in `shard.jsonl` and its peak memory in
// `usage.json`; the process that started it reads them back. The entries are the caller's type (a played game of a
// single lineage, or of a population).

import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ChunkedFileWriter, readFileLines } from "./chunked-file.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { GenerationError, type PlayedEntry, type Shard } from "./league-generation.ts";
import { largest } from "./process-usage.ts";

/** An upper bound on `--policy-workers` and `--game-processes`: each is an operating system process. */
export const MAX_PARALLEL = 64;

/** `--policy-workers` or `--game-processes`: an integer from 1 to `MAX_PARALLEL` (default 1). */
export function parallelCount(text: string | undefined, name: "policy-workers" | "game-processes", refuse: (message: string) => Error): number {
  const given = text ?? "1";
  const value = /^\d+$/.test(given) ? Number(given) : Number.NaN;
  if (!Number.isInteger(value) || value < 1 || value > MAX_PARALLEL) throw refuse(`--${name} must be an integer from 1 to ${MAX_PARALLEL}`);
  return value;
}

/**
 * This process's slice of the plan: `--shard INDEX/COUNT` for a further game process (which is a process of its own, one
 * `--game-processes`), else the first slice of `--game-processes` above 1 (which needs the script to start again), else none.
 */
export function ownShard(flag: string | undefined, processes: number, shardScript: string | undefined, refuse: (message: string) => Error): Shard | undefined {
  if (flag !== undefined) {
    const match = /^(\d+)\/(\d+)$/.exec(flag);
    if (match === null || processes !== 1) throw refuse("--shard is INDEX/COUNT, and only for a process of its own");
    const own = { index: Number(match[1]), of: Number(match[2]) };
    if (own.of < 2 || own.of > MAX_PARALLEL || own.index >= own.of) throw refuse("--shard is INDEX/COUNT, and only for a process of its own");
    return own;
  }
  if (processes > 1) {
    if (shardScript === undefined) throw refuse("--game-processes needs the script to start again");
    return { index: 0, of: processes };
  }
  return undefined;
}

/** Peak memory of the processes of one generation's play (bytes), for the run-metrics record; any part may be missing. */
export interface ResourceUsage {
  game_process_peak_rss_bytes?: number;
  policy_worker_peak_rss_bytes?: number;
}

export const usageOf = (game: number | null, worker: number | null): ResourceUsage => ({
  ...(game === null ? {} : { game_process_peak_rss_bytes: game }),
  ...(worker === null ? {} : { policy_worker_peak_rss_bytes: worker }),
});

/** The largest peaks among several processes' usage. */
export function combineUsage(parts: readonly ResourceUsage[]): ResourceUsage {
  return usageOf(
    largest(parts.map((part) => part.game_process_peak_rss_bytes ?? null)),
    largest(parts.map((part) => part.policy_worker_peak_rss_bytes ?? null)),
  );
}

/** The directory the game processes leave feature rows in: the one this process was given, else a `features` directory of the shard scratch when there is one and the run reads a feature set (#1185). */
export function featuresDirOf(given: string | undefined, scratch: string | undefined, hasFeatures: boolean): string | undefined {
  return given ?? (scratch !== undefined && hasFeatures ? join(scratch, "features") : undefined);
}

/** `[name, value]` as arguments for a child process, or none when the value is absent. */
export function optionalFlag(name: string, value: string | undefined): string[] {
  return value === undefined ? [] : [name, value];
}

/** The team pool the starting process leaves in the shard scratch directory. */
export const POOL_FILE = "pool.json";

/** The game process's result: one canonical JSON entry per line, so no string holds them all (#998). */
export const SHARD_FILE = "shard.jsonl";
/** A game process's peak memory, beside its shard file. */
export const USAGE_FILE = "usage.json";

/** Write a game process's entries, one line each, flushed to disk. */
export function writeShardFile<T = PlayedEntry>(path: string, entries: readonly T[]): void {
  const writer = new ChunkedFileWriter(path);
  try {
    for (const entry of entries) writer.write(`${jcs(entry as unknown as Json)}\n`);
    writer.finish();
  } finally {
    writer.close();
  }
}

/** Read a shard file back, a line at a time. */
export function readShardFile<T = PlayedEntry>(path: string): T[] {
  const entries: T[] = [];
  for (const line of readFileLines(path)) entries.push(JSON.parse(line) as T);
  return entries;
}

/** A further game process: the entries it will have played, and a way to end it. */
export interface ShardProcess<T = PlayedEntry> {
  entries: Promise<T[]>;
  /** Its peak memory once it has ended (empty when it left none). */
  usage: Promise<ResourceUsage>;
  /** Whether the process is still running. */
  alive: () => boolean;
  /** Send the process one line over the pipe it ends with (a closed pipe is no failure: the process is gone and reports it). */
  tell: (line: string) => void;
  stop: () => void;
}

/** How long a failed game process's stderr is awaited after it exits: a process that left a grandchild holding the pipe open must not delay the report (#1100). */
const STDERR_GRACE_MS = 500;

export function startShard<T = PlayedEntry>(script: string, args: readonly string[], out: string, index: number, node: string = process.execPath): ShardProcess<T> {
  const child = spawn(node, [...process.execArgv, script, ...args], { stdio: ["pipe", "ignore", "pipe"] }); // stdin stays open: the child ends when this process does
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => {
    stderr = (stderr + chunk.toString("utf8")).slice(-2000);
  });
  let running = true;
  const entries = new Promise<T[]>((resolve, reject) => {
    const failed = (code: number | null, signal: NodeJS.Signals | null): void => reject(new GenerationError(`game process ${index} failed (${signal === null ? `exit ${code}` : `signal ${signal}`}): ${stderr.trim()}`));
    child.once("error", (error) => {
      running = false;
      reject(new GenerationError(`game process ${index} did not start: ${error.message}`));
    });
    // `exit`, not `close`: `close` waits for every holder of the stderr pipe, so a game process that failed
    // but left a grandchild behind was not reported until that one ended (#1100).
    child.once("exit", (code, signal) => {
      running = false;
      if (code !== 0) {
        const late = setTimeout(() => failed(code, signal), STDERR_GRACE_MS);
        child.once("close", () => {
          clearTimeout(late);
          failed(code, signal);
        });
      }
    });
    child.once("close", (code) => {
      running = false;
      if (code !== 0) return; // reported by the `exit` handler
      try {
        resolve(readShardFile<T>(join(out, SHARD_FILE)));
      } catch (error) {
        reject(new GenerationError(`game process ${index} left no readable result: ${(error as Error).message}`));
      }
    });
  });
  entries.catch(() => undefined); // reported where it is awaited
  const usage = entries.then((): ResourceUsage => {
    try {
      return JSON.parse(readFileSync(join(out, USAGE_FILE), "utf8")) as ResourceUsage;
    } catch {
      return {}; // diagnostics only: a missing file is no failure
    }
  });
  usage.catch(() => undefined);
  child.stdin.on("error", () => undefined); // a child that has gone leaves a broken pipe; its exit is reported through `entries`
  return { entries, usage, alive: () => running, tell: (line) => void child.stdin.write(line), stop: () => void child.kill("SIGKILL") };
}

