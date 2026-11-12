// A cap on concurrent worker start-ups across the game processes of a generation (#1355). A policy worker's handshake
// timer runs while a torch session loads, so dozens of workers starting together on a loaded machine time out. A
// process starts its workers one at a time, and each start holds one of N slots shared by every game process through
// directories in the shard scratch directory (`mkdir` is atomic, so no two processes hold one slot); a worker's
// handshake timer starts only once its start holds a slot. Operational: no byte of a generation depends on it.

import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GenerationError } from "./league-generation.ts";

/** The default of `--max-concurrent-starts`. */
export const DEFAULT_CONCURRENT_STARTS = 2;
/** The upper bound of `--max-concurrent-starts`. */
export const MAX_CONCURRENT_STARTS = 64;
const POLL_MS = 50;

/** `--max-concurrent-starts`: an integer from 1 to `MAX_CONCURRENT_STARTS` (default `DEFAULT_CONCURRENT_STARTS`). */
export function concurrentStarts(text: string | undefined, refuse: (message: string) => Error): number {
  if (text === undefined) return DEFAULT_CONCURRENT_STARTS;
  const value = /^\d+$/.test(text) ? Number(text) : Number.NaN;
  if (!Number.isInteger(value) || value < 1 || value > MAX_CONCURRENT_STARTS) throw refuse(`--max-concurrent-starts must be an integer from 1 to ${MAX_CONCURRENT_STARTS}`);
  return value;
}

/** Runs one start-up at a time in this process and, with a directory, within the global cap. */
export interface StartLimiter {
  run: <T>(start: () => Promise<T>) => Promise<T>;
}

export interface StartLimiterOptions {
  /** The shared slot directory (absent: this process alone, one start at a time). */
  dir?: string | undefined;
  max: number;
  /** Test seams. */
  poll_ms?: number;
  alive?: (pid: number) => boolean;
}

const processAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
};

const held = new Set<string>();
let exitHooked = false;
/** Slots still held when the process ends (a failure that skipped a release) are freed synchronously. */
function releaseOnExit(): void {
  if (exitHooked) return;
  exitHooked = true;
  process.on("exit", () => {
    for (const path of held) rmSync(path, { recursive: true, force: true });
  });
}

/** Take a free slot of the directory, waiting for one; a slot whose holder is gone is taken over. */
async function acquire(dir: string, max: number, poll: number, alive: (pid: number) => boolean): Promise<() => void> {
  for (;;) {
    for (let slot = 0; slot < max; slot++) {
      const path = join(dir, `slot-${slot}`);
      try {
        mkdirSync(path);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        let pid = Number.NaN;
        try {
          pid = Number(readFileSync(join(path, "pid"), "utf8"));
        } catch {
          // The holder is between mkdir and the pid file: leave it.
        }
        if (Number.isInteger(pid) && !alive(pid)) rmSync(path, { recursive: true, force: true });
        continue;
      }
      writeFileSync(join(path, "pid"), String(process.pid));
      held.add(path);
      releaseOnExit();
      return () => {
        held.delete(path);
        rmSync(path, { recursive: true, force: true });
      };
    }
    await new Promise((resolve) => setTimeout(resolve, poll));
  }
}

export function createStartLimiter(options: StartLimiterOptions): StartLimiter {
  if (!Number.isInteger(options.max) || options.max < 1) throw new GenerationError("the number of concurrent starts is a positive integer");
  const { dir, max } = options;
  if (dir !== undefined) mkdirSync(dir, { recursive: true });
  let chain: Promise<unknown> = Promise.resolve();
  return {
    run: <T>(start: () => Promise<T>): Promise<T> => {
      const turn = chain.then(async () => {
        const release = dir === undefined ? () => undefined : await acquire(dir, max, options.poll_ms ?? POLL_MS, options.alive ?? processAlive);
        try {
          return await start();
        } finally {
          release();
        }
      });
      chain = turn.catch(() => undefined);
      return turn;
    },
  };
}
