// A cap on concurrent worker start-ups across the game processes of a generation (#1355). A policy worker's handshake
// timer runs while a torch session loads, so dozens of workers starting together on a loaded machine time out. A
// process starts its workers one at a time, and each start holds one of N slots shared by every game process through
// directories in the shard scratch directory (`mkdir` is atomic, so no two processes hold one slot); a worker's
// handshake timer starts only once its start holds a slot. Operational: no byte of a generation depends on it.

import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GenerationError } from "./league-generation.ts";

/** The default of `--max-concurrent-starts`. */
export const DEFAULT_CONCURRENT_STARTS = 2;
/** The upper bound of `--max-concurrent-starts`. */
export const MAX_CONCURRENT_STARTS = 64;
const POLL_MS = 50;
/** How long a start waits for a slot before it gives up (a generation's workers start in minutes; this only stops a hang). */
export const SLOT_WAIT_MS = 60 * 60 * 1000;
export const SLOT_WAIT_REFUSAL = "no worker start slot became free in time";

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

/** The file operations the slots use, a test seam (the defaults are `node:fs`). */
export interface SlotFs {
  rename: (from: string, to: string) => void;
  rm: (path: string) => void;
  read: (path: string) => string;
}
const realFs: SlotFs = { rename: renameSync, rm: (path) => rmSync(path, { recursive: true, force: true }), read: (path) => readFileSync(path, "utf8") };

export interface StartLimiterOptions {
  /** The shared slot directory (absent: this process alone, one start at a time). */
  dir?: string | undefined;
  max: number;
  /** Test seams. */
  poll_ms?: number;
  wait_ms?: number;
  alive?: (pid: number) => boolean;
  fs?: SlotFs;
}

const processAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
};

/** The slots this process holds (path -> its owner token), so a failure that skipped a release cannot leave one behind. */
const held = new Map<string, string>();
let exitHooked = false;
/** Free every slot this process still holds (called when the process ends). */
export function releaseHeldSlots(): void {
  for (const path of held.keys()) rmSync(path, { recursive: true, force: true });
  held.clear();
}

let counter = 0;
const codeOf = (error: unknown): string | undefined => (error as NodeJS.ErrnoException).code;

/**
 * Take a free slot, waiting for one. A slot is published atomically with its owner token inside: a directory made
 * under a temporary name holds the token file and is renamed to `slot-N`, which fails while a non-empty directory is
 * there, so a slot never exists without its owner. A slot whose owner process is gone is moved aside by a rename (only
 * one taker wins it; the others find it gone), the moved directory is checked to be the one that was read, and it is
 * put back if it turned out to be another, live owner's.
 */
async function acquire(dir: string, max: number, poll: number, alive: (pid: number) => boolean, waitMs: number, fs: SlotFs): Promise<() => void> {
  const deadline = Date.now() + waitMs;
  for (;;) {
    for (let slot = 0; slot < max; slot++) {
      const path = join(dir, `slot-${slot}`);
      const token = `${process.pid}:${Date.now()}:${counter++}`;
      const tmp = join(dir, `.tmp-${token.replaceAll(":", "-")}`);
      mkdirSync(tmp);
      writeFileSync(join(tmp, "owner"), token);
      try {
        fs.rename(tmp, path);
      } catch (error) {
        fs.rm(tmp);
        if (codeOf(error) !== "ENOTEMPTY" && codeOf(error) !== "EEXIST") throw error;
        reclaim(dir, path, alive, fs);
        continue;
      }
      held.set(path, token);
      if (!exitHooked) {
        exitHooked = true;
        process.on("exit", releaseHeldSlots);
      }
      return () => {
        held.delete(path);
        let current: string | undefined;
        try {
          current = fs.read(join(path, "owner"));
        } catch {
          // Gone already.
        }
        if (current === token) fs.rm(path); // only a slot this start still owns
      };
    }
    if (Date.now() >= deadline) throw new GenerationError(SLOT_WAIT_REFUSAL);
    await new Promise((resolve) => setTimeout(resolve, poll));
  }
}

/** Move a slot whose owner process is gone out of the way (nothing happens for a live owner). */
function reclaim(dir: string, path: string, alive: (pid: number) => boolean, fs: SlotFs): void {
  let seen: string;
  try {
    seen = fs.read(join(path, "owner"));
  } catch {
    return; // freed (or taken over) meanwhile
  }
  if (alive(Number(seen.split(":")[0]))) return;
  const aside = join(dir, `.dead-${process.pid}-${counter++}`);
  try {
    fs.rename(path, aside); // only one taker wins; the others get ENOENT
  } catch (error) {
    if (codeOf(error) !== "ENOENT") throw error;
    return;
  }
  let moved = "";
  try {
    moved = fs.read(join(aside, "owner"));
  } catch {
    // Unreadable: it was not a slot of ours to keep.
  }
  if (moved === seen) {
    fs.rm(aside);
    return;
  }
  // Another owner took the slot between the read and the rename: put it back (best effort; a slot taken since is left alone).
  try {
    fs.rename(aside, path);
  } catch {
    fs.rm(aside);
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
        const release = dir === undefined ? () => undefined : await acquire(dir, max, options.poll_ms ?? POLL_MS, options.alive ?? processAlive, options.wait_ms ?? SLOT_WAIT_MS, options.fs ?? realFs);
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
