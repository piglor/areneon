import { mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { GenerationError } from "../src/league-generation.ts";
import { concurrentStarts, createStartLimiter, DEFAULT_CONCURRENT_STARTS, MAX_CONCURRENT_STARTS, releaseHeldSlots, SLOT_WAIT_REFUSAL, type SlotFs } from "../src/start-slots.ts";

const scratch = mkdtempSync(join(tmpdir(), "start-slots-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const refuse = (message: string): Error => new Error(message);
const real = (): SlotFs => ({ rename: renameSync, rm: (path) => rmSync(path, { recursive: true, force: true }), read: (path) => readFileSync(path, "utf8") });
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Starts that record how many are in flight at once. */
function probe(): { start: () => Promise<number>; peak: () => number; done: () => number } {
  let live = 0;
  let peak = 0;
  let done = 0;
  return {
    start: async () => {
      live++;
      peak = Math.max(peak, live);
      await sleep(15);
      live--;
      return ++done;
    },
    peak: () => peak,
    done: () => done,
  };
}

describe("concurrentStarts", () => {
  it("defaults to 2 and takes an integer from 1 to the bound", () => {
    expect(concurrentStarts(undefined, refuse)).toBe(DEFAULT_CONCURRENT_STARTS);
    expect(concurrentStarts("5", refuse)).toBe(5);
    for (const bad of ["0", "-1", "1.5", "x", String(MAX_CONCURRENT_STARTS + 1)]) expect(() => concurrentStarts(bad, refuse)).toThrow(/--max-concurrent-starts/);
  });
});

describe("createStartLimiter", () => {
  it("starts one at a time in a process with no shared directory, in order", async () => {
    const p = probe();
    const limiter = createStartLimiter({ max: 4 });
    expect(await Promise.all([1, 2, 3, 4].map(() => limiter.run(p.start)))).toEqual([1, 2, 3, 4]);
    expect(p.peak()).toBe(1);
  });

  it("never has more than N starts in flight across limiters sharing one directory (stand-in processes)", async () => {
    for (const max of [1, 2, 3]) {
      const dir = join(scratch, `shared-${max}`);
      const p = probe();
      const limiters = Array.from({ length: 6 }, () => createStartLimiter({ dir, max, poll_ms: 2 }));
      await Promise.all(limiters.flatMap((limiter) => [limiter.run(p.start), limiter.run(p.start)]));
      expect(p.done()).toBe(12);
      expect(p.peak()).toBe(max);
      expect(readdirSync(dir)).toEqual([]);
    }
  });

  it("releases a slot when a start fails, and the failure reaches only its caller", async () => {
    const limiter = createStartLimiter({ dir: join(scratch, "failing"), max: 1, poll_ms: 2 });
    await expect(limiter.run(() => Promise.reject(new GenerationError("no handshake")))).rejects.toThrow("no handshake");
    expect(await limiter.run(() => Promise.resolve("next"))).toBe("next");
    expect(readdirSync(join(scratch, "failing"))).toEqual([]);
  });

  const plant = (dir: string, owner: string): void => {
    mkdirSync(join(dir, "slot-0"), { recursive: true });
    writeFileSync(join(dir, "slot-0", "owner"), owner);
  };
  const ownerOf = (dir: string): string => readFileSync(join(dir, "slot-0", "owner"), "utf8");

  it("takes over a slot whose owner has gone, and leaves no temporary or tombstone directory", async () => {
    const dir = join(scratch, "stale");
    plant(dir, "999999:1:1");
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2, alive: (pid) => pid !== 999999 });
    expect(await limiter.run(() => Promise.resolve("took over"))).toBe("took over");
    expect(readdirSync(dir)).toEqual([]);
  });

  it("waits for a live owner until it frees the slot", async () => {
    const dir = join(scratch, "live");
    plant(dir, `${process.pid}:1:1`);
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2 });
    let ran = false;
    const waiting = limiter.run(() => ((ran = true), Promise.resolve(1)));
    await sleep(30);
    expect(ran).toBe(false);
    rmSync(join(dir, "slot-0"), { recursive: true });
    await waiting;
    expect(ran).toBe(true);
  });

  it("tells a gone owner from a live one with the real process check", async () => {
    const dir = join(scratch, "real-check");
    plant(dir, "2147483646:1:1");
    expect(await createStartLimiter({ dir, max: 1, poll_ms: 2 }).run(() => Promise.resolve("gone"))).toBe("gone");
  });

  it("gives up with a fixed error when no slot becomes free (no hang)", async () => {
    const dir = join(scratch, "timeout");
    plant(dir, `${process.pid}:1:1`);
    await expect(createStartLimiter({ dir, max: 1, poll_ms: 2, wait_ms: 20 }).run(() => Promise.resolve(1))).rejects.toThrow(SLOT_WAIT_REFUSAL);
  });

  it("never publishes a slot without its owner: a slot is installed whole, so there is no pid-less slot to wait on", async () => {
    const dir = join(scratch, "atomic");
    const seen: string[] = [];
    const fs = { ...real(), rename: (from: string, to: string) => { seen.push(readFileSync(join(from, "owner"), "utf8")); renameSync(from, to); } };
    await createStartLimiter({ dir, max: 1, fs }).run(() => ((expect(ownerOf(dir)).toBe(seen[0])), Promise.resolve(1)));
    expect(seen).toHaveLength(1);
    expect(seen[0]?.startsWith(`${process.pid}:`)).toBe(true);
  });

  it("lets only one of two takers of a dead slot move it, and the loser retries and acquires after the winner", async () => {
    const dir = join(scratch, "race-dead");
    plant(dir, "999999:1:1");
    let raced = false;
    const fs = {
      ...real(),
      rename: (from: string, to: string) => {
        if (!raced && from.endsWith("slot-0")) {
          raced = true;
          renameSync(from, join(dir, ".dead-other")); // the other taker wins the rename first
          rmSync(join(dir, ".dead-other"), { recursive: true });
        }
        renameSync(from, to);
      },
    };
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2, alive: (pid) => pid !== 999999, fs });
    expect(await limiter.run(() => Promise.resolve("ok"))).toBe("ok");
    expect(raced).toBe(true);
    expect(readdirSync(dir)).toEqual([]);
  });

  it("puts a live owner's slot back when it replaced the dead one between the read and the move", async () => {
    const dir = join(scratch, "race-aba");
    plant(dir, "999999:1:1");
    let swapped = false;
    const fs = {
      ...real(),
      rename: (from: string, to: string) => {
        if (!swapped && from.endsWith("slot-0")) {
          swapped = true;
          rmSync(from, { recursive: true });
          plant(dir, `${process.pid}:live:7`); // a live owner took the slot after the stale read
        }
        renameSync(from, to);
      },
    };
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2, wait_ms: 40, alive: (pid) => pid !== 999999, fs });
    await expect(limiter.run(() => Promise.resolve(1))).rejects.toThrow(SLOT_WAIT_REFUSAL); // the live owner keeps it
    expect(ownerOf(dir)).toBe(`${process.pid}:live:7`);
    expect(readdirSync(dir).sort()).toEqual(["slot-0"]);
  });

  it("puts nothing back over a slot taken since, and a release removes only the slot it still owns", async () => {
    const dir = join(scratch, "release");
    plant(dir, "999999:1:1");
    let step = 0;
    const fs = {
      ...real(),
      rename: (from: string, to: string) => {
        if (step === 0 && from.endsWith("slot-0")) {
          step = 1;
          rmSync(from, { recursive: true });
          plant(dir, `${process.pid}:live:8`);
          renameSync(from, to); // the live slot goes aside
          plant(dir, `${process.pid}:live:9`); // and a third owner installed its own before the put-back
          return;
        }
        renameSync(from, to);
      },
    };
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2, wait_ms: 30, alive: (pid) => pid !== 999999, fs });
    await expect(limiter.run(() => Promise.resolve(1))).rejects.toThrow(SLOT_WAIT_REFUSAL);
    expect(ownerOf(dir)).toBe(`${process.pid}:live:9`);
    // A start whose slot was taken over does not remove the other owner's slot when it releases.
    const mine = createStartLimiter({ dir: join(scratch, "stolen"), max: 1, poll_ms: 2 });
    await mine.run(async () => {
      writeFileSync(join(scratch, "stolen", "slot-0", "owner"), "1:x:y");
    });
    expect(readFileSync(join(scratch, "stolen", "slot-0", "owner"), "utf8")).toBe("1:x:y");
  });

  it("frees the slots a process still holds when it ends", async () => {
    const dir = join(scratch, "exit");
    const started = createStartLimiter({ dir, max: 1, poll_ms: 2 });
    void started.run(() => new Promise(() => undefined)); // never ends: the slot stays held
    await sleep(20);
    expect(readdirSync(dir)).toEqual(["slot-0"]);
    releaseHeldSlots();
    expect(readdirSync(dir)).toEqual([]);
  });

  it("refuses a count that is not a positive integer, and fails on a directory it cannot use", async () => {
    expect(() => createStartLimiter({ max: 0 })).toThrow(GenerationError);
    writeFileSync(join(scratch, "file"), "x");
    expect(() => createStartLimiter({ dir: join(scratch, "file", "nested"), max: 1 })).toThrow();
    const dir = join(scratch, "vanishing");
    const limiter = createStartLimiter({ dir, max: 1 });
    rmSync(dir, { recursive: true });
    await expect(limiter.run(() => Promise.resolve(1))).rejects.toThrow();
  });
});
