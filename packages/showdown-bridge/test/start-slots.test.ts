import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { GenerationError } from "../src/league-generation.ts";
import { concurrentStarts, createStartLimiter, DEFAULT_CONCURRENT_STARTS, MAX_CONCURRENT_STARTS } from "../src/start-slots.ts";

const scratch = mkdtempSync(join(tmpdir(), "start-slots-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const refuse = (message: string): Error => new Error(message);
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

  it("takes over a slot whose holder has gone, and waits for one whose holder lives or has not written its pid", async () => {
    const dir = join(scratch, "stale");
    mkdirSync(join(dir, "slot-0"), { recursive: true });
    writeFileSync(join(dir, "slot-0", "pid"), "999999");
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2, alive: (pid) => pid !== 999999 });
    expect(await limiter.run(() => Promise.resolve("took over"))).toBe("took over");
    // A holder that is alive (or has no pid file yet) keeps the slot until it is freed.
    mkdirSync(join(dir, "slot-0"), { recursive: true });
    let ran = false;
    const waiting = limiter.run(() => ((ran = true), Promise.resolve(1)));
    await sleep(30);
    expect(ran).toBe(false);
    writeFileSync(join(dir, "slot-0", "pid"), String(process.pid));
    await sleep(30);
    expect(ran).toBe(false);
    rmSync(join(dir, "slot-0"), { recursive: true });
    await waiting;
    expect(ran).toBe(true);
  });

  it("tells a live holder (even one it may not signal) from a gone one with the real process check", async () => {
    const dir = join(scratch, "real-check");
    mkdirSync(join(dir, "slot-0"), { recursive: true });
    writeFileSync(join(dir, "slot-0", "pid"), "2147483646");
    const limiter = createStartLimiter({ dir, max: 1, poll_ms: 2 });
    expect(await limiter.run(() => Promise.resolve("gone"))).toBe("gone");
    mkdirSync(join(dir, "slot-0"), { recursive: true });
    writeFileSync(join(dir, "slot-0", "pid"), String(process.pid));
    let ran = false;
    const waiting = limiter.run(() => ((ran = true), Promise.resolve(1)));
    await sleep(30);
    expect(ran).toBe(false);
    rmSync(join(dir, "slot-0"), { recursive: true });
    await waiting;
    expect(ran).toBe(true);
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
