import { spawn } from "node:child_process";
import { existsSync, linkSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { acquireCopyLock, blockedMessage, type HeldLock, type LockBlocked, releaseCopyLock } from "../src/copy-lock.ts";
import { lockPath } from "../src/index.ts";
import { tempDir } from "./helpers.ts";

/** A pid no process has (above the default Linux pid_max). */
const DEAD = 2 ** 22 + 7;

function held(result: HeldLock | LockBlocked): HeldLock {
  if (!("release" in result)) throw new Error(`blocked: ${JSON.stringify(result)}`);
  return result;
}

function stale(path: string): void {
  writeFileSync(path, JSON.stringify({ token: "old", pid: DEAD, supervisor_pid: null, server_pid: null }));
}

describe("the copy lock", () => {
  it("lives next to the copy and is taken by one holder at a time", () => {
    const copy = join(tempDir("lock"), "abc");
    expect(lockPath(copy)).toBe(`${copy}.lock`);
    const first = held(acquireCopyLock(copy));
    expect(acquireCopyLock(copy)).toEqual({ state: "live", pid: process.pid });
    first.update({ supervisor_pid: DEAD, server_pid: DEAD + 1 });
    expect(JSON.parse(readFileSync(first.path, "utf8"))).toEqual({
      token: first.token,
      pid: process.pid,
      supervisor_pid: DEAD,
      server_pid: DEAD + 1,
    });
    releaseCopyLock(first.path, "someone else");
    expect(existsSync(first.path)).toBe(true);
    first.release();
    expect(existsSync(first.path)).toBe(false);
    first.release();
    held(acquireCopyLock(copy)).release();
  });

  it("replaces a stale lock with its own token", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    const lock = held(acquireCopyLock(copy));
    expect(JSON.parse(readFileSync(lock.path, "utf8"))).toMatchObject({ token: lock.token, pid: process.pid });
    expect(existsSync(`${lock.path}.break`)).toBe(false);
  });

  it("lets only one of two takers replace a stale lock", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    let second: HeldLock | LockBlocked | undefined;
    const first = acquireCopyLock(copy, {
      whileBreaking: () => {
        // The second taker arrives while the first holds the breaker.
        second = acquireCopyLock(copy);
      },
    });
    expect(second).toEqual({ state: "busy", breaker: `${lockPath(copy)}.break` });
    expect(held(first).token).toBe((JSON.parse(readFileSync(lockPath(copy), "utf8")) as { token: string }).token);
    // Arriving after the takeover, it finds the lock live.
    expect(acquireCopyLock(copy)).toEqual({ state: "live", pid: process.pid });
  });

  it("does not replace a lock that became live while it took the breaker", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    const result = acquireCopyLock(copy, {
      whileBreaking: () => {
        writeFileSync(
          lockPath(copy),
          JSON.stringify({ token: "new", pid: process.pid, supervisor_pid: null, server_pid: null }),
        );
      },
    });
    expect(result).toEqual({ state: "live", pid: process.pid });
    expect(readFileSync(lockPath(copy), "utf8")).toContain('"new"');
    expect(existsSync(`${lockPath(copy)}.break`)).toBe(false);
  });

  it("updates the lock by rename, never truncating the file a reader may open", () => {
    const dir = tempDir("lock");
    const copy = join(dir, "abc");
    const lock = held(acquireCopyLock(copy));
    const before = readFileSync(lock.path, "utf8");
    // A reader holding the old file still sees the whole old record.
    linkSync(lock.path, `${lock.path}.reader`);
    lock.update({ supervisor_pid: DEAD, server_pid: DEAD + 1 });
    expect(readFileSync(`${lock.path}.reader`, "utf8")).toBe(before);
    expect(JSON.parse(readFileSync(lock.path, "utf8"))).toMatchObject({ token: lock.token, server_pid: DEAD + 1 });
    expect(readdirSync(dir).filter((name) => name.endsWith(".tmp"))).toEqual([]);
  });

  it("takes a stale lock that vanished under the breaker with wx, as a fresh taker would", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    const lock = held(
      acquireCopyLock(copy, {
        whileBreaking: () => rmSync(lockPath(copy)),
      }),
    );
    expect(JSON.parse(readFileSync(lock.path, "utf8"))).toMatchObject({ token: lock.token });
    expect(existsSync(`${lock.path}.break`)).toBe(false);
  });

  it("does not overwrite a lock created between its re-read and its wx take", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    // A dangling symlink reads as absent but refuses a wx create, like a lock
    // a fresh taker created just after the re-read (a real loss reports that
    // taker's lock as `live`; this stand-in has none, so it reports `busy`).
    const result = acquireCopyLock(copy, {
      whileBreaking: () => {
        rmSync(lockPath(copy));
        symlinkSync(join(tempDir("lock"), "gone"), lockPath(copy));
      },
    });
    expect(result).toEqual({ state: "busy", breaker: `${lockPath(copy)}.break` });
    expect(existsSync(`${lockPath(copy)}.break`)).toBe(false);
  });

  it("fails closed on a breaker left by a crash", () => {
    const copy = join(tempDir("lock"), "abc");
    stale(lockPath(copy));
    writeFileSync(`${lockPath(copy)}.break`, "1\n");
    const result = acquireCopyLock(copy);
    expect(result).toEqual({ state: "busy", breaker: `${lockPath(copy)}.break` });
    expect(blockedMessage(copy, result as LockBlocked)).toContain(`remove ${lockPath(copy)}.break`);
  });

  it("propagates an I/O error other than an existing file", () => {
    expect(() => acquireCopyLock(join(tempDir("lock"), "missing-dir", "abc"))).toThrow(/ENOENT/);
  });

  it(
    "gives a stale lock to exactly one of many racing processes",
    async () => {
      const copy = join(tempDir("lock"), "abc");
      stale(lockPath(copy));
      const script = join(tempDir("racer"), "racer.ts");
      writeFileSync(
        script,
        `import { acquireCopyLock } from ${JSON.stringify(join(import.meta.dirname, "../src/copy-lock.ts"))};
const result = acquireCopyLock(${JSON.stringify(copy)});
console.log("release" in result ? "held" : result.state);
process.stdin.resume();
process.stdin.on("end", () => process.exit(0));
`,
      );
      const racers = Array.from({ length: 8 }, () =>
        spawn(process.execPath, [script], { stdio: ["pipe", "pipe", "inherit"] }),
      );
      try {
        const results = await Promise.all(
          racers.map(
            (racer) =>
              new Promise<string>((done) => racer.stdout.once("data", (chunk: Buffer) => done(chunk.toString().trim()))),
          ),
        );
        expect(results.filter((result) => result === "held")).toHaveLength(1);
        expect(results.every((result) => ["held", "live", "busy"].includes(result))).toBe(true);
      } finally {
        for (const racer of racers) racer.stdin.end();
      }
    },
    30_000,
  );

  it("names the holder, the unreadable lock or the breaker in its messages", () => {
    expect(blockedMessage("/r/c", { state: "live", pid: 42 })).toBe(
      "runtime copy /r/c is in use by pid 42 (lock /r/c.lock); " +
        "check that pid, and if it is not a launcher, supervisor or server, remove /r/c.lock",
    );
    expect(blockedMessage("/r/c", { state: "unreadable", reason: "bad" })).toBe(
      "cannot read the lock /r/c.lock (bad); remove it if nothing uses the copy",
    );
    expect(blockedMessage("/r/c", { state: "busy", breaker: "/r/c.lock.break" })).toBe(
      "another process is taking over the stale lock /r/c.lock; retry, or remove /r/c.lock.break " +
        "once no launcher or createRuntimeCopy is running for copy c; " +
        "leftover /r/c.lock.<token>.tmp files are harmless and may be deleted too",
    );
  });
});
