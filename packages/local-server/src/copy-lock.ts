import { randomBytes } from "node:crypto";
import { readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { basename } from "node:path";

/**
 * The lock of a runtime copy: `<runtimeRoot>/<commit>.lock`, next to the copy
 * rather than inside it, so replacing the copy cannot remove it.
 * `createRuntimeCopy` holds it while it puts a copy in place, and `launch`
 * (then its supervisor) for as long as a server may run from the copy.
 */
export function lockPath(copyDir: string): string {
  return `${copyDir}.lock`;
}

interface LockRecord {
  /** Unique per holder: a holder only ever replaces or removes its own record. */
  token: string;
  /** The process that took the lock. */
  pid: number;
  /** The launcher's supervisor, once it runs. */
  supervisor_pid: number | null;
  /** The server's process (and process group) id, once it runs. */
  server_pid: number | null;
}

export type LockBlocked =
  | { state: "live"; pid: number }
  | { state: "unreadable"; reason: string }
  | { state: "busy"; breaker: string };

export interface HeldLock {
  path: string;
  token: string;
  /** Record the supervisor and server pids (only the holder writes). */
  update(fields: { supervisor_pid: number; server_pid: number }): void;
  /** Remove the lock if it is still this holder's. */
  release(): void;
}

/** Test seam: runs while a stale lock is being replaced, with the breaker file held. */
export interface LockHooks {
  whileBreaking?: () => void;
}

/**
 * Take the lock of the runtime copy at `copyDir` for this process.
 *
 * A free lock is created with `wx`, so of two takers only one succeeds. A
 * stale lock (no process it names exists) is replaced under a second `wx`
 * file, `<lock>.break`: the one taker holding it re-reads the lock, and only
 * if it is still stale writes a unique temporary file and renames it over
 * the lock; if the lock has vanished meanwhile, it creates it with `wx`
 * instead, as a fresh taker would. It then re-reads the lock and owns it
 * only if it holds its own token. A breaker file left by a crash blocks every taker (`busy`) until it
 * is removed by hand; this fails closed rather than guessing.
 */
export function acquireCopyLock(copyDir: string, hooks: LockHooks = {}): HeldLock | LockBlocked {
  const path = lockPath(copyDir);
  const record: LockRecord = {
    token: randomBytes(16).toString("hex"),
    pid: process.pid,
    supervisor_pid: null,
    server_pid: null,
  };
  const text = (): string => `${JSON.stringify(record)}\n`;
  const held: HeldLock = {
    path,
    token: record.token,
    update(fields) {
      Object.assign(record, fields);
      // Write aside and rename, so a concurrent reader never sees a truncated record.
      const temp = tempPath(path, record.token);
      writeFileSync(temp, text());
      renameSync(temp, path);
    },
    release() {
      releaseCopyLock(path, record.token);
    },
  };
  if (createExclusive(path, text())) return held;
  const blocked = blocking(lockState(path));
  if (blocked !== null) return blocked;

  const breaker = `${path}.break`;
  if (!createExclusive(breaker, `${process.pid}\n`)) return { state: "busy", breaker };
  try {
    hooks.whileBreaking?.();
    const still = lockState(path);
    const blocked = blocking(still);
    if (blocked !== null) return blocked;
    if (still.state === "none") {
      // The stale lock vanished: a fresh taker may create it with `wx` at any
      // moment, so take it the same way rather than renaming over it.
      createExclusive(path, text());
    } else {
      const temp = tempPath(path, record.token);
      writeFileSync(temp, text(), { flag: "wx" });
      renameSync(temp, path);
    }
  } finally {
    rmSync(breaker, { force: true });
  }
  const after = lockState(path);
  if (after.state === "held" && after.record.token === record.token) return held;
  return blocking(after) ?? { state: "busy", breaker };
}

/** The holder's temporary file, renamed over the lock. */
function tempPath(path: string, token: string): string {
  return `${path}.${token}.tmp`;
}

/** Create `path` with `text` unless it exists; other errors propagate. */
function createExclusive(path: string, text: string): boolean {
  try {
    writeFileSync(path, text, { flag: "wx" });
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    return false;
  }
}

/** Remove the lock at `path` if it still holds `token`. */
export function releaseCopyLock(path: string, token: string): void {
  const state = lockState(path);
  if (state.state === "held" && state.record.token === token) rmSync(path, { force: true });
}

type LockState =
  | { state: "none" }
  | { state: "unreadable"; reason: string }
  | { state: "held"; record: LockRecord; live: number | null };

function lockState(path: string): LockState {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { state: "none" };
    return { state: "unreadable", reason: String(error) };
  }
  const record = parseLock(text);
  if (record === null) return { state: "unreadable", reason: `not a lock record: ${JSON.stringify(text)}` };
  return { state: "held", record, live: livePid(record) };
}

/** Why a lock in this state cannot be taken, or null if it is free or stale. */
function blocking(state: LockState): LockBlocked | null {
  if (state.state === "unreadable") return state;
  if (state.state === "held" && state.live !== null) return { state: "live", pid: state.live };
  return null;
}

/** A process the record names that still exists (the server counts while any of its group does). */
function livePid(record: LockRecord): number | null {
  if (exists(record.pid)) return record.pid;
  if (record.supervisor_pid !== null && exists(record.supervisor_pid)) return record.supervisor_pid;
  if (record.server_pid !== null && exists(-record.server_pid)) return record.server_pid;
  return null;
}

function parseLock(text: string): LockRecord | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) return null;
  const { token, pid, supervisor_pid, server_pid } = value as Record<string, unknown>;
  const optionalPid = (field: unknown): field is number | null => field === null || isPid(field);
  if (typeof token !== "string" || !isPid(pid) || !optionalPid(supervisor_pid) || !optionalPid(server_pid)) {
    return null;
  }
  return { token, pid, supervisor_pid, server_pid };
}

function isPid(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 1;
}

/** Whether a process (`pid` > 0) or any process in a group (`pid` < 0) exists. */
export function exists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** The message for a blocked lock, for the caller's own error type. */
export function blockedMessage(copyDir: string, blocked: LockBlocked): string {
  switch (blocked.state) {
    case "live":
      return (
        `runtime copy ${copyDir} is in use by pid ${blocked.pid} (lock ${lockPath(copyDir)}); ` +
        `check that pid, and if it is not a launcher, supervisor or server, remove ${lockPath(copyDir)}`
      );
    case "unreadable":
      return `cannot read the lock ${lockPath(copyDir)} (${blocked.reason}); remove it if nothing uses the copy`;
    case "busy":
      return (
        `another process is taking over the stale lock ${lockPath(copyDir)}; retry, or remove ${blocked.breaker} ` +
        `once no launcher or createRuntimeCopy is running for commit ${basename(copyDir)}; ` +
        `leftover ${lockPath(copyDir)}.<token>.tmp files are harmless and may be deleted too`
      );
  }
}
