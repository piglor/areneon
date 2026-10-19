import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * The marker a launcher holds inside a runtime copy while one of its servers
 * may run from it. `createRuntimeCopy` refuses to replace a copy whose lock
 * names a live process, and a second launcher refuses to start on it.
 */
export const SERVER_LOCK = "areneon-server.lock";

interface LockRecord {
  /** The process that launched the server and owns the lock. */
  launcher_pid: number;
  /** The server's process (and process group) id, once it is spawned. */
  server_pid: number | null;
}

export type LockState =
  | { state: "none" }
  | { state: "stale" }
  | { state: "live"; pid: number }
  | { state: "unreadable"; reason: string };

/**
 * What the lock in `dir` says. A lock is live while its launcher or any
 * process in its server's process group exists; a lock that cannot be
 * parsed is `unreadable`, and callers fail closed on it.
 */
export function serverLockState(dir: string): LockState {
  let text: string;
  try {
    text = readFileSync(join(dir, SERVER_LOCK), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { state: "none" };
    return { state: "unreadable", reason: String(error) };
  }
  const record = parseLock(text);
  if (record === null) return { state: "unreadable", reason: `not a lock record: ${JSON.stringify(text)}` };
  if (exists(record.launcher_pid)) return { state: "live", pid: record.launcher_pid };
  if (record.server_pid !== null && exists(-record.server_pid)) {
    return { state: "live", pid: record.server_pid };
  }
  return { state: "stale" };
}

export interface HeldLock {
  /** Record the spawned server's pid (its process group id). */
  setServer(pid: number): void;
  /** Remove the lock. */
  release(): void;
}

/**
 * Take the lock in `dir` for this process, replacing a stale one. Returns
 * the lock state that blocked it (`live` or `unreadable`) instead of throwing,
 * so the caller can raise its own error.
 */
export function acquireServerLock(dir: string): HeldLock | Exclude<LockState, { state: "none" | "stale" }> {
  const path = join(dir, SERVER_LOCK);
  const write = (record: LockRecord, flag: "wx" | "w"): void =>
    writeFileSync(path, `${JSON.stringify(record)}\n`, { flag });
  const record: LockRecord = { launcher_pid: process.pid, server_pid: null };
  const current = serverLockState(dir);
  if (current.state === "live" || current.state === "unreadable") return current;
  if (current.state === "stale") rmSync(path, { force: true });
  // `wx`: a launcher that took the lock between the check and here makes
  // this throw EEXIST instead of being overwritten.
  write(record, "wx");
  return {
    setServer(pid) {
      record.server_pid = pid;
      write(record, "w");
    },
    release() {
      rmSync(path, { force: true });
    },
  };
}

function parseLock(text: string): LockRecord | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) return null;
  const { launcher_pid, server_pid } = value as Record<string, unknown>;
  if (!isPid(launcher_pid) || !(server_pid === null || isPid(server_pid))) return null;
  return { launcher_pid, server_pid };
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
