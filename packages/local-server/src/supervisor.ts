import { spawn } from "node:child_process";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import type { Readable, Writable } from "node:stream";
import { exists, releaseCopyLock } from "./copy-lock.ts";

/**
 * The supervisor sits between a launcher and its Showdown server. The
 * launcher holds the supervisor's stdin open; the supervisor starts the
 * server in its own process group and takes that whole group down, waits
 * until none of it is left, marks the session manifest ended, releases the
 * copy's lock and exits, on the first of:
 *
 * - `stop`: the launcher sent a `stop` line;
 * - `eof`: stdin closed, because the launcher stopped or died (SIGKILL too);
 * - `lifetime`: the session's lifetime ran out;
 * - `signal:<name>`: the supervisor got SIGINT, SIGTERM or SIGHUP (Ctrl-C
 *   reaches it, as it shares the launcher's process group);
 * - `server-exited`: the server exited by itself.
 *
 * On stdout it writes one JSON line per event: `spawned` (with the server's
 * pid and the session's start and expiry times) and `ended` (with the
 * reason). The server's own output goes to stderr.
 *
 * Nothing cleans up after a supervisor killed with SIGKILL: the server keeps
 * running until its process group is killed by hand, and the copy's lock
 * stays live while any of that group exists. If the supervisor dies after
 * spawning the server but before the launcher records the server's pid in
 * the lock, the lock does not name the server and goes stale once the
 * launcher exits.
 */
export interface SupervisorConfig {
  /** The server's command line after `node`. */
  argv: string[];
  cwd: string;
  env: Record<string, string>;
  lifetimeMs: number;
  /** How long to wait after SIGTERM, and again after SIGKILL. */
  stopTimeoutMs: number;
  lockPath: string;
  lockToken: string;
  manifestPath: string;
}

export interface SupervisorIO {
  input: Readable;
  events: Writable;
  log: Writable;
  onSignal(handler: (signal: NodeJS.Signals) => void): void;
  /** `process.kill`, replaceable in tests. */
  kill?: (pid: number, signal: NodeJS.Signals) => void;
}

/** Exit codes: 0 the group is gone, 1 the server could not start, 2 the group outlived SIGKILL. */
export async function supervise(config: SupervisorConfig, io: SupervisorIO): Promise<number> {
  const kill = io.kill ?? ((pid: number, signal: NodeJS.Signals) => process.kill(pid, signal));
  const ignore = (): void => {};
  // The launcher may be gone: a write to its closed pipe must not stop the cleanup.
  io.events.on("error", ignore);
  io.log.on("error", ignore);
  const event = (fields: Record<string, unknown>): void => {
    io.events.write(`${JSON.stringify(fields)}\n`);
  };

  // Listen before the server exists, so an early stop or signal is not lost.
  const requested = new Promise<string>((done) => {
    let buffered = "";
    io.input.on("data", (chunk: Buffer | string) => {
      buffered += chunk.toString();
      if (buffered.split("\n").includes("stop")) done("stop");
    });
    io.input.once("end", () => done("eof"));
    io.input.once("close", () => done("eof"));
    io.input.once("error", () => done("eof"));
    io.onSignal((signal) => done(`signal:${signal}`));
  });

  const server = spawn(process.execPath, config.argv, {
    cwd: config.cwd,
    detached: true,
    env: config.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const spawned = await new Promise<Error | null>((done) => {
    server.once("spawn", () => done(null));
    server.once("error", done);
  });
  if (spawned !== null) {
    io.log.write(`cannot start the server: ${String(spawned)}\n`);
    releaseCopyLock(config.lockPath, config.lockToken);
    event({ event: "ended", reason: "spawn-error" });
    return 1;
  }
  const pid = server.pid as number;
  const forward = (chunk: Buffer): void => {
    io.log.write(chunk);
  };
  server.stdout.on("data", forward);
  server.stderr.on("data", forward);
  const startedAt = new Date();
  event({
    event: "spawned",
    pid,
    started_at: startedAt.toISOString(),
    expires_at: new Date(startedAt.getTime() + config.lifetimeMs).toISOString(),
  });

  let lifetime: NodeJS.Timeout | undefined;
  const reason = await Promise.race([
    requested,
    new Promise<string>((done) => {
      server.once("exit", () => done("server-exited"));
      lifetime = setTimeout(() => done("lifetime"), config.lifetimeMs);
    }),
  ]);
  clearTimeout(lifetime);

  const gone = (): boolean => !exists(-pid);
  const waitGone = async (ms: number): Promise<boolean> => {
    const until = Date.now() + ms;
    while (!gone()) {
      if (Date.now() >= until) return false;
      await new Promise((done) => setTimeout(done, 25));
    }
    return true;
  };
  const signal = (name: NodeJS.Signals): void => {
    try {
      kill(-pid, name);
    } catch {
      // Already gone.
    }
  };
  signal("SIGTERM");
  let code = 0;
  if (!(await waitGone(config.stopTimeoutMs))) {
    signal("SIGKILL");
    if (!(await waitGone(config.stopTimeoutMs))) {
      io.log.write(`server process group ${pid} still exists after SIGKILL\n`);
      code = 2;
    }
  }
  server.stdout.off("data", forward);
  server.stderr.off("data", forward);
  if (code === 0) {
    markEnded(config.manifestPath, pid, reason);
    releaseCopyLock(config.lockPath, config.lockToken);
  }
  event({ event: "ended", reason, pid, gone: code === 0 });
  return code;
}

/** Add `ended_at` and `end_reason` to this server's session manifest, if it has one. */
function markEnded(path: string, pid: number, reason: string): void {
  let manifest: Record<string, unknown>;
  try {
    manifest = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  } catch {
    return;
  }
  if (manifest["pid"] !== pid) return;
  const ended = { ...manifest, ended_at: new Date().toISOString(), end_reason: reason };
  writeFileSync(`${path}.tmp`, `${JSON.stringify(ended, null, 2)}\n`);
  renameSync(`${path}.tmp`, path);
}
