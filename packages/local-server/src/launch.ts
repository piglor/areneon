import { type ChildProcessByStdio, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { renameSync, writeFileSync } from "node:fs";
import { get } from "node:http";
import { createServer } from "node:net";
import { join, resolve } from "node:path";
import type { Readable } from "node:stream";
import { PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import { checkServerConfig, loadServerConfig, LOOPBACK_ADDRESS, writeServerConfig } from "./config.ts";
import { verifyRuntimeCopy } from "./runtime-copy.ts";
import { acquireServerLock, exists } from "./server-lock.ts";

/** The session manifest, inside the runtime copy (ADR-005 §4). */
export const SESSION_MANIFEST = "areneon-session.json";

/** ADR-005: the human's client connects to `localhost:8000`. */
export const DEFAULT_PORT = 8000;

/** A server outlives no sparring session: it is stopped after this long. */
export const DEFAULT_LIFETIME_MS = 2 * 60 * 60 * 1000;

export interface LaunchOptions {
  /** The runtime copy to run from (`createRuntimeCopy().dir`). */
  dir: string;
  port?: number;
  expectedCommit?: string;
  /** How long the server may take to serve `GET /`. */
  readyTimeoutMs?: number;
  /** When `stop()` runs by itself. */
  lifetimeMs?: number;
  /** How long `stop()` waits after SIGTERM before it sends SIGKILL. */
  stopTimeoutMs?: number;
}

export interface SessionManifest {
  commit: string;
  /** `treeDigest` of the copy's `dist/`, verified before the start. */
  dist_sha256: string;
  /** SHA-256 of the `config/config.js` the server loaded. */
  config_sha256: string;
  bind_address: string;
  port: number;
  url: string;
  /** The server's command line after `node`, relative to the copy. */
  argv: string[];
  /** The server's pid, which is also its process group id. */
  pid: number;
  started_at: string;
  /** When the launcher stops the server by itself. */
  expires_at: string;
  /** ADR-005 §4: the random per-session suffix of the bot's name. */
  bot_name_suffix: string;
}

export interface LocalServer {
  url: string;
  session: SessionManifest;
  /** Stop the whole server process group and wait until none of it is left. */
  stop(): Promise<void>;
}

export class LaunchError extends Error {
  override name = "LaunchError";
}

type ServerProcess = ChildProcessByStdio<null, Readable, Readable>;

const running = new Set<number>();
let exitHookInstalled = false;

/**
 * Start the pinned Showdown server from a verified runtime copy, bound to
 * 127.0.0.1 only (ADR-005 §4): verify the copy's commit and `dist/` digest,
 * take the copy's server lock, write and check `config/config.js`, start
 * `pokemon-showdown start --skip-build --no-security <port>` from the copy
 * in its own process group, wait until it listens on the loopback address
 * and serves `GET /`, then write the session manifest.
 */
export async function launch(options: LaunchOptions): Promise<LocalServer> {
  const dir = resolve(options.dir);
  const port = options.port ?? DEFAULT_PORT;
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new LaunchError(`port ${port} is not a TCP port`);
  }
  const copy = verifyRuntimeCopy(dir, options.expectedCommit ?? PINNED_SHOWDOWN_COMMIT);
  const lock = acquireServerLock(dir);
  if ("state" in lock) {
    throw new LaunchError(
      lock.state === "live"
        ? `a server (pid ${lock.pid}) is already running from ${dir}`
        : `cannot read the server lock in ${dir} (${lock.reason}); remove it if no server runs there`,
    );
  }
  let child: ServerProcess | undefined;
  try {
    const config = writeServerConfig(dir, port);
    checkServerConfig(loadServerConfig(dir), port);
    await assertPortFree(port);
    const argv = ["pokemon-showdown", "start", "--skip-build", "--no-security", String(port)];
    child = spawn(process.execPath, argv, {
      cwd: dir,
      detached: true,
      env: serverEnv(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    const pid = child.pid as number;
    running.add(pid);
    installExitHook();
    lock.setServer(pid);
    const url = `http://${LOOPBACK_ADDRESS}:${port}`;
    await whenServing(child, port, url, options.readyTimeoutMs ?? 60_000);
    const startedAt = new Date();
    const lifetimeMs = options.lifetimeMs ?? DEFAULT_LIFETIME_MS;
    const session: SessionManifest = {
      commit: copy.commit,
      dist_sha256: copy.distSha256,
      config_sha256: config.sha256,
      bind_address: LOOPBACK_ADDRESS,
      port,
      url,
      argv,
      pid,
      started_at: startedAt.toISOString(),
      expires_at: new Date(startedAt.getTime() + lifetimeMs).toISOString(),
      bot_name_suffix: randomBytes(3).toString("hex"),
    };
    const manifest = join(dir, SESSION_MANIFEST);
    writeFileSync(`${manifest}.tmp`, `${JSON.stringify(session, null, 2)}\n`);
    renameSync(`${manifest}.tmp`, manifest);

    const server = child;
    let stopping: Promise<void> | undefined;
    const stop = (): Promise<void> => {
      stopping ??= (async () => {
        clearTimeout(timer);
        await killGroup(server, options.stopTimeoutMs ?? 5_000);
        lock.release();
      })();
      return stopping;
    };
    const timer = setTimeout(() => void stop(), lifetimeMs);
    timer.unref();
    return { url, session, stop };
  } catch (error) {
    if (child !== undefined) await killGroup(child, options.stopTimeoutMs ?? 5_000);
    lock.release();
    throw error;
  }
}

/**
 * A minimal environment: no PSPORT / PSBINDADDR (which the server's workers
 * read), no cloud PORT / IP, no NODE_OPTIONS.
 */
function serverEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of ["PATH", "HOME", "LANG", "TMPDIR"]) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

/** Refuse a port another process holds, so `GET /` cannot reach a stranger. */
function assertPortFree(port: number): Promise<void> {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.once("error", (error) =>
      fail(new LaunchError(`port ${port} on ${LOOPBACK_ADDRESS} is not free: ${String(error)}`)),
    );
    probe.listen(port, LOOPBACK_ADDRESS, () => probe.close(() => done()));
  });
}

/**
 * Resolve once the server has said it listens on the loopback address and
 * `GET /` answers 200. Reject if it exits first or the deadline passes.
 */
function whenServing(child: ServerProcess, port: number, url: string, timeoutMs: number): Promise<void> {
  const banner = `now listening on ${LOOPBACK_ADDRESS}:${port}`;
  let output = "";
  let listening = false;
  const read = (chunk: Buffer): void => {
    output = (output + chunk.toString("utf8")).slice(-4000);
    listening ||= output.includes(banner);
  };
  child.stdout.on("data", read);
  child.stderr.on("data", read);
  return new Promise((done, fail) => {
    let finished = false;
    let poller: NodeJS.Timeout | undefined;
    const finish = (error?: Error): void => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      clearTimeout(poller);
      child.off("exit", onExit);
      if (error === undefined) done();
      else fail(error);
    };
    const onExit = (code: number | null, signal: string | null): void =>
      finish(new LaunchError(`server exited (${code ?? signal}) before it served ${url}:\n${output.trim()}`));
    child.once("exit", onExit);
    const deadline = setTimeout(
      () => finish(new LaunchError(`server did not serve ${url} within ${timeoutMs} ms:\n${output.trim()}`)),
      timeoutMs,
    );
    const again = (): void => {
      if (!finished) poller = setTimeout(poll, 50);
    };
    const poll = (): void => {
      if (!listening) return again();
      const request = get(`${url}/`, (response) => {
        response.resume();
        if (response.statusCode === 200) finish();
        else again();
      });
      request.setTimeout(1_000, () => request.destroy());
      request.on("error", again);
    };
    poll();
  });
}

/** SIGTERM the server's process group, then SIGKILL it, until none of it is left. */
async function killGroup(child: ServerProcess, timeoutMs: number): Promise<void> {
  const pid = child.pid as number;
  const gone = (): boolean => !exists(-pid);
  const signal = (name: NodeJS.Signals): void => {
    try {
      process.kill(-pid, name);
    } catch {
      // Already gone.
    }
  };
  const waitGone = async (ms: number): Promise<boolean> => {
    const until = Date.now() + ms;
    while (!gone()) {
      if (Date.now() >= until) return false;
      await new Promise((done) => setTimeout(done, 25));
    }
    return true;
  };
  signal("SIGTERM");
  if (!(await waitGone(timeoutMs))) {
    signal("SIGKILL");
    // SIGKILL cannot be caught: wait for it without a deadline.
    await waitGone(Infinity);
  }
  running.delete(pid);
  child.stdout.destroy();
  child.stderr.destroy();
}

/** If the launcher exits with servers still running, take them down with it. */
function installExitHook(): void {
  if (exitHookInstalled) return;
  exitHookInstalled = true;
  process.on("exit", () => {
    for (const pid of running) {
      try {
        process.kill(-pid, "SIGKILL");
      } catch {
        // Already gone.
      }
    }
  });
}
