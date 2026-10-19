import { type ChildProcessByStdio, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { renameSync, rmSync, writeFileSync } from "node:fs";
import { get } from "node:http";
import { createServer } from "node:net";
import { join, resolve } from "node:path";
import type { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";
import { PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import { checkServerConfig, loadServerConfig, LOOPBACK_ADDRESS, writeServerConfig } from "./config.ts";
import { acquireCopyLock, blockedMessage } from "./copy-lock.ts";
import { type DigestedEntry, verifyRuntimeCopy } from "./runtime-copy.ts";
import type { SupervisorConfig } from "./supervisor.ts";

/** The session manifest, inside the runtime copy (ADR-005 §4). */
export const SESSION_MANIFEST = "areneon-session.json";

/** ADR-005: the human's client connects to `localhost:8000`. */
export const DEFAULT_PORT = 8000;

/** A server outlives no sparring session: it is stopped after this long. */
export const DEFAULT_LIFETIME_MS = 2 * 60 * 60 * 1000;

/** The largest timer delay Node accepts. */
const MAX_DELAY_MS = 2 ** 31 - 1;

const SUPERVISOR = fileURLToPath(new URL("../scripts/supervisor.ts", import.meta.url));

export interface LaunchOptions {
  /** The runtime copy to run from (`createRuntimeCopy().dir`). */
  dir: string;
  port?: number;
  expectedCommit?: string;
  /** How long the server may take to serve `GET /` (default 60 s). */
  readyTimeoutMs?: number;
  /** When the supervisor stops the server by itself (default 2 h). */
  lifetimeMs?: number;
  /** How long the supervisor waits after SIGTERM, and again after SIGKILL (default 5 s). */
  stopTimeoutMs?: number;
}

export interface SessionManifest {
  commit: string;
  /** The copy's entry digests (`RuntimeCopy.sha256`), verified before the start. */
  runtime_sha256: Record<DigestedEntry, string>;
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
  /** When the supervisor stops the server by itself. */
  expires_at: string;
  /** ADR-005 §4: the random per-session suffix of the bot's name. */
  bot_name_suffix: string;
  /** Added by the supervisor once the server's process group is gone. */
  ended_at?: string;
  end_reason?: string;
}

export interface LocalServer {
  url: string;
  session: SessionManifest;
  /**
   * Ask the supervisor to stop the server's whole process group, and wait
   * until it has and the supervisor has exited. Throws a `LaunchError` if
   * the group outlived SIGKILL or the supervisor did not exit.
   */
  stop(): Promise<void>;
  /** Settles when the supervisor has exited, for whatever reason. */
  ended: Promise<void>;
}

export class LaunchError extends Error {
  override name = "LaunchError";
}

/**
 * Start the pinned Showdown server from a runtime copy, bound to 127.0.0.1
 * only (ADR-005 §4).
 *
 * It takes the copy's lock (`lockPath`), then verifies the copy's commit and
 * entry digests (`verifyRuntimeCopy`): a copy that fails is refused with a `RuntimeCopyError`,
 * and the caller recreates it with `createRuntimeCopy`. It writes and checks
 * `config/config.js`, refuses a port already held on 127.0.0.1 or ::1, and
 * starts a supervisor (`scripts/supervisor.ts`, see `supervise`) that runs
 * `pokemon-showdown start --skip-build --no-security <port>` from the copy in
 * its own process group. Once the server listens on the loopback address and
 * serves `GET /`, it writes the session manifest. The supervisor owns the
 * server from then on: it stops it on `stop()`, at the end of its lifetime,
 * on SIGINT/SIGTERM/SIGHUP, and when this process dies (its stdin closes),
 * and then releases the lock. A supervisor killed with SIGKILL leaves the
 * server running until its process group is killed by hand; the lock stays
 * live while that group exists, unless the supervisor died before this
 * function recorded the server's pid in it, in which case the lock goes
 * stale once this process exits.
 */
export async function launch(options: LaunchOptions): Promise<LocalServer> {
  const dir = resolve(options.dir);
  const port = options.port ?? DEFAULT_PORT;
  checkRange("port", port, 65535);
  const readyTimeoutMs = options.readyTimeoutMs ?? 60_000;
  const lifetimeMs = options.lifetimeMs ?? DEFAULT_LIFETIME_MS;
  const stopTimeoutMs = options.stopTimeoutMs ?? 5_000;
  checkRange("readyTimeoutMs", readyTimeoutMs, MAX_DELAY_MS);
  checkRange("lifetimeMs", lifetimeMs, MAX_DELAY_MS);
  checkRange("stopTimeoutMs", stopTimeoutMs, MAX_DELAY_MS);

  const lock = acquireCopyLock(dir);
  if (!("release" in lock)) throw new LaunchError(blockedMessage(dir, lock));
  let supervisor: Supervisor | undefined;
  try {
    const copy = verifyRuntimeCopy(dir, options.expectedCommit ?? PINNED_SHOWDOWN_COMMIT);
    const config = writeServerConfig(dir, port);
    checkServerConfig(loadServerConfig(dir), port);
    await assertPortFree(port);
    const manifestPath = join(dir, SESSION_MANIFEST);
    rmSync(manifestPath, { force: true });
    const argv = ["pokemon-showdown", "start", "--skip-build", "--no-security", String(port)];
    const url = `http://${LOOPBACK_ADDRESS}:${port}`;
    supervisor = startSupervisor(
      {
        argv,
        cwd: dir,
        env: serverEnv(),
        lifetimeMs,
        stopTimeoutMs,
        lockPath: lock.path,
        lockToken: lock.token,
        manifestPath,
      },
      `now listening on ${LOOPBACK_ADDRESS}:${port}`,
    );
    const spawned = await supervisor.spawned;
    lock.update({ supervisor_pid: supervisor.pid as number, server_pid: spawned.pid });
    await whenServing(supervisor, url, readyTimeoutMs);
    const session: SessionManifest = {
      commit: copy.commit,
      runtime_sha256: copy.sha256,
      config_sha256: config.sha256,
      bind_address: LOOPBACK_ADDRESS,
      port,
      url,
      argv,
      pid: spawned.pid,
      started_at: spawned.started_at,
      expires_at: spawned.expires_at,
      bot_name_suffix: randomBytes(3).toString("hex"),
    };
    writeFileSync(`${manifestPath}.tmp`, `${JSON.stringify(session, null, 2)}\n`);
    renameSync(`${manifestPath}.tmp`, manifestPath);
    const { stop, exited } = supervisor;
    return { url, session, stop, ended: exited.then(() => undefined) };
  } catch (error) {
    try {
      await supervisor?.stop();
      lock.release();
    } catch {
      // The server may still run: keep the lock live, report the first error.
    }
    throw error;
  }
}

function checkRange(name: string, value: number, max: number): void {
  if (!Number.isInteger(value) || value < 1 || value > max) {
    throw new LaunchError(`${name} must be an integer from 1 to ${max}, not ${value}`);
  }
}

/**
 * A minimal environment: no PSPORT / PSBINDADDR (which the server's workers
 * read), no cloud PORT / IP, no NODE_OPTIONS.
 */
function serverEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const key of ["PATH", "HOME", "LANG", "TMPDIR"]) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

/**
 * Refuse a port another process holds on 127.0.0.1, or on ::1 where IPv6
 * exists, so `GET /` cannot reach a stranger, nor can a client that
 * resolves `localhost` to ::1. Clients should still use `session.url`
 * (127.0.0.1): the server never listens on ::1.
 */
async function assertPortFree(port: number): Promise<void> {
  await probe(port, LOOPBACK_ADDRESS, false);
  await probe(port, "::1", true);
}

function probe(port: number, host: string, optional: boolean): Promise<void> {
  return new Promise((done, fail) => {
    const server = createServer();
    server.once("error", (error: NodeJS.ErrnoException) => {
      if (optional && error.code !== "EADDRINUSE") done();
      else fail(new LaunchError(`port ${port} on ${host} is not free: ${String(error)}`));
    });
    server.listen({ port, host }, () => server.close(() => done()));
  });
}

interface Spawned {
  pid: number;
  started_at: string;
  expires_at: string;
}

interface Exit {
  code: number | null;
  signal: string | null;
  error?: Error;
}

function exitText({ code, signal }: Exit): string {
  return String(code ?? signal);
}

interface Supervisor {
  pid: number | undefined;
  spawned: Promise<Spawned>;
  exited: Promise<Exit>;
  /** Whether the server has printed its listening line. */
  listening(): boolean;
  output(): string;
  stop(): Promise<void>;
}

type SupervisorProcess = ChildProcessByStdio<Writable, Readable, Readable>;

function startSupervisor(config: SupervisorConfig, banner: string): Supervisor {
  const child: SupervisorProcess = spawn(process.execPath, [SUPERVISOR, JSON.stringify(config)], {
    cwd: config.cwd,
    env: config.env,
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdin.on("error", () => {
    // The supervisor is gone; `exited` reports it.
  });
  let output = "";
  let listening = false;
  child.stderr.on("data", (chunk: Buffer) => {
    output = (output + chunk.toString("utf8")).slice(-4000);
    listening ||= output.includes(banner);
  });
  const exited = new Promise<Exit>((done) => {
    child.once("exit", (code, signal) => done({ code, signal }));
    child.once("error", (error) => done({ code: null, signal: null, error }));
  });
  const tail = (): string => output.trim();
  const spawned = new Promise<Spawned>((done, fail) => {
    let lines = "";
    child.stdout.on("data", (chunk: Buffer) => {
      lines += chunk.toString("utf8");
      const complete = lines.split("\n");
      lines = complete.pop() as string;
      for (const line of complete) {
        const event = JSON.parse(line) as { event: string } & Spawned;
        if (event.event === "spawned") done(event);
      }
    });
    void exited.then((exit) =>
      fail(
        new LaunchError(
          exit.error === undefined
            ? `supervisor exited (${exitText(exit)}) before the server started:\n${tail()}`
            : `cannot start the supervisor: ${String(exit.error)}`,
        ),
      ),
    );
  });

  const timeoutMs = 2 * config.stopTimeoutMs + 5_000;
  let stopping: Promise<void> | undefined;
  const stop = (): Promise<void> => {
    stopping ??= (async () => {
      child.stdin.end("stop\n");
      let timer: NodeJS.Timeout | undefined;
      const result = await Promise.race([
        exited,
        new Promise<"timeout">((done) => {
          timer = setTimeout(() => done("timeout"), timeoutMs);
        }),
      ]);
      clearTimeout(timer);
      if (result === "timeout") {
        child.kill("SIGKILL");
        throw new LaunchError(`supervisor ${child.pid} did not exit within ${timeoutMs} ms; the server may still run`);
      }
      if (result.error === undefined && result.code !== 0) {
        throw new LaunchError(
          `supervisor exited (${exitText(result)}); the server may still run:\n${tail()}`,
        );
      }
    })();
    return stopping;
  };
  return { pid: child.pid, spawned, exited, listening: () => listening, output: tail, stop };
}

/**
 * Resolve once the server has said it listens on the loopback address and
 * `GET /` answers 200. Reject if the supervisor exits first or the deadline
 * passes.
 */
function whenServing(supervisor: Supervisor, url: string, timeoutMs: number): Promise<void> {
  return new Promise((done, fail) => {
    let finished = false;
    let poller: NodeJS.Timeout | undefined;
    const finish = (error?: Error): void => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      clearTimeout(poller);
      if (error === undefined) done();
      else fail(error);
    };
    void supervisor.exited.then((exit) =>
      finish(
        new LaunchError(
          `supervisor exited (${exitText(exit)}) before the server served ${url}:\n${supervisor.output()}`,
        ),
      ),
    );
    const deadline = setTimeout(
      () => finish(new LaunchError(`server did not serve ${url} within ${timeoutMs} ms:\n${supervisor.output()}`)),
      timeoutMs,
    );
    const again = (): void => {
      if (!finished) poller = setTimeout(poll, 50);
    };
    const poll = (): void => {
      if (!supervisor.listening()) return again();
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
