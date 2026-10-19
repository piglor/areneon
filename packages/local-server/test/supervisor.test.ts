import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { afterEach, describe, expect, it } from "vitest";
import { acquireCopyLock, type HeldLock } from "../src/copy-lock.ts";
import { supervise, type SupervisorConfig, type SupervisorIO } from "../src/index.ts";
import { tempDir } from "./helpers.ts";

const leftovers: number[] = [];
afterEach(() => {
  for (const pid of leftovers.splice(0)) {
    try {
      process.kill(-pid, "SIGKILL");
    } catch {
      // Gone.
    }
  }
});

function groupAlive(pid: number): boolean {
  try {
    process.kill(-pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** A directory with `server.js`, and a lock taken for a copy next to it. */
function setup(server: string, overrides: Partial<SupervisorConfig> = {}): { config: SupervisorConfig; lock: HeldLock } {
  const dir = tempDir("supervised");
  writeFileSync(join(dir, "server.js"), server);
  const lock = acquireCopyLock(join(dir, "copy"));
  if (!("release" in lock)) throw new Error("lock not taken");
  const config: SupervisorConfig = {
    argv: ["server.js"],
    cwd: dir,
    env: {},
    lifetimeMs: 60_000,
    stopTimeoutMs: 2_000,
    lockPath: lock.path,
    lockToken: lock.token,
    manifestPath: join(dir, "session.json"),
    ...overrides,
  };
  return { config, lock };
}

interface Harness {
  io: SupervisorIO;
  input: PassThrough;
  events(): Record<string, unknown>[];
  log(): string;
  signal(name: NodeJS.Signals): void;
  spawned(): Promise<number>;
}

function harness(kill?: SupervisorIO["kill"]): Harness {
  const input = new PassThrough();
  const events = new PassThrough();
  const log = new PassThrough();
  let eventText = "";
  let logText = "";
  events.on("data", (chunk: Buffer) => (eventText += chunk.toString()));
  log.on("data", (chunk: Buffer) => (logText += chunk.toString()));
  let onSignal: (name: NodeJS.Signals) => void = () => {};
  const parsed = (): Record<string, unknown>[] =>
    eventText
      .split("\n")
      .filter((line) => line !== "")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
  return {
    io: {
      input,
      events,
      log,
      onSignal: (handler) => (onSignal = handler),
      ...(kill === undefined ? {} : { kill }),
    },
    input,
    events: parsed,
    log: () => logText,
    signal: (name) => onSignal(name),
    async spawned() {
      await expect.poll(() => parsed().length, { timeout: 5_000 }).toBeGreaterThan(0);
      const pid = parsed()[0]?.["pid"] as number;
      leftovers.push(pid);
      return pid;
    },
  };
}

const IDLE = "console.log('up'); setInterval(() => {}, 1000);\n";

describe("supervise", () => {
  it.each([
    ["a stop line", "stop", (h: Harness) => h.input.write("noise\nst") && h.input.write("op\n")],
    ["stdin EOF", "eof", (h: Harness) => h.input.end()],
    ["a stdin error", "eof", (h: Harness) => h.input.destroy(new Error("pipe broke"))],
    ["SIGHUP", "signal:SIGHUP", (h: Harness) => h.signal("SIGHUP")],
  ])("on %s stops the server group, marks the manifest and releases the lock", async (_, reason, trigger) => {
    const { config, lock } = setup(IDLE);
    const h = harness();
    const result = supervise(config, h.io);
    const pid = await h.spawned();
    writeFileSync(config.manifestPath, JSON.stringify({ pid, port: 1 }));
    expect(groupAlive(pid)).toBe(true);
    trigger(h);
    expect(await result).toBe(0);
    expect(groupAlive(pid)).toBe(false);
    expect(existsSync(lock.path)).toBe(false);
    expect(JSON.parse(readFileSync(config.manifestPath, "utf8"))).toEqual({
      pid,
      port: 1,
      ended_at: expect.any(String) as string,
      end_reason: reason,
    });
    const [spawned, ended] = h.events();
    expect(spawned).toEqual({
      event: "spawned",
      pid,
      started_at: expect.any(String) as string,
      expires_at: expect.any(String) as string,
    });
    expect(Date.parse(spawned?.["expires_at"] as string) - Date.parse(spawned?.["started_at"] as string)).toBe(60_000);
    expect(ended).toEqual({ event: "ended", reason, pid, gone: true });
    await expect.poll(h.log).toContain("up");
  });

  it("stops when the lifetime ends, and leaves another server's manifest alone", async () => {
    const { config } = setup(IDLE, { lifetimeMs: 300 });
    writeFileSync(config.manifestPath, JSON.stringify({ pid: 12345 }));
    const h = harness();
    expect(await supervise(config, h.io)).toBe(0);
    expect(h.events()[1]).toMatchObject({ reason: "lifetime" });
    expect(JSON.parse(readFileSync(config.manifestPath, "utf8"))).toEqual({ pid: 12345 });
  });

  it("ends when the server exits by itself, with no manifest written yet", async () => {
    const { config, lock } = setup("console.error('bye'); process.exit(4);\n");
    const h = harness();
    expect(await supervise(config, h.io)).toBe(0);
    expect(h.events()[1]).toMatchObject({ reason: "server-exited", gone: true });
    expect(h.log()).toContain("bye");
    expect(existsSync(config.manifestPath)).toBe(false);
    expect(existsSync(lock.path)).toBe(false);
  });

  it("sends SIGKILL to a server that ignores SIGTERM", async () => {
    const { config } = setup(`process.on('SIGTERM', () => {}); ${IDLE}`, { stopTimeoutMs: 300 });
    const h = harness();
    const result = supervise(config, h.io);
    const pid = await h.spawned();
    await expect.poll(h.log).toContain("up");
    h.input.end();
    expect(await result).toBe(0);
    expect(groupAlive(pid)).toBe(false);
  });

  it("exits 2 and keeps the lock when the group outlives SIGKILL", async () => {
    const { config, lock } = setup(IDLE, { stopTimeoutMs: 100 });
    const h = harness(() => {
      // Signals are lost: the group survives.
    });
    const result = supervise(config, h.io);
    const pid = await h.spawned();
    h.input.end();
    expect(await result).toBe(2);
    expect(groupAlive(pid)).toBe(true);
    expect(existsSync(lock.path)).toBe(true);
    expect(h.log()).toContain(`server process group ${pid} still exists after SIGKILL`);
    expect(h.events()[1]).toMatchObject({ reason: "eof", gone: false });
  });

  it("exits 1 and releases the lock when the server cannot start", async () => {
    const { config, lock } = setup(IDLE);
    const h = harness();
    expect(await supervise({ ...config, cwd: join(config.cwd, "missing") }, h.io)).toBe(1);
    expect(h.log()).toContain("cannot start the server");
    expect(h.events()).toEqual([{ event: "ended", reason: "spawn-error" }]);
    expect(existsSync(lock.path)).toBe(false);
  });

  it("keeps going when its output pipes break", async () => {
    const { config } = setup(IDLE);
    const h = harness();
    const result = supervise(config, h.io);
    await h.spawned();
    h.io.events.emit("error", new Error("EPIPE"));
    h.io.log.emit("error", new Error("EPIPE"));
    h.input.write("stop\n");
    expect(await result).toBe(0);
  });
});
