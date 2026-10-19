import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { get } from "node:http";
import { createConnection, createServer, type Server } from "node:net";
import { networkInterfaces } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR, PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import {
  createRuntimeCopy,
  launch,
  LaunchError,
  type LaunchOptions,
  type LocalServer,
  RuntimeCopyError,
  lockPath,
  SERVER_CONFIG_PATH,
  SESSION_MANIFEST,
} from "../src/index.ts";
import { fakeCopy, freePort, tempDir } from "./helpers.ts";

/** A pid no process has (above the default Linux pid_max). */
const DEAD = 2 ** 22 + 7;

const servers: LocalServer[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.stop()));
});

async function start(options: LaunchOptions): Promise<LocalServer> {
  const server = await launch(options);
  servers.push(server);
  return server;
}

/** Whether any process of the group led by `pid` still exists. */
function groupAlive(pid: number): boolean {
  try {
    process.kill(-pid, 0);
    return true;
  } catch {
    return false;
  }
}

function fetch(url: string): Promise<{ status: number; body: string }> {
  return new Promise((done, fail) => {
    get(url, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk: string) => (body += chunk));
      response.on("end", () => done({ status: response.statusCode ?? 0, body }));
    }).on("error", fail);
  });
}

/**
 * A stand-in server: records its argv, cwd, env and pid next to itself, says
 * it listens before it does, answers 503 once and then 200, and (with
 * `stubborn`) ignores SIGTERM.
 */
function fakeServer({ stubborn = false, exitAfterMs = 0 } = {}): string {
  return `#!/usr/bin/env node
const fs = require('fs');
const http = require('http');
const port = Number(process.argv[process.argv.length - 1]);
fs.writeFileSync('started.json', JSON.stringify({ argv: process.argv.slice(1), cwd: process.cwd(), env: process.env, pid: process.pid }));
${stubborn ? "process.on('SIGTERM', () => {});" : ""}
console.log('Worker 1 now listening on 127.0.0.1:' + port);
let first = true;
setTimeout(() => {
  http.createServer((req, res) => {
    res.writeHead(first ? 503 : 200);
    first = false;
    res.end('fake');
  }).listen(port, '127.0.0.1');
}, 150);
${exitAfterMs > 0 ? `setTimeout(() => process.exit(0), ${exitAfterMs});` : ""}
`;
}

describe("launch with a stand-in server", () => {
  it("starts the entry script from the copy and records the session", async () => {
    const copy = fakeCopy(fakeServer());
    const port = await freePort();
    process.env["PSBINDADDR"] = "0.0.0.0";
    let server: LocalServer;
    try {
      server = await start({ dir: copy.dir, port, expectedCommit: copy.commit, lifetimeMs: 60_000 });
    } finally {
      delete process.env["PSBINDADDR"];
    }
    expect(server.url).toBe(`http://127.0.0.1:${port}`);
    const started = JSON.parse(readFileSync(join(copy.dir, "started.json"), "utf8")) as {
      argv: string[];
      cwd: string;
      env: Record<string, string>;
      pid: number;
    };
    expect(started.cwd).toBe(copy.dir);
    expect(started.argv).toEqual([
      join(copy.dir, "pokemon-showdown"),
      "start",
      "--skip-build",
      "--no-security",
      String(port),
    ]);
    expect(Object.keys(started.env).filter((key) => key.startsWith("PS"))).toEqual([]);

    const config = readFileSync(join(copy.dir, SERVER_CONFIG_PATH));
    const session = server.session;
    expect(session).toEqual({
      commit: copy.commit,
      runtime_sha256: copy.sha256,
      config_sha256: createHash("sha256").update(config).digest("hex"),
      bind_address: "127.0.0.1",
      port,
      url: server.url,
      argv: ["pokemon-showdown", "start", "--skip-build", "--no-security", String(port)],
      pid: started.pid,
      started_at: expect.any(String) as string,
      expires_at: expect.any(String) as string,
      bot_name_suffix: expect.stringMatching(/^[0-9a-f]{6}$/) as string,
    });
    expect(Date.parse(session.expires_at) - Date.parse(session.started_at)).toBe(60_000);
    expect(JSON.parse(readFileSync(join(copy.dir, SESSION_MANIFEST), "utf8"))).toEqual(session);
    expect(JSON.parse(readFileSync(lockPath(copy.dir), "utf8"))).toEqual({
      token: expect.stringMatching(/^[0-9a-f]{32}$/) as string,
      pid: process.pid,
      supervisor_pid: expect.any(Number) as number,
      server_pid: started.pid,
    });

    await server.stop();
    expect(groupAlive(started.pid)).toBe(false);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
    const ended = JSON.parse(readFileSync(join(copy.dir, SESSION_MANIFEST), "utf8")) as Record<string, unknown>;
    expect(ended).toEqual({ ...session, ended_at: expect.any(String) as string, end_reason: "stop" });
    await server.stop();
    await server.ended;
  });

  it("refuses a second server on the copy, and a recreated copy, until the first stops", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await expect(
      launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit }),
    ).rejects.toThrow(/is in use by pid/);
    expect(existsSync(lockPath(copy.dir))).toBe(true);
    const recreate = (): unknown =>
      createRuntimeCopy({ vendorDir: copy.vendorDir, runtimeRoot: copy.runtimeRoot, expectedCommit: copy.commit });
    expect(recreate).toThrow(RuntimeCopyError);
    expect(recreate).toThrow(/is in use by pid/);
    await server.stop();
    expect(recreate).not.toThrow();
  });

  it("replaces a stale lock and refuses an unreadable one", async () => {
    const copy = fakeCopy(fakeServer());
    const lock = lockPath(copy.dir);
    writeFileSync(lock, JSON.stringify({ token: "old", pid: DEAD, supervisor_pid: DEAD + 1, server_pid: DEAD + 2 }));
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await server.stop();
    for (const text of [
      "not json",
      "null",
      '{"pid":5,"supervisor_pid":null,"server_pid":null}',
      '{"token":"t","pid":1,"supervisor_pid":null,"server_pid":null}',
      '{"token":"t","pid":5,"supervisor_pid":null,"server_pid":"x"}',
      '{"token":"t","pid":5,"supervisor_pid":"x","server_pid":null}',
    ]) {
      writeFileSync(lock, text);
      await expect(
        launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit }),
      ).rejects.toThrow(/cannot read the lock/);
      expect(readFileSync(lock, "utf8")).toBe(text);
    }
  });

  it("refuses a lock path it cannot read, here a directory, for a launch and a recreated copy", async () => {
    const copy = fakeCopy(fakeServer());
    mkdirSync(lockPath(copy.dir));
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /cannot read the lock/,
    );
    expect(() =>
      createRuntimeCopy({ vendorDir: copy.vendorDir, runtimeRoot: copy.runtimeRoot, expectedCommit: copy.commit }),
    ).toThrow(/cannot read the lock/);
  });

  it("counts a lock live while its supervisor or any of its server group runs", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    const lock = lockPath(copy.dir);
    const held = readFileSync(lock, "utf8");
    const record = JSON.parse(held) as { supervisor_pid: number };
    const attempt = (): Promise<unknown> =>
      launch({ dir: copy.dir, port: 1, expectedCommit: copy.commit });
    writeFileSync(lock, JSON.stringify({ token: "t", pid: DEAD, supervisor_pid: record.supervisor_pid, server_pid: null }));
    await expect(attempt()).rejects.toThrow(`in use by pid ${record.supervisor_pid}`);
    writeFileSync(lock, JSON.stringify({ token: "t", pid: DEAD, supervisor_pid: DEAD, server_pid: server.session.pid }));
    await expect(attempt()).rejects.toThrow(`in use by pid ${server.session.pid}`);
    writeFileSync(lock, held);
  });

  it.each([
    ["SIGTERM to the launcher", "SIGTERM", false],
    ["SIGINT to the launcher", "SIGINT", false],
    ["SIGKILL to the launcher", "SIGKILL", false],
    ["Ctrl-C: SIGINT to the launcher's process group", "SIGINT", true],
  ] as const)("leaves no server and releases the lock after %s", async (_, signal, group) => {
    const copy = fakeCopy(fakeServer());
    const script = join(tempDir("launcher"), "launcher.ts");
    writeFileSync(
      script,
      `import { launch } from ${JSON.stringify(join(import.meta.dirname, "../src/index.ts"))};
const server = await launch(${JSON.stringify({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })});
console.log(server.session.pid);
setInterval(() => {}, 1000);
`,
    );
    const launcher = spawn(process.execPath, [script], { detached: true, stdio: ["ignore", "pipe", "inherit"] });
    try {
      const pid = await new Promise<number>((done, fail) => {
        launcher.stdout.once("data", (chunk: Buffer) => done(Number(chunk.toString().trim())));
        launcher.once("exit", (code) => fail(new Error(`launcher exited ${code}`)));
      });
      expect(groupAlive(pid)).toBe(true);
      process.kill(group ? -(launcher.pid as number) : (launcher.pid as number), signal);
      await expect.poll(() => groupAlive(pid), { timeout: 10_000 }).toBe(false);
      await expect.poll(() => existsSync(lockPath(copy.dir)), { timeout: 10_000 }).toBe(false);
    } finally {
      try {
        process.kill(-(launcher.pid as number), "SIGKILL");
      } catch {
        // Gone.
      }
    }
  });

  it("kills a server that ignores SIGTERM", async () => {
    const copy = fakeCopy(fakeServer({ stubborn: true }));
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, stopTimeoutMs: 200 });
    await server.stop();
    expect(groupAlive(server.session.pid)).toBe(false);
  });

  it("stops the server by itself when its lifetime ends", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, lifetimeMs: 1500 });
    await server.ended;
    expect(groupAlive(server.session.pid)).toBe(false);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
    expect(JSON.parse(readFileSync(join(copy.dir, SESSION_MANIFEST), "utf8"))).toMatchObject({ end_reason: "lifetime" });
    await server.stop();
  });

  it("ends the session when the server exits by itself", async () => {
    const copy = fakeCopy(fakeServer({ exitAfterMs: 1000 }));
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await server.ended;
    expect(existsSync(lockPath(copy.dir))).toBe(false);
    expect(JSON.parse(readFileSync(join(copy.dir, SESSION_MANIFEST), "utf8"))).toMatchObject({
      end_reason: "server-exited",
    });
    await server.stop();
  });

  it.each(["process.exit(3)", "process.kill(process.pid, 'SIGKILL')"])(
    "fails and cleans up when the server exits before it serves: %s",
    async (exit) => {
    const copy = fakeCopy(`#!/usr/bin/env node\nconsole.error('boom');\nsetTimeout(() => ${exit}, 100);\n`);
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /supervisor exited \(0\) before the server served .*boom/s,
    );
    expect(existsSync(lockPath(copy.dir))).toBe(false);
    expect(existsSync(join(copy.dir, SESSION_MANIFEST))).toBe(false);
    },
  );

  it("fails and kills the server when it does not serve in time", async () => {
    const copy = fakeCopy(
      "#!/usr/bin/env node\nrequire('fs').writeFileSync('started.json', String(process.pid));\nsetInterval(() => {}, 1000);\n",
    );
    await expect(
      launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, readyTimeoutMs: 500 }),
    ).rejects.toThrow(/did not serve .* within 500 ms/);
    expect(groupAlive(Number(readFileSync(join(copy.dir, "started.json"), "utf8")))).toBe(false);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
  });

  it("refuses a port another process holds, before starting anything", async () => {
    const copy = fakeCopy(fakeServer());
    const holder: Server = createServer();
    const port = await freePort();
    await new Promise<void>((done) => holder.listen(port, "127.0.0.1", done));
    try {
      await expect(launch({ dir: copy.dir, port, expectedCommit: copy.commit })).rejects.toThrow(
        `port ${port} on 127.0.0.1 is not free`,
      );
    } finally {
      holder.close();
    }
    expect(existsSync(join(copy.dir, "started.json"))).toBe(false);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
  });

  it("refuses a port another process holds on ::1, where IPv6 exists", async () => {
    const copy = fakeCopy(fakeServer());
    const holder: Server = createServer();
    const port = await freePort();
    const listening = await new Promise<boolean>((done) => {
      holder.once("error", () => done(false));
      holder.listen({ port, host: "::1" }, () => done(true));
    });
    if (!listening) return;
    try {
      await expect(launch({ dir: copy.dir, port, expectedCommit: copy.commit })).rejects.toThrow(
        `port ${port} on ::1 is not free`,
      );
    } finally {
      holder.close();
    }
  });

  it.each([
    ["port", 0],
    ["port", 65536],
    ["port", 80.5],
    ["readyTimeoutMs", 0],
    ["lifetimeMs", 0],
    ["lifetimeMs", -1],
    ["lifetimeMs", 1.5],
    ["lifetimeMs", 2 ** 31],
    ["stopTimeoutMs", 0],
    ["stopTimeoutMs", 2 ** 31],
    ["stopTimeoutMs", Number.NaN],
  ])("refuses %s = %s before taking the lock", async (name, value) => {
    const copy = fakeCopy(fakeServer());
    await expect(launch({ dir: copy.dir, expectedCommit: copy.commit, [name]: value })).rejects.toThrow(
      new RegExp(`^${name} must be an integer from 1 to`),
    );
    await expect(launch({ dir: copy.dir, expectedCommit: copy.commit, [name]: value })).rejects.toThrow(LaunchError);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
  });

  it("defaults to port 8000 and verifies the copy before anything else", async () => {
    const dir = tempDir("not-a-copy");
    await expect(launch({ dir })).rejects.toThrow(RuntimeCopyError);
    expect(existsSync(lockPath(dir))).toBe(false);
  });

  it("refuses a copy that fails verification, by default against the pin", async () => {
    const copy = fakeCopy(fakeServer());
    await expect(launch({ dir: copy.dir, port: await freePort() })).rejects.toThrow(RuntimeCopyError);
    writeFileSync(join(copy.dir, "dist/server/index.js"), "// changed\n");
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /digest/,
    );
    expect(existsSync(join(copy.dir, SERVER_CONFIG_PATH))).toBe(false);
  });

  it.each([
    ["entry script", "pokemon-showdown", "#!/usr/bin/env node\nrequire('fs').writeFileSync('ran', '');\n"],
    ["config-example.js", "config/config-example.js", "exports.bindaddress = '0.0.0.0';\n"],
  ])("refuses a tampered %s before writing the config or starting anything", async (_label, path, text) => {
    const copy = fakeCopy(fakeServer());
    writeFileSync(join(copy.dir, path), text);
    const launched = launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await expect(launched).rejects.toThrow(RuntimeCopyError);
    await expect(launched).rejects.toThrow(`${path} digest`);
    expect(existsSync(join(copy.dir, SERVER_CONFIG_PATH))).toBe(false);
    expect(existsSync(join(copy.dir, "ran"))).toBe(false);
    expect(existsSync(join(copy.dir, "started.json"))).toBe(false);
    expect(existsSync(lockPath(copy.dir))).toBe(false);
  });
});

/** Local IPv4 addresses that are not loopback. */
function externalAddresses(): string[] {
  return Object.values(networkInterfaces())
    .flat()
    .filter((info) => info !== undefined && info.family === "IPv4" && !info.internal)
    .map((info) => (info as { address: string }).address);
}

/** The local addresses of TCP sockets listening on `port`, from /proc. */
function listeningAddresses(port: number): string[] {
  const hexPort = port.toString(16).toUpperCase().padStart(4, "0");
  const out: string[] = [];
  for (const file of ["/proc/net/tcp", "/proc/net/tcp6"]) {
    if (!existsSync(file)) continue;
    for (const line of readFileSync(file, "utf8").split("\n").slice(1)) {
      const fields = line.trim().split(/\s+/);
      const [address, socketPort] = (fields[1] ?? "").split(":");
      if (socketPort === hexPort && fields[3] === "0A") out.push(`${file}:${address}`);
    }
  }
  return out;
}

function connects(host: string, port: number): Promise<boolean> {
  return new Promise((done) => {
    const socket = createConnection({ host, port });
    socket.once("connect", () => {
      socket.destroy();
      done(true);
    });
    socket.once("error", () => done(false));
  });
}

describe("launch with the pinned server", () => {
  it(
    "serves the redirect page on 127.0.0.1 only, and stop() leaves no process",
    async () => {
      const copy = createRuntimeCopy({ runtimeRoot: tempDir("real-launch") });
      const port = await freePort();
      const server = await start({ dir: copy.dir, port });
      expect(server.session.commit).toBe(PINNED_SHOWDOWN_COMMIT);

      const page = await fetch(`${server.url}/`);
      expect(page.status).toBe(200);
      expect(page.body).toBe(readFileSync(join(DEFAULT_SHOWDOWN_DIR, "server/static/index.html"), "utf8"));
      expect(page.body).toContain(".psim.us");

      if (process.platform === "linux") {
        expect(listeningAddresses(port)).toEqual(["/proc/net/tcp:0100007F"]);
      }
      for (const address of externalAddresses()) expect(await connects(address, port)).toBe(false);

      await server.stop();
      expect(groupAlive(server.session.pid)).toBe(false);
      expect(await connects("127.0.0.1", port)).toBe(false);
      expect(existsSync(lockPath(copy.dir))).toBe(false);
    },
    120_000,
  );
});
