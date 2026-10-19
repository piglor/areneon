import { execFileSync } from "node:child_process";
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
  SERVER_CONFIG_PATH,
  SERVER_LOCK,
  SESSION_MANIFEST,
} from "../src/index.ts";
import { fakeCopy, freePort, tempDir } from "./helpers.ts";

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
function fakeServer({ stubborn = false } = {}): string {
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
      dist_sha256: copy.distSha256,
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
    expect(JSON.parse(readFileSync(join(copy.dir, SERVER_LOCK), "utf8"))).toEqual({
      launcher_pid: process.pid,
      server_pid: started.pid,
    });

    await server.stop();
    expect(groupAlive(started.pid)).toBe(false);
    expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(false);
    await server.stop();
  });

  it("refuses a second server on the copy, and a recreated copy, until the first stops", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await expect(
      launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit }),
    ).rejects.toThrow(/already running/);
    expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(true);
    const recreate = (): unknown =>
      createRuntimeCopy({ vendorDir: copy.vendorDir, runtimeRoot: copy.runtimeRoot, expectedCommit: copy.commit });
    expect(recreate).toThrow(RuntimeCopyError);
    expect(recreate).toThrow(/is running from/);
    await server.stop();
    expect(recreate).not.toThrow();
  });

  it("replaces a stale lock and refuses an unreadable one", async () => {
    const copy = fakeCopy(fakeServer());
    const lock = join(copy.dir, SERVER_LOCK);
    writeFileSync(lock, JSON.stringify({ launcher_pid: 2 ** 22 + 7, server_pid: 2 ** 22 + 9 }));
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await server.stop();
    for (const text of ["not json", "null", '{"launcher_pid":1,"server_pid":null}', '{"launcher_pid":5,"server_pid":"x"}']) {
      writeFileSync(lock, text);
      await expect(
        launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit }),
      ).rejects.toThrow(/cannot read the server lock/);
      expect(readFileSync(lock, "utf8")).toBe(text);
    }
  });

  it("refuses a lock path it cannot read, here a directory, for a launch and a recreated copy", async () => {
    const copy = fakeCopy(fakeServer());
    mkdirSync(join(copy.dir, SERVER_LOCK));
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /cannot read the server lock/,
    );
    expect(() =>
      createRuntimeCopy({ vendorDir: copy.vendorDir, runtimeRoot: copy.runtimeRoot, expectedCommit: copy.commit }),
    ).toThrow(/cannot read the server lock/);
  });

  it("counts a lock live while its server group runs, even after its launcher is gone", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    const lock = join(copy.dir, SERVER_LOCK);
    const held = readFileSync(lock, "utf8");
    writeFileSync(lock, JSON.stringify({ launcher_pid: 2 ** 22 + 7, server_pid: server.session.pid }));
    await expect(
      launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit }),
    ).rejects.toThrow(`a server (pid ${server.session.pid}) is already running`);
    writeFileSync(lock, held);
  });

  it("takes its servers down when the launching process exits without stop()", async () => {
    const copy = fakeCopy(fakeServer());
    const script = join(tempDir("launcher"), "launcher.ts");
    writeFileSync(
      script,
      `import { launch } from ${JSON.stringify(join(import.meta.dirname, "../src/index.ts"))};
const server = await launch(${JSON.stringify({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })});
console.log(server.session.pid);
process.exit(0);
`,
    );
    const pid = Number(execFileSync(process.execPath, [script], { encoding: "utf8" }).trim());
    expect(pid).toBeGreaterThan(1);
    await expect.poll(() => groupAlive(pid), { timeout: 10_000 }).toBe(false);
  });

  it("kills a server that ignores SIGTERM", async () => {
    const copy = fakeCopy(fakeServer({ stubborn: true }));
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, stopTimeoutMs: 200 });
    await server.stop();
    expect(groupAlive(server.session.pid)).toBe(false);
  });

  it("stops the server by itself when its lifetime ends", async () => {
    const copy = fakeCopy(fakeServer());
    const server = await start({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, lifetimeMs: 300 });
    await expect.poll(() => groupAlive(server.session.pid), { timeout: 10_000 }).toBe(false);
    await expect.poll(() => existsSync(join(copy.dir, SERVER_LOCK)), { timeout: 10_000 }).toBe(false);
  });

  it.each([
    ["process.exit(3)", "3"],
    ["process.kill(process.pid, 'SIGKILL')", "SIGKILL"],
  ])("fails and cleans up when the server exits before it serves: %s", async (exit, how) => {
    const copy = fakeCopy(`#!/usr/bin/env node\nconsole.error('boom');\nsetTimeout(() => ${exit}, 100);\n`);
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      new RegExp(`server exited \\(${how}\\) before it served .*\\n.*boom`, "s"),
    );
    expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(false);
    expect(existsSync(join(copy.dir, SESSION_MANIFEST))).toBe(false);
  });

  it("fails and kills the server when it does not serve in time", async () => {
    const copy = fakeCopy(
      "#!/usr/bin/env node\nrequire('fs').writeFileSync('started.json', String(process.pid));\nsetInterval(() => {}, 1000);\n",
    );
    await expect(
      launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit, readyTimeoutMs: 500 }),
    ).rejects.toThrow(/did not serve .* within 500 ms/);
    expect(groupAlive(Number(readFileSync(join(copy.dir, "started.json"), "utf8")))).toBe(false);
    expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(false);
  });

  it("refuses a port another process holds, before starting anything", async () => {
    const copy = fakeCopy(fakeServer());
    const holder: Server = createServer();
    const port = await freePort();
    await new Promise<void>((done) => holder.listen(port, "127.0.0.1", done));
    try {
      await expect(launch({ dir: copy.dir, port, expectedCommit: copy.commit })).rejects.toThrow(/is not free/);
    } finally {
      holder.close();
    }
    expect(existsSync(join(copy.dir, "started.json"))).toBe(false);
    expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(false);
  });

  it.each([0, 65536, 80.5])("refuses port %s", async (port) => {
    const copy = fakeCopy(fakeServer());
    await expect(launch({ dir: copy.dir, port, expectedCommit: copy.commit })).rejects.toThrow(LaunchError);
  });

  it("defaults to port 8000 and verifies the copy before anything else", async () => {
    const dir = tempDir("not-a-copy");
    await expect(launch({ dir })).rejects.toThrow(RuntimeCopyError);
    expect(existsSync(join(dir, SERVER_LOCK))).toBe(false);
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
      expect(existsSync(join(copy.dir, SERVER_LOCK))).toBe(false);
    },
    120_000,
  );
});
