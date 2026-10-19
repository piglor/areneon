import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { PassThrough } from "node:stream";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SupervisorConfig } from "../src/supervisor.ts";

/**
 * The launcher's handling of a supervisor that fails: `node:child_process`
 * `spawn` is replaced by a scripted stand-in, so the faults are exact.
 */
type Script = (child: FakeChild, config: SupervisorConfig) => void;
let script: Script = () => {};

class FakeChild extends EventEmitter {
  pid: number | undefined = 4_000_000;
  stdin = new PassThrough();
  stdout = new PassThrough();
  stderr = new PassThrough();
  killed: string[] = [];
  kill(signal: string): boolean {
    this.killed.push(signal);
    return true;
  }
}

vi.mock("node:child_process", async (original) => ({
  ...(await original<typeof import("node:child_process")>()),
  spawn: (_command: string, args: string[]) => {
    const child = new FakeChild();
    script(child, JSON.parse(args[1] as string) as SupervisorConfig);
    return child;
  },
}));

/** With `noIpv6`, listening on ::1 fails as on a host without IPv6. */
const net = vi.hoisted(() => ({ noIpv6: false }));
vi.mock("node:net", async (original) => {
  const real = await original<typeof import("node:net")>();
  return {
    ...real,
    createServer: () => {
      const server = real.createServer();
      const listen = server.listen.bind(server) as (...args: unknown[]) => unknown;
      server.listen = ((...args: unknown[]) => {
        const options = args[0] as { host?: string };
        if (net.noIpv6 && typeof options === "object" && options.host === "::1") {
          const error = Object.assign(new Error("listen EADDRNOTAVAIL ::1"), { code: "EADDRNOTAVAIL" });
          process.nextTick(() => server.emit("error", error));
          return server;
        }
        return listen(...args);
      }) as typeof server.listen;
      return server;
    },
  };
});

const { launch, LaunchError, lockPath } = await import("../src/index.ts");
const { fakeCopy, freePort } = await import("./helpers.ts");

const servers: Server[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.close();
});

/** Behave as a supervisor whose server serves on the session's port. */
function serving(child: FakeChild, config: SupervisorConfig): void {
  const port = Number(config.argv.at(-1));
  const server = createServer((_, response) => response.end("ok"));
  servers.push(server);
  server.listen(port, "127.0.0.1", () => {
    child.stdout.write(
      `${JSON.stringify({ event: "spawned", pid: 4_000_001, started_at: "t0", expires_at: "t1" })}\n`,
    );
    child.stderr.write(`now listening on 127.0.0.1:${port}\n`);
  });
}

describe("launch with a faulty supervisor", () => {
  it("turns a spawn error into a LaunchError and releases the lock", async () => {
    script = (child) => {
      child.pid = undefined;
      setImmediate(() => child.emit("error", new Error("spawn EACCES")));
    };
    const copy = fakeCopy("// unused\n");
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      "cannot start the supervisor: Error: spawn EACCES",
    );
    expect(existsSync(lockPath(copy.dir))).toBe(false);
  });

  it("starts on a host without IPv6", async () => {
    net.noIpv6 = true;
    script = serving;
    try {
      const copy = fakeCopy("// unused\n");
      const server = await launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
      expect(server.session.pid).toBe(4_000_001);
    } finally {
      net.noIpv6 = false;
    }
  });

  it("refuses a supervisor killed before the server starts", async () => {
    script = (child) => {
      setImmediate(() => child.emit("exit", null, "SIGKILL"));
    };
    const copy = fakeCopy("// unused\n");
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /supervisor exited \(SIGKILL\) before the server started/,
    );
  });

  it("refuses a supervisor that exits before the server starts", async () => {
    script = (child) => {
      child.stderr.write("cannot start the server\n");
      setImmediate(() => child.emit("exit", 1, null));
    };
    const copy = fakeCopy("// unused\n");
    await expect(launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit })).rejects.toThrow(
      /supervisor exited \(1\) before the server started:\ncannot start the server/,
    );
  });

  it("reports a stop the server outlived, and keeps the lock", async () => {
    script = (child, config) => {
      serving(child, config);
      child.stdin.on("finish", () => setImmediate(() => child.emit("exit", 2, null)));
    };
    const copy = fakeCopy("// unused\n");
    const server = await launch({ dir: copy.dir, port: await freePort(), expectedCommit: copy.commit });
    await expect(server.stop()).rejects.toThrow(/supervisor exited \(2\); the server may still run/);
    expect(existsSync(lockPath(copy.dir))).toBe(true);
  });

  it(
    "kills a supervisor that does not exit, and says so",
    async () => {
      let fake: FakeChild | undefined;
      script = (child, config) => {
        fake = child;
        serving(child, config);
      };
      const copy = fakeCopy("// unused\n");
      const server = await launch({
        dir: copy.dir,
        port: await freePort(),
        expectedCommit: copy.commit,
        stopTimeoutMs: 1,
      });
      await expect(server.stop()).rejects.toThrow(LaunchError);
      await expect(server.stop()).rejects.toThrow(/did not exit within 5002 ms/);
      expect(fake?.killed).toEqual(["SIGKILL"]);
    },
    20_000,
  );
});
