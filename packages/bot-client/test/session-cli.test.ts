import { type ChildProcess, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { LaunchOptions, LocalServer, RuntimeCopyOptions } from "@areneon/local-server";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { type CliDeps, runCli, USAGE } from "../src/cli.ts";
import type { A2BattleRecord, A2BotOptions, A2SessionResult } from "../src/index.ts";
import { FORMAT, freePort, groupAlive, TEAMS_DIR, tempDir } from "./local-battle.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const TEAM = join(TEAMS_DIR, "legal-synthetic.txt");
const RECORD = { room: "battle-x-1" } as A2BattleRecord;

interface Fake {
  deps: CliDeps;
  out: string[];
  err: string[];
  calls: string[];
  bot: A2BotOptions[];
  records: [string, A2BattleRecord][];
  signal: (name: string) => void;
}

/** runCli on fakes: no server starts and nothing connects. */
function fake(overrides: Partial<CliDeps> = {}): Fake {
  const out: string[] = [];
  const err: string[] = [];
  const calls: string[] = [];
  const bot: A2BotOptions[] = [];
  const records: [string, A2BattleRecord][] = [];
  let handler: (name: string) => void = () => {};
  const server = {
    url: "http://127.0.0.1:8123",
    session: { bot_name_suffix: "abc123", pid: 4242 },
    stop: async () => {
      calls.push("stop");
    },
  } as unknown as LocalServer;
  const deps: CliDeps = {
    engine,
    snapshot,
    readFile: (path) => readFileSync(path, "utf8"),
    createRuntimeCopy: (options: RuntimeCopyOptions) => {
      calls.push(`copy ${options.runtimeRoot ?? "default"}`);
      return { dir: "/copy", commit: engine.commit, distSha256: "0" } as never;
    },
    launch: async (options: LaunchOptions) => {
      calls.push(`launch ${options.dir} ${options.port}`);
      return server;
    },
    runBot: async (options) => {
      bot.push(options);
      options.onLog?.({ event: "logged_in", name: options.name });
      options.onBattle?.(RECORD);
      return { battles: [RECORD] };
    },
    onSignal(h) {
      handler = h;
      calls.push("trap");
      return () => calls.push("untrap");
    },
    writeRecord: (dir, record) => records.push([dir, record]),
    defaultOutDir: "/private-data/a2",
    stdout: (line) => out.push(line),
    stderr: (line) => err.push(line),
    ...overrides,
  };
  return { deps, out, err, calls, bot, records, signal: (name) => handler(name) };
}

describe("runCli (the A2 sparring entry point)", () => {
  it("launches the server, runs the bot as a live match, records its battles, then stops the server", async () => {
    const f = fake();
    const code = await runCli(["--challenger", "Sparring Human", "--team", TEAM, "--port", "8123"], f.deps);
    expect(code).toBe(0);
    expect(f.calls).toEqual(["trap", "copy default", "launch /copy 8123", "untrap", "stop"]);
    const [options] = f.bot as [A2BotOptions];
    expect(options).toMatchObject({
      url: "ws://127.0.0.1:8123/showdown/websocket",
      name: "Areneon-bot-abc123",
      challenger: "Sparring Human",
      format: FORMAT,
      data_class: "private",
      mode: "live_match",
    });
    expect(options.team.map((set) => set.species)).toHaveLength(6);
    expect(JSON.parse(f.out[0] as string)).toMatchObject({ event: "ready", bot: "Areneon-bot-abc123", server_pid: 4242 });
    expect(JSON.parse(f.out[1] as string)).toEqual({ event: "logged_in", name: "Areneon-bot-abc123" });
    expect(JSON.parse(f.out.at(-1) as string)).toEqual({ event: "stopped", battles: 1 });
    expect(f.records).toEqual([["/private-data/a2", RECORD]]);
  });

  it("passes the runtime root, record directory, data class and format through", async () => {
    const f = fake();
    const argv = ["--challenger", "h", "--team", TEAM, "--data-class", "synthetic", "--format", FORMAT];
    expect(await runCli([...argv, "--runtime-root", "/rt", "--out", "/out"], f.deps)).toBe(0);
    expect(f.calls).toContain("copy /rt");
    expect(f.calls).toContain("launch /copy 8000");
    expect(f.bot[0]?.data_class).toBe("synthetic");
    expect(f.records[0]?.[0]).toBe("/out");
  });

  it("aborts the bot on a signal and still stops the server", async () => {
    let aborted: AbortSignal | undefined;
    const f = fake({
      runBot: (options) =>
        new Promise<A2SessionResult>((done) => {
          aborted = options.signal;
          options.signal?.addEventListener("abort", () => done({ battles: [] }));
        }),
    });
    const running = runCli(["--challenger", "h", "--team", TEAM], f.deps);
    await expect.poll(() => aborted).toBeDefined();
    f.signal("SIGINT");
    expect(await running).toBe(0);
    expect(aborted?.aborted).toBe(true);
    expect(f.err).toEqual(["SIGINT: stopping the bot and the server"]);
    expect(f.calls.slice(-2)).toEqual(["untrap", "stop"]);
  });

  it("stops the server when the bot fails, and reports the failure", async () => {
    const f = fake({ runBot: () => Promise.reject(new RangeError("boom")) });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(1);
    expect(f.err).toEqual(['a2-bot failed: "RangeError: boom"']);
    expect(f.calls.slice(-2)).toEqual(["untrap", "stop"]);
  });

  it("starts no server when the team is unreadable", async () => {
    const f = fake({
      readFile: () => {
        throw "unreadable";
      },
    });
    expect(await runCli(["--challenger", "h", "--team", "nope"], f.deps)).toBe(1);
    expect(f.err).toEqual(['a2-bot failed: "unreadable"']);
    expect(f.calls).toEqual(["trap", "untrap"]);
  });

  it.each([
    [["--team", TEAM], "--challenger and --team are required"],
    [["--challenger", "h"], "--challenger and --team are required"],
    [["--challenger", "h", "--team", TEAM, "--data-class", "public"], "--data-class must be private or synthetic"],
    [["--challenger", "h", "--team", TEAM, "--port", "0"], "--port must be an integer from 1 to 65535"],
    [["--challenger", "h", "--team", TEAM, "--port", "x"], "--port must be an integer"],
    [["--challenger", "h", "--bogus"], "Unknown option '--bogus'"],
  ])("refuses %j as usage", async (argv, message) => {
    const f = fake();
    expect(await runCli(argv, f.deps)).toBe(2);
    expect(f.err[0]).toContain(message);
    expect(f.err[0]).toContain(USAGE);
    expect(f.calls).toEqual([]);
  });

  it("needs --format when the rules snapshot has more than one", async () => {
    const two = { ...snapshot, formats: { ...snapshot.formats, other: snapshot.formats[FORMAT] } } as typeof snapshot;
    const f = fake({ snapshot: two });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(2);
    expect(f.err[0]).toContain("--format is required: the rules snapshot has 2 formats");
  });
});

const SCRIPT = fileURLToPath(new URL("../scripts/a2-bot.ts", import.meta.url));

/** Every stdout JSON line of `child`, as it comes. */
function events(child: ChildProcess): Record<string, unknown>[] {
  const seen: Record<string, unknown>[] = [];
  let buffer = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    buffer += chunk.toString("utf8");
    const lines = buffer.split("\n");
    buffer = lines.pop() as string;
    for (const line of lines) seen.push(JSON.parse(line) as Record<string, unknown>);
  });
  return seen;
}

describe("the a2-bot process", () => {
  it.each(["SIGINT", "SIGTERM"] as const)(
    "stops the bot and the server on %s, leaving no server process",
    async (signal) => {
      const runtime = tempDir("cli-runtime");
      const out = tempDir("cli-out");
      const child = spawn(
        process.execPath,
        [SCRIPT, "--challenger", "Nobody Here", "--team", TEAM, "--data-class", "synthetic"].concat(
          ["--port", String(await freePort()), "--runtime-root", runtime.dir, "--out", out.dir],
        ),
        { stdio: ["ignore", "pipe", "pipe"] },
      );
      const seen = events(child);
      const exited = new Promise<number | null>((done) => child.once("exit", (code) => done(code)));
      try {
        const find = (event: string) => seen.find((entry) => entry.event === event);
        await expect.poll(() => find("logged_in"), { timeout: 30_000 }).toBeDefined();
        const pid = find("ready")?.server_pid as number;
        expect(groupAlive(pid)).toBe(true);
        child.kill(signal);
        expect(await exited).toBe(0);
        expect(groupAlive(pid)).toBe(false);
      } finally {
        child.kill("SIGKILL");
        runtime.remove();
        out.remove();
      }
    },
    60_000,
  );
});
