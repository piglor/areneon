import { type ChildProcess, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { LaunchOptions, LocalServer, RuntimeCopyOptions } from "@areneon/local-server";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type CliDeps, runCli, USAGE } from "../src/cli.ts";
import type { A2BattleRecord, A2BotOptions, A2SessionResult } from "../src/index.ts";
import { FORMAT, freePort, groupAlive, playRecordedBattle, startLocalServer, TEAMS_DIR, tempDir } from "./local-battle.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const TEAM = join(TEAMS_DIR, "legal-synthetic.txt");
const RECORD = { room: "battle-x-1", ots: "idle", ots_failure: null, fatal: null } as A2BattleRecord;
const REPO = "/repo";
const PRIVATE = "/repo/private-data";

interface Fake {
  deps: CliDeps;
  out: string[];
  err: string[];
  calls: string[];
  bot: A2BotOptions[];
  /** Files written: [path, parsed record, mode], and the directories made. */
  files: [string, A2BattleRecord, number][];
  dirs: [string, number][];
  signal: (name: string) => void;
}

/** runCli on fakes: no server starts and nothing connects. */
function fake(overrides: Partial<CliDeps> = {}): Fake {
  const out: string[] = [];
  const err: string[] = [];
  const calls: string[] = [];
  const bot: A2BotOptions[] = [];
  const files: [string, A2BattleRecord, number][] = [];
  const dirs: [string, number][] = [];
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
    makeDir: (dir, mode) => dirs.push([dir, mode]),
    writeFile: (path, data, mode) => files.push([path, JSON.parse(data) as A2BattleRecord, mode]),
    repoRoot: REPO,
    privateDataDir: PRIVATE,
    defaultOutDir: `${PRIVATE}/a2`,
    now: () => 1234,
    stdout: (line) => out.push(line),
    stderr: (line) => err.push(line),
    ...overrides,
  };
  return { deps, out, err, calls, bot, files, dirs, signal: (name) => handler(name) };
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
      mode: "live_match",
    });
    expect(options.team.map((set) => set.species)).toHaveLength(6);
    expect(JSON.parse(f.out[0] as string)).toMatchObject({ event: "ready", bot: "Areneon-bot-abc123", server_pid: 4242 });
    expect(JSON.parse(f.out[1] as string)).toEqual({ event: "logged_in", name: "Areneon-bot-abc123" });
    expect(JSON.parse(f.out.at(-1) as string)).toEqual({ event: "stopped", battles: 1 });
    // Stored once, though both onBattle and the result carry it.
    expect(f.files).toEqual([["/repo/private-data/a2/battle-x-1-1234.json", RECORD, 0o600]]);
    expect(f.dirs).toEqual([["/repo/private-data/a2", 0o700]]);
  });

  it("passes the runtime root, record directory, data class and format through", async () => {
    const f = fake();
    const argv = ["--challenger", "h", "--team", TEAM, "--format", FORMAT];
    expect(await runCli([...argv, "--runtime-root", "/rt", "--out", "/out"], f.deps)).toBe(0);
    expect(f.calls).toContain("copy /rt");
    expect(f.calls).toContain("launch /copy 8000");
    // The class is proven from team hashes, never given: nothing to pass.
    expect(f.bot[0]).not.toHaveProperty("data_class");
    expect(f.bot[0]?.known_synthetic).toBeUndefined();
    expect(f.bot[0]?.opponent_team).toBeUndefined();
    expect(f.files[0]?.[0]).toBe("/out/battle-x-1-1234.json");
  });

  it("logs a dropped line by its kind only", async () => {
    const f = fake({
      runBot: async (options) => {
        options.onLog?.({ event: "dropped", room: "battle-x-1", line: "|unknownkind|secret text" });
        options.onLog?.({ event: "dropped", room: "battle-x-1", line: "|bare" });
        options.onLog?.({ event: "dropped", room: "battle-x-1", line: "secret|no leading pipe" });
        return { battles: [] };
      },
    });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(0);
    const dropped = f.out.filter((line) => line.includes('"dropped"'));
    expect(dropped.map((line) => JSON.parse(line))).toEqual([
      { event: "dropped", room: "battle-x-1", kind: "|unknownkind" },
      { event: "dropped", room: "battle-x-1", kind: "|bare" },
      { event: "dropped", room: "battle-x-1", kind: "|?" },
    ]);
    expect(f.out.join("\n")).not.toContain("secret");
  });

  it("also stores the battles the session ends without onBattle (unfinished), once each", async () => {
    const unfinished = { room: "battle-x-2", result: "unfinished", ots: "idle", ots_failure: null, fatal: null } as A2BattleRecord;
    const f = fake({
      runBot: async (options) => {
        options.onBattle?.(RECORD);
        return { battles: [RECORD, unfinished] };
      },
    });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(0);
    expect(f.files.map(([path]) => path)).toEqual([
      "/repo/private-data/a2/battle-x-1-1234.json",
      "/repo/private-data/a2/battle-x-2-1234.json",
    ]);
  });

  it("refuses a record whose room is not a battle room id, and fails the run", async () => {
    const bad = (room: string) => ({ room, ots: "idle", ots_failure: null, fatal: null }) as A2BattleRecord;
    const f = fake({
      runBot: async () => ({ battles: [bad("../../etc/x"), bad("battle-Up"), bad("lobby"), bad("battle-ok-9")] }),
    });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(1);
    expect(f.files.map(([path]) => path)).toEqual(["/repo/private-data/a2/battle-ok-9-1234.json"]);
    const refusals = f.err.filter((line) => !line.includes("no Episode"));
    expect(refusals).toHaveLength(3);
    expect(refusals[0]).toContain("not a battle room id");
  });

  it("reports a record that cannot be written, and still stops the server", async () => {
    const f = fake({
      writeFile: () => {
        throw "disk full";
      },
    });
    expect(await runCli(["--challenger", "h", "--team", TEAM], f.deps)).toBe(1);
    expect(f.err).toEqual(['record battle-x-1 not stored: "disk full"']);
    expect(f.calls.at(-1)).toBe("stop");
  });

  it("writes an ots_failure record only under private-data, whatever --out says", async () => {
    const failed = { room: "battle-x-3", ots: "forfeited", fatal: null, ots_failure: { reason: "showteam", detail: "|showteam|p2" } } as A2BattleRecord;
    const run = (extra: string[]) => {
      const f = fake({ runBot: async () => ({ battles: [failed] }) });
      return runCli(["--challenger", "h", "--team", TEAM, ...extra], f.deps).then(
        () => f,
      );
    };
    const outside = await run(["--out", "/elsewhere"]);
    expect(outside.files.map(([path]) => path)).toEqual(["/repo/private-data/a2/battle-x-3-1234.json"]);
    expect(outside.err[0]).toContain("may hold the opponent's team sheet");
    const inside = await run(["--out", "/repo/private-data/other"]);
    expect(inside.files.map(([path]) => path)).toEqual(["/repo/private-data/other/battle-x-3-1234.json"]);
    expect(inside.err.filter((line) => !line.includes("no Episode"))).toEqual([]);
  });

  it.each([
    ["an ots_failure", { ots: "forfeited", ots_failure: { reason: "refused", detail: "x" }, fatal: null }],
    ["a forfeited decline whose failure was not recorded", { ots: "forfeited", ots_failure: null, fatal: null }],
    ["a fatal error", { ots: "idle", ots_failure: null, fatal: { error: "SessionError", channel_complete: true } }],
  ])("writes the record of %s under private-data only", async (_name, fields) => {
    const record = { room: "battle-x-7", ...fields } as A2BattleRecord;
    const f = fake({ runBot: async () => ({ battles: [record] }) });
    await runCli(["--challenger", "h", "--team", TEAM, "--out", "/elsewhere"], f.deps);
    expect(f.files.map(([path]) => path)).toEqual(["/repo/private-data/a2/battle-x-7-1234.json"]);
    expect(f.err[0]).toContain("may hold the opponent's team sheet");
  });

  it("refuses an --out inside the checkout unless it is under private-data (every record is private)", async () => {
    const argv = ["--challenger", "h", "--team", TEAM];
    for (const out of ["/repo/records", "/repo", "/repo/private-data/../records"]) {
      const f = fake();
      expect(await runCli([...argv, "--out", out], f.deps)).toBe(2);
      expect(f.err[0]).toContain("is inside the checkout");
      expect(f.calls).toEqual([]);
    }
    const priv = fake();
    expect(await runCli([...argv, "--out", "/repo/private-data/x"], priv.deps)).toBe(0);
    const outside = fake();
    expect(await runCli([...argv, "--out", "/srv/records"], outside.deps)).toBe(0);
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

  it("stores the records of the battles a fatal error ended, then exits 1 and stops the server", async () => {
    const unfinished = { room: "battle-x-5", result: "unfinished", ots: "idle", ots_failure: null, fatal: { error: "RefusalError" } };
    const failed = { room: "battle-x-6", ots: "forfeited", fatal: null, ots_failure: { reason: "showteam", detail: "|showteam|p2" } } as A2BattleRecord;
    const f = fake({
      runBot: (options) => {
        options.onBattle?.(unfinished as unknown as A2BattleRecord);
        options.onBattle?.(failed);
        return Promise.reject(new RangeError("boom"));
      },
    });
    expect(await runCli(["--challenger", "h", "--team", TEAM, "--out", "/srv/records"], f.deps)).toBe(1);
    expect(f.files.map(([path]) => path)).toEqual([
      "/repo/private-data/a2/battle-x-5-1234.json",
      "/repo/private-data/a2/battle-x-6-1234.json",
    ]);
    expect(f.files[0]?.[2]).toBe(0o600);
    expect(f.err.at(-1)).toBe('a2-bot failed: "RangeError: boom"');
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
    [["--challenger", "h", "--team", TEAM, "--data-class", "synthetic"], "Unknown option '--data-class'"],
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
        [SCRIPT, "--challenger", "Nobody Here", "--team", TEAM].concat(
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

describe("runCli: the bot's own Episode (ADR-009 slice f)", () => {
  let local: Awaited<ReturnType<typeof startLocalServer>>;
  let real: A2BattleRecord;
  beforeAll(async () => {
    local = await startLocalServer();
    ({ mine: real } = await playRecordedBattle(local));
  }, 90_000);
  afterAll(async () => {
    await local?.stop();
  });

  const run = (record: A2BattleRecord, overrides: Partial<CliDeps> = {}, extra: string[] = []) => {
    const f = fake({ runBot: async (options) => (options.onBattle?.(record), { battles: [record] }), ...overrides });
    return runCli(["--challenger", "h", "--team", TEAM, ...extra], f.deps).then((code) => ({ f, code }));
  };

  it("stores a finished battle as a private sparring Episode beside its record, under private-data only", async () => {
    const { f, code } = await run(real, {}, ["--out", "/elsewhere"]);
    expect(code).toBe(0);
    expect(f.err).toEqual([]);
    const [record, episode] = f.files;
    expect(record?.[0]).toBe(`/elsewhere/${real.room}-1234.json`);
    expect(episode?.[0]).toBe(`/repo/private-data/a2/${real.room}-1234.episode.json`);
    expect(episode?.[2]).toBe(0o600);
    expect(f.dirs).toContainEqual(["/repo/private-data/a2", 0o700]);
    expect(episode?.[1]).toMatchObject({
      data_class: "private",
      rights: "owner-private",
      player: real.player,
      provenance: {
        runtime: "A2",
        behaviour_policy_id: "actor-rule@a2",
        collection: { kind: "sparring", session_id: "a2-abc123" },
      },
    });
    expect(Object.keys(episode?.[1] ?? {}).sort()).toEqual(
      ["channel", "data_class", "decisions", "player", "provenance", "result", "rights", "schema_version", "team"],
    );
  });

  it("reports a battle that cannot be an Episode and goes on: its record is stored all the same", async () => {
    const { f, code } = await run({ ...real, defect: true });
    expect(code).toBe(0);
    expect(f.files.map(([path]) => path)).toEqual([`/repo/private-data/a2/${real.room}-1234.json`]);
    expect(f.err).toEqual([`${real.room}: no Episode: the battle had a defect: the engine refused a non-pending candidate`]);
  });

  it("shows only the kind of a reader refusal, never what it quotes", async () => {
    const { f, code } = await run({ ...real, result: real.result === "win" ? "loss" : "win" });
    expect(code).toBe(0);
    expect(f.err).toEqual([`${real.room}: no Episode: the battle did not pass the Episode reader`]);
  });

  it("fails the run when the Episode cannot be written", async () => {
    const { f, code } = await run(real, {
      writeFile: (path) => {
        if (path.endsWith(".episode.json")) throw new Error("disk full");
      },
    });
    expect(code).toBe(1);
    expect(f.err).toEqual([`episode ${real.room} not stored: "Error: disk full"`]);
    const text = await run(real, {
      writeFile: (path) => {
        if (path.endsWith(".episode.json")) throw "no space";
      },
    });
    expect(text.f.err).toEqual([`episode ${real.room} not stored: "no space"`]);
  });
});
