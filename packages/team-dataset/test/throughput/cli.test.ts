// The throughput harness's command line: argument handling, the commands, exit codes. The arms are the fake arm (any command) and a
// stand-in for play-generation.ts, so the whole path from flags to record runs for real processes without a game.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readThroughputRecord } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { GIB_KIB } from "../../src/throughput/guard.ts";
import { runThroughputCli } from "../../src/throughput/cli.ts";
import { FAKE_ARM, fakeDeps, type Script, tempDir } from "./support.ts";

const FAKE_PLAY = new URL("./fake-play.ts", import.meta.url).pathname;
const REPO = new URL("../../../..", import.meta.url).pathname;
const VALID = join(REPO, "fixtures/contracts/v1/throughput-record/valid");

interface Run {
  code: number;
  out: string;
  err: string;
}
async function cli(argv: string[], script: Script = {}): Promise<Run> {
  const out: string[] = [];
  const err: string[] = [];
  const code = await runThroughputCli(argv, { out: (line) => out.push(line), err: (line) => err.push(line), repoRoot: REPO, deps: () => ({ ...fakeDeps(script), sleep: () => Promise.resolve() }) });
  return { code, out: out.join("\n"), err: err.join("\n") };
}
const quick = ["--proc-interval-ms", "50", "--condition-interval-ms", "150", "--start-window-ms", "5"];

describe("guard", () => {
  it("says idle and exits 0 on a quiet machine", async () => {
    const run = await cli(["guard"]);
    expect(run.code).toBe(0);
    expect(run.out).toContain("idle");
  });

  it("lists every broken condition and exits 3 on a busy one", async () => {
    const run = await cli(["guard"], { samples: () => ({ load1_centi: 500, mem_available_kib: GIB_KIB }), others: () => ["pytest tests"] });
    expect(run.code).toBe(3);
    expect(run.err).toContain("load average");
    expect(run.err).toContain("MemAvailable");
    expect(run.err).toContain("another run is going: pytest tests");
  });

  it("checks the conditions of a non-idle run too, which enforces only the memory floor", async () => {
    expect((await cli(["guard", "--mode", "non-idle"], { samples: () => ({ load1_centi: 5000 }) })).code).toBe(0);
    expect((await cli(["guard", "--mode", "non-idle", "--min-mem-gib", "30"])).code).toBe(3);
  });
});

describe("run --arm command", () => {
  it("measures any command that leaves an arm report, once per repeat, and writes one run directory each", async () => {
    const out = tempDir();
    const report = join(out, "arm-report.json");
    const run = await cli(["run", "--arm", "command", "--out", out, "--run-id", "fake", "--repeats", "2", "--arm-report", report, "--arm-id", "fake", "--arm-label", "Fake arm", "--arm-param", "children=1", "--pool-id", "fake-pool", ...quick, "--", process.execPath, FAKE_ARM, "--children", "1", "--burn-ms", "20", "--decisions", "50", "--games", "5", "--report", report]);
    expect(run.err).toBe("");
    expect(run.code).toBe(0);
    expect(readdirSync(out).sort()).toEqual(["arm-report.json", "fake-r1", "fake-r2"]);
    for (const id of ["fake-r1", "fake-r2"]) {
      const record = readThroughputRecord(JSON.parse(readFileSync(join(out, id, "record.json"), "utf8")) as unknown);
      expect(record).toMatchObject({ run_id: id, valid: true, report: { arm: { id: "fake", parameters: [{ name: "children", value: 1 }] }, plan: { pool_id: "fake-pool" }, counts: { games: 5, decisions: 50 } } });
    }
    expect(run.out).toContain("fake-r1: VALID");
    expect(run.out).toContain("fake-r2: VALID");
    expect(run.out).toContain("# Throughput comparison");
    expect(run.out).toContain("| fake children=1 | 2 of 2 |");
  });

  it("exits 1 when a run is void, and still prints it", async () => {
    const out = tempDir();
    const report = join(out, "r.json");
    const run = await cli(["run", "--arm", "command", "--out", out, "--run-id", "bad", "--arm-report", report, "--arm-id", "fake", "--arm-label", "Fake", ...quick, "--", process.execPath, FAKE_ARM, "--burn-ms", "10", "--exit", "4", "--decisions", "3", "--report", report]);
    expect(run.code).toBe(1);
    expect(run.out).toContain("bad-r1: VOID (arm_failed)");
  });

  it("refuses to start on a busy machine with exit 3 and writes no run directory", async () => {
    const out = tempDir();
    const run = await cli(["run", "--arm", "command", "--out", out, "--run-id", "x", "--arm-report", join(out, "r.json"), "--arm-id", "fake", "--arm-label", "Fake", ...quick, "--", process.execPath, FAKE_ARM], { samples: () => ({ load1_centi: 900 }) });
    expect(run.code).toBe(3);
    expect(run.err).toContain("not idle");
    expect(readdirSync(out)).toEqual([]);
  });

  it("waits for an idle machine up to the minutes it is given, trying every 15 seconds", async () => {
    const out = tempDir();
    const report = join(out, "r.json");
    const args = ["run", "--arm", "command", "--out", out, "--run-id", "w", "--arm-report", report, "--arm-id", "fake", "--arm-label", "Fake", "--wait-idle-minutes", "1", ...quick, "--", process.execPath, FAKE_ARM, "--burn-ms", "10", "--decisions", "2", "--report", report];
    // busy at the first two looks (the start sample of each attempt is read 0 and 1), idle afterwards
    expect((await cli(args, { samples: (n) => (n < 2 ? { load1_centi: 900 } : {}) })).code).toBe(0);
    // a minute holds four attempts of 15 s: a machine that never calms down is refused after them
    const never = await cli(args.map((a) => (a === "w" ? "w2" : a)), { samples: () => ({ load1_centi: 900 }) });
    expect(never.code).toBe(3);
    expect(never.err.match(/not idle/g)?.length).toBeGreaterThanOrEqual(1);
  });

  it("refuses arguments it cannot use, exit 2, with the reason", async () => {
    const out = tempDir();
    const bad = async (argv: string[]): Promise<string> => {
      const run = await cli(argv);
      expect(run.code, argv.join(" ")).toBe(2);
      return run.err;
    };
    expect(await bad(["run"])).toContain("--arm is required");
    expect(await bad(["run", "--arm", "gpu"])).toContain("unknown arm");
    expect(await bad(["run", "--arm", "command"])).toContain("--out is required");
    expect(await bad(["run", "--arm", "command", "--out", out])).toContain("--arm-report is required");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-id", "x", "--arm-label", "x"])).toContain("the command to run follows --");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-label", "x", "--", "true"])).toContain("--arm-id is required");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-id", "x", "--arm-label", "x", "--arm-param", "nope", "--", "true"])).toContain("--arm-param");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-id", "x", "--arm-label", "x", "--mode", "loud", "--", "true"])).toContain("--mode");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-id", "x", "--arm-label", "x", "--repeats", "0", "--", "true"])).toContain("--repeats");
    expect(await bad(["run", "--arm", "command", "--out", out, "--arm-report", "r", "--arm-id", "x", "--arm-label", "x", "--run-id", "has space", "--", "true"])).toContain("--run-id");
    expect(await bad(["run", "--arm", "command", "--bogus"])).toContain("bogus");
    expect(await bad(["frobnicate"])).toContain("unknown command");
    expect(await bad([])).toContain("usage");
  });

  it("refuses an output directory that is a file", async () => {
    const dir = tempDir();
    writeFileSync(join(dir, "f"), "x");
    const run = await cli(["run", "--arm", "command", "--out", join(dir, "f"), "--arm-report", "r", "--arm-id", "x", "--arm-label", "x", "--", "true"]);
    expect(run.code).toBe(2);
    expect(run.err).toContain("not a directory");
  });
});

describe("run --arm showdown", () => {
  /** A repository root of its own (the arm's working directory), the stand-in play script, and a run config naming a team dataset. */
  function project(): { repo: string; config: string; out: string; policyId: string; base: string[] } {
    const repo = tempDir();
    const config = join(repo, "config.json");
    writeFileSync(config, JSON.stringify({ team_dataset: { path: "/data/train-v1", dataset_hash: "7".repeat(64) } }));
    const policyId = (JSON.parse(readFileSync(join(VALID, "../../league-run-record/valid/tiny.json"), "utf8")) as { pair_policy_id: string }).pair_policy_id;
    const out = join(repo, "out");
    const base = ["run", "--arm", "showdown", "--out", out, "--run-id", "sd", "--repo-root", repo, "--play-script", FAKE_PLAY, "--config", config, "--run-dir", join(repo, "run"), "--preview", "1".repeat(64), "--battle", "2".repeat(64), "--engine-data", join(repo, "engine-data.json"), "--worker-python", "python3", "--policy-id", policyId, "--encoder-version", "enc1:x", ...quick];
    return { repo, config, out, policyId, base };
  }

  it("runs the play at each number of game processes in interleaved rounds, one record each, and removes the play's bulk output", async () => {
    const { out, base } = project();
    const run = await cli([...base, "--game-processes", "4,6", "--repeats", "2"]);
    expect(run.err).toBe("");
    expect(run.code).toBe(0);
    expect(readdirSync(out).sort()).toEqual(["sd-gp4-r1", "sd-gp4-r2", "sd-gp6-r1", "sd-gp6-r2"]);
    const record = readThroughputRecord(JSON.parse(readFileSync(join(out, "sd-gp6-r2", "record.json"), "utf8")) as unknown);
    expect(record.report.arm.parameters).toEqual([{ name: "game_processes", value: 6 }, { name: "generation", value: 1 }, { name: "policy_workers", value: 1 }]);
    expect(record.report.counts).toMatchObject({ decisions: 1006, games_on_showdown: record.report.counts.games });
    expect(record.report.plan).toMatchObject({ pool_id: "train-v1", pool_hash: "7".repeat(64) });
    expect(record.cpu.by_role.some((r) => r.role === "node")).toBe(true); // the stand-in is not named play-generation.ts, so its role is its program
    expect(existsSync(join(out, "sd-gp6-r2", "arm"))).toBe(false);
    const order = run.out.split("\n").filter((line) => /^sd-gp\d-r\d: VALID/.test(line)).map((line) => line.split(":")[0]);
    expect(order).toEqual(["sd-gp4-r1", "sd-gp6-r1", "sd-gp4-r2", "sd-gp6-r2"]); // interleaved: each round plays every setting
    expect(run.out).toContain("showdown game_processes=4,generation=1,policy_workers=1");
    expect(run.out).toContain("showdown game_processes=6,generation=1,policy_workers=1");
  });

  it("keeps the play's output when asked", async () => {
    const { out, base } = project();
    expect((await cli([...base, "--keep-arm-output"])).code).toBe(0);
    expect(existsSync(join(out, "sd-gp4-r1", "arm", "league-run.json"))).toBe(true);
  });

  it("asks the Python package for the ids when they are not given", async () => {
    const { base } = project();
    const noIds = base.filter((a, i, all) => a !== "--policy-id" && a !== "--encoder-version" && all[i - 1] !== "--policy-id" && all[i - 1] !== "--encoder-version");
    const run = await cli([...noIds, "--ids-python", "/nonexistent/python"]);
    expect(run.code).toBe(2);
    expect(run.err).toContain("policy ids");
    const missing = await cli(noIds);
    expect(missing.code).toBe(2);
    expect(missing.err).toContain("--policy-id and --encoder-version, or --ids-python");
  });

  it("refuses what it cannot use", async () => {
    const { base } = project();
    const bad = async (extra: string[], without: string[] = []): Promise<string> => {
      const argv = base.filter((a, i, all) => !without.includes(a) && !without.includes(all[i - 1] as string)).concat(extra);
      const run = await cli(argv);
      expect(run.code, extra.join(" ")).toBe(2);
      return run.err;
    };
    expect(await bad(["--game-processes", "0"])).toContain("--game-processes");
    expect(await bad(["--game-processes", "x"])).toContain("--game-processes");
    expect(await bad([], ["--config"])).toContain("--config is required");
    expect(await bad(["--config", "/nonexistent/config.json"], ["--config"])).toContain("cannot read the run config");
    expect(await bad(["--nice", "99"])).toContain("--nice");
    expect(await bad(["--node-heap-mb", "0"])).toContain("--node-heap-mb");
  });
});

describe("the real machine", () => {
  it("checks the real conditions of a non-idle run (memory only) and measures a command on the real /proc", async () => {
    const out: string[] = [];
    const err: string[] = [];
    const io = { out: (line: string) => out.push(line), err: (line: string) => err.push(line), repoRoot: REPO };
    expect(await runThroughputCli(["guard", "--mode", "non-idle", "--min-mem-gib", "1", "--start-window-ms", "20"], io)).toBe(0);
    expect(out.join("\n")).toContain("non-idle: every condition holds");
    const dir = tempDir();
    const report = join(dir, "r.json");
    const code = await runThroughputCli(["run", "--arm", "command", "--out", join(dir, "o"), "--run-id", "real", "--mode", "non-idle", "--min-mem-gib", "1", "--arm-id", "fake", "--arm-label", "Fake", "--arm-report", report, "--proc-interval-ms", "20", "--condition-interval-ms", "100", "--start-window-ms", "20", "--", process.execPath, FAKE_ARM, "--children", "1", "--burn-ms", "60", "--hold-ms", "150", "--decisions", "9", "--games", "3", "--report", report], io);
    expect(err.join("\n")).toBe("");
    expect(code).toBe(0);
    const record = readThroughputRecord(JSON.parse(readFileSync(join(dir, "o", "real-r1", "record.json"), "utf8")) as unknown);
    expect(record.valid).toBe(true);
    expect(record.cpu.total_ms).toBeGreaterThanOrEqual(2 * 60 - 20); // two processes burnt 60 ms each
    expect(record.code.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(record.code.engine_pin).toMatch(/^[0-9a-f]{40}$/);
    expect(record.machine.logical_cpus).toBeGreaterThan(0);
  });

  it("lets an error that is not the caller's mistake propagate", async () => {
    const io = { out: () => undefined, err: () => undefined, repoRoot: REPO, deps: () => { throw new Error("boom"); } };
    await expect(runThroughputCli(["guard"], io)).rejects.toThrow("boom");
  });
});

describe("verify and summarise", () => {
  it("reads records and prints their hashes, exit 0, or refuses a record and exits 1", async () => {
    const good = await cli(["verify", join(VALID, "showdown-idle-4-processes.json"), join(VALID, "gpu-arm-with-steady-rate.json")]);
    expect(good.code).toBe(0);
    expect(good.out.match(/^[0-9a-f]{64} /gm)?.length).toBe(2);
    const bad = await cli(["verify", join(VALID, "../reader-invalid/rate-not-recomputed.json")]);
    expect(bad.code).toBe(1);
    expect(bad.err).toContain("rate-not-recomputed.json");
    expect((await cli(["verify", "/nonexistent/record.json"])).code).toBe(1);
    expect((await cli(["verify"])).code).toBe(2);
  });

  it("summarises records given as files or as a directory of run directories", async () => {
    const dir = tempDir();
    for (const [name, file] of [["a", "showdown-idle-4-processes.json"], ["b", "non-idle-void-memory.json"]] as const) {
      mkdirSync(join(dir, name));
      writeFileSync(join(dir, name, "record.json"), readFileSync(join(VALID, file)));
    }
    const fromDir = await cli(["summarise", dir]);
    expect(fromDir.code).toBe(0);
    expect(fromDir.out).toContain("# Throughput comparison");
    expect(fromDir.out).toContain("void runs");
    const fromFiles = await cli(["summarise", join(dir, "a", "record.json")]);
    expect(fromFiles.out).toContain("| showdown game_processes=4,policy_workers=1 | 1 of 1 |");
    expect((await cli(["summarise"])).code).toBe(2);
    expect((await cli(["summarise", "/nonexistent"])).code).toBe(1);
  });
});
