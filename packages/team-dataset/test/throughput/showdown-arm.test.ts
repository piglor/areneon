// The Showdown arm of the throughput harness: the play-generation command it runs (as the generation loop runs it) and the
// ArmReport it makes from that script's output and the run record the script leaves.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { readArmReport } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { classifyShowdown, parsePlayOutput, poolOf, resolveIds, showdownArm, type ShowdownArmOptions } from "../../src/throughput/showdown-arm.ts";
import { tempDir } from "./support.ts";

const LEAGUE = new URL("../../../../fixtures/contracts/v1/league-run-record/valid/tiny.json", import.meta.url);
const league = JSON.parse(readFileSync(LEAGUE, "utf8")) as Record<string, unknown>;

const options = (dir: string, extra: Partial<ShowdownArmOptions> = {}): ShowdownArmOptions => ({
  repoRoot: "/repo",
  node: "/usr/bin/node",
  nice: null,
  config: "/runs/config.json",
  runDir: "/runs/lr1",
  preview: "p".repeat(64),
  battle: "b".repeat(64),
  policyId: league.pair_policy_id as string,
  encoderVersion: "enc1:abc",
  engineData: "/repo/fixtures/engine-data.json",
  workerPython: "/repo/.venv/bin/python",
  generation: 1,
  gameProcesses: 4,
  policyWorkers: 1,
  nodeHeapMb: null,
  poolId: "train-v1",
  poolHash: "7".repeat(64),
  outDir: join(dir, "arm"),
  playScript: "packages/team-dataset/scripts/play-generation.ts",
  ...extra,
});

const profile = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({
  profile: {
    decisions: 1900,
    game_processes: { processes: 4, cpu_ms: 100_000, e_sim_ms: 21_000, e_judge_ms: 9000, e_probe_ms: 52_000, o_obs_ms: 6000, ipc_wait_ms: 90_000, pool_load_ms: 136_000, merge_ms: 12_000, per_process: [] },
    policy_workers: { workers: 4, queries: 1230, cpu_ms: 30_000, w_enc_ms: 7000, w_fwd_ms: 11_000, w_enc_cpu_ms: 6500, w_fwd_cpu_ms: 10_500, per_process: [] },
    ...extra,
  },
});
const stdoutOf = (lines: unknown[]): string => `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`;
const summaryLine = { generation: 1, run_config_hash: league.run_config_hash, training_set_hash: "d".repeat(64), examples: 1234 };

/** A finished play run on disk: the arm's stdout and the league-run.json the script leaves in its --out directory. */
function finished(dir: string, stdout: string, record: unknown = league): { runDir: string; stdoutPath: string; stderrPath: string } {
  mkdirSync(join(dir, "arm"), { recursive: true });
  writeFileSync(join(dir, "arm", "league-run.json"), JSON.stringify(record));
  writeFileSync(join(dir, "arm.out"), stdout);
  return { runDir: dir, stdoutPath: join(dir, "arm.out"), stderrPath: join(dir, "arm.err") };
}

describe("showdownArm: the command", () => {
  it("runs play-generation.ts with the flags the generation loop gives it, plus --profile, from the repository root", () => {
    const arm = showdownArm(options("/x"));
    expect(arm.command).toBe("/usr/bin/node");
    expect(arm.cwd).toBe("/repo");
    expect(arm.args).toEqual([
      "/repo/packages/team-dataset/scripts/play-generation.ts",
      "--config", "/runs/config.json",
      "--generation", "1",
      "--policy-id", league.pair_policy_id,
      "--preview", "p".repeat(64),
      "--battle", "b".repeat(64),
      "--run-dir", "/runs/lr1",
      "--engine-data", "/repo/fixtures/engine-data.json",
      "--encoder-version", "enc1:abc",
      "--worker-python", "/repo/.venv/bin/python",
      "--out", "/x/arm",
      "--policy-workers", "1",
      "--game-processes", "4",
      "--profile",
    ]);
  });

  it("wraps the command in nice when asked and caps the V8 heap when told", () => {
    const arm = showdownArm(options("/x", { nice: 19, nodeHeapMb: 640, gameProcesses: 6, policyWorkers: 2 }));
    expect(arm.command).toBe("nice");
    expect(arm.args.slice(0, 5)).toEqual(["-n", "19", "/usr/bin/node", "--max-old-space-size=640", "/repo/packages/team-dataset/scripts/play-generation.ts"]);
    expect(arm.args).toContain("6");
    expect(arm.args.slice(arm.args.indexOf("--policy-workers"), arm.args.indexOf("--policy-workers") + 4)).toEqual(["--policy-workers", "2", "--game-processes", "6"]);
  });

  it("describes the arm by its knobs (sorted) and the plan by its pool, with no GPU", () => {
    const { description } = showdownArm(options("/x"));
    expect(description.arm).toEqual({ id: "showdown", label: "Showdown path: play-generation.ts, 4 game processes x 1 policy worker", parameters: [{ name: "game_processes", value: 4 }, { name: "generation", value: 1 }, { name: "policy_workers", value: 1 }] });
    expect(description.plan).toEqual({ pool_id: "train-v1", pool_hash: "7".repeat(64), plan_hash: null, config_hash: null });
    expect(description.gpu_used).toBe(false);
    expect(showdownArm(options("/x", { nodeHeapMb: 640 })).description.arm.parameters.map((p) => p.name)).toEqual(["game_processes", "generation", "node_heap_mb", "policy_workers"]);
  });

  it("names the processes of the play: game processes, policy workers, others by program", () => {
    expect(classifyShowdown(["node", "/repo/packages/team-dataset/scripts/play-generation.ts", "--shard", "1/4"], "MainThread")).toBe("game_process");
    expect(classifyShowdown(["/repo/.venv/bin/python", "-I", "-m", "areneon_ml.worker", "--run-dir", "x"], "python")).toBe("policy_worker");
    expect(classifyShowdown(["nice", "-n", "19"], "nice")).toBe("nice");
  });
});

describe("parsePlayOutput", () => {
  it("finds the summary, the profile and ignores lines that are not JSON", () => {
    const parsed = parsePlayOutput(`${stdoutOf([summaryLine, { resource_usage: { game_process: 1 } }, profile()])}not json at all\n`);
    expect(parsed.summary.examples).toBe(1234);
    expect(parsed.profile?.decisions).toBe(1900);
  });

  it("refuses output with no summary line, and takes a missing profile as none", () => {
    expect(() => parsePlayOutput(stdoutOf([{ profile: {} }]))).toThrow("printed no summary line");
    expect(parsePlayOutput(stdoutOf([summaryLine])).profile).toBeUndefined();
  });

  it("refuses a profile line of another shape", () => {
    expect(() => parsePlayOutput(stdoutOf([summaryLine, { profile: { decisions: "many" } }]))).toThrow("profile line");
  });
});

describe("showdownArm: the report", () => {
  it("counts games from the run record and decisions from the training set, and states the phases and stages from the profile", async () => {
    const dir = tempDir();
    const context = finished(dir, stdoutOf([summaryLine, profile()]));
    const report = await showdownArm(options(dir)).collect({ ...context, exitCode: 0, wallMs: 326_000 });
    expect(readArmReport(JSON.parse(JSON.stringify(report)))).toEqual(report); // a valid ArmReport
    expect(report.counts).toEqual({ games: league.games, games_forfeited: league.forfeits, games_on_showdown: league.games, decisions: 1234, scripted_decisions: 666 });
    expect([league.games, league.forfeits]).toEqual([6, 1]); // the sample record the test reads: 6 games, 1 of them a forfeit
    expect(report.plan).toEqual({ pool_id: "train-v1", pool_hash: "7".repeat(64), plan_hash: league.plan_hash, config_hash: league.run_config_hash });
    expect(report.phases).toEqual([
      { name: "pool_load", kind: "warmup", wall_ms: 34_000, cpu_ms: null },
      { name: "play", kind: "play", wall_ms: 326_000 - 34_000 - 12_000, cpu_ms: null },
      { name: "merge", kind: "teardown", wall_ms: 12_000, cpu_ms: null },
    ]);
    expect(report.stages).toEqual([
      { name: "engine_step", ms: 21_000 },
      { name: "features", ms: 52_000 },
      { name: "legal_mask", ms: 9000 },
      { name: "observation", ms: 6000 },
      { name: "policy_encode", ms: 7000 },
      { name: "policy_forward", ms: 11_000 },
    ]);
  });

  it("states no phases, stages or scripted decisions when the script printed no profile", async () => {
    const dir = tempDir();
    const report = await showdownArm(options(dir)).collect({ ...finished(dir, stdoutOf([summaryLine])), exitCode: 0, wallMs: 1000 });
    expect(report.phases).toEqual([]);
    expect(report.stages).toEqual([]);
    expect(report.counts.scripted_decisions).toBe(0);
  });

  it("refuses output that does not belong to the run it was given", async () => {
    const dir = tempDir();
    const arm = showdownArm(options(dir));
    await expect(arm.collect({ ...finished(dir, stdoutOf([{ ...summaryLine, run_config_hash: "e".repeat(64) }, profile()])), exitCode: 0, wallMs: 1 })).rejects.toThrow("run config hash");
    await expect(showdownArm(options(dir, { policyId: `policy:${"9".repeat(64)}` })).collect({ ...finished(dir, stdoutOf([summaryLine, profile()])), exitCode: 0, wallMs: 1 })).rejects.toThrow("policy id");
    await expect(arm.collect({ ...finished(dir, stdoutOf([summaryLine, profile()]), { not: "a league run record" }), exitCode: 0, wallMs: 1 })).rejects.toThrow();
  });

  it("refuses a play that left no run record", async () => {
    const dir = tempDir();
    writeFileSync(join(dir, "arm.out"), stdoutOf([summaryLine]));
    await expect(showdownArm(options(dir)).collect({ runDir: dir, stdoutPath: join(dir, "arm.out"), stderrPath: join(dir, "arm.err"), exitCode: 0, wallMs: 1 })).rejects.toThrow("league-run.json");
  });
});

describe("poolOf", () => {
  it("names the pool by the team dataset the run config points at, or says it is the interim pool", () => {
    expect(poolOf({ team_dataset: { path: "/data/teams/train-v1", dataset_hash: "a".repeat(64) } })).toEqual({ poolId: "train-v1", poolHash: "a".repeat(64) });
    expect(poolOf({})).toEqual({ poolId: "interim", poolHash: null });
    expect(poolOf({ team_dataset: { path: "/data/ with space/Train v1/", dataset_hash: "a".repeat(64) } }).poolId).toBe("Train-v1");
  });
});

describe("resolveIds", () => {
  it("asks the Python package for the policy id and the encoder version, with the worktree's code first on the path", () => {
    const calls: { command: string; args: readonly string[]; env: Record<string, string> }[] = [];
    const ids = resolveIds({ python: "/venv/bin/python", repoRoot: "/repo", config: "/c.json", engineData: "/e.json", preview: "p", battle: "b" }, (command, args, env) => {
      calls.push({ command, args, env });
      return `${JSON.stringify({ policy_id: "policy:x", encoder_version: "enc1:y" })}\n`;
    });
    expect(ids).toEqual({ policyId: "policy:x", encoderVersion: "enc1:y" });
    expect(calls[0]).toEqual({
      command: "/venv/bin/python",
      args: ["-s", "-P", "-m", "areneon_ml.loop.play_ids", "--config", "/c.json", "--engine-data", "/e.json", "--preview", "p", "--battle", "b"],
      env: { PYTHONPATH: "/repo/ml:/repo/packages/schema/python:/repo" },
    });
  });

  it("refuses an answer that is not the two ids", () => {
    const args = { python: "py", repoRoot: "/r", config: "c", engineData: "e", preview: "p", battle: "b" };
    expect(() => resolveIds(args, () => "garbage")).toThrow("policy ids");
    expect(() => resolveIds(args, () => JSON.stringify({ policy_id: "x" }))).toThrow("policy ids");
  });
});
