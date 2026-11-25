// The throughput harness's runner against a fake arm: exact counts, CPU accounting of the whole process tree, the idle guard's
// refusals and breaches, aborts, survivors and the GPU figures. The arm is a real process tree (fake-arm.ts) that burns known CPU.
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ratesOf, readThroughputRecord, throughputRecordHash } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { commandArm, defaultClassifier, emptyReport } from "../../src/throughput/arms.ts";
import { GIB_KIB, thresholdsFor } from "../../src/throughput/guard.ts";
import { Proc } from "../../src/throughput/procfs.ts";
import { ConditionsNotMet, defaultRunOptions, runArm } from "../../src/throughput/runner.ts";
import { FAKE_ARM, fakeDeps, tempDir } from "./support.ts";

const description = { arm: { id: "fake", label: "Fake arm", parameters: [] }, plan: { pool_id: "fake-pool", pool_hash: null, plan_hash: null, config_hash: null }, gpu_used: false };
const setup = (flags: string[], options: Parameters<typeof defaultRunOptions>[0] = {}, withReport = true) => {
  const dir = tempDir();
  const report = join(dir, "arm-report.json");
  const arm = commandArm({ command: process.execPath, args: [FAKE_ARM, ...flags, ...(withReport ? ["--report", report] : [])], cwd: dir, reportPath: report, description });
  return { dir, arm, options: defaultRunOptions({ runId: "t1", outDir: join(dir, "run"), procIntervalMs: 50, conditionIntervalMs: 150, startWindowMs: 5, gpuIntervalMs: 50, killGraceMs: 500, ...options }) };
};
const isGone = (pid: number): boolean => {
  const stat = new Proc().stat(pid);
  return stat === null || stat.state === "Z";
};

describe("runArm on a fake arm", () => {
  it("counts the games and decisions exactly and the CPU of every process of the tree", async () => {
    const { dir, arm, options } = setup(["--children", "2", "--burn-ms", "200", "--hold-ms", "500", "--games", "12", "--decisions", "345"]);
    const result = await runArm(arm, options, fakeDeps());
    const { record } = result;
    expect(record.valid).toBe(true);
    expect(record.void_reasons).toEqual([]);
    expect(record.report.counts).toEqual({ games: 12, games_forfeited: 0, games_on_showdown: 0, decisions: 345, scripted_decisions: 0 });
    // Three processes each burnt 200 ms of CPU; the kernel counts in 10 ms ticks and a start-up costs a little more.
    expect(record.cpu.total_ms).toBeGreaterThanOrEqual(3 * 200 - 30);
    expect(record.cpu.total_ms).toBeLessThan(3 * 200 + 3 * 600);
    expect(record.cpu.total_ms).toBe(record.cpu.user_ms + record.cpu.system_ms);
    expect(record.cpu.processes_seen).toBeGreaterThanOrEqual(3);
    expect(record.cpu.survivors).toBe(0);
    expect(record.cpu.by_role.map((r) => r.role)).toContain("node");
    expect(record.window.wall_ms).toBeGreaterThanOrEqual(500);
    expect(record.window).toMatchObject({ exit_code: 0, aborted: false });
    expect(record.rates).toEqual(ratesOf(record));
    expect(record.rates.decisions_per_core_hour).toBe(Math.floor((345 * 3_600_000) / record.cpu.total_ms));
    expect(record.rates.decisions_per_gpu_hour).toBeNull();
    expect(record.code).toEqual({ commit: "a".repeat(40), engine_pin: "b".repeat(40), tree_clean: true });
    expect(record.machine).toEqual({ logical_cpus: 4, mem_total_kib: 16 * GIB_KIB, cpu_model: "Fake CPU" });
    expect(record.conditions.mode).toBe("idle");
    expect(record.conditions.samples).toBeGreaterThanOrEqual(3); // start, at least one minute sample, end
    expect(record.cpu.harness_ms).toBeGreaterThanOrEqual(0);
    // The files: the record (the one the reader accepts), the journal and the summary.
    const written = JSON.parse(readFileSync(join(dir, "run", "record.json"), "utf8")) as unknown;
    expect(readThroughputRecord(written)).toEqual(record);
    expect(result.hash).toBe(throughputRecordHash(record));
    for (const name of ["summary.md", "conditions.jsonl", "procs.jsonl", "command.txt", "arm.out", "arm.err"]) expect(existsSync(join(dir, "run", name)), name).toBe(true);
    expect(readFileSync(join(dir, "run", "summary.md"), "utf8")).toContain("t1");
    expect(readFileSync(join(dir, "run", "conditions.jsonl"), "utf8").trim().split("\n").length).toBe(record.conditions.samples);
  });

  it("is void when the arm fails, and keeps the report it left", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--decisions", "5", "--games", "1", "--exit", "3"]);
    const { record } = await runArm(arm, options, fakeDeps());
    expect(record.valid).toBe(false);
    expect(record.void_reasons).toEqual(["arm_failed"]);
    expect(record.window.exit_code).toBe(3);
    expect(record.report.counts.decisions).toBe(5);
  });

  it("is void with an empty report when the arm leaves none", async () => {
    const { arm, options } = setup(["--burn-ms", "10"], {}, false);
    const result = await runArm(arm, options, fakeDeps());
    expect(result.record.valid).toBe(false);
    expect(result.record.void_reasons).toEqual(["no_decisions"]);
    expect(result.record.report).toEqual(emptyReport(description));
    expect(result.collectError).toMatch(/arm report/);
  });

  it("is void when the arm leaves a process behind, and ends that process", async () => {
    const file = join(tempDir(), "pid");
    const { arm, options } = setup(["--burn-ms", "10", "--survivor", "--survivor-file", file, "--decisions", "1"]);
    const result = await runArm(arm, options, fakeDeps());
    expect(result.record.cpu.survivors).toBeGreaterThanOrEqual(1);
    expect(result.record.void_reasons).toContain("survivors");
    const pid = Number(readFileSync(file, "utf8"));
    for (let i = 0; i < 40 && !isGone(pid); i++) await new Promise((resolve) => setTimeout(resolve, 50));
    expect(isGone(pid)).toBe(true);
  });

  it("refuses to start on a busy machine and writes nothing", async () => {
    const { dir, arm, options } = setup(["--burn-ms", "10"]);
    const deps = fakeDeps({ samples: (n) => (n === 0 ? { load1_centi: 500, mem_available_kib: GIB_KIB } : {}), others: (n) => (n === 0 ? ["pytest tests"] : []) });
    const error = await runArm(arm, options, deps).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ConditionsNotMet);
    expect((error as ConditionsNotMet).failures.map((f) => f.code).sort()).toEqual(["load", "memory_start", "other_run"]);
    expect(existsSync(join(dir, "run"))).toBe(false);
  });

  it("refuses an output directory that already exists", async () => {
    const { dir, arm, options } = setup(["--burn-ms", "10"]);
    await runArm(arm, options, fakeDeps());
    await expect(runArm(commandArm({ command: process.execPath, args: [FAKE_ARM], cwd: dir, reportPath: join(dir, "r.json"), description }), options, fakeDeps())).rejects.toThrow("already exists");
  });

  it("aborts the arm when the memory floor breaks, and the run is void", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "20000", "--decisions", "1"]);
    const started = performance.now();
    const { record } = await runArm(arm, options, fakeDeps({ samples: (n) => (n >= 2 ? { mem_available_kib: 3 * GIB_KIB } : {}) }));
    expect(performance.now() - started).toBeLessThan(10_000);
    expect(record.window.aborted).toBe(true);
    expect(record.conditions.breaches[0]?.code).toBe("memory_run");
    expect(record.void_reasons).toEqual(expect.arrayContaining(["aborted", "memory_run", "arm_failed"]));
    expect(record.conditions.min_mem_available_kib).toBe(3 * GIB_KIB);
  });

  it("catches a memory floor within a walk of the process tree, not at the next minute sample", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "20000", "--decisions", "1"], { conditionIntervalMs: 60_000 });
    const started = performance.now();
    const { record } = await runArm(arm, options, fakeDeps({ available: (n) => (n >= 2 ? 3 * GIB_KIB : 20 * GIB_KIB), samples: (n) => (n >= 1 ? { mem_available_kib: 3 * GIB_KIB } : {}) }));
    expect(performance.now() - started).toBeLessThan(10_000);
    expect(record.window.aborted).toBe(true);
    expect(record.void_reasons).toEqual(expect.arrayContaining(["aborted", "memory_run"]));
    expect(record.conditions.samples).toBe(3); // start, the sample the quick reading brought forward, end
  });

  it("only records the breach when told not to abort, and the arm runs to its end", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "600", "--decisions", "1"], { abortOnMemory: false });
    const { record } = await runArm(arm, options, fakeDeps({ samples: (n) => (n >= 2 ? { mem_available_kib: 3 * GIB_KIB } : {}) }));
    expect(record.window).toMatchObject({ aborted: false, exit_code: 0 });
    expect(record.void_reasons).toEqual(["memory_run"]);
  });

  it("is void when another run starts during the arm, and records the first breach of each kind once", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "700", "--decisions", "1"]);
    const { record } = await runArm(arm, options, fakeDeps({ others: (n) => (n >= 1 ? ["python -m pytest"] : []) }));
    expect(record.void_reasons).toEqual(["other_run"]);
    expect(record.conditions.breaches).toHaveLength(1);
    expect(record.window.aborted).toBe(false);
  });

  it("runs on a busy machine in non-idle mode and says so", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--decisions", "1"], { mode: "non-idle", thresholds: thresholdsFor("non-idle") });
    const deps = fakeDeps({ samples: (n) => (n === 0 ? { load1_centi: 1500, cpu_busy_ppm: 900_000 } : {}), others: () => ["play-generation.ts"] });
    const { record } = await runArm(arm, options, deps);
    expect(record.conditions.mode).toBe("non-idle");
    expect(record.valid).toBe(true);
    expect(record.conditions.start.load1_centi).toBe(1500);
  });

  it("states the GPU figures from the monitor's readings, the helper CPU apart from the arm's, and decisions per GPU-hour", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "300", "--decisions", "100", "--gpu"]);
    const gpuArm = { ...arm, description: { ...description, gpu_used: true } };
    const deps = fakeDeps({ gpuOnce: { util: 0, memMib: 300 }, gpuReadings: [{ util: 40, memMib: 1000 }, { util: 60, memMib: 2000 }] });
    const { record } = await runArm(gpuArm, options, deps);
    expect(record.gpu).toEqual({ used: true, available: true, allocated_ms: record.window.wall_ms, util_mean_ppm: 500_000, mem_peak_mib: 2000, samples: 2 });
    expect(record.cpu.helper_ms).toBe(70); // 7 ticks of the monitor, never part of the arm's total
    expect(record.rates.decisions_per_gpu_hour).toBe(Math.floor((100 * 3_600_000) / record.window.wall_ms));
    expect(record.conditions.start).toMatchObject({ gpu_util_pct: 0, gpu_mem_used_mib: 300 });
  });
});

describe("runArm: the arm and the harness stopping", () => {
  it("is void with arm_failed when the command cannot be started, and signals nobody", async () => {
    const dir = tempDir();
    const arm = commandArm({ command: "/nonexistent/arm", args: [], cwd: dir, reportPath: join(dir, "r.json"), description });
    const stopped: string[] = [];
    const deps = { ...fakeDeps(), signals: { on: (_name: "SIGINT" | "SIGTERM", handler: () => void) => { handler(); stopped.push("signal"); }, off: () => undefined } };
    const { record } = await runArm(arm, defaultRunOptions({ runId: "nope", outDir: join(dir, "run"), procIntervalMs: 50, conditionIntervalMs: 150, startWindowMs: 5 }), deps);
    expect(stopped.length).toBeGreaterThan(0); // a stop request arrived with no group to stop
    expect(record.void_reasons).toEqual(expect.arrayContaining(["arm_failed", "no_decisions"]));
    expect(record.window).toMatchObject({ exit_code: -1, aborted: false });
    expect(record.cpu.processes_seen).toBe(0);
  });

  it("ends the arm and says so when the harness is told to stop", async () => {
    const { arm, options } = setup(["--burn-ms", "10", "--hold-ms", "20000", "--decisions", "1"]);
    const registered = new Map<string, () => void>();
    const removed: string[] = [];
    const signals = { on: (name: "SIGINT" | "SIGTERM", handler: () => void) => void registered.set(name, handler), off: (name: "SIGINT" | "SIGTERM") => void removed.push(name) };
    const started = performance.now();
    const running = runArm(arm, options, { ...fakeDeps(), signals });
    await new Promise((resolve) => setTimeout(resolve, 400));
    registered.get("SIGTERM")?.();
    const { record } = await running;
    expect(performance.now() - started).toBeLessThan(10_000);
    expect(record.window).toMatchObject({ aborted: true, exit_code: -1 });
    expect(record.void_reasons).toEqual(expect.arrayContaining(["aborted", "arm_failed"]));
    expect(removed.sort()).toEqual(["SIGINT", "SIGTERM"]); // the handlers do not outlive the run
  });

  it("writes the command that ran, quoted so it can be pasted", async () => {
    const dir = tempDir();
    mkdirSync(join(dir, "with space"));
    const report = join(dir, "with space", "arm-report.json");
    const arm = commandArm({ command: process.execPath, args: [FAKE_ARM, "--burn-ms", "5", "--decisions", "1", "--report", report], cwd: dir, reportPath: report, description });
    await runArm(arm, defaultRunOptions({ runId: "q", outDir: join(dir, "run"), procIntervalMs: 50, conditionIntervalMs: 150, startWindowMs: 5 }), fakeDeps());
    expect(readFileSync(join(dir, "run", "command.txt"), "utf8")).toContain(`'${report}'`);
  });
});

describe("arms", () => {
  it("names a process by the program it runs, else by its kernel name, as a token", () => {
    expect(defaultClassifier(["/usr/bin/node", "x"], "MainThread")).toBe("node"); // Node's kernel name is MainThread
    expect(defaultClassifier(["python3", "-m", "x"], "python3")).toBe("python3");
    expect(defaultClassifier([], "Web Content (1)")).toBe("Web-Content--1-");
    expect(defaultClassifier([], "")).toBe("process");
  });

  it("makes an empty report from the arm's description: nothing counted", () => {
    expect(emptyReport(description)).toEqual({ schema_version: 1, ...description, counts: { games: 0, games_forfeited: 0, games_on_showdown: 0, decisions: 0, scripted_decisions: 0 }, phases: [], stages: [] });
  });
});
