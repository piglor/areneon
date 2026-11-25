// The idle-machine guard of the throughput harness (ADR-028 section 1): the conditions are pure functions of one machine sample,
// so every boundary is tested exactly. A run that breaks an enforced condition is void.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { BUSY_PATTERNS, checkRun, checkStart, findOtherRuns, GIB_KIB, isBusyCommand, MachineSampler, NOT_ENFORCED, thresholdsFor, type Sample } from "../../src/throughput/guard.ts";
import { Proc } from "../../src/throughput/procfs.ts";

const sample = (extra: Partial<Sample> = {}): Sample => ({
  at_epoch_ms: 1_700_000_000_000,
  load1_centi: 40,
  load5_centi: 90,
  mem_available_kib: 20 * GIB_KIB,
  swap_used_kib: 0,
  cpu_busy_ppm: 20_000,
  gpu_util_pct: 0,
  gpu_mem_used_mib: 300,
  ...extra,
});

describe("thresholdsFor", () => {
  it("takes the ADR-028 section 1 values for an idle run", () => {
    expect(thresholdsFor("idle")).toEqual({
      max_load1_centi: 100, // load average at most 1.0 before the start
      max_cpu_busy_ppm: 70_000, // at most about two of 28 threads busy
      min_mem_available_start_kib: 8 * GIB_KIB,
      min_mem_available_run_kib: 4 * GIB_KIB,
      max_swap_growth_kib: 0,
      forbid_other_runs: true,
      require_idle_gpu: true,
      max_gpu_util_pct: 5,
      max_gpu_mem_used_mib: 2000,
    });
  });

  it("enforces only the memory floors in a labelled non-idle run, and lets the caller raise or lower the numbers", () => {
    const loose = thresholdsFor("non-idle");
    expect(loose).toMatchObject({ forbid_other_runs: false, require_idle_gpu: false, min_mem_available_start_kib: 8 * GIB_KIB, min_mem_available_run_kib: 4 * GIB_KIB, max_swap_growth_kib: NOT_ENFORCED, max_load1_centi: NOT_ENFORCED });
    expect(thresholdsFor("non-idle", { min_mem_available_start_kib: 10 * GIB_KIB }).min_mem_available_start_kib).toBe(10 * GIB_KIB);
  });
});

describe("checkStart (before the arm starts)", () => {
  const idle = thresholdsFor("idle");

  it("passes a quiet machine", () => {
    expect(checkStart(sample(), [], idle)).toEqual([]);
  });

  it("refuses a load average above 1.00 and accepts exactly 1.00", () => {
    expect(checkStart(sample({ load1_centi: 100 }), [], idle)).toEqual([]);
    expect(checkStart(sample({ load1_centi: 101 }), [], idle).map((f) => f.code)).toEqual(["load"]);
  });

  it("refuses MemAvailable below 8 GiB and accepts exactly 8 GiB", () => {
    expect(checkStart(sample({ mem_available_kib: 8 * GIB_KIB }), [], idle)).toEqual([]);
    expect(checkStart(sample({ mem_available_kib: 8 * GIB_KIB - 1 }), [], idle).map((f) => f.code)).toEqual(["memory_start"]);
  });

  it("refuses a busy CPU (above 7%) and accepts exactly 7%", () => {
    expect(checkStart(sample({ cpu_busy_ppm: 70_000 }), [], idle)).toEqual([]);
    expect(checkStart(sample({ cpu_busy_ppm: 70_001 }), [], idle).map((f) => f.code)).toEqual(["cpu_busy"]);
  });

  it("refuses when another run, training, evaluation or benchmark is going, naming the command", () => {
    const failures = checkStart(sample(), ["node packages/team-dataset/scripts/play-generation.ts --config x"], idle);
    expect(failures).toEqual([{ code: "other_run", detail: "another run is going: node packages/team-dataset/scripts/play-generation.ts --config x" }]);
  });

  it("refuses a busy GPU (above 5% or 2000 MiB), and a machine without a card passes", () => {
    expect(checkStart(sample({ gpu_util_pct: 5, gpu_mem_used_mib: 2000 }), [], idle)).toEqual([]);
    expect(checkStart(sample({ gpu_util_pct: 6 }), [], idle).map((f) => f.code)).toEqual(["gpu_busy"]);
    expect(checkStart(sample({ gpu_mem_used_mib: 2001 }), [], idle).map((f) => f.code)).toEqual(["gpu_busy"]);
    expect(checkStart(sample({ gpu_util_pct: null, gpu_mem_used_mib: null }), [], idle)).toEqual([]);
  });

  it("lists every failure at once", () => {
    const failures = checkStart(sample({ load1_centi: 500, mem_available_kib: GIB_KIB, cpu_busy_ppm: 500_000, gpu_util_pct: 90 }), ["pytest tests"], idle);
    expect(failures.map((f) => f.code).sort()).toEqual(["cpu_busy", "gpu_busy", "load", "memory_start", "other_run"]);
  });

  it("enforces only memory in non-idle mode", () => {
    const loose = thresholdsFor("non-idle");
    expect(checkStart(sample({ load1_centi: 1500, cpu_busy_ppm: 900_000, gpu_util_pct: 99 }), ["play-generation.ts"], loose)).toEqual([]);
    expect(checkStart(sample({ mem_available_kib: 8 * GIB_KIB - 1 }), [], loose).map((f) => f.code)).toEqual(["memory_start"]);
  });
});

describe("checkRun (at each sample during the arm)", () => {
  const idle = thresholdsFor("idle");
  const start = sample({ swap_used_kib: 1000 });

  it("passes a steady machine", () => {
    expect(checkRun(sample({ swap_used_kib: 1000 }), start, [], idle)).toEqual([]);
  });

  it("breaks the memory floor below 4 GiB and holds at exactly 4 GiB", () => {
    expect(checkRun(sample({ mem_available_kib: 4 * GIB_KIB, swap_used_kib: 1000 }), start, [], idle)).toEqual([]);
    expect(checkRun(sample({ mem_available_kib: 4 * GIB_KIB - 1, swap_used_kib: 1000 }), start, [], idle)).toEqual(["memory_run"]);
  });

  it("breaks on any swap growth over the start and allows the configured growth", () => {
    expect(checkRun(sample({ swap_used_kib: 1001 }), start, [], idle)).toEqual(["swap_growth"]);
    expect(checkRun(sample({ swap_used_kib: 900 }), start, [], idle)).toEqual([]);
    expect(checkRun(sample({ swap_used_kib: 1100 }), start, [], thresholdsFor("idle", { max_swap_growth_kib: 100 }))).toEqual([]);
  });

  it("breaks when another run starts, only in a mode that forbids it", () => {
    expect(checkRun(sample({ swap_used_kib: 1000 }), start, ["vitest run"], idle)).toEqual(["other_run"]);
    expect(checkRun(sample({ swap_used_kib: 1000 }), start, ["vitest run"], thresholdsFor("non-idle"))).toEqual([]);
  });

  it("does not breach on load: the arm itself loads the machine", () => {
    expect(checkRun(sample({ load1_centi: 2800, swap_used_kib: 1000 }), start, [], idle)).toEqual([]);
  });
});

describe("isBusyCommand", () => {
  it("matches the tools that run, train, evaluate or benchmark, and the owner's extra patterns", () => {
    for (const cmd of [
      ["python", "-m", "areneon_ml.loop", "run"],
      ["python", "-I", "-m", "areneon_ml.worker"],
      ["node", "packages/team-dataset/scripts/play-generation.ts"],
      ["node", "packages/showdown-bridge/scripts/feature-bench.ts"],
      ["node", "packages/team-dataset/scripts/scripted-h2h.ts"],
      ["node", "node_modules/vitest/vitest.mjs", "run"],
      ["/usr/bin/python3", "-m", "pytest"],
    ]) expect(isBusyCommand(cmd, BUSY_PATTERNS), cmd.join(" ")).toBe(true);
    expect(isBusyCommand(["bash", "queue-lr1.sh"], BUSY_PATTERNS)).toBe(false);
    expect(isBusyCommand(["bash", "queue-lr1.sh"], [...BUSY_PATTERNS, "queue-lr1.sh"])).toBe(true);
    expect(isBusyCommand(["vim", "notes.txt"], BUSY_PATTERNS)).toBe(false);
    expect(isBusyCommand([], BUSY_PATTERNS)).toBe(false);
  });
});

describe("over a fake /proc", () => {
  let root: string;
  const writeProc = (pid: number, ppid: number, args: string[], pgrp = pid): void => {
    mkdirSync(join(root, String(pid)), { recursive: true });
    const fields = ["S", ppid, pgrp, pgrp, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 1, 0, 100, 1000, 10, ...Array.from({ length: 28 }, () => 0)];
    writeFileSync(join(root, String(pid), "stat"), `${pid} (x) ${fields.join(" ")}\n`);
    writeFileSync(join(root, String(pid), "cmdline"), `${args.join("\0")}\0`);
  };
  const writeMachine = (stat: string, mem = "MemTotal: 16777216 kB\nMemAvailable: 12582912 kB\nSwapTotal: 4096 kB\nSwapFree: 1096 kB\n"): void => {
    writeFileSync(join(root, "loadavg"), "0.35 0.80 1.00 1/10 5\n");
    writeFileSync(join(root, "meminfo"), mem);
    writeFileSync(join(root, "stat"), stat);
  };
  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), "guard-proc-"));
    writeProc(100, 1, ["bash", "harness.sh"]); // the harness's parent
    writeProc(200, 100, ["node", "throughput-bench.ts", "run"]); // the harness (pid 200 below)
    writeProc(300, 1, ["node", "packages/team-dataset/scripts/play-generation.ts"]); // someone else's run
    writeProc(400, 1, ["node", "play-generation.ts"], 777); // the arm's own: its group is 777
    writeProc(500, 1, ["vim", "notes"]);
  });
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("finds other runs, leaving out the harness, its ancestors and the arm's own process group", () => {
    const proc = new Proc(root);
    expect(findOtherRuns(proc, BUSY_PATTERNS, { exclude: new Set([100, 200]), group: 777 })).toEqual(["node packages/team-dataset/scripts/play-generation.ts"]);
    expect(findOtherRuns(proc, BUSY_PATTERNS, { exclude: new Set([100, 200, 300]), group: 777 })).toEqual([]);
    // Before an arm exists there is no group: the arm's look-alike counts as another run.
    expect(findOtherRuns(proc, BUSY_PATTERNS, { exclude: new Set([100, 200, 300]), group: null })).toEqual(["node play-generation.ts"]);
  });

  it("samples load, memory and swap, and the CPU busy share between two samples", () => {
    writeMachine("cpu  100 0 100 800 0 0 0 0 0 0\n");
    const sampler = new MachineSampler(new Proc(root), () => 1_700_000_000_000);
    const first = sampler.sample(null);
    expect(first).toEqual({ at_epoch_ms: 1_700_000_000_000, load1_centi: 35, load5_centi: 80, mem_available_kib: 12_582_912, swap_used_kib: 3000, cpu_busy_ppm: null, gpu_util_pct: null, gpu_mem_used_mib: null });
    writeMachine("cpu  150 0 150 900 0 0 0 0 0 0\n"); // 100 busy over 200 total: half
    expect(sampler.sample({ util: 37, memMib: 1200 }).cpu_busy_ppm).toBe(500_000);
    expect(sampler.sample({ util: 37, memMib: 1200 })).toMatchObject({ gpu_util_pct: 37, gpu_mem_used_mib: 1200, cpu_busy_ppm: 0 }); // no ticks since the last sample: nothing busy
  });

  it("measures the busy share over a window of its own for the start sample", async () => {
    writeMachine("cpu  100 0 100 800 0 0 0 0 0 0\n");
    const sampler = new MachineSampler(new Proc(root), () => 1);
    const wait = (): Promise<void> => {
      writeMachine("cpu  100 0 100 1000 0 0 0 0 0 0\n"); // 200 idle ticks and nothing busy while the window runs
      return Promise.resolve();
    };
    expect((await sampler.sampleOverWindow(null, 50, wait)).cpu_busy_ppm).toBe(0);
  });
});
