// The idle-machine guard of the throughput harness (ADR-028 section 1). A throughput figure is a gate figure only if the machine was
// idle: load average at most 1.0 before the start, no other run, training, evaluation or benchmark, MemAvailable at least 8 GiB at the
// start and at least 4 GiB throughout, no growth of swap in use, readings every minute. A run that breaks an enforced condition is void.
//
// The conditions are pure functions of one machine sample (`checkStart`, `checkRun`); the thresholds say what is enforced, so a labelled
// non-idle run (a relative measurement beside other work) enforces only the memory floors and records the rest.
import type { ThroughputRecord } from "@areneon/showdown-bridge";
import { type Proc } from "./procfs.ts";

export type Sample = ThroughputRecord["conditions"]["start"];
export type Thresholds = ThroughputRecord["conditions"]["thresholds"];
export type BreachCode = ThroughputRecord["conditions"]["breaches"][number]["code"];
export type Mode = ThroughputRecord["conditions"]["mode"];

/** One GPU reading: utilisation in percent and memory in MiB. */
export interface GpuReading {
  util: number;
  memMib: number;
}

export const GIB_KIB = 1024 * 1024;
/** A threshold the mode does not enforce: no machine reaches it. */
export const NOT_ENFORCED = Number.MAX_SAFE_INTEGER;

/** The ADR-028 section 1 conditions (idle), or only its memory floors (non-idle: a labelled relative measurement beside other work). */
export function thresholdsFor(mode: Mode, overrides: Partial<Thresholds> = {}): Thresholds {
  const memory = { min_mem_available_start_kib: 8 * GIB_KIB, min_mem_available_run_kib: 4 * GIB_KIB };
  const base: Thresholds =
    mode === "idle"
      ? { max_load1_centi: 100, max_cpu_busy_ppm: 70_000, ...memory, max_swap_growth_kib: 0, forbid_other_runs: true, require_idle_gpu: true, max_gpu_util_pct: 5, max_gpu_mem_used_mib: 2000 }
      : { max_load1_centi: NOT_ENFORCED, max_cpu_busy_ppm: 1_000_000, ...memory, max_swap_growth_kib: NOT_ENFORCED, forbid_other_runs: false, require_idle_gpu: false, max_gpu_util_pct: 100, max_gpu_mem_used_mib: NOT_ENFORCED };
  return { ...base, ...overrides };
}

export interface StartFailure {
  code: "load" | "cpu_busy" | "memory_start" | "other_run" | "gpu_busy";
  detail: string;
}

/** The conditions that must hold before the arm starts; every failure, not just the first. `others` are the commands of other runs found on the machine. */
export function checkStart(sample: Sample, others: readonly string[], thresholds: Thresholds): StartFailure[] {
  const failures: StartFailure[] = [];
  const fail = (code: StartFailure["code"], detail: string): number => failures.push({ code, detail });
  if (sample.load1_centi > thresholds.max_load1_centi) fail("load", `the 1-minute load average is ${sample.load1_centi / 100}, above ${thresholds.max_load1_centi / 100}`);
  if (sample.cpu_busy_ppm !== null && sample.cpu_busy_ppm > thresholds.max_cpu_busy_ppm) fail("cpu_busy", `the CPU is ${sample.cpu_busy_ppm / 10_000}% busy, above ${thresholds.max_cpu_busy_ppm / 10_000}%`);
  if (sample.mem_available_kib < thresholds.min_mem_available_start_kib) fail("memory_start", `MemAvailable is ${(sample.mem_available_kib / GIB_KIB).toFixed(1)} GiB, below ${(thresholds.min_mem_available_start_kib / GIB_KIB).toFixed(1)} GiB`);
  if (thresholds.forbid_other_runs) for (const command of others) fail("other_run", `another run is going: ${command}`);
  if (thresholds.require_idle_gpu && ((sample.gpu_util_pct ?? 0) > thresholds.max_gpu_util_pct || (sample.gpu_mem_used_mib ?? 0) > thresholds.max_gpu_mem_used_mib)) {
    fail("gpu_busy", `the GPU is in use (${sample.gpu_util_pct}% utilised, ${sample.gpu_mem_used_mib} MiB)`);
  }
  return failures;
}

/** The enforced conditions broken at one sample during the run (load is not one: the arm itself loads the machine). */
export function checkRun(sample: Sample, start: Sample, others: readonly string[], thresholds: Thresholds): BreachCode[] {
  const breaches: BreachCode[] = [];
  if (sample.mem_available_kib < thresholds.min_mem_available_run_kib) breaches.push("memory_run");
  if (sample.swap_used_kib > start.swap_used_kib + thresholds.max_swap_growth_kib) breaches.push("swap_growth");
  if (thresholds.forbid_other_runs && others.length > 0) breaches.push("other_run");
  return breaches;
}

/** Commands that run, train, evaluate or benchmark (substrings of the command line). The owner's queue scripts are added with `--busy-pattern`. */
export const BUSY_PATTERNS: readonly string[] = [
  "areneon_ml.loop",
  "areneon_ml.worker",
  "areneon_ml.learner",
  "areneon_ml.progress",
  "play-generation.ts",
  "feature-bench",
  "scripted-h2h",
  "best-response",
  "reducer-sweep",
  "teacher-run",
  "throughput-bench",
  "vitest",
  "pytest",
];

export const isBusyCommand = (cmdline: readonly string[], patterns: readonly string[]): boolean => {
  const text = cmdline.join(" ");
  return text.length > 0 && patterns.some((pattern) => text.includes(pattern));
};

/**
 * The commands of other runs on the machine: processes that match a busy pattern, are not the harness or one of its ancestors (`exclude`)
 * and are not in the arm's own process group (`group`, null before the arm exists).
 */
export function findOtherRuns(proc: Proc, patterns: readonly string[], { exclude, group }: { exclude: ReadonlySet<number>; group: number | null }): string[] {
  const found: string[] = [];
  for (const pid of proc.pids()) {
    if (exclude.has(pid)) continue;
    const stat = proc.stat(pid);
    if (stat === null || (group !== null && stat.pgrp === group)) continue;
    const cmdline = proc.cmdline(pid);
    if (cmdline !== null && isBusyCommand(cmdline, patterns)) found.push(cmdline.join(" ").slice(0, 200));
  }
  return found;
}

/** Reads machine samples: load, memory, swap and the system-wide busy share since the previous reading. */
export class MachineSampler {
  private last: { busy: number; total: number } | null = null;
  private readonly proc: Proc;
  private readonly now: () => number;

  constructor(proc: Proc, now: () => number = Date.now) {
    this.proc = proc;
    this.now = now;
  }

  /** One sample; `cpu_busy_ppm` is the busy share since the previous reading (null for the first, and 0 when no tick has passed). */
  sample(gpu: GpuReading | null): Sample {
    const times = this.proc.cpuTimes();
    const busy = this.last === null ? null : this.share(this.last, times);
    this.last = times;
    return this.build(gpu, busy);
  }

  /** A start sample: the busy share over a window of its own (`sleep` for `windowMs`), so the first sample has one. */
  async sampleOverWindow(gpu: GpuReading | null, windowMs: number, sleep: (ms: number) => Promise<void>): Promise<Sample> {
    this.last = this.proc.cpuTimes();
    await sleep(windowMs);
    return this.sample(gpu);
  }

  private share(before: { busy: number; total: number }, after: { busy: number; total: number }): number {
    const total = after.total - before.total;
    return total <= 0 ? 0 : Math.round(((after.busy - before.busy) * 1_000_000) / total);
  }

  private build(gpu: GpuReading | null, busy: number | null): Sample {
    const load = this.proc.loadavg();
    const memory = this.proc.meminfo();
    return {
      at_epoch_ms: this.now(),
      load1_centi: load.load1Centi,
      load5_centi: load.load5Centi,
      mem_available_kib: memory.availableKib,
      swap_used_kib: memory.swapTotalKib - memory.swapFreeKib,
      cpu_busy_ppm: busy,
      gpu_util_pct: gpu === null ? null : gpu.util,
      gpu_mem_used_mib: gpu === null ? null : gpu.memMib,
    };
  }
}
