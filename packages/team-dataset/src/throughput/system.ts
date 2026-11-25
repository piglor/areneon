// The real machine behind the runner's `Deps`: /proc, nvidia-smi, git and the pinned engine's commit.
import { execFileSync } from "node:child_process";
import { PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import { BUSY_PATTERNS, findOtherRuns, type GpuReading, MachineSampler, type Sample } from "./guard.ts";
import { NvidiaSmi } from "./monitor.ts";
import { Proc } from "./procfs.ts";
import { type Deps, type Machine } from "./runner.ts";

/** Samples the real machine and looks for other runs by the busy patterns. */
class SystemMachine implements Machine {
  private readonly sampler: MachineSampler;
  private readonly proc: Proc;
  private readonly patterns: readonly string[];

  constructor(proc: Proc, patterns: readonly string[], now: () => number) {
    this.proc = proc;
    this.patterns = patterns;
    this.sampler = new MachineSampler(proc, now);
  }

  sampleOverWindow(gpu: GpuReading | null, windowMs: number, sleep: (ms: number) => Promise<void>): Promise<Sample> {
    return this.sampler.sampleOverWindow(gpu, windowMs, sleep);
  }

  sample(gpu: GpuReading | null): Sample {
    return this.sampler.sample(gpu);
  }

  availableKib(): number {
    return this.proc.meminfo().availableKib;
  }

  others(exclude: ReadonlySet<number>, group: number | null): string[] {
    return findOtherRuns(this.proc, this.patterns, { exclude, group });
  }
}

const git = (repoRoot: string, ...args: string[]): string => execFileSync("git", ["-C", repoRoot, ...args], { encoding: "utf8" }).trim();

/** The real thing. `extraPatterns` are added to the commands that count as another run. */
export function defaultDeps(repoRoot: string, extraPatterns: readonly string[] = []): Deps {
  const proc = new Proc();
  const now = (): number => Date.now();
  return {
    proc,
    machine: new SystemMachine(proc, [...BUSY_PATTERNS, ...extraPatterns], now),
    gpu: new NvidiaSmi(),
    now,
    mono: () => performance.now(),
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    // Tracked files only: an untracked scratch file does not change the code that runs.
    code: () => ({ commit: git(repoRoot, "rev-parse", "HEAD"), engine_pin: PINNED_SHOWDOWN_COMMIT, tree_clean: git(repoRoot, "status", "--porcelain", "--untracked-files=no") === "" }),
    machineInfo: () => {
      const cpu = proc.cpuinfo();
      return { logical_cpus: cpu.logicalCpus, mem_total_kib: proc.meminfo().totalKib, cpu_model: cpu.model.replace(/[^ -~]/g, "?").slice(0, 200) };
    },
    pageBytes: 4096, // the page size of Linux on x86-64 and arm64 (the kernel's rss is in pages)
    signals: process,
  };
}
