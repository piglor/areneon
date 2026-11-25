// The runner of the throughput harness: one arm, one run. It starts the arm as a child in a process group of its own, watches it, and
// writes one ThroughputRecord from what it measured itself plus the ArmReport the arm left.
//
// Host CPU, exactly. The kernel adds the CPU time of a process, its threads and every descendant it waited for to its parent's
// `cutime` and `cstime` when the parent reaps it. The harness is the parent of the arm, so the change of its own `cutime + cstime`
// between starting the arm and the arm's exit is the CPU of the arm's whole process tree to the clock tick (10 ms). Two things must
// therefore not happen in between: the harness reaps no other child (the GPU monitor, the only helper that lives that long, is
// reaped after the figure is read; every other helper runs before the arm starts or after the figure is read) and the arm leaves no
// orphan, whose time nobody would credit (found by its process group, ended, and the run is void).
import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type ArmReport, jcs, ratesOf, readThroughputRecord, type ThroughputRecord, throughputRecordHash, voidReasonsOf } from "@areneon/showdown-bridge";
import { type ArmSpec, emptyReport } from "./arms.ts";
import { UsageError } from "./errors.ts";
import { checkRun, checkStart, type GpuReading, type Mode, type Sample, type StartFailure, thresholdsFor, type Thresholds } from "./guard.ts";
import { type GpuMonitor, survivorsOf, TreeWatcher } from "./monitor.ts";
import { type Proc, ticksToMs } from "./procfs.ts";
import { renderRunSummary } from "./summary.ts";

type Breach = ThroughputRecord["conditions"]["breaches"][number];
type Json = Parameters<typeof jcs>[0];

/** The machine the harness reads: samples, and the commands of other runs on it. */
export interface Machine {
  sampleOverWindow(gpu: GpuReading | null, windowMs: number, sleep: (ms: number) => Promise<void>): Promise<Sample>;
  sample(gpu: GpuReading | null): Sample;
  /** MemAvailable now, in KiB: a cheap reading the runner takes every walk so a memory floor is caught in seconds, not at the next minute sample. */
  availableKib(): number;
  others(exclude: ReadonlySet<number>, group: number | null): string[];
}

/** Everything the runner touches outside itself, so a test can script the machine and the GPU and keep the real processes. */
export interface Deps {
  proc: Proc;
  machine: Machine;
  gpu: { once(): Promise<GpuReading | null>; monitor(intervalMs: number, proc: Proc): GpuMonitor };
  now(): number;
  mono(): number;
  sleep(ms: number): Promise<void>;
  code(): ThroughputRecord["code"];
  machineInfo(): ThroughputRecord["machine"];
  pageBytes: number;
  signals?: { on(signal: "SIGINT" | "SIGTERM", handler: () => void): void; off(signal: "SIGINT" | "SIGTERM", handler: () => void): void };
}

export interface RunOptions {
  runId: string;
  /** The run directory: created here, and refused when it exists. */
  outDir: string;
  mode: Mode;
  thresholds: Thresholds;
  /** Between two walks of the arm's process tree. */
  procIntervalMs: number;
  /** Between two machine samples (ADR-028: one minute). */
  conditionIntervalMs: number;
  /** The window the start sample's CPU busy share is measured over. */
  startWindowMs: number;
  gpuIntervalMs: number;
  /** End the arm when the memory floor breaks, so a run beside other work cannot take the machine down. */
  abortOnMemory: boolean;
  /** SIGTERM to SIGKILL for an arm that is being ended. */
  killGraceMs: number;
}

export function defaultRunOptions(given: Partial<RunOptions> = {}): RunOptions {
  const mode = given.mode ?? "idle";
  return {
    runId: "run",
    outDir: "run",
    mode,
    thresholds: thresholdsFor(mode),
    procIntervalMs: 1000,
    conditionIntervalMs: 60_000,
    startWindowMs: 5000,
    gpuIntervalMs: 1000,
    abortOnMemory: true,
    killGraceMs: 10_000,
    ...given,
  };
}

/** The idle-machine conditions do not hold: nothing was started and nothing was written. */
export class ConditionsNotMet extends Error {
  override name = "ConditionsNotMet";
  readonly failures: readonly StartFailure[];

  constructor(failures: readonly StartFailure[]) {
    super(`the machine is not idle: ${failures.map((failure) => failure.detail).join("; ")}`);
    this.failures = failures;
  }
}

export interface RunResult {
  record: ThroughputRecord;
  dir: string;
  hash: string;
  /** Why the arm's own report could not be read (the record then holds an empty one), else null. */
  collectError: string | null;
}

/** The pids of this process and its ancestors: the harness's own look-alikes, never "another run". */
function ancestors(proc: Proc): Set<number> {
  const found = new Set<number>();
  for (let pid: number | undefined = process.pid; pid !== undefined && pid > 0 && !found.has(pid); pid = proc.stat(pid)?.ppid) found.add(pid);
  return found;
}

const shellQuote = (word: string): string => (/^[\w./:=@%+-]+$/.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`);

export async function runArm(spec: ArmSpec, options: RunOptions, deps: Deps): Promise<RunResult> {
  if (existsSync(options.outDir)) throw new UsageError(`the run directory already exists: ${options.outDir}`);
  const exclude = ancestors(deps.proc);
  // Helpers that spawn and reap a process (git, nvidia-smi) run before the arm starts or after its CPU is read.
  const code = deps.code();
  const machineInfo = deps.machineInfo();
  const gpuStart = await deps.gpu.once();
  const start = await deps.machine.sampleOverWindow(gpuStart, options.startWindowMs, deps.sleep);
  const failures = checkStart(start, deps.machine.others(exclude, null), options.thresholds);
  if (failures.length > 0) throw new ConditionsNotMet(failures);

  mkdirSync(options.outDir, { recursive: true });
  const stdoutPath = join(options.outDir, "arm.out");
  const stderrPath = join(options.outDir, "arm.err");
  writeFileSync(join(options.outDir, "command.txt"), `${[spec.command, ...spec.args].map(shellQuote).join(" ")}\n`);
  const monitor = deps.gpu.monitor(options.gpuIntervalMs, deps.proc);
  const before = deps.proc.stat(process.pid) as NonNullable<ReturnType<Proc["stat"]>>;
  const out = openSync(stdoutPath, "w");
  const err = openSync(stderrPath, "w");
  const child = spawn(spec.command, [...spec.args], { cwd: spec.cwd, env: { ...process.env, ...spec.env }, stdio: ["ignore", out, err], detached: true });
  closeSync(out);
  closeSync(err);
  const startedAt = deps.mono();
  const startedEpoch = deps.now();
  const pid = child.pid ?? -1;
  const exited = new Promise<number>((resolve) => {
    child.on("error", () => resolve(-1));
    child.on("exit", (exitCode) => resolve(exitCode ?? -1));
  });

  let aborted = false;
  let killTimer: NodeJS.Timeout | undefined;
  const send = (signal: NodeJS.Signals): void => {
    try {
      process.kill(-pid, signal);
    } catch {
      // the group is gone already
    }
  };
  const abort = (): void => {
    if (aborted || pid <= 0) return; // never signal a group through a failed start: process.kill(-(-1)) would signal pid 1
    aborted = true;
    send("SIGTERM");
    killTimer = setTimeout(() => send("SIGKILL"), options.killGraceMs);
  };
  deps.signals?.on("SIGINT", abort);
  deps.signals?.on("SIGTERM", abort);

  const watcher = new TreeWatcher(deps.proc, pid, spec.classify, deps.pageBytes);
  const samples: Sample[] = [start];
  const breaches: Breach[] = [];
  const lookAtMachine = (): void => {
    const sample = deps.machine.sample(monitor.latest());
    samples.push(sample);
    for (const breach of checkRun(sample, start, deps.machine.others(exclude, pid), options.thresholds)) {
      if (!breaches.some((known) => known.code === breach)) breaches.push({ code: breach, at_ms: Math.round(deps.mono() - startedAt) });
      if (breach === "memory_run" && options.abortOnMemory) abort();
    }
  };
  let floorSeen = false;
  const walk = setInterval(() => {
    watcher.tick();
    // The memory floor is also watched at every walk: a minute is long enough for a run beside other work to take the machine down.
    if (!floorSeen && deps.machine.availableKib() < options.thresholds.min_mem_available_run_kib) {
      floorSeen = true;
      lookAtMachine();
    }
  }, options.procIntervalMs);
  const look = setInterval(lookAtMachine, options.conditionIntervalMs);
  watcher.tick();
  const exitCode = await exited;
  // The figures that must be read the moment the arm is gone: its wall clock and the credited CPU of the harness's reaped children.
  const wallMs = Math.round(deps.mono() - startedAt);
  const after = deps.proc.stat(process.pid) as NonNullable<ReturnType<Proc["stat"]>>;
  clearInterval(walk);
  clearInterval(look);
  clearTimeout(killTimer);
  deps.signals?.off("SIGINT", abort);
  deps.signals?.off("SIGTERM", abort);

  const stray = pid < 0 ? [] : survivorsOf(deps.proc, pid, exclude);
  for (const survivor of stray) {
    try {
      process.kill(survivor, "SIGKILL");
    } catch {
      // it ended by itself
    }
  }
  const helper = await monitor.stop();
  const gpuEnd = await deps.gpu.once();
  const end = deps.machine.sample(gpuEnd);
  samples.push(end);

  let report: ArmReport;
  let collectError: string | null = null;
  try {
    report = await spec.collect({ runDir: options.outDir, stdoutPath, stderrPath, exitCode, wallMs });
  } catch (error) {
    report = emptyReport(spec.description);
    collectError = (error as Error).message;
  }

  const roles = watcher.result();
  const userMs = ticksToMs(after.cutime - before.cutime);
  const systemMs = ticksToMs(after.cstime - before.cstime);
  const totalMs = userMs + systemMs;
  const sampledMs = roles.byRole.reduce((sum, role) => sum + role.cpu_ms, 0);
  const readings = monitor.readings();
  const record: ThroughputRecord = {
    schema_version: 1,
    data_class: "synthetic",
    run_id: options.runId,
    report,
    machine: machineInfo,
    code,
    conditions: {
      mode: options.mode,
      thresholds: options.thresholds,
      start,
      end,
      interval_ms: options.conditionIntervalMs,
      samples: samples.length,
      min_mem_available_kib: Math.min(...samples.map((sample) => sample.mem_available_kib)),
      max_swap_used_kib: Math.max(...samples.map((sample) => sample.swap_used_kib)),
      breaches,
    },
    window: { started_epoch_ms: startedEpoch, wall_ms: wallMs, exit_code: exitCode, aborted },
    cpu: {
      total_ms: totalMs,
      user_ms: userMs,
      system_ms: systemMs,
      helper_ms: ticksToMs(helper.ticks),
      harness_ms: ticksToMs(after.utime + after.stime - before.utime - before.stime),
      sampled_ms: sampledMs,
      unsampled_ms: Math.max(0, totalMs - sampledMs),
      processes_seen: roles.processesSeen,
      peak_tree_rss_bytes: roles.peakTreeRssBytes,
      survivors: stray.length,
      by_role: roles.byRole,
    },
    gpu: {
      used: report.gpu_used,
      available: gpuStart !== null || gpuEnd !== null || readings.length > 0,
      allocated_ms: report.gpu_used ? wallMs : 0,
      util_mean_ppm: readings.length === 0 ? null : Math.round((readings.reduce((sum, reading) => sum + reading.util, 0) * 10_000) / readings.length),
      mem_peak_mib: readings.length === 0 ? null : Math.max(...readings.map((reading) => reading.memMib)),
      samples: readings.length,
    },
    rates: { decisions_per_core_hour: 0, games_per_core_hour: 0, decisions_per_wall_hour: 0, decisions_per_gpu_hour: null, decisions_per_core_hour_steady: null },
    valid: true,
    void_reasons: [],
  };
  record.void_reasons = voidReasonsOf(record);
  record.valid = record.void_reasons.length === 0;
  record.rates = ratesOf(record);
  readThroughputRecord(JSON.parse(JSON.stringify(record)) as unknown); // a record the reader refuses is a defect of the harness, found before it is written

  const lines = (items: readonly unknown[]): string => items.map((item) => `${JSON.stringify(item)}\n`).join("");
  writeFileSync(join(options.outDir, "record.json"), `${jcs(record as unknown as Json)}\n`);
  writeFileSync(join(options.outDir, "conditions.jsonl"), lines(samples));
  writeFileSync(join(options.outDir, "procs.jsonl"), lines(watcher.rows()));
  writeFileSync(join(options.outDir, "summary.md"), renderRunSummary(record, samples));
  return { record, dir: options.outDir, hash: throughputRecordHash(record), collectError };
}
