// What the throughput harness watches while an arm runs: the arm's process tree (CPU by role, memory), the processes it leaves behind
// and the GPU. The tree watcher only reads /proc; the GPU is read from one long-lived nvidia-smi whose own CPU the runner subtracts.
import { type ChildProcess, spawn } from "node:child_process";
import { type Proc, ticksToMs } from "./procfs.ts";

/** One role of the arm's processes, as the record states it. */
export interface RoleFigures {
  role: string;
  processes: number;
  cpu_ms: number;
  peak_rss_bytes: number;
}

/** One process of the arm, last reading each (the journal's `procs.jsonl`). */
export interface ProcessRow {
  pid: number;
  ppid: number;
  role: string;
  cpu_ms: number;
  peak_rss_bytes: number;
  threads_peak: number;
}

export type Classifier = (cmdline: readonly string[], comm: string) => string;

interface Seen {
  pid: number;
  ppid: number;
  role: string;
  ticks: number;
  peakRssBytes: number;
  threadsPeak: number;
}

/**
 * Walks /proc and follows the descendants of one root process. A process is told apart from a later one that reuses its pid by its
 * start time; its CPU is the last reading (so a process that ends between two walks is short by the part after its last walk, which the
 * runner's exact total accounts for), its role is decided again at every walk (a forked child shows its parent's command line until it
 * execs) and its memory is its peak.
 */
export class TreeWatcher {
  private readonly seen = new Map<string, Seen>();
  private peakTreeRssBytes = 0;
  private readonly proc: Proc;
  private readonly rootPid: number;
  private readonly classify: Classifier;
  private readonly pageBytes: number;

  constructor(proc: Proc, rootPid: number, classify: Classifier, pageBytes: number) {
    this.proc = proc;
    this.rootPid = rootPid;
    this.classify = classify;
    this.pageBytes = pageBytes;
  }

  /** One walk; returns how many processes of the tree were alive. */
  tick(): number {
    const stats = new Map<number, NonNullable<ReturnType<Proc["stat"]>>>();
    for (const pid of this.proc.pids()) {
      const stat = this.proc.stat(pid);
      if (stat !== null) stats.set(pid, stat);
    }
    const tree = new Set<number>(stats.has(this.rootPid) ? [this.rootPid] : []);
    for (let grew = tree.size > 0; grew; ) {
      grew = false;
      for (const [pid, stat] of stats) {
        if (!tree.has(pid) && tree.has(stat.ppid)) {
          tree.add(pid);
          grew = true;
        }
      }
    }
    let rss = 0;
    for (const pid of tree) {
      const stat = stats.get(pid) as NonNullable<ReturnType<Proc["stat"]>>;
      const key = `${pid}:${stat.starttime}`;
      const bytes = stat.rssPages * this.pageBytes;
      rss += bytes;
      const before = this.seen.get(key);
      this.seen.set(key, {
        pid,
        ppid: stat.ppid,
        role: this.classify(this.proc.cmdline(pid) ?? [], stat.comm),
        ticks: stat.utime + stat.stime,
        peakRssBytes: Math.max(before?.peakRssBytes ?? 0, bytes),
        threadsPeak: Math.max(before?.threadsPeak ?? 0, stat.threads),
      });
    }
    this.peakTreeRssBytes = Math.max(this.peakTreeRssBytes, rss);
    return tree.size;
  }

  result(): { byRole: RoleFigures[]; processesSeen: number; peakTreeRssBytes: number } {
    const roles = new Map<string, RoleFigures>();
    for (const entry of this.seen.values()) {
      const figures = roles.get(entry.role) ?? { role: entry.role, processes: 0, cpu_ms: 0, peak_rss_bytes: 0 };
      figures.processes += 1;
      figures.cpu_ms += ticksToMs(entry.ticks);
      figures.peak_rss_bytes = Math.max(figures.peak_rss_bytes, entry.peakRssBytes);
      roles.set(entry.role, figures);
    }
    return { byRole: [...roles.values()].sort((a, b) => (a.role < b.role ? -1 : 1)), processesSeen: this.seen.size, peakTreeRssBytes: this.peakTreeRssBytes };
  }

  rows(): ProcessRow[] {
    return [...this.seen.values()].map((entry) => ({ pid: entry.pid, ppid: entry.ppid, role: entry.role, cpu_ms: ticksToMs(entry.ticks), peak_rss_bytes: entry.peakRssBytes, threads_peak: entry.threadsPeak }));
  }
}

/** Live processes still in the arm's process group or session once the arm has exited (zombies and the excluded pids do not count). */
export function survivorsOf(proc: Proc, group: number, exclude: ReadonlySet<number>): number[] {
  const found: number[] = [];
  for (const pid of proc.pids()) {
    if (exclude.has(pid)) continue;
    const stat = proc.stat(pid);
    if (stat !== null && stat.state !== "Z" && (stat.pgrp === group || stat.session === group)) found.push(pid);
  }
  return found;
}

/** One nvidia-smi answer: the busiest card's utilisation and the memory of all cards; null for text that is not a reading. */
export function parseGpuReading(text: string): { util: number; memMib: number } | null {
  let util = 0;
  let memMib = 0;
  let cards = 0;
  for (const line of text.split("\n")) {
    if (line.trim() === "") continue;
    const match = /^\s*(\d+)\s*,\s*(\d+)\s*$/.exec(line);
    if (match === null) return null;
    util = Math.max(util, Number(match[1]));
    memMib += Number(match[2]);
    cards += 1;
  }
  return cards === 0 ? null : { util, memMib };
}

const QUERY = ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"];

/** A running nvidia-smi loop: its readings so far and, when stopped, the CPU ticks it used. */
export interface GpuMonitor {
  latest(): { util: number; memMib: number } | null;
  readings(): { util: number; memMib: number }[];
  stop(): Promise<{ ticks: number }>;
}

/** nvidia-smi as a source of GPU readings; `command` and `baseArgs` let a test stand in for it. A machine without the tool simply has no readings. */
export class NvidiaSmi {
  private readonly command: string;
  private readonly baseArgs: readonly string[];
  private readonly env: Record<string, string>;

  constructor(command: string = "nvidia-smi", baseArgs: readonly string[] = [], env: Record<string, string> = {}) {
    this.command = command;
    this.baseArgs = baseArgs;
    this.env = env;
  }

  private start(extra: readonly string[]): ChildProcess {
    return spawn(this.command, [...this.baseArgs, ...QUERY, ...extra], { stdio: ["ignore", "pipe", "ignore"], env: { ...process.env, ...this.env } });
  }

  /** One reading (spawns and reaps a process: call it before the arm starts or after the totals are read, never while the CPU total accumulates). */
  once(): Promise<{ util: number; memMib: number } | null> {
    return new Promise((resolve) => {
      const child = this.start([]);
      let out = "";
      child.stdout?.on("data", (chunk: Buffer) => (out += chunk.toString()));
      child.on("error", () => resolve(null));
      child.on("close", (code) => resolve(code === 0 ? parseGpuReading(out) : null));
    });
  }

  /** Start one long-lived `nvidia-smi -lms` and keep its readings; `stop()` reads its CPU and ends it. */
  monitor(intervalMs: number, proc: Proc): GpuMonitor {
    const child = this.start(["-lms", String(intervalMs)]);
    const readings: { util: number; memMib: number }[] = [];
    let alive = true;
    let spawned = true;
    let tail = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      const lines = (tail + chunk.toString()).split("\n");
      tail = lines.pop() ?? "";
      for (const line of lines) {
        const reading = parseGpuReading(line);
        if (reading !== null) readings.push(reading);
      }
    });
    const exited = new Promise<void>((resolve) => {
      child.on("error", () => {
        spawned = false;
        alive = false;
        resolve();
      });
      child.on("exit", () => {
        alive = false;
        resolve();
      });
    });
    let stopped: { ticks: number } | null = null;
    return {
      latest: () => readings[readings.length - 1] ?? null,
      readings: () => readings,
      stop: async () => {
        if (stopped !== null) return stopped;
        let ticks = 0;
        const pid = child.pid;
        if (spawned && alive && pid !== undefined) {
          const stat = proc.stat(pid);
          ticks = stat === null ? 0 : stat.utime + stat.stime;
          child.kill("SIGTERM");
        }
        await exited;
        stopped = { ticks };
        return stopped;
      },
    };
  }
}
