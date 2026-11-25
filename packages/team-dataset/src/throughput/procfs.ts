// /proc readers of the throughput harness (Linux only): the kernel's text formats parsed into integers, and a reader that takes the
// /proc root as an argument so a test can build a fake tree. Nothing here spawns a process, so reading never adds CPU time to the
// harness's reaped children (the figure the arm's CPU total is taken from).
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** The kernel counts CPU time in /proc in USER_HZ = 100 ticks a second, whatever its own timer frequency. */
export const TICKS_PER_SECOND = 100;
export const ticksToMs = (ticks: number): number => (ticks * 1000) / TICKS_PER_SECOND;

export class ProcfsError extends Error {
  override name = "ProcfsError";
}

/** The fields of /proc/<pid>/stat the harness uses (times in clock ticks, `rssPages` in pages). */
export interface ProcStat {
  pid: number;
  comm: string;
  state: string;
  ppid: number;
  pgrp: number;
  session: number;
  utime: number;
  stime: number;
  cutime: number;
  cstime: number;
  threads: number;
  starttime: number;
  rssPages: number;
}

const integer = (text: string | undefined): number => {
  if (text === undefined || !/^-?\d+$/.test(text)) throw new ProcfsError("not a /proc stat line");
  return Number(text);
};

/** Parse one /proc/<pid>/stat line. The comm is in parentheses and may hold spaces and parentheses, so the LAST closing parenthesis ends it. */
export function parseStat(text: string): ProcStat {
  const open = text.indexOf(" (");
  const close = text.lastIndexOf(")");
  if (open < 0 || close < open) throw new ProcfsError("not a /proc stat line");
  const rest = text.slice(close + 1).trim().split(/\s+/);
  // rest[0] is field 3 (state), so field n is rest[n - 3].
  if (rest.length < 22) throw new ProcfsError("not a /proc stat line");
  return {
    pid: integer(text.slice(0, open)),
    comm: text.slice(open + 2, close),
    state: rest[0] as string,
    ppid: integer(rest[1]),
    pgrp: integer(rest[2]),
    session: integer(rest[3]),
    utime: integer(rest[11]),
    stime: integer(rest[12]),
    cutime: integer(rest[13]),
    cstime: integer(rest[14]),
    threads: integer(rest[17]),
    starttime: integer(rest[19]),
    rssPages: integer(rest[21]),
  };
}

/** A load average to hundredths from its decimal text, with no floating point. */
function centi(text: string | undefined): number {
  const match = text === undefined ? null : /^(\d+)(?:\.(\d+))?$/.exec(text);
  if (match === null) throw new ProcfsError("not a /proc loadavg line");
  return Number(match[1]) * 100 + Number(((match[2] ?? "") + "00").slice(0, 2));
}

/** The 1 and 5 minute load averages times 100. */
export function parseLoadavg(text: string): { load1Centi: number; load5Centi: number } {
  const [one, five] = text.trim().split(/\s+/);
  return { load1Centi: centi(one), load5Centi: centi(five) };
}

export interface MemInfo {
  totalKib: number;
  availableKib: number;
  swapTotalKib: number;
  swapFreeKib: number;
}

/** MemTotal, MemAvailable, SwapTotal and SwapFree of /proc/meminfo, in KiB. */
export function parseMeminfo(text: string): MemInfo {
  const find = (key: string): number => {
    const match = new RegExp(`^${key}:\\s+(\\d+)\\s*kB`, "m").exec(text);
    if (match === null) throw new ProcfsError(`/proc/meminfo has no ${key}`);
    return Number(match[1]);
  };
  return { totalKib: find("MemTotal"), availableKib: find("MemAvailable"), swapTotalKib: find("SwapTotal"), swapFreeKib: find("SwapFree") };
}

/** The aggregate `cpu` line of /proc/stat: busy is everything but idle and iowait over the eight time fields (user nice system idle iowait irq softirq steal). */
export function parseCpuStat(text: string): { busy: number; total: number } {
  const line = text.split("\n").find((candidate) => /^cpu\s/.test(candidate));
  if (line === undefined) throw new ProcfsError("not a /proc stat file");
  const [user, nice, system, idle, iowait, irq, softirq, steal] = line.trim().split(/\s+/).slice(1, 9).map(Number) as [number, number, number, number, number, number, number, number];
  const total = user + nice + system + idle + iowait + irq + softirq + steal;
  return { busy: total - idle - iowait, total };
}

/** The processor count and the first model name of /proc/cpuinfo. */
export function parseCpuinfo(text: string): { logicalCpus: number; model: string } {
  const logicalCpus = (text.match(/^processor\s*:/gm) ?? []).length;
  if (logicalCpus === 0) throw new ProcfsError("/proc/cpuinfo lists no processor");
  const model = /^model name\s*:\s*(.+)$/m.exec(text)?.[1]?.trim();
  return { logicalCpus, model: model ?? "unknown" };
}

/** A /proc/<pid>/cmdline: NUL-separated arguments. */
export function cmdlineOf(raw: string): string[] {
  const args = raw.split("\0");
  if (args[args.length - 1] === "") args.pop();
  return args;
}

/** A reader of one /proc tree (the real one by default). A process that is gone is `null`, never an error: it exits between the listing and the read. */
export class Proc {
  readonly root: string;

  constructor(root: string = "/proc") {
    this.root = root;
  }

  private read(...parts: string[]): string {
    return readFileSync(join(this.root, ...parts), "utf8");
  }

  private tryRead(...parts: string[]): string | null {
    try {
      return this.read(...parts);
    } catch {
      return null;
    }
  }

  private machine<T>(name: string, parse: (text: string) => T): T {
    try {
      return parse(this.read(name));
    } catch (error) {
      if (error instanceof ProcfsError) throw error;
      throw new ProcfsError(`cannot read /proc/${name}`);
    }
  }

  pids(): number[] {
    return readdirSync(this.root).filter((name) => /^\d+$/.test(name)).map(Number);
  }

  stat(pid: number): ProcStat | null {
    const text = this.tryRead(String(pid), "stat");
    if (text === null) return null;
    try {
      return parseStat(text);
    } catch {
      return null; // a half-written line of a process that is exiting
    }
  }

  cmdline(pid: number): string[] | null {
    const text = this.tryRead(String(pid), "cmdline");
    return text === null ? null : cmdlineOf(text);
  }

  loadavg(): { load1Centi: number; load5Centi: number } {
    return this.machine("loadavg", parseLoadavg);
  }

  meminfo(): MemInfo {
    return this.machine("meminfo", parseMeminfo);
  }

  cpuTimes(): { busy: number; total: number } {
    return this.machine("stat", parseCpuStat);
  }

  cpuinfo(): { logicalCpus: number; model: string } {
    return this.machine("cpuinfo", parseCpuinfo);
  }
}
