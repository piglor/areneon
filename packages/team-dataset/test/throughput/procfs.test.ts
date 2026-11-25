// /proc readers of the throughput harness: parsers against hand-written kernel text, and the Proc reader over a fake /proc tree.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { cmdlineOf, parseCpuStat, parseCpuinfo, parseLoadavg, parseMeminfo, parseStat, Proc, ProcfsError, TICKS_PER_SECOND, ticksToMs } from "../../src/throughput/procfs.ts";

/** A /proc/<pid>/stat line: fields 1..52, the comm in parentheses (it may hold spaces and parentheses). */
const statLine = (pid: number, comm: string, f: { state?: string; ppid?: number; pgrp?: number; session?: number; utime?: number; stime?: number; cutime?: number; cstime?: number; threads?: number; starttime?: number; rss?: number }): string => {
  const fields = [
    f.state ?? "S", f.ppid ?? 1, f.pgrp ?? pid, f.session ?? pid, 0, -1, 4194560, 100, 0, 0, 0, // 3..13: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
    f.utime ?? 0, f.stime ?? 0, f.cutime ?? 0, f.cstime ?? 0, // 14..17
    20, 0, f.threads ?? 1, 0, f.starttime ?? 1000, 1_000_000, f.rss ?? 0, // 18..24: priority nice num_threads itrealvalue starttime vsize rss
    ...Array.from({ length: 28 }, () => 0), // 25..52
  ];
  return `${pid} (${comm}) ${fields.join(" ")}\n`;
};

describe("ticks", () => {
  it("are 100 a second, so 10 ms each", () => {
    expect(TICKS_PER_SECOND).toBe(100);
    expect(ticksToMs(0)).toBe(0);
    expect(ticksToMs(123)).toBe(1230);
  });
});

describe("parseStat", () => {
  it("reads the fields the harness needs by their position", () => {
    const stat = parseStat(statLine(4242, "node", { state: "R", ppid: 7, pgrp: 4242, session: 4242, utime: 150, stime: 30, cutime: 20, cstime: 5, threads: 11, starttime: 987_654, rss: 4567 }));
    expect(stat).toEqual({ pid: 4242, comm: "node", state: "R", ppid: 7, pgrp: 4242, session: 4242, utime: 150, stime: 30, cutime: 20, cstime: 5, threads: 11, starttime: 987_654, rssPages: 4567 });
  });

  it("finds the end of a comm that holds spaces and parentheses (the last closing parenthesis ends it)", () => {
    const stat = parseStat(statLine(9, "a b) (c", { ppid: 3, utime: 8 }));
    expect(stat.comm).toBe("a b) (c");
    expect(stat.ppid).toBe(3);
    expect(stat.utime).toBe(8);
  });

  it("refuses a line that is not a stat line, with a rule and no value", () => {
    for (const bad of ["", "12 node S 1", "12 (node) S 1 2", `12 (node) ${"x ".repeat(30)}`]) expect(() => parseStat(bad), bad).toThrow(ProcfsError);
    expect(() => parseStat("12 (node) S 1")).toThrow("not a /proc stat line");
  });
});

describe("parseLoadavg", () => {
  it("states the 1 and 5 minute averages in hundredths without a float", () => {
    expect(parseLoadavg("10.69 9.17 8.73 12/1873 99999\n")).toEqual({ load1Centi: 1069, load5Centi: 917 });
    expect(parseLoadavg("0.00 0.05 0.5 1/10 3\n")).toEqual({ load1Centi: 0, load5Centi: 5 });
    expect(parseLoadavg("1.5 2 3 1/1 1")).toEqual({ load1Centi: 150, load5Centi: 200 });
  });

  it("refuses text that is not a load average", () => {
    expect(() => parseLoadavg("")).toThrow(ProcfsError);
    expect(() => parseLoadavg("a b c")).toThrow("not a /proc loadavg line");
  });
});

describe("parseMeminfo", () => {
  const text = "MemTotal:       24117248 kB\nMemFree:         3000000 kB\nMemAvailable:   12582912 kB\nSwapTotal:       4194304 kB\nSwapFree:        4194000 kB\nDirty: 0 kB\n";
  it("reads memory and swap in KiB", () => {
    expect(parseMeminfo(text)).toEqual({ totalKib: 24_117_248, availableKib: 12_582_912, swapTotalKib: 4_194_304, swapFreeKib: 4_194_000 });
  });

  it("refuses a /proc/meminfo with a field missing", () => {
    expect(() => parseMeminfo(text.replace("MemAvailable", "MemAvail"))).toThrow("MemAvailable");
  });
});

describe("parseCpuStat", () => {
  it("counts busy as everything but idle and iowait over the eight time fields", () => {
    // user nice system idle iowait irq softirq steal guest guest_nice
    const line = "cpu  100 20 30 700 50 5 5 90 40 0\ncpu0 1 2 3 4 5 6 7 8 9 0\n";
    expect(parseCpuStat(line)).toEqual({ busy: 100 + 20 + 30 + 5 + 5 + 90, total: 100 + 20 + 30 + 700 + 50 + 5 + 5 + 90 });
  });

  it("refuses text without an aggregate cpu line", () => {
    expect(() => parseCpuStat("intr 1 2 3\n")).toThrow("not a /proc stat file");
  });
});

describe("parseCpuinfo", () => {
  it("counts processors and names the first model", () => {
    const text = "processor\t: 0\nmodel name\t: Example CPU @ 3.0GHz\n\nprocessor\t: 1\nmodel name\t: Example CPU @ 3.0GHz\n";
    expect(parseCpuinfo(text)).toEqual({ logicalCpus: 2, model: "Example CPU @ 3.0GHz" });
  });

  it("names an unknown model rather than failing, and refuses no processors", () => {
    expect(parseCpuinfo("processor : 0\n")).toEqual({ logicalCpus: 1, model: "unknown" });
    expect(() => parseCpuinfo("")).toThrow("no processor");
  });
});

describe("cmdlineOf", () => {
  it("splits NUL-separated arguments and drops the trailing empty one", () => {
    expect(cmdlineOf("node\0--flag\0a b\0")).toEqual(["node", "--flag", "a b"]);
    expect(cmdlineOf("")).toEqual([]);
  });
});

describe("Proc over a fake /proc", () => {
  let root: string;
  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), "fake-proc-"));
    writeFileSync(join(root, "loadavg"), "0.52 0.40 0.30 1/200 77\n");
    writeFileSync(join(root, "meminfo"), "MemTotal: 1000 kB\nMemAvailable: 800 kB\nSwapTotal: 100 kB\nSwapFree: 90 kB\n");
    writeFileSync(join(root, "stat"), "cpu  10 0 10 80 0 0 0 0 0 0\n");
    writeFileSync(join(root, "cpuinfo"), "processor : 0\nmodel name : Fake\n");
    for (const [pid, comm, ppid] of [[10, "init", 1], [11, "child", 10]] as const) {
      mkdirSync(join(root, String(pid)));
      writeFileSync(join(root, String(pid), "stat"), statLine(pid, comm, { ppid, utime: pid }));
      writeFileSync(join(root, String(pid), "cmdline"), `${comm}\0--arg\0`);
    }
    mkdirSync(join(root, "sys")); // a non-numeric entry is not a process
  });
  afterAll(() => rmSync(root, { recursive: true, force: true }));

  it("lists processes, reads their stat and command line, and answers null for one that is gone", () => {
    const proc = new Proc(root);
    expect(proc.pids().sort()).toEqual([10, 11]);
    expect(proc.stat(11)).toMatchObject({ pid: 11, comm: "child", ppid: 10, utime: 11 });
    expect(proc.cmdline(11)).toEqual(["child", "--arg"]);
    expect(proc.stat(99)).toBeNull();
    mkdirSync(join(root, "12"));
    writeFileSync(join(root, "12", "stat"), "12 (half-written) S 1"); // a process that is exiting
    expect(proc.stat(12)).toBeNull();
    expect(proc.cmdline(99)).toBeNull();
  });

  it("reads the machine files", () => {
    const proc = new Proc(root);
    expect(proc.loadavg()).toEqual({ load1Centi: 52, load5Centi: 40 });
    expect(proc.meminfo()).toEqual({ totalKib: 1000, availableKib: 800, swapTotalKib: 100, swapFreeKib: 90 });
    expect(proc.cpuTimes()).toEqual({ busy: 20, total: 100 });
    expect(proc.cpuinfo()).toEqual({ logicalCpus: 1, model: "Fake" });
  });

  it("refuses an unreadable machine file instead of guessing", () => {
    expect(() => new Proc(join(root, "nowhere")).loadavg()).toThrow(ProcfsError);
  });

  it("reads the real /proc of this machine", () => {
    const proc = new Proc();
    const self = proc.stat(process.pid);
    expect(self).not.toBeNull();
    expect(self?.comm.length).toBeGreaterThan(0);
    expect(proc.cmdline(process.pid)?.length).toBeGreaterThan(0);
    expect(proc.meminfo().availableKib).toBeGreaterThan(0);
  });
});
