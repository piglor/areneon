// The process-tree watcher and the GPU monitor of the throughput harness, over a fake /proc tree and a fake nvidia-smi.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { NvidiaSmi, parseGpuReading, survivorsOf, TreeWatcher } from "../../src/throughput/monitor.ts";
import { Proc } from "../../src/throughput/procfs.ts";

let root: string;
const proc = (): Proc => new Proc(root);
type Fields = { ppid: number; pgrp?: number; session?: number; state?: string; utime?: number; stime?: number; starttime?: number; rssPages?: number; threads?: number };

/** Write (or overwrite) a fake process: its stat and command line. */
const put = (pid: number, comm: string, args: string[], f: Fields): void => {
  mkdirSync(join(root, String(pid)), { recursive: true });
  const fields = [f.state ?? "S", f.ppid, f.pgrp ?? pid, f.session ?? pid, 0, -1, 0, 0, 0, 0, 0, f.utime ?? 0, f.stime ?? 0, 0, 0, 20, 0, f.threads ?? 1, 0, f.starttime ?? 100, 1000, f.rssPages ?? 0, ...Array.from({ length: 28 }, () => 0)];
  writeFileSync(join(root, String(pid), "stat"), `${pid} (${comm}) ${fields.join(" ")}\n`);
  writeFileSync(join(root, String(pid), "cmdline"), `${args.join("\0")}\0`);
};
const drop = (pid: number): void => rmSync(join(root, String(pid)), { recursive: true, force: true });
const roleOf = (cmd: readonly string[]): string => (cmd.some((a) => a.includes("worker")) ? "worker" : cmd.some((a) => a.includes("game")) ? "game" : "other");

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), "monitor-proc-"));
});
afterAll(() => rmSync(root, { recursive: true, force: true }));

describe("TreeWatcher", () => {
  it("counts the CPU of every descendant by role, as last seen, and leaves out processes outside the tree", () => {
    put(1000, "node", ["node", "driver"], { ppid: 1, utime: 10, stime: 5, rssPages: 100 });
    put(1001, "node", ["node", "game.ts"], { ppid: 1000, utime: 100, stime: 20, rssPages: 1000, threads: 9 });
    put(1002, "python", ["python", "worker"], { ppid: 1001, utime: 50, stime: 10, rssPages: 500 });
    put(1003, "python", ["python", "worker"], { ppid: 1002, utime: 7, stime: 1, rssPages: 20 }); // a grandchild
    put(2000, "node", ["node", "game.ts"], { ppid: 1, utime: 9999, stime: 0 }); // not the arm's
    const watcher = new TreeWatcher(proc(), 1000, roleOf, 4096);
    expect(watcher.tick()).toBe(4);
    expect(watcher.result()).toEqual({
      byRole: [
        { role: "game", processes: 1, cpu_ms: 1200, peak_rss_bytes: 1000 * 4096 },
        { role: "other", processes: 1, cpu_ms: 150, peak_rss_bytes: 100 * 4096 },
        { role: "worker", processes: 2, cpu_ms: 680, peak_rss_bytes: 500 * 4096 },
      ],
      processesSeen: 4,
      peakTreeRssBytes: (100 + 1000 + 500 + 20) * 4096,
    });
  });

  it("keeps the last CPU of a process that exited, and sums a process again only while it lives", () => {
    const watcher = new TreeWatcher(proc(), 1000, roleOf, 4096);
    watcher.tick();
    put(1001, "node", ["node", "game.ts"], { ppid: 1000, utime: 300, stime: 20, rssPages: 400 }); // more CPU, less memory
    watcher.tick();
    drop(1002);
    drop(1003); // both workers are gone before the next walk
    watcher.tick();
    const result = watcher.result();
    expect(result.byRole.find((r) => r.role === "game")).toEqual({ role: "game", processes: 1, cpu_ms: 3200, peak_rss_bytes: 1000 * 4096 });
    expect(result.byRole.find((r) => r.role === "worker")).toEqual({ role: "worker", processes: 2, cpu_ms: 680, peak_rss_bytes: 500 * 4096 });
    expect(result.peakTreeRssBytes).toBe((100 + 1000 + 500 + 20) * 4096); // the first walk was the largest
    expect(result.processesSeen).toBe(4);
  });

  it("tells a new process that reused a pid by its start time", () => {
    put(1002, "python", ["python", "worker"], { ppid: 1000, utime: 5, stime: 0, starttime: 5000 });
    const watcher = new TreeWatcher(proc(), 1000, roleOf, 4096);
    watcher.tick();
    put(1002, "python", ["python", "worker"], { ppid: 1000, utime: 2, stime: 0, starttime: 6000 }); // another process, the same pid
    watcher.tick();
    expect(watcher.result().byRole.find((r) => r.role === "worker")).toMatchObject({ processes: 2, cpu_ms: 70 });
  });

  it("classifies a process again at every walk: a forked child shows its parent's command line until it execs", () => {
    drop(1002);
    put(1005, "node", ["node", "game.ts"], { ppid: 1000, utime: 1 }); // a copy of the parent's command line
    const watcher = new TreeWatcher(proc(), 1000, roleOf, 4096);
    watcher.tick();
    put(1005, "python", ["python", "worker"], { ppid: 1000, utime: 2, starttime: 100 });
    watcher.tick();
    const roles = watcher.result().byRole;
    expect(roles.find((r) => r.role === "worker")).toMatchObject({ processes: 1, cpu_ms: 20 });
  });

  it("sees nothing of a root that is gone", () => {
    expect(new TreeWatcher(proc(), 424_242, roleOf, 4096).tick()).toBe(0);
  });

  it("writes one row per process for the journal, last reading each", () => {
    const watcher = new TreeWatcher(proc(), 1000, roleOf, 4096);
    watcher.tick();
    const rows = watcher.rows();
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]).toEqual(expect.objectContaining({ pid: expect.any(Number), role: expect.any(String), cpu_ms: expect.any(Number), peak_rss_bytes: expect.any(Number), threads_peak: expect.any(Number) }));
  });
});

describe("survivorsOf", () => {
  it("finds live processes of the arm's process group or session, and ignores zombies and others", () => {
    put(3000, "sh", ["sh"], { ppid: 1, pgrp: 777, session: 777 });
    put(3001, "node", ["node", "stray"], { ppid: 1, pgrp: 3001, session: 777 }); // moved its group, kept the session
    put(3002, "node", ["node", "dead"], { ppid: 1, pgrp: 777, session: 777, state: "Z" });
    put(3003, "vim", ["vim"], { ppid: 1, pgrp: 3003, session: 3003 });
    expect(survivorsOf(proc(), 777, new Set()).sort()).toEqual([3000, 3001]);
    expect(survivorsOf(proc(), 777, new Set([3000])).sort()).toEqual([3001]);
  });
});

describe("parseGpuReading", () => {
  it("reads utilisation and memory, taking the busiest utilisation and the summed memory of several cards", () => {
    expect(parseGpuReading("37, 1200\n")).toEqual({ util: 37, memMib: 1200 });
    expect(parseGpuReading("10, 100\n90, 900\n")).toEqual({ util: 90, memMib: 1000 });
  });

  it("is null for text that is not a reading", () => {
    for (const bad of ["", "[N/A], [N/A]", "no devices were found", "1, x"]) expect(parseGpuReading(bad), bad).toBeNull();
  });
});

describe("NvidiaSmi (against a fake tool)", () => {
  let fake: string;
  beforeAll(() => {
    fake = join(root, "fake-smi.mjs");
    writeFileSync(
      fake,
      [
        "const loop = process.argv.includes('-lms');",
        "const mode = process.env.FAKE_SMI;",
        "if (mode === 'fail') process.exit(9);",
        "if (mode === 'garbage') { console.log('no devices'); process.exit(0); }",
        "console.log('37, 1200');",
        "if (loop) setInterval(() => console.log('50, 2400'), 20);",
        "else process.exit(0);",
      ].join("\n"),
    );
  });

  it("answers one reading, or null when the tool is missing, fails or prints something else", async () => {
    expect(await new NvidiaSmi(process.execPath, [fake]).once()).toEqual({ util: 37, memMib: 1200 });
    expect(await new NvidiaSmi("/nonexistent/nvidia-smi").once()).toBeNull();
    expect(await new NvidiaSmi(process.execPath, [fake], { FAKE_SMI: "fail" }).once()).toBeNull();
    expect(await new NvidiaSmi(process.execPath, [fake], { FAKE_SMI: "garbage" }).once()).toBeNull();
  });

  it("monitors in one long-lived process, keeps its readings and reports its own CPU when it stops", async () => {
    const monitor = new NvidiaSmi(process.execPath, [fake]).monitor(20, new Proc());
    for (let waited = 0; monitor.readings().length < 4 && waited < 20_000; waited += 20) await new Promise((resolve) => setTimeout(resolve, 20)); // a loaded machine starts Node slowly
    expect(monitor.latest()).toEqual({ util: 50, memMib: 2400 });
    const stopped = await monitor.stop();
    expect(monitor.readings().length).toBeGreaterThan(2);
    expect(monitor.readings()[0]).toEqual({ util: 37, memMib: 1200 });
    expect(stopped.ticks).toBeGreaterThanOrEqual(0);
    expect(await monitor.stop()).toEqual(stopped); // stopping twice is harmless
  });

  it("has no readings and no CPU when the tool is missing", async () => {
    const monitor = new NvidiaSmi("/nonexistent/nvidia-smi").monitor(20, proc());
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(monitor.latest()).toBeNull();
    expect(await monitor.stop()).toEqual({ ticks: 0 });
  });
});
