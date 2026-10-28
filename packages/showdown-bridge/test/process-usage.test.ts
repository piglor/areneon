import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { largest, ownPeakRss, parsePeakRss, peakRssOf } from "../src/process-usage.ts";

describe("peak memory of a process (#1069)", () => {
  it("reads VmHWM from a /proc status file, in bytes", () => {
    expect(parsePeakRss("Name:\tnode\nVmHWM:\t  1234 kB\nVmRSS:\t 1000 kB\n")).toBe(1234 * 1024);
    expect(parsePeakRss("Name:\tnode\nVmRSS:\t 1000 kB\n")).toBeNull();
    expect(parsePeakRss("VmHWM:\tmany kB\n")).toBeNull();
  });

  it("is null where nothing can be read: no pid, no /proc, a file of another shape", () => {
    expect(peakRssOf(undefined)).toBeNull();
    expect(peakRssOf(4242, () => { throw new Error("no such process"); })).toBeNull();
    expect(peakRssOf(4242, () => "nothing useful")).toBeNull();
    expect(peakRssOf(4242, (path) => (path === "/proc/4242/status" ? "VmHWM:\t2 kB" : ""))).toBe(2048);
  });

  it("reads this very process on Linux", () => {
    const own = peakRssOf(process.pid);
    if (process.platform === "linux") expect(own).toBeGreaterThan(10_000_000);
    expect(ownPeakRss()).toBeGreaterThan(10_000_000);
    // The rusage figure is inherited across fork and exec (the driver's peak showed as the game process's, #1110): /proc wins.
    expect(ownPeakRss(() => ({ maxRSS: 9_000_000 }), () => "VmHWM:\t5 kB")).toBe(5 * 1024);
    // Where /proc cannot be read, the rusage figure stands in.
    expect(ownPeakRss(() => ({ maxRSS: 3 }), () => { throw new Error("no /proc"); })).toBe(3072);
  });

  it("does not inherit the parent's peak across exec (#1110)", () => {
    if (process.platform !== "linux") return;
    const probe = "const{readFileSync}=require('fs');const m=/VmHWM:\\s+(\\d+)/.exec(readFileSync('/proc/self/status','utf8'));console.log(m[1]*1024)";
    // This process holds far more than a fresh node does; a child's own peak must not be this process's.
    const ballast = Buffer.alloc(300 * 1024 * 1024, 1);
    const child = Number(execFileSync(process.execPath, ["-e", probe], { encoding: "utf8" }));
    expect(ballast.length).toBeGreaterThan(0);
    expect(child).toBeLessThan(ownPeakRss());
  });

  it("takes the largest figure that is present", () => {
    expect(largest([])).toBeNull();
    expect(largest([null, null])).toBeNull();
    expect(largest([null, 3, 9, 4])).toBe(9);
  });
});
