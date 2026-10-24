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
    expect(ownPeakRss(() => ({ maxRSS: 3 }))).toBe(3072);
  });

  it("takes the largest figure that is present", () => {
    expect(largest([])).toBeNull();
    expect(largest([null, null])).toBeNull();
    expect(largest([null, 3, 9, 4])).toBe(9);
  });
});
