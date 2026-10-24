// Peak memory of a process, for the run-metrics record (#1069). Diagnostics only: nothing here is
// hashed, read by the loop or part of a contract's identity, and a platform without /proc says null.

import { readFileSync } from "node:fs";

/** Parse the `VmHWM` line (peak resident set, in kB) of a /proc status file as bytes; null when it is absent or odd. */
export function parsePeakRss(status: string): number | null {
  const match = /^VmHWM:\s+(\d+)\s+kB$/m.exec(status);
  return match === null ? null : Number(match[1]) * 1024;
}

/** The peak resident set of process `pid` in bytes (Linux `/proc`), or null when it cannot be read. */
export function peakRssOf(pid: number | undefined, read: (path: string) => string = (path) => readFileSync(path, "utf8")): number | null {
  if (pid === undefined) return null;
  try {
    return parsePeakRss(read(`/proc/${pid}/status`));
  } catch {
    return null;
  }
}

/** The largest of the numbers that are present, or null when none is. */
export function largest(values: readonly (number | null)[]): number | null {
  const present = values.filter((value): value is number => value !== null);
  return present.length === 0 ? null : Math.max(...present);
}

/** This process's own peak resident set in bytes (`process.resourceUsage().maxRSS` is in kB). */
export function ownPeakRss(usage: () => { maxRSS: number } = () => process.resourceUsage()): number {
  return usage().maxRSS * 1024;
}
