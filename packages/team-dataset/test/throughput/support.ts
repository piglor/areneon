// Shared by the throughput harness tests: scripted machine, GPU and code stand-ins and the fake arm's command.
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GIB_KIB, type Sample } from "../../src/throughput/guard.ts";
import { type Deps } from "../../src/throughput/runner.ts";
import { Proc } from "../../src/throughput/procfs.ts";

export const FAKE_ARM = new URL("./fake-arm.ts", import.meta.url).pathname;
export const tempDir = (): string => mkdtempSync(join(tmpdir(), "throughput-"));

export const quiet = (extra: Partial<Sample> = {}): Sample => ({
  at_epoch_ms: 1_700_000_000_000,
  load1_centi: 30,
  load5_centi: 80,
  mem_available_kib: 20 * GIB_KIB,
  swap_used_kib: 0,
  cpu_busy_ppm: 10_000,
  gpu_util_pct: null,
  gpu_mem_used_mib: null,
  ...extra,
});

export interface Script {
  /** The sample the n-th reading (0 is the start sample, 1.. the minute samples and the end) returns. */
  samples?: (n: number) => Partial<Sample>;
  /** MemAvailable at the n-th quick reading (the runner takes one per walk of the process tree); by default the same as the samples'. */
  available?: (n: number) => number;
  /** The other runs found at the n-th look (0 is the look before the start). */
  others?: (n: number) => string[];
  gpuOnce?: { util: number; memMib: number } | null;
  gpuReadings?: { util: number; memMib: number }[];
}

/** Deps with a scripted machine, GPU and code, over the real /proc, spawn and clock. */
export function fakeDeps(script: Script = {}): Deps {
  let readings = 0;
  let looks = 0;
  let quick = 0;
  return {
    proc: new Proc(),
    machine: {
      sampleOverWindow: async (gpu) => ({ ...quiet(), ...script.samples?.(readings++), gpu_util_pct: gpu?.util ?? null, gpu_mem_used_mib: gpu?.memMib ?? null }),
      sample: (gpu) => ({ ...quiet(), ...script.samples?.(readings++), gpu_util_pct: gpu?.util ?? null, gpu_mem_used_mib: gpu?.memMib ?? null }),
      availableKib: () => script.available?.(quick++) ?? quiet().mem_available_kib,
      others: () => script.others?.(looks++) ?? [],
    },
    gpu: {
      once: async () => script.gpuOnce ?? null,
      monitor: () => {
        const list = script.gpuReadings ?? [];
        return { latest: () => list[list.length - 1] ?? null, readings: () => list, stop: async () => ({ ticks: 7 }) };
      },
    },
    now: () => Date.now(),
    mono: () => performance.now(),
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    code: () => ({ commit: "a".repeat(40), engine_pin: "b".repeat(40), tree_clean: true }),
    machineInfo: () => ({ logical_cpus: 4, mem_total_kib: 16 * GIB_KIB, cpu_model: "Fake CPU" }),
    pageBytes: 4096,
  };
}
