// The ThroughputRecord and ArmReport readers (ADR-030 step 5 baseline harness) against hand-worked numbers and the shared sample corpus.
/* eslint-disable @typescript-eslint/no-explicit-any */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { SAMPLES_DIR } from "../src/samples.ts";
import { throughputSamples } from "../src/throughput-samples.ts";
import { CLOCK_TICK_MS, HOUR_MS, perHour, ratesOf, readArmReport, readThroughputRecord, ThroughputRecordError, throughputRecordHash, voidReasonsOf, type ThroughputRecord } from "../src/throughput-record.ts";

const DIR = join(SAMPLES_DIR, "v1", "throughput-record");
const files = (kind: string): string[] => readdirSync(join(DIR, kind)).sort();
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(join(DIR, kind, name), "utf8")) as unknown;
const valid = (): ThroughputRecord => readThroughputRecord(sample("valid", "showdown-idle-4-processes.json"));
const mutated = (change: (copy: Record<string, any>) => void): unknown => {
  const copy = structuredClone(valid()) as unknown as Record<string, any>;
  change(copy);
  return copy;
};

describe("perHour (count x 3,600,000 / ms, truncated, 0 when the time is 0)", () => {
  it("is exact on hand-worked values", () => {
    expect(HOUR_MS).toBe(3_600_000);
    expect(perHour(10_800, 1_520_000)).toBe(25_578); // 10,800 x 3,600,000 / 1,520,000 = 25,578.9..
    expect(perHour(1, 3_600_000)).toBe(1);
    expect(perHour(1, 3_600_001)).toBe(0);
    expect(perHour(5, 0)).toBe(0);
    expect(perHour(0, 100)).toBe(0);
  });

  it("does not lose precision where a double would (count x 3.6e6 above 2^53)", () => {
    // 9,007,199,254,740,000 decisions would overflow a double product; BigInt keeps the exact floor.
    expect(perHour(9_007_199_254_740, 3_600_000)).toBe(9_007_199_254_740);
    expect(perHour(9_007_199_254_740, 1_800_000)).toBe(18_014_398_509_480);
  });
});

describe("ratesOf", () => {
  const base = valid();
  it("states every rate from the record's own measures", () => {
    const rates = ratesOf(base);
    expect(rates.decisions_per_core_hour).toBe(perHour(base.report.counts.decisions, base.cpu.total_ms));
    expect(rates.games_per_core_hour).toBe(perHour(base.report.counts.games, base.cpu.total_ms));
    expect(rates.decisions_per_wall_hour).toBe(perHour(base.report.counts.decisions, base.window.wall_ms));
    expect(rates.decisions_per_gpu_hour).toBeNull(); // the Showdown arm does not use the GPU
    expect(rates.decisions_per_core_hour_steady).toBeNull(); // its phases do not state their CPU
    expect(rates).toEqual(base.rates);
  });

  it("states decisions per GPU-hour over the allocated GPU time only for an arm that uses the GPU", () => {
    const gpu = readThroughputRecord(sample("valid", "gpu-arm-with-steady-rate.json"));
    expect(gpu.gpu.used).toBe(true);
    expect(gpu.rates.decisions_per_gpu_hour).toBe(perHour(gpu.report.counts.decisions, gpu.gpu.allocated_ms));
    expect(gpu.gpu.allocated_ms).toBe(gpu.window.wall_ms);
  });

  it("subtracts the CPU of every warmup and teardown phase for the steady rate, only when each states it", () => {
    const gpu = readThroughputRecord(sample("valid", "gpu-arm-with-steady-rate.json"));
    const outside = gpu.report.phases.filter((p) => p.kind !== "play").reduce((sum, p) => sum + (p.cpu_ms as number), 0);
    expect(outside).toBeGreaterThan(0);
    expect(gpu.rates.decisions_per_core_hour_steady).toBe(perHour(gpu.report.counts.decisions, gpu.cpu.total_ms - outside));
    expect(gpu.rates.decisions_per_core_hour_steady).toBeGreaterThan(gpu.rates.decisions_per_core_hour);
    // One unknown phase CPU, or no phase outside play, leaves it null.
    const unknown = structuredClone(gpu);
    (unknown.report.phases[0] as { cpu_ms: number | null }).cpu_ms = null;
    expect(ratesOf(unknown).decisions_per_core_hour_steady).toBeNull();
    const playOnly = structuredClone(gpu);
    playOnly.report.phases = playOnly.report.phases.filter((p) => p.kind === "play");
    expect(ratesOf(playOnly).decisions_per_core_hour_steady).toBeNull();
    // Phases that claim more CPU than the run spent leave nothing to divide by.
    const over = structuredClone(gpu);
    (over.report.phases[0] as { cpu_ms: number | null }).cpu_ms = over.cpu.total_ms + 1;
    expect(ratesOf(over).decisions_per_core_hour_steady).toBeNull();
  });

  it("is 0 for a run that took no CPU or no time", () => {
    const empty = structuredClone(base);
    empty.cpu.total_ms = 0;
    empty.window.wall_ms = 0;
    expect(ratesOf(empty)).toMatchObject({ decisions_per_core_hour: 0, games_per_core_hour: 0, decisions_per_wall_hour: 0 });
  });
});

describe("voidReasonsOf (a run is void for exactly these reasons)", () => {
  const base = valid();
  it("is empty for a clean run", () => {
    expect(voidReasonsOf(base)).toEqual([]);
  });

  it("names each reason, sorted and unique", () => {
    const failed = structuredClone(base);
    failed.window.exit_code = 1;
    failed.window.aborted = true;
    failed.cpu.survivors = 2;
    failed.report.counts.decisions = 0;
    failed.conditions.breaches = [
      { code: "swap_growth", at_ms: 5 },
      { code: "memory_run", at_ms: 9 },
      { code: "swap_growth", at_ms: 12 },
      { code: "other_run", at_ms: 20 },
    ];
    expect(voidReasonsOf(failed)).toEqual(["aborted", "arm_failed", "memory_run", "no_decisions", "other_run", "survivors", "swap_growth"]);
  });

  it("calls CPU accounting a mismatch when nothing was counted or the walks saw more than the kernel credited, beyond one tick a process", () => {
    const none = structuredClone(base);
    none.cpu.total_ms = 0;
    expect(voidReasonsOf(none)).toContain("accounting_mismatch");
    const slack = CLOCK_TICK_MS * base.cpu.processes_seen;
    const within = structuredClone(base);
    within.cpu.sampled_ms = within.cpu.total_ms + slack;
    expect(voidReasonsOf(within)).not.toContain("accounting_mismatch");
    const beyond = structuredClone(base);
    beyond.cpu.sampled_ms = beyond.cpu.total_ms + slack + 1;
    expect(voidReasonsOf(beyond)).toContain("accounting_mismatch");
  });
});

describe("readThroughputRecord", () => {
  it("reads every valid sample and hashes its JCS bytes", () => {
    expect(files("valid")).toEqual(["gpu-arm-with-steady-rate.json", "non-idle-void-memory.json", "showdown-idle-4-processes.json"]);
    for (const name of files("valid")) {
      const record = readThroughputRecord(sample("valid", name));
      expect(record.data_class).toBe("synthetic");
      expect(throughputRecordHash(record)).toMatch(/^[0-9a-f]{64}$/);
      expect(record.valid).toBe(record.void_reasons.length === 0);
    }
  });

  it("keeps a void run as a record: valid false with its reasons", () => {
    const record = readThroughputRecord(sample("valid", "non-idle-void-memory.json"));
    expect(record.valid).toBe(false);
    expect(record.void_reasons).toEqual(["memory_run"]);
    expect(record.conditions.mode).toBe("non-idle");
  });

  it("refuses every invalid sample by the schema and every reader-invalid one by a rule", () => {
    for (const name of files("invalid")) expect(() => readThroughputRecord(sample("invalid", name)), name).toThrow(/invalid ThroughputRecord/);
    for (const name of files("reader-invalid")) {
      const value = sample("reader-invalid", name);
      expect(() => readThroughputRecord(value), name).toThrow(ThroughputRecordError);
      expect(() => readThroughputRecord(value), name).not.toThrow(/invalid ThroughputRecord/); // the schema passes it
    }
  });

  it("states a rule and never repeats a value", () => {
    const record = mutated((m) => (m.rates.decisions_per_core_hour += 7));
    expect(() => readThroughputRecord(record)).toThrow("decisions_per_core_hour is not the count over the CPU time");
    expect(() => readThroughputRecord(record)).not.toThrow(String(valid().rates.decisions_per_core_hour + 7));
  });

  it("recomputes every derived field", () => {
    const rules: [string, (m: Record<string, any>) => void, string][] = [
      ["games_per_core_hour", (m) => (m.rates.games_per_core_hour += 1), "games_per_core_hour"],
      ["decisions_per_wall_hour", (m) => (m.rates.decisions_per_wall_hour += 1), "decisions_per_wall_hour"],
      ["steady rate", (m) => (m.rates.decisions_per_core_hour_steady = 5), "decisions_per_core_hour_steady"],
      ["total", (m) => (m.cpu.user_ms += 10), "total_ms is not user_ms plus system_ms"],
      ["sampled", (m) => (m.cpu.sampled_ms += 10), "sampled_ms is not the sum of the roles"],
      ["unsampled", (m) => (m.cpu.unsampled_ms += 10), "unsampled_ms is not the CPU the walks did not see"],
      ["processes", (m) => (m.cpu.processes_seen += 1), "processes_seen is not the sum of the roles"],
      ["peak rss", (m) => (m.cpu.peak_tree_rss_bytes = 1), "peak_tree_rss_bytes is below a role's peak"],
      ["roles unsorted", (m) => m.cpu.by_role.reverse(), "by_role must be sorted by role, each once"],
      ["gpu used", (m) => (m.gpu.used = true), "gpu.used is not the arm's gpu_used"],
      ["gpu time", (m) => (m.gpu.allocated_ms = 1), "allocated_ms is not the wall clock"],
      ["gpu samples", (m) => (m.gpu.samples = 0), "no GPU readings means no mean and no peak"],
      ["gpu absent", (m) => (m.gpu.available = false), "an absent GPU has no readings"],
      ["valid flag", (m) => (m.valid = false), "valid is not whether void_reasons is empty"],
      ["void reasons", (m) => ((m.valid = false), (m.void_reasons = ["aborted"])), "void_reasons are not the ones the record implies"],
      ["breaches unsorted", (m) => (m.conditions.breaches = [{ code: "memory_run", at_ms: 9 }, { code: "other_run", at_ms: 3 }]), "breaches must be in time order"],
      ["breach after the run", (m) => (m.conditions.breaches = [{ code: "memory_run", at_ms: m.window.wall_ms + 1 }]), "a breach after the run ended"],
      ["min memory", (m) => (m.conditions.min_mem_available_kib = m.conditions.start.mem_available_kib + 1), "min_mem_available_kib is above a sample"],
      ["max swap", (m) => (m.conditions.start.swap_used_kib = m.conditions.max_swap_used_kib + 1), "max_swap_used_kib is below a sample"],
      ["samples", (m) => (m.conditions.samples = 1), "samples must include the start and the end"],
      ["parameters", (m) => m.report.arm.parameters.reverse(), "parameters must be sorted by name, each once"],
      ["phases", (m) => m.report.phases.push({ ...m.report.phases[0] }), "phases must name each phase once"],
      ["stages", (m) => m.report.stages.reverse(), "stages must be sorted by name, each once"],
      ["forfeits", (m) => (m.report.counts.games_forfeited = m.report.counts.games + 1), "forfeited games cannot exceed games"],
      ["showdown games", (m) => (m.report.counts.games_on_showdown = m.report.counts.games + 1), "games on Showdown cannot exceed games"],
    ];
    for (const [name, change, message] of rules) expect(() => readThroughputRecord(mutated(change)), name).toThrow(message);
  });
});

describe("readArmReport (what an arm leaves for the harness)", () => {
  it("is the report a record embeds, validated on its own", () => {
    const report = valid().report;
    expect(readArmReport(structuredClone(report))).toEqual(report);
  });

  it("refuses a malformed report by the schema and by its rules", () => {
    const report = structuredClone(valid().report) as unknown as Record<string, any>;
    expect(() => readArmReport({ ...report, extra: 1 })).toThrow(/invalid ArmReport/);
    expect(() => readArmReport({ ...report, counts: { ...report.counts, decisions: 1.5 } })).toThrow(/invalid ArmReport/);
    expect(() => readArmReport({ ...report, counts: { ...report.counts, games_forfeited: report.counts.games + 1 } })).toThrow("forfeited games cannot exceed games");
    expect(() => readArmReport({ ...report, stages: [...report.stages].reverse() })).toThrow("stages must be sorted by name, each once");
  });
});

describe("the corpus builder", () => {
  it("makes samples the reader classifies as their directory says", () => {
    const { valid: ok, invalid, readerInvalid } = throughputSamples();
    for (const value of Object.values(ok)) expect(() => readThroughputRecord(value)).not.toThrow();
    for (const value of Object.values(invalid)) expect(() => readThroughputRecord(value)).toThrow(/invalid ThroughputRecord/);
    for (const value of Object.values(readerInvalid)) expect(() => readThroughputRecord(value)).toThrow(ThroughputRecordError);
  });
});
