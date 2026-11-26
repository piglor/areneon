// The ThroughputRecord's share of the sample corpus (ADR-008 section 2, ADR-030 step 5 baseline harness).
// Placeholders that pin a shape: counts and times of nothing real, hashes that are single repeated digits.

import { ratesOf, voidReasonsOf, type ThroughputRecord } from "./throughput-record.ts";

type Cases = Record<string, unknown>;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Draft = Record<string, any>;
const hexOf = (c: string, n = 64): string => c.repeat(n);
const mutated = (value: unknown, change: (copy: Draft) => void): Draft => {
  const copy = structuredClone(value) as Draft;
  change(copy);
  return copy;
};

const sampleAt = (at: number, extra: Draft = {}): Draft => ({
  at_epoch_ms: at,
  load1_centi: 40,
  load5_centi: 90,
  mem_available_kib: 20_000_000,
  swap_used_kib: 0,
  cpu_busy_ppm: 20_000,
  gpu_util_pct: 0,
  gpu_mem_used_mib: 300,
  ...extra,
});

const idleThresholds = {
  max_load1_centi: 100,
  max_cpu_busy_ppm: 70_000,
  min_mem_available_start_kib: 8 * 1024 * 1024,
  min_mem_available_run_kib: 4 * 1024 * 1024,
  max_swap_growth_kib: 0,
  forbid_other_runs: true,
  require_idle_gpu: true,
  max_gpu_util_pct: 5,
  max_gpu_mem_used_mib: 2000,
};

/** A record with its derived fields (void reasons, valid flag, rates) filled in from its measures, as the harness writes it. */
function finished(draft: Draft): Draft {
  const record = draft as ThroughputRecord;
  record.void_reasons = voidReasonsOf(record);
  record.valid = record.void_reasons.length === 0;
  record.rates = ratesOf(record);
  return draft;
}

const showdownReport = (): Draft => ({
  schema_version: 1,
  arm: { id: "showdown", label: "Showdown path: play-generation.ts, 4 game processes", parameters: [{ name: "game_processes", value: 4 }, { name: "policy_workers", value: 1 }] },
  plan: { pool_id: "example-pool", pool_hash: hexOf("1"), plan_hash: hexOf("2"), config_hash: hexOf("3") },
  counts: { games: 768, games_forfeited: 0, games_on_showdown: 768, decisions: 10_800, scripted_decisions: 4000 },
  phases: [
    { name: "pool_load", kind: "warmup", wall_ms: 34_000, cpu_ms: null },
    { name: "play", kind: "play", wall_ms: 280_000, cpu_ms: null },
    { name: "merge", kind: "teardown", wall_ms: 12_000, cpu_ms: null },
  ],
  stages: [
    { name: "engine_step", ms: 210_000 },
    { name: "features", ms: 520_000 },
    { name: "legal_mask", ms: 90_000 },
    { name: "observation", ms: 60_000 },
    { name: "policy_encode", ms: 70_000 },
    { name: "policy_forward", ms: 110_000 },
  ],
  gpu_used: false,
});

const base = (report: Draft, window: Draft, cpu: Draft, gpu: Draft, conditions: Draft = {}): Draft =>
  finished({
    schema_version: 1,
    data_class: "synthetic",
    run_id: "example-run",
    report,
    machine: { logical_cpus: 8, mem_total_kib: 16_777_216, cpu_model: "Example CPU" },
    code: { commit: hexOf("a", 40), engine_pin: hexOf("b", 40), tree_clean: true },
    conditions: {
      mode: "idle",
      thresholds: idleThresholds,
      start: sampleAt(1_700_000_000_000),
      end: sampleAt(1_700_000_000_000 + window.wall_ms, { cpu_busy_ppm: 560_000 }),
      interval_ms: 60_000,
      samples: 7,
      min_mem_available_kib: 14_000_000,
      max_swap_used_kib: 0,
      breaches: [],
      ...conditions,
    },
    window,
    cpu,
    gpu,
    rates: {},
    valid: true,
    void_reasons: [],
  });

function showdownIdle(): Draft {
  return base(
    showdownReport(),
    { started_epoch_ms: 1_700_000_000_000, wall_ms: 326_000, exit_code: 0, aborted: false },
    {
      total_ms: 1_520_000,
      user_ms: 1_490_000,
      system_ms: 30_000,
      helper_ms: 0,
      harness_ms: 3200,
      sampled_ms: 1_500_000,
      unsampled_ms: 20_000,
      processes_seen: 8,
      peak_tree_rss_bytes: 3_800_000_000,
      survivors: 0,
      by_role: [
        { role: "game_process", processes: 4, cpu_ms: 1_200_000, peak_rss_bytes: 800_000_000 },
        { role: "policy_worker", processes: 4, cpu_ms: 300_000, peak_rss_bytes: 450_000_000 },
      ],
    },
    { used: false, available: true, allocated_ms: 0, util_mean_ppm: 12_000, mem_peak_mib: 320, samples: 326 },
  );
}

function gpuArm(): Draft {
  const report: Draft = {
    schema_version: 1,
    arm: { id: "gpu", label: "GPU arm: whole game on the device", parameters: [{ name: "games_in_flight", value: 4096 }] },
    plan: { pool_id: "example-pool", pool_hash: hexOf("1"), plan_hash: hexOf("2"), config_hash: hexOf("3") },
    counts: { games: 2000, games_forfeited: 0, games_on_showdown: 0, decisions: 1_000_000, scripted_decisions: 0 },
    phases: [
      { name: "graph_capture", kind: "warmup", wall_ms: 4000, cpu_ms: 3500 },
      { name: "kernel_compile", kind: "warmup", wall_ms: 30_000, cpu_ms: 29_000 },
      { name: "play", kind: "play", wall_ms: 20_000, cpu_ms: null },
      { name: "record_write", kind: "teardown", wall_ms: 2000, cpu_ms: 1500 },
    ],
    stages: [
      { name: "engine_step", ms: 6000 },
      { name: "policy_forward", ms: 9000 },
    ],
    gpu_used: true,
  };
  return base(
    report,
    { started_epoch_ms: 1_700_000_000_000, wall_ms: 56_000, exit_code: 0, aborted: false },
    {
      total_ms: 90_000,
      user_ms: 84_000,
      system_ms: 6000,
      helper_ms: 250,
      harness_ms: 900,
      sampled_ms: 89_500,
      unsampled_ms: 500,
      processes_seen: 2,
      peak_tree_rss_bytes: 6_000_000_000,
      survivors: 0,
      by_role: [{ role: "host", processes: 2, cpu_ms: 89_500, peak_rss_bytes: 3_500_000_000 }],
    },
    { used: true, available: true, allocated_ms: 56_000, util_mean_ppm: 830_000, mem_peak_mib: 9100, samples: 56 },
  );
}

function nonIdleVoid(): Draft {
  const record = showdownIdle();
  return finished(
    mutated(record, (m) => {
      m.run_id = "example-non-idle";
      m.conditions.mode = "non-idle";
      // What a labelled non-idle run enforces: only the memory floors (the others are set where no machine reaches them).
      m.conditions.thresholds = { ...idleThresholds, max_load1_centi: 9_007_199_254_740_991, max_cpu_busy_ppm: 1_000_000, max_swap_growth_kib: 9_007_199_254_740_991, forbid_other_runs: false, require_idle_gpu: false, max_gpu_util_pct: 100, max_gpu_mem_used_mib: 9_007_199_254_740_991 };
      m.conditions.start.load1_centi = 950;
      m.conditions.min_mem_available_kib = 3_900_000;
      m.conditions.breaches = [{ code: "memory_run", at_ms: 120_000 }];
    }),
  );
}

export function throughputSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const good = showdownIdle();
  const gpu = gpuArm();
  const of = (change: (m: Draft) => void): Draft => mutated(good, change);
  const withRates = (change: (m: Draft) => void): Draft => finished(of(change));
  return {
    valid: {
      "throughput-record/showdown-idle-4-processes": good,
      "throughput-record/gpu-arm-with-steady-rate": gpu,
      "throughput-record/non-idle-void-memory": nonIdleVoid(),
    },
    invalid: {
      "throughput-record/public-class": of((m) => (m.data_class = "public")),
      "throughput-record/extra-field": of((m) => (m.notes = "x")),
      "throughput-record/next-version": of((m) => (m.schema_version = 2)),
      "throughput-record/fractional-cpu-ms": of((m) => (m.cpu.total_ms = 1_520_000.5)),
      "throughput-record/negative-decisions": of((m) => (m.report.counts.decisions = -1)),
      "throughput-record/short-config-hash": of((m) => (m.report.plan.config_hash = "abc")),
      "throughput-record/unknown-mode": of((m) => (m.conditions.mode = "busy")),
      "throughput-record/unknown-void-reason": of((m) => ((m.valid = false), (m.void_reasons = ["bored"]))),
      "throughput-record/unknown-phase-kind": of((m) => (m.report.phases[0].kind = "boot")),
      "throughput-record/free-text-run-id": of((m) => (m.run_id = "has spaces")),
      "throughput-record/no-cpu": of((m) => delete m.cpu),
    },
    readerInvalid: {
      "throughput-record/rate-not-recomputed": of((m) => (m.rates.decisions_per_core_hour += 1)),
      "throughput-record/total-not-user-plus-system": of((m) => (m.cpu.user_ms += 10)),
      "throughput-record/unsampled-not-remainder": of((m) => (m.cpu.unsampled_ms += 10)),
      "throughput-record/roles-unsorted": of((m) => m.cpu.by_role.reverse()),
      "throughput-record/valid-contradicts-void-reasons": of((m) => (m.valid = false)),
      "throughput-record/void-reasons-not-implied": of((m) => ((m.valid = false), (m.void_reasons = ["aborted"]))),
      "throughput-record/forfeits-over-games": of((m) => (m.report.counts.games_forfeited = m.report.counts.games + 1)),
      "throughput-record/gpu-used-differs-from-report": of((m) => (m.gpu.used = true)),
      "throughput-record/breach-after-the-run": withRates((m) => (m.conditions.breaches = [{ code: "memory_run", at_ms: m.window.wall_ms + 1 }])),
      "throughput-record/steady-rate-with-unknown-phase-cpu": mutated(gpu, (m) => (m.report.phases[0].cpu_ms = null)),
      "throughput-record/stages-unsorted": of((m) => m.report.stages.reverse()),
    },
  };
}
