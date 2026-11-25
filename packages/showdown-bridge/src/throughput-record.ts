// The ThroughputRecord and its ArmReport (ADR-030 step 5, the baseline harness): one measured run of one
// arm of the throughput comparison. The schema checks the shape; the rules the contract profile cannot say
// are checked here: every derived figure is recomputed (rates, sums, void reasons), lists are sorted and
// unique, and the record's own flags agree with its numbers. A message states a rule and never repeats a value.
//
// The harness (packages/team-dataset/scripts/throughput-bench.ts) is the only writer of the record; an arm
// (the Showdown path, a native engine, a GPU engine) writes the ArmReport it embeds. Wall-clock and CPU values
// differ on every run, so a record is a measurement, never an input to a run or a hash that matters.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/throughput-record.schema.json" with { type: "json" };
import type { ArmReport, Rates, ThroughputRecord, VoidReason } from "@areneon/schema/types/v1/throughput-record.ts";
import { type Json, jcs } from "./canonical-json.ts";

export type { ArmReport, Rates, ThroughputRecord, VoidReason };
export type Phase = ArmReport["phases"][number];
export type Role = ThroughputRecord["cpu"]["by_role"][number];

/** Milliseconds in an hour: the unit every rate is stated in. */
export const HOUR_MS = 3_600_000;

/** The kernel credits CPU time to /proc in clock ticks of USER_HZ = 100 per second: the resolution of every CPU figure. */
export const CLOCK_TICK_MS = 10;

export class ThroughputRecordError extends Error {
  override name = "ThroughputRecordError";
}

const fail = (message: string): never => {
  throw new ThroughputRecordError(message);
};

const ajv = ajvFor("contract");
const validRecord = ajv.compile(schema);
// The ArmReport is a definition of the same schema: an arm validates its file against it alone.
const validReport = ajv.compile({ $ref: `${schema.$id}#/$defs/ArmReport` });

/** SHA-256 of a record's JCS bytes. */
export const throughputRecordHash = (record: ThroughputRecord): string => createHash("sha256").update(jcs(record as unknown as Json)).digest("hex");

/** `count x 3,600,000 / ms`, truncated (exact in integers whatever the size); 0 when no time was measured. */
export function perHour(count: number, ms: number): number {
  if (ms <= 0) return 0;
  return Number((BigInt(count) * BigInt(HOUR_MS)) / BigInt(ms));
}

type Measured = Pick<ThroughputRecord, "report" | "cpu" | "window" | "gpu">;

/** The rates a record states, from its own measures. */
export function ratesOf(record: Measured): Rates {
  const { counts, phases } = record.report;
  const outside = phases.filter((phase) => phase.kind !== "play");
  const known = outside.length > 0 && outside.every((phase) => phase.cpu_ms !== null);
  const steadyMs = known ? record.cpu.total_ms - outside.reduce((sum, phase) => sum + (phase.cpu_ms as number), 0) : 0;
  return {
    decisions_per_core_hour: perHour(counts.decisions, record.cpu.total_ms),
    games_per_core_hour: perHour(counts.games, record.cpu.total_ms),
    decisions_per_wall_hour: perHour(counts.decisions, record.window.wall_ms),
    decisions_per_gpu_hour: record.report.gpu_used ? perHour(counts.decisions, record.gpu.allocated_ms) : null,
    decisions_per_core_hour_steady: known && steadyMs > 0 ? perHour(counts.decisions, steadyMs) : null,
  };
}

/** Why a run is void, sorted and unique: the reasons a record's own numbers imply. */
export function voidReasonsOf(record: Pick<ThroughputRecord, "report" | "cpu" | "window" | "conditions">): VoidReason[] {
  const reasons = new Set<VoidReason>();
  if (record.window.exit_code !== 0) reasons.add("arm_failed");
  if (record.window.aborted) reasons.add("aborted");
  if (record.cpu.survivors > 0) reasons.add("survivors");
  if (record.report.counts.decisions === 0) reasons.add("no_decisions");
  for (const breach of record.conditions.breaches) reasons.add(breach.code);
  // One tick of slack a process: the kernel scales a process's CPU split at exit, so a walk can read a tick more than it finally credits.
  if (record.cpu.total_ms === 0 || record.cpu.sampled_ms > record.cpu.total_ms + CLOCK_TICK_MS * record.cpu.processes_seen) reasons.add("accounting_mismatch");
  return [...reasons].sort();
}

const sortedByName = (names: readonly string[]): boolean => names.every((name, i) => i === 0 || (names[i - 1] as string) < name);
const sameList = (a: readonly unknown[], b: readonly unknown[]): boolean => a.length === b.length && a.every((value, i) => value === b[i]);

function checkReport(report: ArmReport): void {
  if (!sortedByName(report.arm.parameters.map((p) => p.name))) fail("parameters must be sorted by name, each once");
  if (new Set(report.phases.map((phase) => phase.name)).size !== report.phases.length) fail("phases must name each phase once");
  if (!sortedByName(report.stages.map((stage) => stage.name))) fail("stages must be sorted by name, each once");
  if (report.counts.games_forfeited > report.counts.games) fail("forfeited games cannot exceed games");
  if (report.counts.games_on_showdown > report.counts.games) fail("games on Showdown cannot exceed games");
}

/** Validate the report an arm left: the schema, then its rules. */
export function readArmReport(value: unknown): ArmReport {
  if (!validReport(value)) return fail(`invalid ArmReport: ${ajv.errorsText(validReport.errors)}`);
  const report = value as unknown as ArmReport;
  checkReport(report);
  return report;
}

function checkCpu(record: ThroughputRecord): void {
  const { cpu } = record;
  if (cpu.total_ms !== cpu.user_ms + cpu.system_ms) fail("total_ms is not user_ms plus system_ms");
  if (!sortedByName(cpu.by_role.map((role) => role.role))) fail("by_role must be sorted by role, each once");
  if (cpu.sampled_ms !== cpu.by_role.reduce((sum, role) => sum + role.cpu_ms, 0)) fail("sampled_ms is not the sum of the roles");
  if (cpu.unsampled_ms !== Math.max(0, cpu.total_ms - cpu.sampled_ms)) fail("unsampled_ms is not the CPU the walks did not see");
  if (cpu.processes_seen !== cpu.by_role.reduce((sum, role) => sum + role.processes, 0)) fail("processes_seen is not the sum of the roles");
  if (cpu.by_role.some((role) => role.peak_rss_bytes > cpu.peak_tree_rss_bytes)) fail("peak_tree_rss_bytes is below a role's peak");
}

function checkGpu(record: ThroughputRecord): void {
  const { gpu, report, window } = record;
  if (gpu.used !== report.gpu_used) fail("gpu.used is not the arm's gpu_used");
  if (gpu.allocated_ms !== (gpu.used ? window.wall_ms : 0)) fail("allocated_ms is not the wall clock of an arm that uses the GPU, and 0 for one that does not");
  if (!gpu.available && gpu.samples > 0) fail("an absent GPU has no readings");
  if ((gpu.samples === 0) !== (gpu.util_mean_ppm === null && gpu.mem_peak_mib === null)) fail("no GPU readings means no mean and no peak, and readings mean both");
}

function checkConditions(record: ThroughputRecord): void {
  const { conditions, window } = record;
  const { start, end } = conditions;
  if (conditions.samples < 2) fail("samples must include the start and the end");
  if (conditions.min_mem_available_kib > Math.min(start.mem_available_kib, end.mem_available_kib)) fail("min_mem_available_kib is above a sample");
  if (conditions.max_swap_used_kib < Math.max(start.swap_used_kib, end.swap_used_kib)) fail("max_swap_used_kib is below a sample");
  const times = conditions.breaches.map((breach) => breach.at_ms);
  if (times.some((at, i) => i > 0 && at < (times[i - 1] as number))) fail("breaches must be in time order");
  if (times.some((at) => at > window.wall_ms)) fail("a breach after the run ended is not a breach of the run");
}

/** Validate a record: the schema, then every rule it cannot say. */
export function readThroughputRecord(value: unknown): ThroughputRecord {
  if (!validRecord(value)) return fail(`invalid ThroughputRecord: ${ajv.errorsText(validRecord.errors)}`);
  const record = value as unknown as ThroughputRecord;
  checkReport(record.report);
  checkCpu(record);
  checkGpu(record);
  checkConditions(record);
  const reasons = voidReasonsOf(record);
  if (!sameList(reasons, record.void_reasons)) fail("void_reasons are not the ones the record implies");
  if (record.valid !== (record.void_reasons.length === 0)) fail("valid is not whether void_reasons is empty");
  const rates = ratesOf(record);
  for (const key of Object.keys(rates) as (keyof Rates)[]) {
    if (rates[key] !== record.rates[key]) fail(`${key} is not the count over the CPU time, wall clock or GPU time it is measured against`);
  }
  return record;
}
