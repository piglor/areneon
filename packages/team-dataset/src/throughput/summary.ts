// Markdown for the throughput harness: one run (what the owner journals) and a comparison of runs (median and range per arm).
import type { ThroughputRecord } from "@areneon/showdown-bridge";
import { NOT_ENFORCED, type Sample } from "./guard.ts";

/** The unit of comparison, stated in every summary so no figure travels without it. */
export const DEFINITION =
  "decisions per host-core-hour = full decisions x 3,600,000 / host CPU ms of the arm's whole process tree (user plus system, every process and thread, from launch to exit); " +
  "a full decision is one policy query answered with the engine step, legal mask, features, encoding, policy forward and sampling its request kind has. " +
  "Per-hour rates are truncated integers; start-up and tear-down are included unless the steady rate is stated.";

const num = (value: number): string => value.toLocaleString("en-US");
const orNa = (value: number | null): string => (value === null ? "n/a" : num(value));
const gib = (kib: number): string => (kib / (1024 * 1024)).toFixed(1);
const iso = (epochMs: number): string => `${new Date(epochMs).toISOString().slice(0, 19)}Z`;
const limit = (value: number, show: (n: number) => string): string => (value >= NOT_ENFORCED ? "not enforced" : show(value));

/** The middle value; the mean of the two middle values truncated for an even count; null for none. */
export function medianInt(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? (sorted[middle] as number) : Math.floor(((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2);
}

const table = (header: readonly string[], rows: readonly (readonly string[])[]): string[] => [`| ${header.join(" | ")} |`, `| ${header.map(() => "---").join(" | ")} |`, ...rows.map((row) => `| ${row.join(" | ")} |`)];

/** What the owner journals for one run: verdict, headline, counts, CPU by role, phases, stages and the condition samples. */
export function renderRunSummary(record: ThroughputRecord, samples: readonly Sample[]): string {
  const { report, cpu, window, rates, conditions, gpu } = record;
  const lines: string[] = [`# Throughput run ${record.run_id}`, ""];
  lines.push(record.valid ? "**VALID**" : `**VOID**: ${record.void_reasons.join(", ")}. A void run is never a gate figure.`);
  lines.push("");
  lines.push(`Arm ${report.arm.id}: ${report.arm.label}. Parameters: ${report.arm.parameters.map((p) => `${p.name}=${p.value}`).join(", ") || "none"}. Mode: ${conditions.mode}.`);
  if (conditions.mode === "non-idle") lines.push("", "**NON-IDLE**: a relative measurement beside other work, not an ADR-028 section 1 measurement; the figures are never gate values.");
  lines.push("", `Plan: pool ${report.plan.pool_id} (${report.plan.pool_hash ?? "no hash"}), plan ${report.plan.plan_hash ?? "no hash"}, config ${report.plan.config_hash ?? "no hash"}.`);
  lines.push(`Code: commit ${record.code.commit}${record.code.tree_clean ? "" : " (working tree not clean)"}, engine pin ${record.code.engine_pin}. Machine: ${record.machine.logical_cpus} logical CPUs, ${gib(record.machine.mem_total_kib)} GiB, ${record.machine.cpu_model}.`);

  lines.push("", "## Headline", "", ...table(["measure", "value"], [
    ["decisions per host-core-hour", num(rates.decisions_per_core_hour)],
    ["games per host-core-hour", num(rates.games_per_core_hour)],
    ["decisions per wall-hour", num(rates.decisions_per_wall_hour)],
    ["decisions per GPU-hour", orNa(rates.decisions_per_gpu_hour)],
    ["decisions per host-core-hour, warm-up and tear-down excluded", orNa(rates.decisions_per_core_hour_steady)],
  ]));
  lines.push("", `Definition: ${DEFINITION}`);

  lines.push("", "## Counts", "", ...table(["count", "value"], [
    ["games", num(report.counts.games)],
    ["games forfeited", num(report.counts.games_forfeited)],
    ["games on Showdown", num(report.counts.games_on_showdown)],
    ["decisions", num(report.counts.decisions)],
    ["scripted decisions", num(report.counts.scripted_decisions)],
  ]));

  lines.push("", "## Host CPU", "", `Wall clock ${num(window.wall_ms)} ms; exit code ${window.exit_code}${window.aborted ? " (ended by the harness)" : ""}. Total ${num(cpu.total_ms)} ms (user ${num(cpu.user_ms)}, system ${num(cpu.system_ms)}). Seen by the walks ${num(cpu.sampled_ms)} ms; the rest, ${num(cpu.unsampled_ms)} ms, is CPU of processes that ended between two walks. Harness itself ${num(cpu.harness_ms)} ms and monitors ${num(cpu.helper_ms)} ms (neither is part of the arm). Peak resident memory of the tree ${gib(Math.round(cpu.peak_tree_rss_bytes / 1024))} GiB; processes left behind: ${cpu.survivors}.`, "");
  lines.push(...table(["role", "processes", "cpu ms", "peak rss bytes"], cpu.by_role.map((role) => [role.role, num(role.processes), num(role.cpu_ms), num(role.peak_rss_bytes)])));

  if (report.phases.length > 0) lines.push("", "## Phases", "", ...table(["phase", "kind", "wall ms", "cpu ms"], report.phases.map((phase) => [phase.name, phase.kind, num(phase.wall_ms), orNa(phase.cpu_ms)])));
  if (report.stages.length > 0) lines.push("", "## Stages (process-milliseconds)", "", ...table(["stage", "ms"], report.stages.map((stage) => [stage.name, num(stage.ms)])));
  if (gpu.available) lines.push("", "## GPU", "", `Arm uses the GPU: ${gpu.used ? "yes" : "no"}. The card as a whole, any process: ${gpu.samples} readings; mean utilisation ${gpu.util_mean_ppm === null ? "n/a" : `${(gpu.util_mean_ppm / 10_000).toFixed(1)}%`}, peak memory ${orNa(gpu.mem_peak_mib)} MiB.`);

  const t = conditions.thresholds;
  lines.push("", "## Conditions (ADR-028 section 1)", "", `Mode ${conditions.mode}: load average at most ${limit(t.max_load1_centi, (n) => String(n / 100))}, MemAvailable at least ${gib(t.min_mem_available_start_kib)} GiB at the start and ${gib(t.min_mem_available_run_kib)} GiB throughout, swap growth at most ${limit(t.max_swap_growth_kib, (n) => `${num(n)} KiB`)}, other runs ${t.forbid_other_runs ? "forbidden" : "allowed"}. Lowest MemAvailable ${gib(conditions.min_mem_available_kib)} GiB; highest swap in use ${num(conditions.max_swap_used_kib)} KiB; ${conditions.samples} samples every ${num(conditions.interval_ms)} ms.`);
  lines.push("", conditions.breaches.length === 0 ? "Breaches: none." : `Breaches: ${conditions.breaches.map((b) => `${b.code} at ${num(b.at_ms)} ms`).join(", ")}.`);
  if (samples.length > 0) {
    lines.push("", ...table(["time (UTC)", "load1", "load5", "MemAvailable GiB", "swap KiB", "cpu busy %", "gpu % / MiB"], samples.map((s) => [iso(s.at_epoch_ms), (s.load1_centi / 100).toFixed(2), (s.load5_centi / 100).toFixed(2), gib(s.mem_available_kib), num(s.swap_used_kib), s.cpu_busy_ppm === null ? "n/a" : (s.cpu_busy_ppm / 10_000).toFixed(1), s.gpu_util_pct === null ? "n/a" : `${s.gpu_util_pct} / ${s.gpu_mem_used_mib}`])));
  }
  return `${lines.join("\n")}\n`;
}

const armKey = (record: ThroughputRecord): string => `${record.report.arm.id} ${record.report.arm.parameters.map((p) => `${p.name}=${p.value}`).join(",")}`.trim();

/** The facts two runs of one arm must share to be compared. */
const IDENTITY: readonly (readonly [string, (record: ThroughputRecord) => string])[] = [
  ["pool_hash", (r) => String(r.report.plan.pool_hash)],
  ["plan_hash", (r) => String(r.report.plan.plan_hash)],
  ["config_hash", (r) => String(r.report.plan.config_hash)],
  ["commit", (r) => r.code.commit],
  ["engine_pin", (r) => r.code.engine_pin],
  ["mode", (r) => r.conditions.mode],
];

/** Median and range of decisions per host-core-hour per arm over the valid runs; void runs are listed apart, and runs that played different plans are flagged. */
export function renderComparison(records: readonly ThroughputRecord[]): string {
  if (records.length === 0) return "# Throughput comparison\n\nno records.\n";
  const groups = new Map<string, ThroughputRecord[]>();
  for (const record of records) groups.set(armKey(record), [...(groups.get(armKey(record)) ?? []), record]);
  const rows: string[][] = [];
  const voids: string[] = [];
  const notes: string[] = [];
  for (const [key, group] of [...groups].sort(([a], [b]) => (a < b ? -1 : 1))) {
    const valid = group.filter((record) => record.valid);
    const rates = valid.map((record) => record.rates.decisions_per_core_hour);
    const walls = valid.map((record) => record.rates.decisions_per_wall_hour);
    rows.push([key, `${valid.length} of ${group.length}`, orNa(medianInt(rates)), rates.length === 0 ? "n/a" : `${num(Math.min(...rates))} to ${num(Math.max(...rates))}`, orNa(medianInt(walls))]);
    for (const record of group.filter((r) => !r.valid)) voids.push(`${record.run_id} (${record.void_reasons.join(", ")})`);
    const differing = IDENTITY.filter(([, read]) => new Set(valid.map(read)).size > 1).map(([name]) => name);
    if (differing.length > 0) notes.push(`${key}: not comparable, the runs differ in ${differing.join(", ")}.`);
  }
  const lines = ["# Throughput comparison", "", ...table(["arm", "valid runs", "decisions per host-core-hour, median", "range", "decisions per wall-hour, median"], rows)];
  if (voids.length > 0) lines.push("", `void runs (left out of the medians): ${voids.join("; ")}.`);
  if (notes.length > 0) lines.push("", ...notes);
  lines.push("", `Definition: ${DEFINITION}`);
  return `${lines.join("\n")}\n`;
}
