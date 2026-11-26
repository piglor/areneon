// The Markdown summaries of the throughput harness: one run, and the median and range of several (voids left out, mixed plans flagged).
import { readFileSync } from "node:fs";
import { type ThroughputRecord, ratesOf, voidReasonsOf } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { DEFINITION, medianInt, renderComparison, renderRunSummary } from "../../src/throughput/summary.ts";

const VALID = new URL("../../../../fixtures/contracts/v1/throughput-record/valid/", import.meta.url);
const load = (name: string): ThroughputRecord => JSON.parse(readFileSync(new URL(name, VALID), "utf8")) as ThroughputRecord;
const finish = (record: ThroughputRecord): ThroughputRecord => {
  record.void_reasons = voidReasonsOf(record);
  record.valid = record.void_reasons.length === 0;
  record.rates = ratesOf(record);
  return record;
};
const variant = (run: string, change: (r: ThroughputRecord) => void): ThroughputRecord => {
  const copy = structuredClone(load("showdown-idle-4-processes.json"));
  copy.run_id = run;
  change(copy);
  return finish(copy);
};

describe("medianInt", () => {
  it("is the middle value, the floor of the mean of the two middle ones for an even count, and null for none", () => {
    expect(medianInt([5, 1, 3])).toBe(3);
    expect(medianInt([4, 1, 3, 2])).toBe(2); // (2 + 3) / 2 = 2.5, truncated
    expect(medianInt([7])).toBe(7);
    expect(medianInt([])).toBeNull();
  });
});

describe("renderRunSummary", () => {
  const record = load("showdown-idle-4-processes.json");
  const text = renderRunSummary(record, [record.conditions.start, record.conditions.end]);

  it("states the verdict, the arm and the headline number with its definition", () => {
    expect(text).toContain("# Throughput run example-run");
    expect(text).toContain("**VALID**");
    expect(text).toContain("showdown");
    expect(text).toContain(`| decisions per host-core-hour | ${record.rates.decisions_per_core_hour.toLocaleString("en-US")} |`);
    expect(text).toContain(DEFINITION);
    expect(DEFINITION).toContain("3,600,000");
  });

  it("lists the counts, the host CPU by role, the phases, the stages and the conditions", () => {
    expect(text).toContain("| games | 768 |");
    expect(text).toContain("| game_process | 4 |");
    expect(text).toContain("| policy_worker | 4 |");
    expect(text).toContain("| pool_load | warmup |");
    expect(text).toContain("| policy_forward |");
    expect(text).toContain("idle");
    expect(text).toContain("MemAvailable");
    expect(text.match(/^\| 2023-/gm)?.length).toBe(2); // the two condition samples, as dates
  });

  it("says VOID with every reason, and labels a non-idle run as a relative measurement", () => {
    const void_ = load("non-idle-void-memory.json");
    const voidText = renderRunSummary(void_, []);
    expect(voidText).toContain("**VOID**");
    expect(voidText).toContain("memory_run");
    expect(voidText).toContain("NON-IDLE");
    expect(voidText).toContain("never a gate figure");
    expect(voidText).toContain("load average at most not enforced");
    expect(voidText).toContain("swap growth at most not enforced");
  });

  it("leaves the GPU rate out for an arm that does not use the GPU, and states the steady rate when there is one", () => {
    expect(text).toContain("| decisions per GPU-hour | n/a |");
    expect(text).toContain("| decisions per host-core-hour, warm-up and tear-down excluded | n/a |");
    const gpu = renderRunSummary(load("gpu-arm-with-steady-rate.json"), []);
    expect(gpu).not.toContain("| decisions per GPU-hour | n/a |");
    expect(gpu).not.toContain("excluded | n/a |");
  });
});

describe("renderComparison", () => {
  it("gives per arm the median and range of the valid runs and lists the void ones apart", () => {
    const a = variant("r1", (r) => (r.report.counts.decisions = 10_000));
    const b = variant("r2", (r) => (r.report.counts.decisions = 12_000));
    const c = variant("r3", (r) => (r.report.counts.decisions = 14_000));
    const broken = variant("r4", (r) => (r.window.exit_code = 1));
    const text = renderComparison([a, b, c, broken]);
    const rates = [a, b, c].map((r) => r.rates.decisions_per_core_hour);
    expect(text).toContain(`| showdown game_processes=4,policy_workers=1 | 3 of 4 | ${medianInt(rates)?.toLocaleString("en-US")} | ${Math.min(...rates).toLocaleString("en-US")} to ${Math.max(...rates).toLocaleString("en-US")} |`);
    expect(text).toContain("void runs");
    expect(text).toContain("r4 (arm_failed)");
  });

  it("separates arms by their parameters, and flags runs of one arm that played different plans", () => {
    const four = variant("a", () => undefined);
    const six = variant("b", (r) => (r.report.arm.parameters = [{ name: "game_processes", value: 6 }, { name: "policy_workers", value: 1 }]));
    const other = variant("c", (r) => (r.report.plan.pool_hash = "9".repeat(64)));
    const text = renderComparison([four, six, other]);
    expect(text).toContain("game_processes=6,policy_workers=1");
    expect(text).toContain("not comparable");
    expect(text).toContain("pool_hash");
  });

  it("says so when there is nothing to compare", () => {
    expect(renderComparison([])).toContain("no records");
  });
});
