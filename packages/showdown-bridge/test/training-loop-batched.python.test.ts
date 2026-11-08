// A tiny training run whose self-play forward pass is batched (ADR-028 section 2 F2, #1195), end to end on CPU: the Python driver,
// Node, two game processes and the real policy workers started with --inference-max-batch. Skipped without the project's Python
// environment unless ARENEON_REQUIRE_PYTHON_WORKER=1 (the typescript CI job sets it).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { policyId } from "../src/policy-protocol.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(GOLDEN_DIR, "engine-data.json");
const TINY = join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json");
const BATCH = { kind: "batched", max_batch: 2 } as const;
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };

interface RunSummary {
  checkpoints: { battle: string; preview: string };
  policy_id: string;
}
const run = (config: string, runDir: string, ...extra: string[]): RunSummary =>
  JSON.parse(
    execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA, "--generations", "2", "--node", process.execPath, "--worker-python", PYTHON, ...extra], { encoding: "utf8", env }),
  ) as RunSummary;
const gen = (n: number): string => `g${String(n).padStart(6, "0")}`;
const json = <T>(runDir: string, ...path: string[]): T => JSON.parse(readFileSync(join(runDir, ...path), "utf8")) as T;

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("a batched training run, tiny and on CPU", () => {
  let dir: string;
  let batched: string;
  let plain: string;
  let summary: RunSummary;

  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-batched-"));
    const config = JSON.parse(readFileSync(TINY, "utf8")) as { generation: Record<string, unknown> };
    config.generation.inference = BATCH;
    const file = join(dir, "batched.json");
    writeFileSync(file, JSON.stringify(config));
    batched = join(dir, "batched");
    summary = run(file, batched, "--game-processes", "2");
    plain = join(dir, "plain");
    run(TINY, plain);
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays under the batched policy id, in the league record and the final summary", () => {
    expect(summary.policy_id).toBe(policyId(summary.checkpoints, "sample", {}, undefined, undefined, undefined, BATCH));
    const record = json<{ pair_policy_id: string }>(batched, "generations", gen(2), "league-run.json");
    const previous = json<{ checkpoints: { battle: string; preview: string } }>(batched, "generations", gen(1), "manifest.json").checkpoints;
    expect(record.pair_policy_id).toBe(policyId(previous, "sample", {}, undefined, undefined, undefined, BATCH));
    expect(record.pair_policy_id).not.toBe(policyId(previous, "sample"));
  });

  it("keeps the plan of the unbatched run (seeds, teams and baseline seats), which does not depend on how the games are scored", () => {
    for (const g of [1, 2]) {
      expect(readFileSync(join(batched, "generations", gen(g), "plan.json"), "utf8")).toBe(readFileSync(join(plain, "generations", gen(g), "plan.json"), "utf8"));
    }
  });

  it("records how far batching strayed from batch 1 in the run metrics, outside every hash", () => {
    const metrics = json<{ inference_divergence?: { max_batch: number; decisions: number; sampled: number; differing: number; max_prior_diff_ppm: number } }>(batched, "metrics", `${gen(1)}.json`);
    const found = metrics.inference_divergence;
    expect(found).toBeDefined();
    expect(found?.max_batch).toBe(2);
    expect(found?.decisions).toBeGreaterThan(0);
    expect(found?.differing).toBeLessThanOrEqual(found?.sampled ?? 0);
    expect(found?.sampled).toBeLessThanOrEqual(found?.decisions ?? 0);
    expect(json<Record<string, unknown>>(plain, "metrics", `${gen(1)}.json`).inference_divergence).toBeUndefined();
    expect(readFileSync(join(batched, "generations", gen(1), "manifest.json"), "utf8")).not.toContain("inference_divergence");
  });
});
