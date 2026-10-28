// The training loop with a candidate feature set (ADR-022 slice D3): the Python driver plays through Node and the real
// policy worker with the real probe, trains, and a run resumed after generation 1 reaches the learner state of an
// uninterrupted one. Needs the project's Python environment, like training-loop.python.test.ts (skipped without it).
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/engine.ts";
import { featureSetup } from "../src/league-generation.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig } from "../src/training-run-config.ts";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const DATA = join(GOLDEN_DIR, "engine-data.json");
const env = { ...(process.env.SHOWDOWN_DIR ? { SHOWDOWN_DIR: process.env.SHOWDOWN_DIR } : {}) };

interface Summary {
  generation: number;
  learner_state_hash: string;
  checkpoints: Record<string, string>;
}
const run = (config: string, runDir: string, generations: number, ...extra: string[]): Summary =>
  JSON.parse(
    execFileSync(PYTHON, ["-I", "-m", "areneon_ml.loop", "run", "--config", config, "--run-dir", runDir, "--engine-data", DATA, "--generations", String(generations), "--node", process.execPath, "--worker-python", PYTHON, ...extra], { encoding: "utf8", env }),
  ) as Summary;

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the training loop with candidate features, tiny and on CPU", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-loop-features-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("plays, trains, and a run resumed after generation 1 reaches the same learner state as an uninterrupted one", () => {
    const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { model: { encoder: { options?: unknown } } };
    raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: "0".repeat(64) } };
    let hash = "";
    try {
      featureSetup(loadEngine(), readTrainingRunConfig(raw));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    raw.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
    const config = join(dir, "config.json");
    writeFileSync(config, JSON.stringify(raw));
    const whole = run(config, join(dir, "whole"), 2);
    run(config, join(dir, "resumed"), 1);
    const resumed = run(config, join(dir, "resumed"), 2);
    expect(whole.generation).toBe(2);
    expect(resumed.learner_state_hash).toBe(whole.learner_state_hash);
    expect(resumed.checkpoints).toEqual(whole.checkpoints);
    const stream = readFileSync(join(dir, "whole", "generations", "g000002", "stream.jsonl"), "utf8");
    expect(stream).toContain(`"feature_set_hash":"${hash}"`);
    // Two game processes recompute the rows of their own Episodes (#1185): the same stream, state and checkpoints.
    const sharded = run(config, join(dir, "sharded"), 2, "--game-processes", "2");
    expect(sharded.learner_state_hash).toBe(whole.learner_state_hash);
    expect(sharded.checkpoints).toEqual(whole.checkpoints);
    expect(readFileSync(join(dir, "sharded", "generations", "g000002", "stream.jsonl"), "utf8")).toBe(stream);
  }, 900_000); // two generations of six games with the real probe, played three times
});
