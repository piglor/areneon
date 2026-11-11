// The training loop with a candidate feature set (ADR-022 slice D3): the Python driver plays through Node and the real
// policy worker with the real probe, trains, and a run resumed after generation 1 reaches the learner state of an
// uninterrupted one. Needs the project's Python environment, like training-loop.python.test.ts (skipped without it).
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the training loop with candidate features (probe-damage/1, /2 and probe-outcome/1, the last also with history_facts), tiny and on CPU", () => {
  const dir = mkdtempSync(join(tmpdir(), "areneon-loop-features-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const cases = [
    { name: "probe-damage/1", set: "probe-damage/1", history: false },
    { name: "probe-damage/2", set: "probe-damage/2", history: false },
    { name: "probe-outcome/1", set: "probe-outcome/1", history: false },
    { name: "probe-outcome/1 with history_facts", set: "probe-outcome/1", history: true },
  ] as const;
  it.each(cases)("plays, trains, and a run resumed after generation 1 reaches the same learner state as an uninterrupted one ($name)", ({ name, set, history }) => {
    const base = join(dir, name.replace(/[^a-z0-9]+/g, "-"));
    const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { model: { encoder: { options?: unknown } } };
    const options = (hash: string) => ({ candidate_features: { feature_set: set, feature_set_hash: hash }, ...(history ? { history_facts: true } : {}) });
    raw.model.encoder.options = options("0".repeat(64));
    let hash = "";
    try {
      featureSetup(loadEngine(), readTrainingRunConfig(raw));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    raw.model.encoder.options = options(hash);
    const config = join(base, "config.json");
    mkdirSync(base, { recursive: true });
    writeFileSync(config, JSON.stringify(raw));
    const whole = run(config, join(base, "whole"), 2);
    run(config, join(base, "resumed"), 1);
    const resumed = run(config, join(base, "resumed"), 2);
    expect(whole.generation).toBe(2);
    expect(resumed.learner_state_hash).toBe(whole.learner_state_hash);
    expect(resumed.checkpoints).toEqual(whole.checkpoints);
    const stream = readFileSync(join(base, "whole", "generations", "g000002", "stream.jsonl"), "utf8");
    expect(stream).toContain(`"feature_set_hash":"${hash}"`);
    // Two game processes recompute the rows of their own Episodes (#1185): the same stream, state and checkpoints.
    const sharded = run(config, join(base, "sharded"), 2, "--game-processes", "2");
    expect(sharded.learner_state_hash).toBe(whole.learner_state_hash);
    expect(sharded.checkpoints).toEqual(whole.checkpoints);
    expect(readFileSync(join(base, "sharded", "generations", "g000002", "stream.jsonl"), "utf8")).toBe(stream);
  }, 900_000); // two generations of six games with the real probe, played three times, for each set
});
