import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runConfigHash, readTrainingRunConfig, TrainingRunConfigError } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const sample = (name: string): Record<string, unknown> =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/valid/${name}.json`), "utf8")) as Record<string, unknown>;

describe("the TrainingRunConfig reader (ADR-010 §5)", () => {
  it("reads the valid samples and hands back a copy", () => {
    const raw = sample("tiny");
    const config = readTrainingRunConfig(raw);
    expect(config).toEqual(raw);
    expect(config).not.toBe(raw);
    expect(readTrainingRunConfig(sample("no-baselines")).generation.baseline_games).toEqual([]);
  });

  it("reads the committed run configs and the sample that names a team dataset", () => {
    const named = readTrainingRunConfig(sample("team-dataset"));
    expect(named.team_dataset?.path).toBe(".tmp/datasets/train-v1");
    const dir = join(SAMPLES_DIR, "../../config/training");
    for (const file of readdirSync(dir)) {
      const config = readTrainingRunConfig(JSON.parse(readFileSync(join(dir, file), "utf8")));
      expect(config.learner.device).toBe("cuda");
      expect(config.generation.baseline_games.map((b) => b.actor).sort()).toEqual(["actor-rule", "random-legal"]);
      expect(config.team_dataset).toBeDefined();
    }
  });

  it("reads damage-aware as a scripted baseline and refuses a fourth entry (ADR-018 section 7)", () => {
    const config = readTrainingRunConfig(sample("damage-aware"));
    expect(config.generation.baseline_games.map((b) => b.actor)).toEqual(["random-legal", "actor-rule", "damage-aware"]);
    const tiny = sample("tiny") as { generation: { baseline_games: unknown[] } };
    const four = ["a", "b", "c", "d"].map((n) => ({ actor: "actor-rule", policy_id: `actor-rule@${n}`, games: 1 }));
    expect(() => readTrainingRunConfig({ ...tiny, generation: { ...tiny.generation, baseline_games: four } })).toThrow(TrainingRunConfigError);
  });

  it("reads the optional KL target and refuses zero (ADR-010 erratum A1)", () => {
    expect(readTrainingRunConfig(sample("tiny")).ppo.target_kl_micro).toBeUndefined();
    expect(readTrainingRunConfig(sample("target-kl")).ppo.target_kl_micro).toBe(20000);
    const tiny = sample("tiny") as { ppo: Record<string, unknown> };
    expect(() => readTrainingRunConfig({ ...tiny, ppo: { ...tiny.ppo, target_kl_micro: 0 } })).toThrow(TrainingRunConfigError);
  });

  it("reads the optional learning-rate schedule and refuses a malformed one (ADR-010 erratum A1)", () => {
    expect(readTrainingRunConfig(sample("tiny")).ppo.learning_rate_schedule).toBeUndefined();
    expect(readTrainingRunConfig(sample("constant-schedule")).ppo.learning_rate_schedule).toEqual({ kind: "constant" });
    expect(readTrainingRunConfig(sample("linear-schedule")).ppo.learning_rate_schedule).toEqual({
      kind: "linear",
      final_micro: 100,
      generations: 4,
    });
    const tiny = sample("tiny") as { ppo: Record<string, unknown> };
    for (const bad of [{ kind: "linear", final_micro: 100, generations: 0 }, { kind: "linear", final_micro: 100 }, { kind: "cosine" }, { kind: "constant", generations: 1 }]) {
      expect(() => readTrainingRunConfig({ ...tiny, ppo: { ...tiny.ppo, learning_rate_schedule: bad } })).toThrow(TrainingRunConfigError);
    }
  });

  it("refuses a record the schema refuses, naming nothing it was not given", () => {
    expect(() => readTrainingRunConfig({ ...sample("tiny"), notes: "x" })).toThrow(TrainingRunConfigError);
    expect(() => readTrainingRunConfig(null)).toThrow(/invalid TrainingRunConfig/);
  });

  it("hashes the JCS bytes: key order does not matter, a changed setting does", async () => {
    const config = readTrainingRunConfig(sample("tiny"));
    const reordered = readTrainingRunConfig(Object.fromEntries(Object.entries(config).reverse()));
    expect(await runConfigHash(reordered)).toBe(await runConfigHash(config));
    expect(await runConfigHash(config)).toMatch(/^[0-9a-f]{64}$/);
    const changed = readTrainingRunConfig({ ...config, ppo: { ...config.ppo, epochs: config.ppo.epochs + 1 } });
    expect(await runConfigHash(changed)).not.toBe(await runConfigHash(config));
  });
});
