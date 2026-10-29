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

  it("reads a warm start, hashes it apart from none and refuses the same checkpoint for both roles (ADR-010 erratum A2)", async () => {
    const warm = readTrainingRunConfig(sample("warm-start"));
    expect(warm.warm_start).toEqual({ preview: "a".repeat(64), battle: "b".repeat(64) });
    expect(readTrainingRunConfig(sample("tiny"))).not.toHaveProperty("warm_start");
    expect(await runConfigHash(warm)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("tiny"))));
    const same = { ...sample("tiny"), warm_start: { preview: "a".repeat(64), battle: "a".repeat(64) } };
    expect(() => readTrainingRunConfig(same)).toThrow(TrainingRunConfigError);
    for (const name of ["short-id", "upper-case", "missing-battle", "extra-field"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/warm-start-${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
  });

  it("reads a Battle-only warm start with its warm-up and anchor, and keeps the both-role record as it was (#1165)", async () => {
    const only = readTrainingRunConfig(sample("warm-start-battle-only"));
    expect(only.warm_start).toEqual({ battle: "b".repeat(64) });
    const both = readTrainingRunConfig(sample("warm-start"));
    expect(await runConfigHash(only)).not.toBe(await runConfigHash(both));
    const companions = readTrainingRunConfig(sample("battle-only-warmup-anchor"));
    expect(companions.ppo.value_warmup).toEqual({ generations: 2 });
    expect(companions.ppo.anchor).toBeDefined();
    const preview = { ...sample("tiny"), warm_start: { preview: "a".repeat(64) } };
    expect(() => readTrainingRunConfig(preview)).toThrow(TrainingRunConfigError);
  });

  it("reads a value warm-up only with a warm start and hashes it apart (ADR-019 section 4)", async () => {
    const warm = readTrainingRunConfig(sample("value-warmup"));
    expect(warm.ppo.value_warmup).toEqual({ generations: 2 });
    expect(readTrainingRunConfig(sample("warm-start")).ppo).not.toHaveProperty("value_warmup");
    expect(await runConfigHash(warm)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("warm-start"))));
    const tiny = sample("tiny") as { ppo: Record<string, unknown> };
    expect(() => readTrainingRunConfig({ ...tiny, ppo: { ...tiny.ppo, value_warmup: { generations: 2 } } })).toThrow(/needs warm_start/);
    for (const name of ["zero", "extra-field", "fractional"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/value-warmup-${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
  });

  it("reads a KL anchor only with a warm start and hashes it apart (ADR-019 section 4)", async () => {
    const constant = readTrainingRunConfig(sample("anchor-constant"));
    expect(constant.ppo.anchor).toEqual({ kl_micro: 100000, schedule: { kind: "constant" } });
    const linear = readTrainingRunConfig(sample("anchor-linear"));
    expect(linear.ppo.anchor?.schedule).toEqual({ kind: "linear", final_micro: 0, generations: 10 });
    expect(readTrainingRunConfig(sample("warm-start")).ppo).not.toHaveProperty("anchor");
    expect(await runConfigHash(constant)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("warm-start"))));
    const tiny = sample("tiny") as { ppo: Record<string, unknown> };
    expect(() => readTrainingRunConfig({ ...tiny, ppo: { ...tiny.ppo, anchor: { kl_micro: 1, schedule: { kind: "constant" } } } })).toThrow(/needs warm_start/);
    for (const name of ["zero-kl", "no-schedule", "linear-zero-generations", "extra-field"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/anchor-${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
  });

  it("reads damage-aware as a scripted baseline and refuses a fourth entry (ADR-018 section 7)", () => {
    const config = readTrainingRunConfig(sample("damage-aware"));
    expect(config.generation.baseline_games.map((b) => b.actor)).toEqual(["random-legal", "actor-rule", "damage-aware"]);
    const tiny = sample("tiny") as { generation: { baseline_games: unknown[] } };
    const four = ["a", "b", "c", "d"].map((n) => ({ actor: "actor-rule", policy_id: `actor-rule@${n}`, games: 1 }));
    expect(() => readTrainingRunConfig({ ...tiny, generation: { ...tiny.generation, baseline_games: four } })).toThrow(TrainingRunConfigError);
  });

  it("reads empty model options, hashes them apart from absent ones and refuses any named option (ADR-010 erratum A4 item 2)", async () => {
    const empty = readTrainingRunConfig(sample("empty-options"));
    expect(empty.model.options).toEqual({});
    expect(empty.model.encoder.options).toEqual({});
    expect(await runConfigHash(empty)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("tiny"))));
    expect(readTrainingRunConfig(sample("tiny")).model).not.toHaveProperty("options");
    for (const name of ["unknown-model-option", "unknown-encoder-option"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
  });

  it("reads the candidate_features object, hashes it apart and refuses the other shapes (ADR-022 tier A)", async () => {
    const featured = readTrainingRunConfig(sample("candidate-features"));
    expect(featured.model.encoder.options).toEqual({ candidate_features: { feature_set: "probe-damage/1", feature_set_hash: "c".repeat(64) } });
    expect(await runConfigHash(featured)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("tiny"))));
    for (const name of ["candidate-features-true", "candidate-features-short-hash", "candidate-features-unknown-set"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
  });

  it("reads generation.play, hashes it apart and refuses the other shapes and the explicit default (ADR-023 section 1)", async () => {
    expect(readTrainingRunConfig(sample("tiny")).generation).not.toHaveProperty("play");
    const tempered = readTrainingRunConfig(sample("play-temperature"));
    expect(tempered.generation.play).toEqual({ temperature_ppm: 500000 });
    expect(await runConfigHash(tempered)).not.toBe(await runConfigHash(readTrainingRunConfig(sample("tiny"))));
    for (const name of ["play-temperature-zero", "play-temperature-fractional", "play-extra-field", "play-empty"]) {
      const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/invalid/${name}.json`), "utf8"));
      expect(() => readTrainingRunConfig(raw)).toThrow(TrainingRunConfigError);
    }
    const generation = sample("tiny").generation as Record<string, unknown>;
    expect(() => readTrainingRunConfig({ ...sample("tiny"), generation: { ...generation, play: { temperature_ppm: 1000000 } } })).toThrow("is the default");
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

  it("reads the optional entropy schedule, a final of 0 included, and refuses a malformed one (ADR-023 section 3)", () => {
    expect(readTrainingRunConfig(sample("tiny")).ppo.entropy_schedule).toBeUndefined();
    expect(readTrainingRunConfig(sample("entropy-constant")).ppo.entropy_schedule).toEqual({ kind: "constant" });
    expect(readTrainingRunConfig(sample("entropy-linear")).ppo.entropy_schedule).toEqual({ kind: "linear", final_micro: 0, generations: 4 });
    const tiny = sample("tiny") as { ppo: Record<string, unknown> };
    for (const bad of [{ kind: "linear", final_micro: 0, generations: 0 }, { kind: "linear", final_micro: -1, generations: 2 }, { kind: "linear", final_micro: 0 }, { kind: "cosine" }, { kind: "constant", generations: 1 }]) {
      expect(() => readTrainingRunConfig({ ...tiny, ppo: { ...tiny.ppo, entropy_schedule: bad } })).toThrow(TrainingRunConfigError);
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
