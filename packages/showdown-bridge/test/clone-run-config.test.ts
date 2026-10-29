import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { CloneRunConfigError, cloneRunConfigHash, readCloneRunConfig } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const sample = (name: string): Record<string, unknown> =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/clone-run-config/valid/${name}.json`), "utf8")) as Record<string, unknown>;

describe("the CloneRunConfig reader (ADR-019 track 1)", () => {
  it("reads the valid samples and hands back a copy", () => {
    const raw = sample("tiny");
    const config = readCloneRunConfig(raw);
    expect(config).toEqual(raw);
    expect(config).not.toBe(raw);
    expect(readCloneRunConfig(sample("no-holdout")).clone.holdout.ppm).toBe(0);
  });

  it("refuses a record the schema refuses, naming nothing it was not given", () => {
    expect(() => readCloneRunConfig({ ...sample("tiny"), notes: "x" })).toThrow(CloneRunConfigError);
    expect(() => readCloneRunConfig(null)).toThrow(/invalid CloneRunConfig/);
  });

  it("hashes the JCS bytes: key order does not matter, a changed setting does", async () => {
    const config = readCloneRunConfig(sample("tiny"));
    const reordered = readCloneRunConfig(Object.fromEntries(Object.entries(config).reverse()));
    expect(await cloneRunConfigHash(reordered)).toBe(await cloneRunConfigHash(config));
    expect(await cloneRunConfigHash(config)).toMatch(/^[0-9a-f]{64}$/);
    const changed = readCloneRunConfig({ ...config, clone: { ...config.clone, epochs: config.clone.epochs + 1 } });
    expect(await cloneRunConfigHash(changed)).not.toBe(await cloneRunConfigHash(config));
  });

  it("takes the model options as the TrainingRunConfig does: only candidate_matchup true, absent keeps the hash", async () => {
    const plain = readCloneRunConfig(sample("tiny"));
    const matchup = readCloneRunConfig(sample("candidate-matchup"));
    expect(matchup.model.encoder.options).toEqual({ candidate_matchup: true });
    expect(readCloneRunConfig(sample("empty-options")).model.options).toEqual({});
    expect(await cloneRunConfigHash(matchup)).not.toBe(await cloneRunConfigHash(plain));
    const wide = { ...plain.model, options: { wide: 1 } };
    expect(() => readCloneRunConfig({ ...plain, model: wide })).toThrow(CloneRunConfigError);
    const off = { ...plain.model, encoder: { ...plain.model.encoder, options: { candidate_matchup: false } } };
    expect(() => readCloneRunConfig({ ...plain, model: off })).toThrow(CloneRunConfigError);
  });

  it("takes the candidate_features object as the TrainingRunConfig does (ADR-022 tier A)", async () => {
    const plain = readCloneRunConfig(sample("tiny"));
    const featured = readCloneRunConfig(sample("candidate-features"));
    expect(featured.model.encoder.options).toEqual({ candidate_features: { feature_set: "probe-damage/1", feature_set_hash: "c".repeat(64) } });
    expect(await cloneRunConfigHash(featured)).not.toBe(await cloneRunConfigHash(plain));
    const second = readCloneRunConfig(sample("candidate-features-2"));
    expect(second.model.encoder.options).toEqual({ candidate_features: { feature_set: "probe-damage/2", feature_set_hash: "c".repeat(64) } });
    expect(await cloneRunConfigHash(second)).not.toBe(await cloneRunConfigHash(featured));
    const unknown = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/clone-run-config/invalid/candidate-features-unknown-set.json"), "utf8"));
    expect(() => readCloneRunConfig(unknown)).toThrow(CloneRunConfigError);
  });
});
