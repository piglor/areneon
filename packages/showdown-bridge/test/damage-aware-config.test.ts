import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import {
  DAMAGE_AWARE_CONFIG_DIR,
  type DamageAwareConfig,
  DamageAwareConfigError,
  damageAwareConfigHash,
  loadDamageAwareConfig,
  loadDamageAwareConfigV2,
  readDamageAwareConfig,
  readDamageAwareConfigV2,
} from "../src/damage-aware-config.ts";
import { beliefConfigHash, readBeliefConfig } from "../src/belief-config.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { sha256Hex } from "../src/rules.ts";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const vocabulary = goldenTeamVocabulary();
const config = loadDamageAwareConfig("v1", vocabulary);
const sample = (kind: string, name: string, contract = "damage-aware-config"): unknown =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", contract, kind, `${name}.json`), "utf8"));
const OPPONENT_MODEL_DIR = join(DAMAGE_AWARE_CONFIG_DIR, "..", "opponent-model", vocabulary.format_id);
const edited = (change: (copy: DamageAwareConfig) => void): DamageAwareConfig => {
  const copy = structuredClone(config);
  change(copy);
  return copy;
};

describe("the committed config", () => {
  it("holds the v1, v2 and v3 records in config/baselines, reads against the vocabulary and equals the corpus sample", () => {
    expect(readdirSync(DAMAGE_AWARE_CONFIG_DIR)).toEqual(["damage-aware.v1.json", "damage-aware.v2.json", "damage-aware.v3.json"]);
    expect(config).toEqual(sample("valid", "v1"));
  });

  it("hashes as the SHA-256 of its JCS bytes", () => {
    expect(damageAwareConfigHash(config)).toBe(sha256Hex(jcs(config as unknown as Json)));
    expect(damageAwareConfigHash(edited((c) => (c.epsilon_ppm = 1)))).not.toBe(damageAwareConfigHash(config));
  });

  it("evaluates with no exploration", () => {
    expect(config.epsilon_ppm).toBe(0);
  });
});

describe("readDamageAwareConfig", () => {
  it("refuses a document that is not the contract", () => {
    expect(() => readDamageAwareConfig("x", vocabulary)).toThrow("invalid DamageAwareConfig");
    expect(() => readDamageAwareConfig({ ...config, extra: 1 }, vocabulary)).toThrow(DamageAwareConfigError);
  });

  it("refuses a value that is not an integer and one out of range", () => {
    expect(() => readDamageAwareConfig(edited((c) => (c.scoring.ko_weight_ppm = 0.5)), vocabulary)).toThrow("invalid DamageAwareConfig");
    expect(() => readDamageAwareConfig(edited((c) => (c.epsilon_ppm = 1_000_001)), vocabulary)).toThrow("invalid DamageAwareConfig");
  });

  it("refuses an id the team vocabulary does not hold", () => {
    expect(() => readDamageAwareConfig(edited((c) => (c.assumed_set.nature = "grumpy")), vocabulary)).toThrow("nature grumpy");
    expect(() => readDamageAwareConfig(edited((c) => (c.assumed_set.item = { kind: "fixed", item: "notanitem" })), vocabulary)).toThrow("item notanitem");
  });

  it("accepts a fixed item of the vocabulary", () => {
    const item = vocabulary.items[0] as string;
    expect(readDamageAwareConfig(edited((c) => (c.assumed_set.item = { kind: "fixed", item })), vocabulary).assumed_set.item).toEqual({ kind: "fixed", item });
  });

  it("refuses another format and a spread over the format's limit", () => {
    expect(() => readDamageAwareConfig(edited((c) => (c.format_id = "gen9ou")), vocabulary)).toThrow("format_id");
    expect(() => readDamageAwareConfig(edited((c) => (c.assumed_set.stat_points.hp = 255)), vocabulary)).toThrow("over the format's limit");
  });
});

describe("the corpus samples", () => {
  it.each([
    ["unknown-nature", "nature"],
    ["unknown-fixed-item", "item"],
    ["stat-points-over-limit", "limit"],
    ["other-format", "format_id"],
  ])("the reader refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readDamageAwareConfig(sample("reader-invalid", name), vocabulary)).toThrow(message);
  });

  it.each(readdirSync(join(SAMPLES_DIR, "v1", "damage-aware-config", "invalid")).map((f) => f.replace(".json", "")))(
    "the reader refuses the invalid sample %s",
    (name) => {
      expect(() => readDamageAwareConfig(sample("invalid", name), vocabulary)).toThrow("invalid DamageAwareConfig");
    },
  );
});

describe("the v2 record (the belief)", () => {
  const second = loadDamageAwareConfigV2(vocabulary);
  const read = (value: unknown) => readDamageAwareConfigV2(value, vocabulary);

  it("equals the corpus sample, shares v1's weights and names the committed prior and belief config by their hashes", () => {
    expect(second).toEqual(sample("valid", "v2", "damage-aware-config-2"));
    expect(second.assumed_set.worlds).toBe(8);
    expect({ hp_rounding: second.hp_rounding, scoring: second.scoring, epsilon_ppm: second.epsilon_ppm }).toEqual({ hp_rounding: config.hp_rounding, scoring: config.scoring, epsilon_ppm: config.epsilon_ppm });
    const prior = readUsagePrior(JSON.parse(readFileSync(join(OPPONENT_MODEL_DIR, "generator-matched.uniform-v1.json"), "utf8")), vocabulary);
    expect(prior.format_id).toBe(second.format_id);
    expect(usagePriorHash(prior)).toBe(second.assumed_set.prior_hash);
    const belief = readBeliefConfig(JSON.parse(readFileSync(join(OPPONENT_MODEL_DIR, "belief.v1.json"), "utf8")), vocabulary);
    expect(beliefConfigHash(belief)).toBe(second.assumed_set.belief_config_hash);
  });

  it("is a new record: its hash is not v1's and v1's bytes are what they were", () => {
    expect(damageAwareConfigHash(second)).not.toBe(damageAwareConfigHash(config));
    expect(damageAwareConfigHash(config)).toBe("1df63656d4747a06f3e8b47e18959c1f7763f83ac500d317e3ae0e4a5db9ec14");
  });

  it("refuses a document that is not the contract, v1 field rules in its place and another format", () => {
    expect(() => read("x")).toThrow("invalid DamageAwareConfigV2");
    expect(() => read(config)).toThrow(DamageAwareConfigError);
    expect(() => read({ ...second, assumed_set: config.assumed_set })).toThrow("invalid DamageAwareConfigV2");
    expect(() => read({ ...second, format_id: "gen9ou" })).toThrow("format_id");
    expect(() => readDamageAwareConfig(second, vocabulary)).toThrow("invalid DamageAwareConfig");
  });

  it.each(readdirSync(join(SAMPLES_DIR, "v1", "damage-aware-config-2", "invalid")).map((f) => f.replace(".json", "")))("the reader refuses the invalid sample %s", (name) => {
    expect(() => read(sample("invalid", name, "damage-aware-config-2"))).toThrow("invalid DamageAwareConfigV2");
  });

  it("the reader refuses the reader-invalid sample other-format", () => {
    expect(() => read(sample("reader-invalid", "other-format", "damage-aware-config-2"))).toThrow("format_id");
  });
});
