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
  readDamageAwareConfig,
} from "../src/damage-aware-config.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { sha256Hex } from "../src/rules.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const vocabulary = goldenTeamVocabulary();
const config = loadDamageAwareConfig("v1", vocabulary);
const sample = (kind: string, name: string): unknown =>
  JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", "damage-aware-config", kind, `${name}.json`), "utf8"));
const edited = (change: (copy: DamageAwareConfig) => void): DamageAwareConfig => {
  const copy = structuredClone(config);
  change(copy);
  return copy;
};

describe("the committed config", () => {
  it("is the only record in config/baselines, reads against the vocabulary and equals the corpus sample", () => {
    expect(readdirSync(DAMAGE_AWARE_CONFIG_DIR)).toEqual(["damage-aware.v1.json"]);
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
