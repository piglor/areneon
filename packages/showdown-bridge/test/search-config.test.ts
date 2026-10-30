// The SearchConfig, the search identity and the policy id that carries it (ADR-026 section 5, slice R part 1):
// the committed record, the reader's refusals, an identity that moves with every input, and ids that agree with
// the Python twin (vectors computed there) while every existing id stays byte-identical.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { WORLD_FACTS_FORMAT } from "../../public-state/src/world-facts.ts";
import { DAMAGE_VALUE_FORMAT } from "../src/damage-aware-score.ts";
import { PolicyProtocolError, policyId } from "../src/policy-protocol.ts";
import { REPLY_MODEL_FORMAT } from "../src/reply-model.ts";
import { loadSearchConfig, LEAF_FORMAT, readSearchConfig, SearchConfigError, searchConfigHash, searchIdentity, searchIdentityHash, type SearchIdentityInputs } from "../src/search-config.ts";
import { SEARCH_HP_ROUNDING, SEARCH_PURPOSE, GENERATOR_MATCHED_LABEL } from "../src/world-sampler.ts";
import { WORLD_BUILDER_FORMAT } from "../src/world-builder.ts";

const FORMAT = "gen9championsvgc2026regmc";
const SAMPLES = new URL("../../../fixtures/contracts/v1/search-config/", import.meta.url);
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(new URL(`${kind}/${name}.json`, SAMPLES), "utf8"));
const names = (kind: string): string[] => ["rerank-1", "uniform-replies-hp-leaf"].filter(() => kind === "valid");

const h = (c: string): string => c.repeat(64);
const config = loadSearchConfig(FORMAT);
const inputs: SearchIdentityInputs = {
  belief_identity_hash: config.belief_identity_hash,
  mechanics_config_hash: h("1"),
  mechanics_adapter_format: 2,
  damage_aware_config_hash: h("2"),
  damage_value_format: 1,
  world_writers_hash: h("3"),
  world_builder_format: 2,
  world_facts_format: "world-facts/3",
  reply_model_format: 1,
  leaf_format: 1,
  reader_sha256: h("4"),
  version_manifest: { showdown_commit: "c".repeat(40), format_id: FORMAT, regulation: "M-C", ruleset_hash: h("5"), format_rules_hash: h("6") },
};
// Computed with Python (areneon_ml.loop.records), outside the TS code.
const CONFIG_HASH = "739f4b1f1a30411b3d5fb3d37749739a6673bcaf6195941f3a57f6dc1e70d49c";
// Moved by #248's format bumps (WORLD_BUILDER_FORMAT 1 -> 2, world-facts/2 -> 3), which are identity inputs; recomputed with Python's hashlib over the
// canonical JSON (the old inputs still give edd348bd...). The policy-id vectors below take a search hash as an opaque string, so they keep that one.
const SEARCH_HASH = "3508d46a98552f0898f662de77bfa30067c13eaefe085145f39766d0bc4e1ce6";
const ID_VECTOR_SEARCH = "edd348bdf8b9f5069ada74afae22d3ba99e1b349e2748b245de73384c9d666b9";

describe("the committed SearchConfig", () => {
  it("holds ADR-026's defaults and hashes as the Python side does", () => {
    expect(config).toMatchObject({ top_k: 4, worlds: { n: 8, redraws: 4 }, replies: { max_damage_ppm: 500000, uniform_draws: 2 }, min_valid_worlds: 4, leaf: { kind: "value-head" } });
    expect(searchConfigHash(config)).toBe(CONFIG_HASH);
  });

  it("agrees with the world sampler's constants and the prior label", () => {
    expect(config.worlds.hp_rounding).toBe(SEARCH_HP_ROUNDING);
    expect(config.seed_derivation).toBe(SEARCH_PURPOSE);
    expect(config.worlds.prior).toEqual({ kind: "generator-matched", label: GENERATOR_MATCHED_LABEL });
  });

  it("is refused for another format", () => {
    expect(() => loadSearchConfig("gen9other")).toThrow(SearchConfigError);
  });

  it("reads every valid sample and refuses the invalid and reader-invalid ones", () => {
    for (const name of names("valid")) expect(() => readSearchConfig(sample("valid", name))).not.toThrow();
    for (const name of ["extra-field", "fractional-top-k", "zero-top-k", "unknown-algorithm", "unknown-leaf", "unknown-hp-rounding", "bad-seed-derivation", "short-belief-hash", "missing-replies", "next-version"]) {
      expect(() => readSearchConfig(sample("invalid", name)), name).toThrow(/invalid SearchConfig/);
    }
    for (const name of ["min-worlds-over-n", "inexact-uniform-share", "no-uniform-without-full-weight"]) {
      expect(() => readSearchConfig(sample("reader-invalid", name)), name).toThrow(SearchConfigError);
    }
  });

  it("accepts a max-damage reply alone", () => {
    const alone = structuredClone(config);
    alone.replies = { max_damage_ppm: 1_000_000, uniform_draws: 0 };
    expect(readSearchConfig(alone).replies.uniform_draws).toBe(0);
  });
});

describe("the search identity", () => {
  it("is the hash the Python twin computes", () => {
    expect(searchIdentityHash(config, inputs)).toBe(SEARCH_HASH);
    expect(Object.keys(searchIdentity(config, inputs)).length).toBe(13);
  });

  it("moves with every input and with every config value", () => {
    const base = searchIdentityHash(config, inputs);
    const seen = new Set([base]);
    const changes: [string, Partial<SearchIdentityInputs>][] = [
      ["mechanics", { mechanics_config_hash: h("9") }],
      ["adapter format", { mechanics_adapter_format: 3 }],
      ["damage-aware config", { damage_aware_config_hash: h("9") }],
      ["damage value", { damage_value_format: 2 }],
      ["writers", { world_writers_hash: h("9") }],
      ["builder", { world_builder_format: 3 }],
      ["facts", { world_facts_format: "world-facts/4" }],
      ["reply model", { reply_model_format: 2 }],
      ["leaf", { leaf_format: 2 }],
      ["reader", { reader_sha256: h("9") }],
      ["manifest", { version_manifest: { ...inputs.version_manifest, format_rules_hash: h("9") } }],
    ];
    for (const [label, change] of changes) {
      const next = searchIdentityHash(config, { ...inputs, ...change });
      expect(seen.has(next), label).toBe(false);
      seen.add(next);
    }
    const edits: ((c: typeof config) => void)[] = [
      (c) => (c.top_k = 5),
      (c) => (c.top_k_rule = "distinct-slot"),
      (c) => (c.worlds.n = 9),
      (c) => (c.worlds.redraws = 5),
      (c) => (c.worlds.hp_rounding = "nearest"),
      (c) => (c.replies.uniform_draws = 1),
      (c) => (c.leaf = { kind: "hp-ko-differential" }),
      (c) => (c.budget_ms = 20000),
      (c) => (c.a2_clock_floor_ms = 1),
      (c) => (c.min_valid_worlds = 3),
    ];
    for (const edit of edits) {
      const copy = structuredClone(config);
      edit(copy);
      const next = searchIdentityHash(copy, inputs);
      expect(seen.has(next)).toBe(false);
      seen.add(next);
    }
    const other = structuredClone(config);
    other.belief_identity_hash = h("7");
    expect(searchIdentityHash(other, { ...inputs, belief_identity_hash: h("7") })).not.toBe(base);
  });

  it("refuses a belief identity the config does not name and malformed inputs", () => {
    expect(() => searchIdentityHash(config, { ...inputs, belief_identity_hash: h("8") })).toThrow(/belief identity/);
    expect(() => searchIdentityHash(config, { ...inputs, reader_sha256: "abc" })).toThrow(/64 lowercase hex/);
    expect(() => searchIdentityHash(config, { ...inputs, leaf_format: 0 })).toThrow(/positive integer/);
    expect(() => searchIdentityHash(config, { ...inputs, world_facts_format: "" })).toThrow(/non-empty/);
  });

  it("names the formats the modules export today", () => {
    expect({ WORLD_BUILDER_FORMAT, REPLY_MODEL_FORMAT, DAMAGE_VALUE_FORMAT, LEAF_FORMAT, WORLD_FACTS_FORMAT }).toEqual({ WORLD_BUILDER_FORMAT: 2, REPLY_MODEL_FORMAT: 1, DAMAGE_VALUE_FORMAT: 1, LEAF_FORMAT: 1, WORLD_FACTS_FORMAT: "world-facts/3" });
  });
});

interface Existing { preview: string; battle: string; mode: "sample" | "argmax"; temperature_ppm?: number; truncation?: { kind: "threshold"; min_ppm: number }; features?: string; id: string }
const existing = JSON.parse(readFileSync(new URL("../../../fixtures/policy-ids/existing.json", import.meta.url), "utf8")) as Existing[];

describe("the policy id with search", () => {
  it("leaves every existing id byte-identical (ids computed before the change)", () => {
    expect(existing.length).toBeGreaterThanOrEqual(9);
    for (const c of existing) expect(policyId(c, c.mode, { ...(c.temperature_ppm === undefined ? {} : { temperature_ppm: c.temperature_ppm }), ...(c.truncation === undefined ? {} : { truncation: c.truncation }) }, c.features)).toBe(c.id);
  });

  it("adds search only when present, and equals the Python twin", () => {
    const pair = { preview: h("a"), battle: h("b") };
    expect(policyId(pair, "argmax", {}, undefined, ID_VECTOR_SEARCH)).toBe("policy:86611a1d2e6ce20f8e0a1a1f4b61cf3a88d27229bc18ef2a80ec4990b1797453");
    expect(policyId(pair, "sample", { temperature_ppm: 700000, truncation: { kind: "threshold", min_ppm: 30000 } }, h("f"), ID_VECTOR_SEARCH)).toBe("policy:7cbab00c7c5aac1f95f622fec925b0ddf5bf0097d0d05352911b6ceb6174fb26");
  });

  it("differs for another underlying mode, another play setting and another search identity", () => {
    const pair = { preview: h("a"), battle: h("b") };
    const ids = [
      policyId(pair, "sample"),
      policyId(pair, "sample", {}, undefined, SEARCH_HASH),
      policyId(pair, "argmax", {}, undefined, ID_VECTOR_SEARCH),
      policyId(pair, "sample", { temperature_ppm: 500000 }, undefined, SEARCH_HASH),
      policyId(pair, "sample", {}, undefined, searchIdentityHash(config, { ...inputs, leaf_format: 2 })),
    ];
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("refuses a search that is not a hash", () => {
    expect(() => policyId({ preview: h("a"), battle: h("b") }, "sample", {}, undefined, "nope")).toThrow(PolicyProtocolError);
  });
});
