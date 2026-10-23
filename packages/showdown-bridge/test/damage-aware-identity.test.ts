import { describe, expect, it } from "vitest";
import { damageAwareConfigHash, loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { DAMAGE_AWARE_NAME, damageAwareIdentity, damageAwareIdentityHash, damageAwarePolicyId } from "../src/damage-aware-identity.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine, loadRulesSnapshot, resolveFormat } from "../src/index.ts";
import { loadMechanicsConfig, mechanicsConfigHash } from "../src/mechanics-config.ts";
import { MECHANICS_ADAPTER_FORMAT } from "../src/mechanics-result.ts";

const engine = loadEngine();
const config = loadDamageAwareConfig("v1", goldenTeamVocabulary());
const mechanics = loadMechanicsConfig(GOLDEN_FORMAT);
const manifest = resolveFormat(engine, GOLDEN_FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const parts = { config, mechanics, manifest, reader_sha256: "a".repeat(64) };

describe("the baseline identity (ADR-018 section 5)", () => {
  it("holds the five parts the play depends on", () => {
    const identity = damageAwareIdentity(parts);
    expect(identity).toEqual({
      baseline_config_hash: damageAwareConfigHash(config),
      mechanics_config_hash: mechanicsConfigHash(mechanics),
      adapter_format: MECHANICS_ADAPTER_FORMAT,
      reader_sha256: "a".repeat(64),
      version_manifest: manifest,
    });
    expect(damageAwareIdentityHash(identity)).toMatch(/^[0-9a-f]{64}$/);
  });
  it("changes with any part: a new baseline, never an edit", () => {
    const base = damageAwareIdentityHash(damageAwareIdentity(parts));
    const hashes = new Set([
      base,
      damageAwareIdentityHash(damageAwareIdentity({ ...parts, config: { ...config, epsilon_ppm: 1 } })),
      damageAwareIdentityHash(damageAwareIdentity({ ...parts, config: { ...config, scoring: { ...config.scoring, ko_weight_ppm: 1 } } })),
      damageAwareIdentityHash(damageAwareIdentity({ ...parts, mechanics: { ...mechanics, ko_blockers: { ...mechanics.ko_blockers, items: [...mechanics.ko_blockers.items, "x"] } } })),
      damageAwareIdentityHash(damageAwareIdentity({ ...parts, reader_sha256: "b".repeat(64) })),
      damageAwareIdentityHash(damageAwareIdentity({ ...parts, manifest: { ...manifest, showdown_commit: "0".repeat(40) } })),
    ]);
    expect(hashes.size).toBe(6);
    // and the same parts give the same hash
    expect(damageAwareIdentityHash(damageAwareIdentity(parts))).toBe(base);
  });
  it("names the policy damage-aware@<commit>", () => {
    expect(damageAwarePolicyId("v1")).toBe("damage-aware@v1");
    expect(damageAwarePolicyId("0a1b2c")).toBe(`${DAMAGE_AWARE_NAME}@0a1b2c`);
    expect(() => damageAwarePolicyId("")).toThrow(RangeError);
    expect(() => damageAwarePolicyId("a b")).toThrow(RangeError);
  });
});
