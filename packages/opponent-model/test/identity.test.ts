import { describe, expect, it } from "vitest";
import { BELIEF_FORMAT, beliefIdentityHash, type BeliefIdentityParts } from "../src/index.ts";
import { parseSpreadKey } from "../src/fields.ts";

const parts: BeliefIdentityParts = {
  prior_hash: "a".repeat(64),
  belief_config_hash: "b".repeat(64),
  vocabulary_hash: "c".repeat(64),
  reader_sha256: "d".repeat(64),
  version_manifest: { showdown_commit: "e".repeat(40), format_id: "testformat", regulation: "Reg X", ruleset_hash: "1".repeat(64), format_rules_hash: "2".repeat(64) },
};

describe("belief_identity_hash (ADR-025 section 5)", () => {
  it("is the SHA-256 of the JCS of {prior_hash, belief_config_hash, belief_format, vocabulary_hash, reader_sha256, version_manifest} (a vector made with Python)", () => {
    expect(BELIEF_FORMAT).toBe("belief/5");
    expect(beliefIdentityHash(parts)).toBe("72381ea33fb8d0a2a520297a4b321365def0f6d88f3cb4273fea92aa2b622918");
  });

  it("changes with every part: a new prior, config, vocabulary, reader or pin is a new identity", () => {
    const base = beliefIdentityHash(parts);
    const other = (change: Partial<BeliefIdentityParts>) => beliefIdentityHash({ ...parts, ...change });
    const hashes = [
      other({ prior_hash: "0".repeat(64) }),
      other({ belief_config_hash: "0".repeat(64) }),
      other({ vocabulary_hash: "0".repeat(64) }),
      other({ reader_sha256: "0".repeat(64) }),
      other({ version_manifest: { ...parts.version_manifest, showdown_commit: "0".repeat(40) } }),
      other({ version_manifest: { ...parts.version_manifest, format_rules_hash: "0".repeat(64) } }),
    ];
    expect(new Set([base, ...hashes]).size).toBe(7);
  });

  it("refuses a hash that is not a lower-case SHA-256", () => {
    for (const key of ["prior_hash", "belief_config_hash", "vocabulary_hash", "reader_sha256"] as const) {
      expect(() => beliefIdentityHash({ ...parts, [key]: "A".repeat(64) })).toThrow(RangeError);
      expect(() => beliefIdentityHash({ ...parts, [key]: "a".repeat(63) })).toThrow(RangeError);
    }
  });
});

describe("spread keys", () => {
  it("round-trip the nature and the six Stat Points; the other-spread entry is no key", () => {
    expect(parseSpreadKey("adamant:1-2-3-4-5-6")).toEqual({ nature: "adamant", stat_points: { hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 } });
    expect(parseSpreadKey("other")).toBeNull();
    expect(parseSpreadKey("adamant:1-2-3")).toBeNull();
  });
});
