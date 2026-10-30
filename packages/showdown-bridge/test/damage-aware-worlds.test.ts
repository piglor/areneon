import { describe, expect, it } from "vitest";
import { damageAwareConfigHash, loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { DAMAGE_AWARE_NAME, damageAwareIdentity, damageAwareIdentityHash } from "../src/damage-aware-identity.ts";
import { BELIEF_DEFECTS, refusalCode, WORLDS_PURPOSE, worldsStream } from "../src/damage-aware-worlds.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, loadRulesSnapshot, resolveFormat } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { vocabulary } from "./damage-aware-support.ts";
import { configV2 } from "./damage-aware-world-support.ts";

const hex = (bytes: Uint8Array): string => Buffer.from(bytes).toString("hex");

describe("the world stream of a request (ADR-025 section 4)", () => {
  it("is the SHA-256 of the JCS of the player's own request ids and the purpose (a vector made with Python's hashlib)", () => {
    expect(WORLDS_PURPOSE).toBe("damage-aware-worlds/1");
    const stream = worldsStream({ request_hash: "ab".repeat(32), request_seq: 3, player: "p1" });
    expect(stream).toHaveLength(32);
    expect(hex(stream)).toBe("bbb8a19ab0d0f68200ed88b21e51fe3f8512932f533a5fb7e764fbc584711de0");
  });

  it("changes with each of the request's three ids and with nothing else", () => {
    const base = { request_hash: "ab".repeat(32), request_seq: 3, player: "p1" };
    const streams = new Set([base, { ...base, request_hash: "cd".repeat(32) }, { ...base, request_seq: 4 }, { ...base, player: "p2" }].map((r) => hex(worldsStream(r))));
    expect(streams.size).toBe(4);
    // a field the request record carries besides those three (the decoded body, a phase) is not read
    expect(hex(worldsStream({ ...base, phase: "move", request: {} } as never))).toBe(hex(worldsStream(base)));
  });
});

describe("a belief refusal", () => {
  const refusal = (name: string, code: unknown): Error => Object.assign(new Error("x"), { name, code });
  it("is an Error named BeliefError with a string code, and nothing else", () => {
    expect(refusalCode(refusal("BeliefError", "reader_flagged"))).toBe("reader_flagged");
    expect(refusalCode(refusal("BeliefError", 3))).toBeNull();
    expect(refusalCode(refusal("RangeError", "reader_flagged"))).toBeNull();
    expect(refusalCode("reader_flagged")).toBeNull();
    expect(refusalCode(null)).toBeNull();
  });

  it("treats a prior of another format, an unread Item Clause and a missing spread sampler as defects of the run, not as fallbacks", () => {
    expect([...BELIEF_DEFECTS].sort()).toEqual(["format_mismatch", "item_clause_unsupported", "spread_sampler_required"]);
    expect(BELIEF_DEFECTS.has("reader_flagged")).toBe(false);
  });
});

describe("the baseline identity with a belief (ADR-018 erratum E-18.3)", () => {
  const engine = loadEngine();
  const config = loadDamageAwareConfig("v1", vocabulary);
  const mechanics = loadMechanicsConfig(GOLDEN_FORMAT);
  const manifest = resolveFormat(engine, GOLDEN_FORMAT, loadRulesSnapshot(engine.commit)).manifest;
  const parts = { config, mechanics, manifest, reader_sha256: "a".repeat(64) };
  const belief = "c".repeat(64);

  it("keeps v1's identity bytes exactly: no belief key, and the pinned hash is the one it gives (it moved with mechanics_config_hash in #1166 and with the adapter format in #1243)", () => {
    const identity = damageAwareIdentity(parts);
    expect("belief_identity_hash" in identity).toBe(false);
    expect(damageAwareConfigHash(config)).toBe("1df63656d4747a06f3e8b47e18959c1f7763f83ac500d317e3ae0e4a5db9ec14");
    expect(damageAwareIdentityHash(identity)).toBe("dd8d8687374dd9f0bfd01b21ce8dbd3a05e2a3584235b9fdff78dbbf6a087a79");
  });

  it("gains belief_identity_hash only for a record that names a belief, and then changes with it", () => {
    const second = { ...parts, config: configV2, belief_identity_hash: belief };
    const identity = damageAwareIdentity(second);
    expect(identity.belief_identity_hash).toBe(belief);
    expect(identity.baseline_config_hash).toBe(damageAwareConfigHash(configV2));
    const hashes = new Set([damageAwareIdentityHash(identity), damageAwareIdentityHash(damageAwareIdentity({ ...second, belief_identity_hash: "d".repeat(64) })), damageAwareIdentityHash(damageAwareIdentity(parts))]);
    expect(hashes.size).toBe(3);
    expect(damageAwareIdentityHash(damageAwareIdentity(second))).toBe(damageAwareIdentityHash(identity));
  });

  it("refuses a belief hash with a v1 record, a v2 record without one, and one that is not a SHA-256", () => {
    expect(() => damageAwareIdentity({ ...parts, belief_identity_hash: belief })).toThrow(RangeError);
    expect(() => damageAwareIdentity({ ...parts, config: configV2 })).toThrow(RangeError);
    expect(() => damageAwareIdentity({ ...parts, config: configV2, belief_identity_hash: "C".repeat(64) })).toThrow("SHA-256");
    expect(DAMAGE_AWARE_NAME).toBe("damage-aware");
  });
});
