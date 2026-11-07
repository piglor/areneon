// The PreviewSolveConfig, the solver identity and the policy id that carries it (ADR-029 section 4, ticket #1259): the committed
// record, the reader's refusals, an identity that moves with every input, the harness derivations, and ids that agree with the
// Python twin (vectors computed there with hashlib) while every existing id stays byte-identical.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { HYPOTHETICAL_BATTLE_FORMAT } from "../src/hypothetical-preview.ts";
import { PAYOFF_LEAF_FORMAT } from "../src/preview-payoff.ts";
import { drawFromMixture, previewSolveStream, previewSolveU } from "../src/preview-solve.ts";
import { loadPreviewSolveConfig, PreviewSolveConfigError, previewSolveConfigHash, previewSolveIdentity, previewSolveIdentityHash, type PreviewSolveIdentityInputs, readPreviewSolveConfig } from "../src/preview-solve-config.ts";
import { PolicyProtocolError, policyId } from "../src/policy-protocol.ts";

const FORMAT = "gen9championsvgc2026regmc";
const SAMPLES = new URL("../../../fixtures/contracts/v1/preview-solve-config/", import.meta.url);
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(new URL(`${kind}/${name}.json`, SAMPLES), "utf8"));

const h = (c: string): string => c.repeat(64);
const config = loadPreviewSolveConfig(FORMAT);
const inputs: PreviewSolveIdentityInputs = {
  belief_identity_hash: config.belief_identity_hash,
  reader_sha256: h("4"),
  version_manifest: { showdown_commit: "c".repeat(40), format_id: FORMAT, regulation: "M-C", ruleset_hash: h("5"), format_rules_hash: h("6") },
  solver_format: "preview-solver/1",
  leaf_format: 1,
  hypothetical_battle_format: 1,
};
// Vectors computed with Python hashlib over the sorted-key compact JSON (tests/ml/loop/test_preview_solve_identity.py).
const CONFIG_HASH = "415714401460127e920daefdc52ad85707fef8fa3d1d98cdd5ff01fbba27ee0e";
const IDENTITY_HASH = "89ae2b2359f6d1d272b03efc8d7959f9c0a2135d87ae4eefb48270b519caf932";
const SEARCH_VECTOR = "edd348bdf8b9f5069ada74afae22d3ba99e1b349e2748b245de73384c9d666b9";

describe("the committed PreviewSolveConfig", () => {
  it("is the ADR's defaults and hashes to the Python vector", () => {
    expect(config).toMatchObject({ k_own: 4, k_opp: 4, selection_rule: "distinct-leads", worlds: { n: 4, min_valid: 3 }, leaf: { kind: "value-at-first-request" }, solver: { kind: "regret-matching", iterations: 4096 }, seed_derivation: "preview-solve/1" });
    expect(previewSolveConfigHash(config)).toBe(CONFIG_HASH);
  });

  it("is read from the corpus samples, valid ones and the reader-invalid ones refused", () => {
    for (const name of ["preview-solve-1", "rollout-plus", "exact-support"]) expect(() => readPreviewSolveConfig(sample("valid", name))).not.toThrow();
    for (const [name, message] of [["min-valid-over-n", "min_valid exceeds"], ["iterative-without-iterations", "iterations is present exactly"], ["exact-support-with-iterations", "iterations is present exactly"]] as const) {
      expect(() => readPreviewSolveConfig(sample("reader-invalid", name)), name).toThrow(message);
    }
  });

  it("refuses a config of another format and an invalid one", () => {
    expect(() => loadPreviewSolveConfig("gen9other")).toThrow();
    expect(() => readPreviewSolveConfig({ ...config, extra: 1 })).toThrow(PreviewSolveConfigError);
  });
});

describe("the solver identity", () => {
  it("equals the Python vector and moves with every input", () => {
    expect(previewSolveIdentityHash(config, inputs)).toBe(IDENTITY_HASH);
    const seen = new Set([IDENTITY_HASH]);
    const changes: Partial<PreviewSolveIdentityInputs>[] = [
      { reader_sha256: h("9") },
      { version_manifest: { ...inputs.version_manifest, regulation: "M-D" } },
      { solver_format: "preview-solver/2" },
      { leaf_format: 2 },
      { hypothetical_battle_format: 2 },
    ];
    for (const change of changes) {
      const next = previewSolveIdentityHash(config, { ...inputs, ...change });
      expect(seen.has(next)).toBe(false);
      seen.add(next);
    }
    const edits: ((c: typeof config) => void)[] = [(c) => (c.k_own = 3), (c) => (c.worlds.n = 5), (c) => (c.leaf = { kind: "rollout", turns: 1 }), (c) => (c.solver.iterations = 1024)];
    for (const edit of edits) {
      const copy = structuredClone(config);
      edit(copy);
      const next = previewSolveIdentityHash(copy, inputs);
      expect(seen.has(next)).toBe(false);
      seen.add(next);
    }
  });

  it("holds exactly its seven inputs", () => {
    expect(Object.keys(previewSolveIdentity(config, inputs)).sort()).toEqual(["belief_identity_hash", "hypothetical_battle_format", "leaf_format", "preview_solve_config_hash", "reader_sha256", "solver_format", "version_manifest"]);
  });

  it("refuses a belief identity the config does not name and malformed inputs", () => {
    expect(() => previewSolveIdentityHash(config, { ...inputs, belief_identity_hash: h("8") })).toThrow(/belief identity/);
    expect(() => previewSolveIdentityHash(config, { ...inputs, reader_sha256: "abc" })).toThrow(/64 lowercase hex/);
    expect(() => previewSolveIdentityHash(config, { ...inputs, leaf_format: 0 })).toThrow(/positive integer/);
    expect(() => previewSolveIdentityHash(config, { ...inputs, solver_format: "" })).toThrow(/non-empty/);
  });

  it("names the formats the modules export today", () => {
    expect({ PAYOFF_LEAF_FORMAT, HYPOTHETICAL_BATTLE_FORMAT }).toEqual({ PAYOFF_LEAF_FORMAT: 1, HYPOTHETICAL_BATTLE_FORMAT: 1 });
  });
});

interface Existing { preview: string; battle: string; mode: "sample" | "argmax"; temperature_ppm?: number; truncation?: { kind: "threshold"; min_ppm: number }; features?: string; id: string }
const existing = JSON.parse(readFileSync(new URL("../../../fixtures/policy-ids/existing.json", import.meta.url), "utf8")) as Existing[];
const pair = { preview: h("a"), battle: h("b") };

describe("the policy id with the solver", () => {
  it("leaves every existing id byte-identical (ids computed before the change)", () => {
    expect(existing.length).toBeGreaterThanOrEqual(9);
    for (const c of existing) expect(policyId(c, c.mode, { ...(c.temperature_ppm === undefined ? {} : { temperature_ppm: c.temperature_ppm }), ...(c.truncation === undefined ? {} : { truncation: c.truncation }) }, c.features)).toBe(c.id);
    expect(policyId(pair, "argmax", {}, undefined, SEARCH_VECTOR)).toBe("policy:86611a1d2e6ce20f8e0a1a1f4b61cf3a88d27229bc18ef2a80ec4990b1797453");
  });

  it("adds preview_solve only when present, and equals the Python twin", () => {
    expect(policyId(pair, "argmax", {}, undefined, undefined, IDENTITY_HASH)).toBe("policy:917a7880a7a04b493d463d68fee28131bc7484f381de95e60329a8a3da8d5499");
    expect(policyId(pair, "sample", { temperature_ppm: 700000, truncation: { kind: "threshold", min_ppm: 30000 } }, h("f"), SEARCH_VECTOR, IDENTITY_HASH)).toBe("policy:9e18393e4c548c72aa64192bb4c137f1ab4ce6ee163cfa552584863587bbdf92");
  });

  it("is independent of search: the solver, search and both are three different policies", () => {
    const ids = [policyId(pair, "argmax"), policyId(pair, "argmax", {}, undefined, undefined, IDENTITY_HASH), policyId(pair, "argmax", {}, undefined, SEARCH_VECTOR), policyId(pair, "argmax", {}, undefined, SEARCH_VECTOR, IDENTITY_HASH)];
    expect(new Set(ids).size).toBe(4);
  });

  it("refuses a solver identity that is not a hash", () => {
    expect(() => policyId(pair, "sample", {}, undefined, undefined, "nope")).toThrow(PolicyProtocolError);
  });
});

describe("the harness derivations", () => {
  const parts = { seed: "sodium,00112233445566778899aabbccddeeff", game: 3, player: "p1", policy_id: `policy:${h("c")}`, request_seq: 1 };
  it("derive the stream and u as the Python vectors do (their own purposes)", () => {
    expect(Buffer.from(previewSolveStream(parts)).toString("hex")).toBe("131fc6935a8b7981367b07504a096e03a7a0e8e797b0fc9f247aba08715f0143");
    expect(previewSolveU(parts)).toBe(4855325505044104);
    expect(previewSolveU({ ...parts, request_seq: 2 })).not.toBe(4855325505044104);
  });

  it("draw the first strategy whose cumulative ppm exceeds floor(u * 10^6 / 2^53)", () => {
    const mixture = [250_000, 0, 500_000, 250_000];
    const at = (ppm: number): number => drawFromMixture(mixture, Math.ceil((ppm * 2 ** 53) / 1_000_000));
    expect([at(0), at(249_999), at(250_000), at(749_999), at(750_000), at(999_999)]).toEqual([0, 0, 2, 2, 3, 3]);
    expect(drawFromMixture([1_000_000], 2 ** 53 - 1)).toBe(0);
    // A mixture that does not reach the target (it should sum to 10^6) falls to its last entry rather than going out of range.
    expect(drawFromMixture([1, 1], 2 ** 53 - 1)).toBe(1);
  });
});
