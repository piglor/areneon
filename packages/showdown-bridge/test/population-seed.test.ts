import { describe, expect, it } from "vitest";
import { POPULATION_SEED_DERIVATION, POPULATION_SEED_PURPOSES, populationDraw, type PopulationSeedPurpose } from "../src/index.ts";

// Computed outside this code base (Python hashlib; the same six numbers are pinned in
// tests/ml/loop/test_population_seed.py):
//   int.from_bytes(hashlib.sha256(json.dumps({"run_key": k, "generation": g, "member": m,
//   "purpose": p, "index": i}, sort_keys=True, separators=(",", ":")).encode()).digest()[:8], "big")
// A change to any of these numbers is a new derivation version.
const VECTORS: [string, number, string, PopulationSeedPurpose, number, bigint][] = [
  ["tiny-run", 0, "m0", "member-init", 0, 13019428712076808456n],
  ["tiny-run", 0, "m1", "member-init", 1, 18329199380263047422n],
  ["tiny-run", 3, "m2", "opponent-draw", 17, 12259119117443046183n],
  ["pop-main", 12, "x0", "explore", 2, 6998671094935325674n],
  ["pop-main", 7, "m3", "bootstrap", 999, 18168013126753230032n],
  ["pop-main", 7, "m3", "member-learner", 0, 6024445643219269802n],
];

describe("population seed derivation (ADR-021 section 5)", () => {
  it("is named population-seed/1 and has the five purposes", () => {
    expect(POPULATION_SEED_DERIVATION).toBe("population-seed/1");
    expect(POPULATION_SEED_PURPOSES).toEqual(["member-init", "member-learner", "opponent-draw", "explore", "bootstrap"]);
  });

  it.each(VECTORS)("reproduces the pinned vector for %s, generation %i, %s, %s, index %i", (run_key, generation, member, purpose, index, draw) => {
    expect(populationDraw(run_key, generation, member, purpose, index)).toBe(draw);
  });

  it("gives every input its own draw", () => {
    const variants: Parameters<typeof populationDraw>[] = [
      ["a", 1, "m0", "explore", 0],
      ["b", 1, "m0", "explore", 0],
      ["a", 2, "m0", "explore", 0],
      ["a", 1, "m1", "explore", 0],
      ["a", 1, "x0", "explore", 0],
      ["a", 1, "m0", "bootstrap", 0],
      ["a", 1, "m0", "explore", 1],
    ];
    expect(new Set(variants.map((v) => populationDraw(...v))).size).toBe(variants.length);
  });

  it("refuses an input out of range", () => {
    expect(() => populationDraw("a", -1, "m0", "explore", 0)).toThrow(RangeError);
    expect(() => populationDraw("a", 0, "m0", "explore", -1)).toThrow(RangeError);
    expect(() => populationDraw("a", 0.5, "m0", "explore", 0)).toThrow(RangeError);
    for (const member of ["0", "m01", "m100", "y0", ""]) {
      expect(() => populationDraw("a", 0, member, "explore", 0)).toThrow("member must be");
    }
    expect(() => populationDraw("a", 0, "m0", "shuffle" as PopulationSeedPurpose, 0)).toThrow("purpose");
  });
});
