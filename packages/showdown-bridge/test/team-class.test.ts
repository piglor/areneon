import { describe, expect, it } from "vitest";
import { classOfSets, syntheticTeamHashes, teamHash } from "../src/team-class.ts";

const sets = [{ species: "Pikachu" }, { species: "Incineroar" }];

describe("syntheticTeamHashes", () => {
  it("builds the known set from lists of hashes, and a member's team is synthetic", () => {
    const known = syntheticTeamHashes([teamHash(sets)], new Set(["a".repeat(64)]));
    expect(known.has(teamHash(sets))).toBe(true);
    expect(known.has("a".repeat(64))).toBe(true);
    expect(classOfSets(sets, known)).toBe("synthetic");
  });

  it("leaves a team outside the lists private", () => {
    expect(classOfSets(sets, syntheticTeamHashes([teamHash([{ species: "Pikachu" }])]))).toBe("private");
    expect(classOfSets(sets, syntheticTeamHashes())).toBe("private");
  });

  it("refuses a value that is not a SHA-256 digest", () => {
    for (const bad of ["", "abc", "A".repeat(64), "g".repeat(64)]) expect(() => syntheticTeamHashes([bad])).toThrow(TypeError);
  });
});
