import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { leagueGameStreams } from "../src/league-seed.ts";
import { jcs } from "../src/canonical-json.ts";
import { PROGRESS_PAIRING, PROGRESS_SEED_DERIVATION, progressGameStreams, progressPairs } from "../src/progress-seed.ts";

// The ADR-017 test vector for run key public-main-001, computed with Python hashlib.
const VECTOR = [
  [0, "sodium,f1f6d174ab06d8da49d0d29c894c1efa", 2351535428],
  [1, "sodium,4fb0138fce45349660355601f54d606f", 1369276180],
  [2, "sodium,331b23b6b714a4feb87b0db05b6204fd", 2554743786],
] as const;

describe("progress-seed/1", () => {
  it("names its derivation and pairing", () => {
    expect(PROGRESS_SEED_DERIVATION).toBe("progress-seed/1");
    expect(PROGRESS_PAIRING).toBe("progress-pairing/1");
  });

  it("reproduces the vector the ADR computed with Python hashlib", () => {
    for (const [game, seed, baseline_seed] of VECTOR) expect(progressGameStreams("public-main-001", game)).toEqual({ seed, baseline_seed });
  });

  it("is SHA-256 of the JCS of {run_key, game, purpose}: first 16 bytes the seed, bytes 16 to 19 the baseline", () => {
    const digest = createHash("sha256").update('{"game":9,"purpose":"progress-game","run_key":"r"}').digest();
    expect(progressGameStreams("r", 9)).toEqual({ seed: `sodium,${digest.subarray(0, 16).toString("hex")}`, baseline_seed: digest.readUInt32BE(16) });
  });

  it("never equals a league seed, which hashes another purpose", () => {
    const league = new Set<string>();
    for (let generation = 1; generation <= 5; generation++) for (let game = 0; game < 200; game++) league.add(leagueGameStreams("run", generation, game).seed);
    for (let game = 0; game < 200; game++) expect(league.has(progressGameStreams("run", game).seed)).toBe(false);
  });

  it("refuses a game that is not a non-negative integer", () => {
    expect(() => progressGameStreams("r", -1)).toThrow(RangeError);
    expect(() => progressGameStreams("r", 1.5)).toThrow(RangeError);
  });
});

describe("progress-pairing/1", () => {
  const teams = Array.from({ length: 21 }, (_, i) => createHash("sha256").update(`team ${i}`).digest("hex"));

  it("cuts the teams into disjoint ordered pairs, leaving an odd one out", () => {
    const pairs = progressPairs("run", teams);
    expect(pairs).toHaveLength(10);
    const used = pairs.flat();
    expect(new Set(used).size).toBe(20);
    for (const team of used) expect(teams).toContain(team);
  });

  it("orders by SHA-256 of the JCS of {run_key, canonical_team_hash, purpose}, whatever order the teams come in", () => {
    const key = (team: string): string => createHash("sha256").update(jcs({ run_key: "run", canonical_team_hash: team, purpose: "progress-pairing" })).digest("hex");
    const expected = [...teams].sort((a, b) => (key(a) < key(b) ? -1 : 1));
    expect(progressPairs("run", teams).flat()).toEqual(expected.slice(0, 20));
    expect(progressPairs("run", [...teams].reverse())).toEqual(progressPairs("run", teams));
  });

  it("depends on the run key and refuses a team twice", () => {
    expect(progressPairs("other", teams)).not.toEqual(progressPairs("run", teams));
    expect(() => progressPairs("run", [...teams, teams[0] as string])).toThrow(RangeError);
    expect(progressPairs("run", [])).toEqual([]);
  });
});
