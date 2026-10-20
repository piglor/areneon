import { describe, expect, it } from "vitest";
import { LEAGUE_SEED_DERIVATION, leagueBattleSeed, leagueGameStreams, policyStreamLabel } from "../src/index.ts";
import { sampleU } from "../src/policy-actor.ts";

// Computed outside this code base (Python hashlib over json.dumps(sort_keys, compact)): a pinned test vector
// of `league-seed/1`. A change to any of these numbers is a new derivation version.
const VECTORS = [
  { run_key: "tiny-run", generation: 1, game: 0, seed: "sodium,b73e6191281f92d07753d42296dfb7c0", teams: [3869884051, 1010911428], baseline: 30648425 },
  { run_key: "tiny-run", generation: 1, game: 5, seed: "sodium,cb9117ffd3513b4158002651bc9692bb", teams: [2268974604, 3949729686], baseline: 3798732311 },
  { run_key: "tiny-run", generation: 2, game: 0, seed: "sodium,bbbfbe1e69918c198699250e57ea85cf", teams: [2286293003, 1004693005], baseline: 1217926759 },
  { run_key: "league-main", generation: 7, game: 31, seed: "sodium,ae40cd0b66b292755d3fb44100de3cec", teams: [3169554341, 2084957086], baseline: 2668631861 },
];

describe("league seed derivation (ADR-009 §3)", () => {
  it("is named league-seed/1", () => expect(LEAGUE_SEED_DERIVATION).toBe("league-seed/1"));

  it.each(VECTORS)("reproduces the pinned vector for $run_key, generation $generation, game $game", (v) => {
    const streams = leagueGameStreams(v.run_key, v.generation, v.game);
    expect(streams).toEqual({ seed: v.seed, team_draws: { p1: v.teams[0], p2: v.teams[1] }, baseline_seed: v.baseline });
    expect(leagueBattleSeed(v.run_key, v.generation, v.game)).toBe(v.seed);
    expect(streams.seed).toMatch(/^sodium,[0-9a-f]{32}$/);
  });

  it("gives every key, generation and game its own seed", () => {
    const seeds = new Set([
      leagueBattleSeed("a", 1, 0),
      leagueBattleSeed("b", 1, 0),
      leagueBattleSeed("a", 2, 0),
      leagueBattleSeed("a", 1, 1),
    ]);
    expect(seeds.size).toBe(4);
  });

  it("refuses an index that is not an integer in range", () => {
    expect(() => leagueGameStreams("a", 0, 0)).toThrow(RangeError);
    expect(() => leagueGameStreams("a", 1, -1)).toThrow(RangeError);
    expect(() => leagueGameStreams("a", 1.5, 0)).toThrow(RangeError);
    expect(() => leagueGameStreams("a", 1, Number.NaN)).toThrow(RangeError);
  });
});

describe("the per-seat policy stream (protocol D9)", () => {
  const seed = "sodium,00112233445566778899aabbccddeeff";
  const policy_id = `policy:${"a".repeat(64)}`;

  it("labels a seat by seed, game, seat and policy", () => {
    expect(policyStreamLabel(seed, 3, "p2", policy_id)).toEqual({ seed, game: 3, player: "p2", policy_id });
  });

  it("keeps the sampler's draw as an independent implementation computes it", () => {
    expect(sampleU({ seed, game: 3, player: "p2", policy_id, request_seq: 4 })).toBe(3558399440586964);
  });

  it("gives two games of one battle seed, two seats and two policies independent streams", () => {
    const draw = (game: number, player: "p1" | "p2", id: string) => sampleU({ seed, game, player, policy_id: id, request_seq: 1 });
    const draws = new Set([draw(1, "p1", policy_id), draw(3, "p1", policy_id), draw(1, "p2", policy_id), draw(1, "p1", "other@x")]);
    expect(draws.size).toBe(4);
  });
});
