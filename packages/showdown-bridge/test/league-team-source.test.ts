import { describe, expect, it } from "vitest";
import { leagueGameStreams, privateTeamSource } from "../src/index.ts";
import { type GamePlan, GenerationError, planGames, playEntries, playGeneration, poolShape, refusePrivatePool, syntheticDrawnHashes, type TrainingPool } from "../src/league-generation.ts";

// Computed outside this code base (Python hashlib over json.dumps(sort_keys, compact), first four bytes big-endian):
// n and n * 1000 / 2^32 for `league-team-source/1`. A seat is private when that position is below the share.
const VECTORS = [
  { generation: 1, game: 0, seat: "p1", position: 38.13175274990499 },
  { generation: 1, game: 0, seat: "p2", position: 141.76270854659379 },
  { generation: 3, game: 17, seat: "p1", position: 652.8143889736384 },
  { generation: 3, game: 17, seat: "p2", position: 701.6126539092511 },
  { generation: 2, game: 5, seat: "p1", position: 81.03507221676409 },
] as const;

const opponents = (n: number): GamePlan["opponent"][] => Array.from({ length: n }, () => "self" as const);

describe("league-team-source/1 (ADR-015 erratum E1 item 3)", () => {
  it.each(VECTORS)("reproduces the pinned draw for generation $generation, game $game, $seat", (v) => {
    const at = (share: number): boolean => privateTeamSource("vec-run", v.generation, v.game, v.seat, share);
    expect(at(Math.floor(v.position))).toBe(false); // below the position: not private
    expect(at(Math.floor(v.position) + 1)).toBe(true);
  });

  it("is never private at 0 and always private at 1000, and refuses a bad share or index", () => {
    for (let game = 0; game < 50; game++) {
      expect(privateTeamSource("r", 1, game, "p1", 0)).toBe(false);
      expect(privateTeamSource("r", 1, game, "p2", 1000)).toBe(true);
    }
    expect(() => privateTeamSource("r", 1, 0, "p1", 1001)).toThrow(RangeError);
    expect(() => privateTeamSource("r", 1, 0, "p1", -1)).toThrow(RangeError);
    expect(() => privateTeamSource("r", 1, 0, "p1", 0.5)).toThrow(RangeError);
    expect(() => privateTeamSource("r", 0, 0, "p1", 5)).toThrow(RangeError);
    expect(() => privateTeamSource("r", 1, -1, "p1", 5)).toThrow(RangeError);
  });

  it("depends on the run key, generation, game and seat", () => {
    const bits = (key: string, generation: number, seat: "p1" | "p2"): string => Array.from({ length: 64 }, (_, g) => (privateTeamSource(key, generation, g, seat, 500) ? "1" : "0")).join("");
    expect(new Set([bits("a", 1, "p1"), bits("b", 1, "p1"), bits("a", 2, "p1"), bits("a", 1, "p2")]).size).toBe(4);
  });
});

describe("planGames with a private component", () => {
  const synthetic = 37;
  const privateCount = 11;
  const shape = (share: number) => ({ size: synthetic + privateCount, private: { count: privateCount, share_permille: share } });

  it("without a private component every byte of the plan is the old one (a number or a shape without it)", () => {
    const old = planGames("run", 2, 48, opponents(300));
    expect(planGames("run", 2, { size: 48 }, opponents(300))).toEqual(old);
    for (const [game, entry] of old.entries()) {
      const streams = leagueGameStreams("run", 2, game);
      expect(entry).toEqual({ game, seed: streams.seed, teams: { p1: streams.team_draws.p1 % 48, p2: streams.team_draws.p2 % 48 }, opponent: "self", policy_seat: "both", baseline_seed: streams.baseline_seed });
    }
  });

  it("is a pure function of the config, generation and both pools, and keeps seeds and baseline streams", () => {
    const plan = planGames("run", 3, shape(250), opponents(400));
    expect(planGames("run", 3, shape(250), opponents(400))).toEqual(plan);
    const plain = planGames("run", 3, synthetic, opponents(400));
    plan.forEach((entry, game) => {
      expect(entry.seed).toBe((plain[game] as GamePlan).seed);
      expect(entry.baseline_seed).toBe((plain[game] as GamePlan).baseline_seed);
      const streams = leagueGameStreams("run", 3, game);
      for (const seat of ["p1", "p2"] as const) {
        const index = entry.teams[seat];
        // The index inside the component is the existing league-seed/1 draw.
        if (privateTeamSource("run", 3, game, seat, 250)) expect(index).toBe(synthetic + (streams.team_draws[seat] % privateCount));
        else expect(index).toBe(streams.team_draws[seat] % synthetic);
      }
    });
    expect(planGames("run", 3, shape(400), opponents(400))).not.toEqual(plan);
  });

  it("honours the share within a binomial bound over a large plan", () => {
    for (const share of [100, 300, 900]) {
      const plan = planGames("share-run", 1, shape(share), opponents(20000));
      const seats = plan.flatMap((entry) => [entry.teams.p1, entry.teams.p2]);
      const privateSeats = seats.filter((index) => index >= synthetic).length;
      const p = share / 1000;
      const sd = Math.sqrt(seats.length * p * (1 - p));
      expect(Math.abs(privateSeats - seats.length * p)).toBeLessThan(5 * sd);
      // Both components are used throughout.
      expect(new Set(seats.filter((i) => i >= synthetic)).size).toBe(privateCount);
    }
  });

  it("takes everything from one component at 0 or 1000 and refuses an empty component", () => {
    const none = planGames("run", 1, shape(0), opponents(200));
    expect(none.every((entry) => entry.teams.p1 < synthetic && entry.teams.p2 < synthetic)).toBe(true);
    const all = planGames("run", 1, shape(1000), opponents(200));
    expect(all.every((entry) => entry.teams.p1 >= synthetic && entry.teams.p2 >= synthetic)).toBe(true);
    expect(() => planGames("run", 1, { size: 5, private: { count: 0, share_permille: 5 } }, opponents(1))).toThrow(RangeError);
    expect(() => planGames("run", 1, { size: 5, private: { count: 5, share_permille: 5 } }, opponents(1))).toThrow(RangeError);
  });

  it("poolShape reads the size and the private component of a pool", () => {
    const base = { teams: [[], []], hashes: ["a", "b"], known_synthetic: new Set<string>() } as unknown as TrainingPool;
    expect(poolShape(base)).toEqual({ size: 2 });
    expect(poolShape({ ...base, private: { count: 1, share_permille: 7, holdout: [true] } })).toEqual({ size: 2, private: { count: 1, share_permille: 7 } });
  });
});

describe("a pool with a private component in a generation", () => {
  const pool = (withPrivate: boolean): TrainingPool =>
    ({
      teams: [[], [], [], []],
      hashes: ["s0", "s1", "p0", "p1"],
      known_synthetic: new Set(["s0", "s1"]),
      ...(withPrivate ? { private: { count: 2, share_permille: 500, holdout: [false, false] } } : {}),
    }) as unknown as TrainingPool;

  it("is refused by playGeneration and playEntries with a fixed error, before any Episode is written", async () => {
    const options = { pool: pool(true) } as never;
    await expect(playGeneration(options)).rejects.toThrow(GenerationError);
    await expect(playEntries(options)).rejects.toThrow("a pool with a private component cannot be played by this generation yet");
    expect(() => refusePrivatePool(pool(true))).toThrow(GenerationError);
    expect(() => refusePrivatePool(pool(false))).not.toThrow();
  });

  it("names synthetic teams only in the cumulative team set", () => {
    const plan = [
      { teams: { p1: 0, p2: 2 } },
      { teams: { p1: 3, p2: 1 } },
      { teams: { p1: 1, p2: 0 } },
    ] as unknown as GamePlan[];
    expect(syntheticDrawnHashes(plan, pool(true))).toEqual(["s0", "s1"]);
    // Without a private component every drawn team is named (the old behaviour).
    expect(syntheticDrawnHashes(plan, pool(false))).toEqual(["p0", "p1", "s0", "s1"]);
    // A real plan over a mixed pool never carries a private hash.
    const mixed = planGames("drawn-run", 1, poolShape(pool(true)), opponents(500));
    const hashes = syntheticDrawnHashes(mixed, pool(true));
    expect(hashes.every((h) => h.startsWith("s"))).toBe(true);
    expect(hashes).toEqual(["s0", "s1"]);
  });
});
