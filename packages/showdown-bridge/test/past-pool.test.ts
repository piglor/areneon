import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { drawPast, emptyPastPool, PAST_POOL_FORMAT, type PastCell, type PastPoolFile, type PastSnapshot, pfspWeight, pfspWeights, PastPoolError, poolKeys, readPastPool } from "../src/past-pool.ts";
import { PAST_POOL_MISSING, planPopulation, PopulationPlanError } from "../src/population-plan.ts";
import { runConfigHash } from "../src/training-run-config.ts";
import { populationConfig, populationOf } from "./population-support.ts";

const sha = (value: unknown): string => createHash("sha256").update(jcs(value as Json)).digest("hex");
const policy = (digit: string): string => `policy:${digit.repeat(64)}`;

// The sample population's pool rules: every 5 generations, at most 32, at most 4 per member, p = 2, floor 50,000.
const config = populationConfig(2, 4, [500_000, 250_000, 250_000]);
const settings = populationOf(config).past_pool;

const snapshot = (member: string, generation: number, digit: string): PastSnapshot => ({ id: `${member}@g${generation}`, member, generation, policy_id: policy(digit), preview: "a".repeat(64), battle: "b".repeat(64) });
const SNAPSHOTS = [snapshot("m0", 0, "1"), snapshot("m1", 0, "2"), snapshot("m0", 5, "3"), snapshot("m1", 5, "4")];
const CELLS: PastCell[] = [
  { member: "m0", snapshot: "m0@g0", scores: [0, 0, 2, 0, 0] },
  { member: "m0", snapshot: "m1@g0", scores: [0, 0, 0, 0, 4] },
  { member: "m1", snapshot: "m0@g0", scores: [1, 0, 0, 0, 0] },
  { member: "m1", snapshot: "m1@g0", scores: [0, 0, 1, 0, 1] },
];
const file = async (change: Partial<PastPoolFile> = {}): Promise<PastPoolFile> => ({ schema_version: 1, format: PAST_POOL_FORMAT, run_id: config.seeds.run_key, run_config_hash: await runConfigHash(config), generation: 5, snapshots: SNAPSHOTS, cells: CELLS, ...change });

describe("poolKeys: eviction order and the generation-0 rule", () => {
  it("takes a snapshot of every member at every multiple of snapshot_every, generation 0 included", () => {
    expect(poolKeys(2, 5, 32, 0)).toEqual([[0, 0], [0, 1]]);
    expect(poolKeys(2, 5, 32, 4)).toEqual([[0, 0], [0, 1]]);
    expect(poolKeys(2, 5, 32, 10)).toEqual([[0, 0], [0, 1], [5, 0], [5, 1], [10, 0], [10, 1]]);
  });

  it("evicts the oldest first once the cap is passed", () => {
    expect(poolKeys(2, 5, 3, 5)).toEqual([[0, 0], [0, 1], [5, 1]]);
    expect(poolKeys(2, 5, 3, 10)).toEqual([[0, 0], [0, 1], [10, 1]]);
    expect(poolKeys(2, 5, 4, 15)).toEqual([[0, 0], [0, 1], [15, 0], [15, 1]]);
    expect(poolKeys(1, 2, 3, 8)).toEqual([[0, 0], [6, 0], [8, 0]]);
  });

  it("never evicts a generation-0 pair, so the pool may exceed the cap by exactly those", () => {
    expect(poolKeys(3, 5, 2, 0)).toEqual([[0, 0], [0, 1], [0, 2]]);
    expect(poolKeys(3, 5, 2, 25)).toEqual([[0, 0], [0, 1], [0, 2]]);
    expect(poolKeys(2, 1, 1, 6)).toEqual([[0, 0], [0, 1]]);
  });
});

describe("pfspWeight: the integer f_hard rule", () => {
  it("is (10^6 - s)^p / 10^(6 (p - 1)), floored, from the mean pair score s in ppm", () => {
    // s = 500,000: 500,000^2 / 10^6 = 250,000.
    expect(pfspWeight(settings, [0, 0, 2, 0, 0])).toBe(250_000);
    expect(pfspWeight(settings, [0, 1, 0, 1, 0])).toBe(250_000);
    // s = 750,000 over two pairs: 250,000^2 / 10^6 = 62,500 (above the floor).
    expect(pfspWeight(settings, [0, 0, 1, 0, 1])).toBe(62_500);
    // s = floor(1,000,000 / 3) = 333,333 (pairs at 250,000, 250,000 and 500,000): 666,667^2 / 10^6 = 444,444.88..., floored.
    expect(pfspWeight(settings, [0, 2, 1, 0, 0])).toBe(444_444);
    // s = 0: the largest the rule gives.
    expect(pfspWeight(settings, [4, 0, 0, 0, 0])).toBe(1_000_000);
  });

  it("never goes below the floor", () => {
    expect(pfspWeight(settings, [0, 0, 0, 0, 4])).toBe(50_000);
    expect(pfspWeight(settings, [0, 0, 0, 1, 9])).toBe(50_000);
    expect(pfspWeight({ ...settings, floor_ppm: 0 }, [0, 0, 0, 0, 4])).toBe(0);
  });

  it("gives a snapshot with no games the largest weight, 10^6", () => {
    expect(pfspWeight(settings, undefined)).toBe(1_000_000);
    expect(pfspWeight(settings, [0, 0, 0, 0, 0])).toBe(1_000_000);
  });

  it("follows the power: p = 3 at s = 500,000 is 500,000^3 / 10^12 = 125,000, and p = 16 stays exact", () => {
    expect(pfspWeight({ ...settings, pfsp_power: 3 }, [0, 0, 2, 0, 0])).toBe(125_000);
    expect(pfspWeight({ ...settings, pfsp_power: 1 }, [0, 0, 2, 0, 0])).toBe(500_000);
    expect(pfspWeight({ ...settings, pfsp_power: 16, floor_ppm: 0 }, [0, 0, 2, 0, 0])).toBe(15); // 500,000^16 / 10^90 = 15.2...
  });

  it("reads each member's own cell", async () => {
    const pool = await file();
    expect(pfspWeights(settings, pool, "m0", SNAPSHOTS)).toEqual([250_000, 50_000, 1_000_000, 1_000_000]);
    expect(pfspWeights(settings, pool, "m1", SNAPSHOTS)).toEqual([1_000_000, 62_500, 1_000_000, 1_000_000]);
  });
});

describe("drawPast", () => {
  it("draws the snapshots Python's twin draws (vectors computed with areneon_ml.loop.past_pool)", async () => {
    const pool = await file();
    const ids = (member: string, pairs: number, current: string, generation = 6): string[] => drawPast(config, generation, member, pairs, current, pool).map((entry) => entry.id);
    expect(ids("m0", 1, policy("3"))).toEqual(["m1@g5"]);
    expect(ids("m0", 3, policy("3"))).toEqual(["m1@g5", "m1@g5", "m0@g0"]);
    expect(ids("m0", 6, policy("3"))).toEqual(["m1@g5", "m1@g5", "m0@g0", "m1@g5", "m1@g5", "m1@g0"]);
    expect(ids("m1", 1, policy("4"))).toEqual(["m0@g0"]);
    expect(ids("m1", 3, policy("4"))).toEqual(["m0@g0", "m0@g0", "m0@g0"]);
    expect(ids("m1", 6, policy("4"))).toEqual(["m0@g0", "m0@g0", "m0@g0", "m0@g5", "m0@g0", "m0@g0"]);
    expect(ids("m0", 3, policy("3"), 7)).toEqual(["m1@g5", "m1@g5", "m1@g0"]);
    expect(ids("m1", 6, policy("4"), 7)).toEqual(["m0@g0", "m0@g5", "m0@g5", "m0@g5", "m0@g5", "m0@g5"]);
  });

  it("never draws the member's own current pair, and draws nothing from an empty pool or for no pairs", async () => {
    const pool = await file();
    expect(drawPast(config, 6, "m0", 20, policy("3"), pool).every((entry) => entry.policy_id !== policy("3"))).toBe(true);
    expect(drawPast(config, 6, "m0", 5, policy("3"), emptyPastPool(config))).toEqual([]);
    expect(drawPast(config, 6, "m0", 0, policy("3"), pool)).toEqual([]);
  });

  it("meets at most max_past_per_member distinct snapshots, and at most as many as it has pairs", async () => {
    const pool = await file();
    const capped = { ...config, population: { ...populationOf(config), past_pool: { ...settings, max_past_per_member: 2 } } };
    for (const generation of [6, 7, 8, 9, 10, 11, 12]) {
      expect(new Set(drawPast(capped, generation, "m0", 40, policy("3"), pool).map((entry) => entry.id)).size).toBeLessThanOrEqual(2);
      expect(new Set(drawPast(config, generation, "m0", 1, policy("3"), pool).map((entry) => entry.id)).size).toBe(1);
    }
  });

  it("draws equally when every weight is 0 (a floor of 0 and a score of 1,000,000 against all of them)", async () => {
    const zero = { ...config, population: { ...populationOf(config), past_pool: { ...settings, floor_ppm: 0 } } };
    const cells: PastCell[] = ["m0@g0", "m1@g0", "m1@g5"].map((id) => ({ member: "m0", snapshot: id, scores: [0, 0, 0, 0, 3] as PastCell["scores"] }));
    const found = new Set(Array.from({ length: 20 }, (_, g) => drawPast(zero, g + 1, "m0", 1, policy("3"), { ...emptyPastPool(config, 5), snapshots: SNAPSHOTS, cells }).map((entry) => entry.id)[0]));
    expect(found.size).toBeGreaterThan(1);
  });
});

describe("readPastPool", () => {
  const hash = runConfigHash(config);
  it("reads a good file back unchanged", async () => {
    expect(readPastPool(JSON.parse(JSON.stringify(await file())), config, await hash)).toEqual(await file());
  });

  it("refuses another run or config, a malformed shape and a pool the config does not give", async () => {
    const good = await file();
    const read = async (change: object): Promise<unknown> => readPastPool({ ...good, ...change }, config, await hash);
    await expect(read({ run_id: "other" })).rejects.toBeInstanceOf(PastPoolError);
    expect(() => readPastPool({ ...good, run_config_hash: "0".repeat(64) }, config, "1".repeat(64))).toThrow(/another run or config/);
    expect(() => readPastPool([], config, "x")).toThrow(/not a past-pool file/);
    await expect(read({ extra: 1 })).rejects.toThrow(/not a past-pool file/);
    await expect(read({ format: "past-pool/2" })).rejects.toThrow(/version 1/);
    await expect(read({ generation: -1 })).rejects.toThrow(/non-negative/);
    await expect(read({ snapshots: "x" })).rejects.toThrow(/lists/);
    await expect(read({ snapshots: SNAPSHOTS.slice(1) })).rejects.toThrow(/not the pool the config gives/);
    await expect(read({ snapshots: [...SNAPSHOTS].reverse() })).rejects.toThrow(/not the pool the config gives/);
    await expect(read({ snapshots: [{ ...SNAPSHOTS[0], id: "m0@g1" }, ...SNAPSHOTS.slice(1)] })).rejects.toThrow(/id must be its member and generation/);
    await expect(read({ snapshots: [{ ...SNAPSHOTS[0], extra: 1 }, ...SNAPSHOTS.slice(1)] })).rejects.toThrow(/wrong keys/);
    await expect(read({ snapshots: [{ ...SNAPSHOTS[0], policy_id: "nope" }, ...SNAPSHOTS.slice(1)] })).rejects.toThrow(/malformed/);
    await expect(read({ snapshots: [snapshot("m0", 1, "1")] })).rejects.toThrow(/snapshot_every/);
    await expect(read({ snapshots: [snapshot("m5", 0, "1")] })).rejects.toThrow(/member the run does not have/);
  });

  it("refuses unsorted cells, a cell for a snapshot outside the pool and a cell without pairs", async () => {
    const read = async (cells: unknown): Promise<unknown> => readPastPool({ ...(await file()), cells }, config, await hash);
    await expect(read([CELLS[1], CELLS[0]])).rejects.toThrow(/sorted/);
    await expect(read([{ member: "m0", snapshot: "m0@g10", scores: [1, 0, 0, 0, 0] }])).rejects.toThrow(/not in the pool/);
    await expect(read([{ member: "m0", snapshot: "m0@g0", scores: [0, 0, 0, 0, 0] }])).rejects.toThrow(/at least one pair/);
    await expect(read([{ member: "m0", snapshot: "m0@g0", scores: [1, 0, 0, 0, -1] }])).rejects.toThrow(/non-negative/);
    await expect(read([{ member: "m9", snapshot: "m0@g0", scores: [1, 0, 0, 0, 0] }])).rejects.toThrow(/member the run does not have/);
    await expect(read([{ member: "m0", snapshot: "m0@g0", scores: [1, 0, 0, 0, 0], extra: 1 }])).rejects.toThrow(/wrong keys/);
  });
});

describe("the plan with a past pool", () => {
  const current = { m0: policy("3"), m1: policy("4") };

  it("is a pure function of the committed pool: past pairs follow the self-play pairs, each a drawn snapshot", async () => {
    const pool = await file();
    const plan = planPopulation(config, 6, 50, { pool, current });
    expect(planPopulation(config, 6, 50, { pool, current })).toEqual(plan);
    const past = plan.filter((entry) => entry.kind === "past" && entry.half === 0);
    expect(past.map((entry) => [entry.member, entry.seats.p2])).toEqual([
      ["m0", { snapshot: "m1@g5", policy_id: policy("4") }],
      ["m1", { snapshot: "m0@g0", policy_id: policy("1") }],
    ]);
    // Both halves of a pair: the member's seat alternates, the snapshot is the other seat, the teams stay put.
    const [first, second] = plan.filter((entry) => entry.kind === "past" && entry.member === "m0");
    expect(first?.seats).toEqual({ p1: { member: "m0" }, p2: { snapshot: "m1@g5", policy_id: policy("4") } });
    expect(second?.seats).toEqual({ p1: { snapshot: "m1@g5", policy_id: policy("4") }, p2: { member: "m0" } });
    expect(second?.teams).toEqual(first?.teams);
    expect(second?.seed).toBe(first?.seed);
    // The order within a member: ladder, self, past, anchors; the pairs number on through the generation.
    const ofM0 = plan.filter((entry) => entry.member === "m0" && entry.half === 0).map((entry) => entry.kind);
    expect(ofM0).toEqual(["ladder", "ladder", "self", "past", "anchor"]);
    expect(plan.map((entry) => entry.pair)).toEqual(plan.map((_, index) => Math.floor(index / 2)));
    // The pinned hash of this plan from the fixture pool: a change to what a plan holds is a new plan format.
    expect(sha(plan)).toBe(PINNED_PLAN_HASH);
  });

  it("changes with the cells it was given (the weights) and with the pool's membership", async () => {
    const base = planPopulation(config, 6, 50, { pool: await file(), current });
    const other = planPopulation(config, 6, 50, { pool: await file({ cells: [] }), current });
    const fewer = planPopulation(config, 6, 50, { pool: await file({ snapshots: SNAPSHOTS.slice(0, 2), cells: CELLS.filter((c) => c.snapshot.endsWith("g0")), generation: 4 }), current });
    expect(sha(other)).not.toBe(sha(base));
    expect(sha(fewer)).not.toBe(sha(base));
    expect(sha(planPopulation(config, 7, 50, { pool: await file(), current }))).not.toBe(sha(base));
  });

  it("gives a member with nothing to meet its past pairs as self-play, and refuses a share without a pool", async () => {
    const empty = planPopulation(config, 1, 50, { pool: emptyPastPool(config), current });
    expect(empty.filter((entry) => entry.kind === "past")).toEqual([]);
    expect(empty.filter((entry) => entry.kind === "self" && entry.half === 0).length).toBe(2 * 2);
    expect(empty).toHaveLength(planPopulation(config, 1, 50, { pool: await file(), current }).length);
    expect(() => planPopulation(config, 1, 50)).toThrow(new PopulationPlanError(PAST_POOL_MISSING));
    // A config without a share plans as it did, with or without a pool.
    const none = populationConfig(2, 4);
    expect(planPopulation(none, 1, 50, { pool: await file(), current })).toEqual(planPopulation(none, 1, 50));
  });
});

const PINNED_PLAN_HASH = "7b5da6f371c03dedeca32876385e27890301ab93f4bee47f73a848f18e7294f6";
