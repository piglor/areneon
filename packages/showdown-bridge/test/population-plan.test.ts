import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { leagueGameStreams } from "../src/league-seed.ts";
import { EXPLOITER_REFUSAL, memberLabel, ODD_ANCHOR_REFUSAL, PAST_POOL_REFUSAL, planPopulation, PopulationPlanError, seatCounts } from "../src/population-plan.ts";
import { planGeneration } from "../src/league-generation.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";
import { populationConfig, populationOf, sample } from "./population-support.ts";

const sha = (value: unknown): string => createHash("sha256").update(jcs(value as Json)).digest("hex");

describe("seatCounts", () => {
  it("splits a member's seats exactly by the shares when they divide", () => {
    expect(seatCounts(populationOf(populationConfig(4, 512, [500_000, 250_000, 250_000])))).toEqual({ ladder: 256, self: 128, past: 128 });
    expect(seatCounts(populationOf(populationConfig(3, 8, [500_000, 500_000, 0])))).toEqual({ ladder: 4, self: 4, past: 0 });
  });

  it("gives the leftover seats to the largest remainders and always sums to the seats", () => {
    const counts = seatCounts(populationOf(populationConfig(2, 5, [333_334, 333_333, 333_333])));
    expect(counts).toEqual({ ladder: 2, self: 2, past: 1 });
    for (const seats of [1, 2, 3, 7, 10, 31]) {
      const c = seatCounts(populationOf(populationConfig(2, seats, [400_000, 350_000, 250_000])));
      expect(c.ladder + c.self + c.past).toBe(seats);
    }
  });

  it("moves the ladder share to self-play for a population of one", () => {
    expect(seatCounts(populationOf(populationConfig(1, 8, [500_000, 500_000, 0])))).toEqual({ ladder: 0, self: 8, past: 0 });
  });
});

describe("planPopulation", () => {
  const config = populationConfig(2, 4);
  const plan = planPopulation(config, 1, 50);

  it("is a pure function of config, generation and pool size, pinned by its hash", () => {
    expect(planPopulation(config, 1, 50)).toEqual(plan);
    expect(planPopulation(config, 2, 50)).not.toEqual(plan);
    expect(planPopulation(config, 1, 51)).not.toEqual(plan);
    // The SHA-256 of the JCS of this small plan; a change to any byte of a plan is a new plan format (POPULATION_PLAN_FORMAT).
    expect(sha(plan)).toBe("4c0bcbf9c0ab6f411294b7cb797a5960d35cf5c9781eba2b1b31416ab76db6eb");
  });

  it("plays every pair twice on one battle seed with the seats swapped and the teams staying on their seats", () => {
    expect(plan).toHaveLength(2 * 2 * (2 + 2 + 1));
    for (let pair = 0; pair < plan.length / 2; pair++) {
      const [first, second] = [plan[2 * pair], plan[2 * pair + 1]];
      expect(first?.game).toBe(2 * pair);
      expect(second?.game).toBe(2 * pair + 1);
      expect([first?.pair, second?.pair]).toEqual([pair, pair]);
      expect(first?.seed).toBe(second?.seed);
      if (first?.kind === "self") {
        // The same policy on both seats: the pair swaps the two (distinct) teams' seats.
        expect(second?.teams).toEqual({ p1: first.teams.p2, p2: first.teams.p1 });
        expect(first.teams.p1).not.toBe(first.teams.p2);
      } else {
        expect(first?.teams).toEqual(second?.teams);
        expect(first?.seats.p1).toEqual(second?.seats.p2);
        expect(first?.seats.p2).toEqual(second?.seats.p1);
      }
      const streams = leagueGameStreams("tiny-run", 1, pair);
      expect(first?.seed).toBe(streams.seed);
      if (first?.kind !== "self") expect(first?.teams).toEqual({ p1: streams.team_draws.p1 % 50, p2: streams.team_draws.p2 % 50 });
    }
  });

  it("gives each member exactly the shares of its seats, as pairs it scheduled", () => {
    for (const member of ["m0", "m1"]) {
      const pairs = plan.filter((g) => g.member === member && g.half === 0);
      expect(pairs.filter((g) => g.kind === "ladder")).toHaveLength(2);
      expect(pairs.filter((g) => g.kind === "self")).toHaveLength(2);
      expect(pairs.filter((g) => g.kind === "anchor")).toHaveLength(1);
      for (const g of pairs.filter((x) => x.kind === "ladder")) {
        // The opponent is another member, drawn uniformly from them.
        const others = [g.seats.p1, g.seats.p2].map((s) => ("member" in s ? s.member : "")).filter((m) => m !== member);
        expect(others).toHaveLength(1);
      }
      for (const g of pairs.filter((x) => x.kind === "self")) expect(g.seats).toEqual({ p1: { member }, p2: { member } });
      for (const g of pairs.filter((x) => x.kind === "anchor")) expect(g.seats).toEqual({ p1: { member }, p2: { actor: "random-legal", policy_id: "random-legal@tiny" } });
    }
  });

  it("meets every other member over a larger plan, and never itself on the ladder", () => {
    const wide = planPopulation(populationConfig(4, 60, [1_000_000, 0, 0], 2), 1, 50);
    for (let k = 0; k < 4; k++) {
      const met = new Set(wide.filter((g) => g.kind === "ladder" && g.member === memberLabel(k) && g.half === 0).map((g) => (g.seats.p2 as { member: string }).member));
      expect(met).toEqual(new Set([0, 1, 2, 3].filter((o) => o !== k).map(memberLabel)));
    }
  });

  it("changes with the run key", () => {
    const other = readTrainingRunConfig({ ...(config as unknown as object), seeds: { ...config.seeds, run_key: "another-run" } });
    expect(planPopulation(other, 1, 50).map((g) => g.seed)).not.toEqual(plan.map((g) => g.seed));
  });

  it("refuses a past-pool share, exploiter seats, an odd anchor count, no population and bad numbers, in fixed words", () => {
    expect(() => planPopulation(populationConfig(2, 4, [500_000, 250_000, 250_000]), 1, 50)).toThrow(new PopulationPlanError(PAST_POOL_REFUSAL));
    expect(PAST_POOL_REFUSAL).toBe("population: a non-zero past_pool_ppm is refused until the past pool is built (ADR-021 slice 10)");
    const withExploiter = { ...config, population: { ...populationOf(config), exploiters: 1 } } as TrainingRunConfig;
    expect(() => planPopulation(withExploiter, 1, 50)).toThrow(new PopulationPlanError(EXPLOITER_REFUSAL));
    expect(() => planPopulation(populationConfig(2, 4, [500_000, 500_000, 0], 3), 1, 50)).toThrow(new PopulationPlanError(ODD_ANCHOR_REFUSAL));
    expect(() => planPopulation(readTrainingRunConfig(sample("tiny")), 1, 50)).toThrow(PopulationPlanError);
    expect(() => planPopulation(config, 0, 50)).toThrow(RangeError);
    expect(() => planPopulation(config, 1, 0)).toThrow(RangeError);
  });

  it("leaves planGeneration and a config without a population alone", () => {
    const tiny = readTrainingRunConfig(sample("tiny"));
    expect(planGeneration(tiny, 1, 50)).toHaveLength(6);
    expect(planGeneration(config, 1, 50).length).toBe(config.generation.games + 2);
  });
});
