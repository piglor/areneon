import { describe, expect, it } from "vitest";
import { candidateFeaturesOutcome } from "../src/candidate-features-outcome.ts";
import { createFeatureMemo, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { clearProbeCaches, keptScenario, probeCaches, restoreScenario, scenarioCacheStats, scenarioFormatId, startScenario } from "../src/mechanics-scenario.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { playerChannel } from "../src/player-channel.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { config, corpus, engine, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";

// #1453: the probe caches (the resolved scenario format with its rule table, and the kept scenario start of every probe path) never change a
// result. With them off every probe starts its scenario itself; the rows are equal bytes on every request of seeded battles, both orders.

const configs: FeatureConfigs = { vocabulary, config, mechanics: withMoveOutcome(mechanics), outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT) }, snapshot, data_class: "synthetic" };

interface Point {
  channel: string[];
  team: Parameters<typeof candidateFeaturesOutcome>[3];
  candidates: { choice: string; pending: boolean }[];
}

async function points(index: number): Promise<Point[]> {
  const battle = corpus[index] as (typeof corpus)[number];
  const out: Point[] = [];
  const base = randomLegalActor(1000 + index);
  const p1: Actor = (request, context) => {
    if (request.phase === "move" && request.legal !== undefined && context?.team !== undefined) {
      const legal = request.legal;
      out.push({ channel: [...context.channel], team: context.team, candidates: [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) })) });
    }
    return base(request, context);
  };
  await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams, actors: { p1, p2: randomLegalActor(2000 + index) }, maxTurns: 12 });
  return out;
}

const rowsOf = (point: Point): string =>
  jcs(candidateFeaturesOutcome(engine, configs, playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: point.channel }), point.team, point.candidates, createFeatureMemo()) as unknown as Json);

describe("the probe caches change no row", () => {
  it("gives equal bytes with the caches on (cold, then warm) and off, on seeded battles", async () => {
    const all: Point[] = [];
    for (const index of [0, 3, 5]) all.push(...(await points(index)).filter((_, i) => i % 2 === 0));
    expect(all.length).toBeGreaterThan(8);
    probeCaches.enabled = false;
    const reference = all.map(rowsOf);
    probeCaches.enabled = true;
    clearProbeCaches(engine);
    const before = { ...scenarioCacheStats };
    const cold = all.map(rowsOf);
    const warm = all.map(rowsOf);
    expect(cold).toEqual(reference);
    expect(warm).toEqual(reference);
    // Not vacuous: the second pass reused kept starts and the resolved format.
    expect(scenarioCacheStats.start_reused - before.start_reused).toBeGreaterThan(all.length);
    expect(scenarioCacheStats.format_reused).toBeGreaterThan(before.format_reused);
  }, 600_000);
});

/** A battle as canonical JSON: key order is not part of a battle. */
const canon = (battle: { toJSON(): object }): string => jcs(JSON.parse(JSON.stringify(battle.toJSON())) as Json);

describe("the kept scenario", () => {
  const set = (corpus[0] as (typeof corpus)[number]).teams.p1.slice(0, 2);
  const teams = { p1: set, p2: set };
  const scenarioId = scenarioFormatId(mechanics);

  it("builds a battle equal to a fresh start, and every call its own battle", () => {
    const fresh = JSON.stringify(startScenario(engine, scenarioId, teams).toJSON());
    const want = canon(startScenario(engine, scenarioId, teams));
    const a = keptScenario(engine, scenarioId, teams);
    const b = keptScenario(engine, scenarioId, teams);
    expect(a).not.toBe(b);
    expect(canon(a)).toBe(want);
    expect(canon(b)).toBe(want);
    expect(canon(restoreScenario(engine, fresh))).toBe(want);
  });

  it("starts the scenario itself every time with the caches off", () => {
    probeCaches.enabled = false;
    try {
      const before = { ...scenarioCacheStats };
      keptScenario(engine, scenarioId, teams);
      keptScenario(engine, scenarioId, teams);
      expect(scenarioCacheStats).toEqual(before);
    } finally {
      probeCaches.enabled = true;
    }
  });

  it("restores the engine's format lookup after every call, a failed one included", () => {
    const formats = engine.sim.Dex.formats;
    const original = formats.get;
    startScenario(engine, scenarioId, teams);
    expect(formats.get).toBe(original);
    expect(() => restoreScenario(engine, "not json")).toThrow();
    expect(formats.get).toBe(original);
  });

  it("keeps a bounded number of starts: past the limit the table is emptied and later calls still answer", () => {
    const base = (corpus[0] as (typeof corpus)[number]).teams.p1[0] as { level?: number };
    for (let level = 1; level <= 300; level++) {
      const varied = [{ ...base, level: (level % 100) + 1, nature: level % 2 === 0 ? "Hardy" : "Adamant", name: `n${level}` }] as unknown as typeof set;
      keptScenario(engine, scenarioId, { p1: [...varied, ...varied], p2: set });
    }
    expect(canon(keptScenario(engine, scenarioId, teams))).toBe(canon(startScenario(engine, scenarioId, teams)));
  }, 120_000);
});
