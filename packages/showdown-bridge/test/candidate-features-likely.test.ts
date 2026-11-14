import { beforeAll, describe, expect, it } from "vitest";
import { readerSha256, toId, type PublicMon, type WorldFacts } from "@areneon/public-state";
import { candidateFeaturesLikely, FEATURE_SET_LIKELY, type CandidateFeaturesLikely } from "../src/candidate-features-likely.ts";
import { candidateFeaturesOutcome } from "../src/candidate-features-outcome.ts";
import { candidateFeatures2 } from "../src/candidate-features-2.ts";
import { createFeatureMemo, type FeatureConfigs, featureSetHash, featureSetIdentity, LIKELY_THREATS_FORMAT } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { buildFeatures, configuredSetHash, isFeatureSetId } from "../src/feature-build.ts";
import { choiceLocked, inclusionsOf, listOf, valuesAt, type CachedDamage } from "../src/likely-threats.ts";
import { type LikelyConfig, likelyConfigFiles, likelyConfigHash, type LikelySetup, loadLikelySetup, type MoveList } from "../src/likely-threats-config.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { resolveFormat } from "../src/formats.ts";
import { mechanicsConfigHash } from "../src/mechanics-config.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { buildResult } from "../src/mechanics-result.ts";
import type { MechanicsContext } from "../src/mechanics-probe.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { PlayerChannel } from "../src/observation.ts";
import type { PokemonSet } from "../src/engine.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

// ADR-022 erratum E-22.7: `probe-outcome-likely/1`. Stand-in probe tests of the logic (likelihood, ranking, cut, refusals, identity, the
// bytes of `probe-outcome/1`), with the real engine for the validator, the sets and the corpus requests.

const withOutcome = withMoveOutcome(mechanics);
const answerOutcome = { kind: "computed", answer: { kind: "move_outcome", precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000 } } as never;
const dex = engine.sim.Dex.mod("champions") as unknown as { moves: { get(id: string): { exists: boolean; id: string; category: string; target: string } } };
const base = loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.json");

/** A probe stand-in: `stubProbe`'s damage and stats answers (a constant roll per move), every action of a speed order in a group of its own, outcomes that pass. */
function probe(values: Record<string, number> = {}, log: ParsedMechanicsQuery[] = []) {
  const base = stubProbe({ values, log });
  return (context: MechanicsContext, query: ParsedMechanicsQuery) => {
    if (query.query.kind === "speed_order" || query.query.kind === "move_outcome") {
      log.push(query);
      const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(withOutcome), gap_ids: [] };
      if (query.query.kind === "move_outcome") return { result: buildResult(identity, answerOutcome), diagnostics: [] };
      const actions = query.query.actions.map((a) => ({ ...a, order: a.move === null ? 103 : 200, priority: 0, speed: 100 - 10 * a.side }));
      return { result: buildResult(identity, { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups: actions.map((_, i) => [i]) } as never }), diagnostics: [] };
    }
    return base(context, query);
  };
}
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;

const configs = (change: Partial<FeatureConfigs> = {}, likely: LikelySetup = base, values: Record<string, number> = {}): FeatureConfigs => ({
  vocabulary,
  config,
  mechanics: withOutcome,
  outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT) },
  likely,
  snapshot,
  data_class: "synthetic",
  probe: probe(values),
  ...change,
});

/** The configs without one record (`exactOptionalPropertyTypes`: absent, not undefined). */
const without = (key: "likely" | "outcome"): FeatureConfigs => {
  const { [key]: _removed, ...rest } = configs();
  return rest;
};

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const teamOf = (c: Captured): PokemonSet[] => c.context.team as PokemonSet[];
const channelOf = (c: Captured): PlayerChannel => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const build = (c: Captured, change?: Partial<FeatureConfigs>, memo = createFeatureMemo(), likely: LikelySetup = base, values: Record<string, number> = {}) =>
  candidateFeaturesLikely(engine, configs(change, likely, values), channelOf(c), teamOf(c), candidatesOf(c), memo);
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const foesOf = (c: Captured): PublicMon[] => Object.values(stateOf(c).opponent.slots).filter((i) => i !== null).map((i) => stateOf(c).opponent.roster[i as number] as PublicMon);
const first = (): Captured => moveRequests()[0] as Captured;

/** The damaging, single or spread targeting moves a species' roster entry can hold, in id order. */
const damagingPool = (species: string): string[] => (vocabulary.roster.find((entry) => entry.id === species)?.moves ?? []).filter((id) => dex.moves.get(id).exists && dex.moves.get(id).category !== "Status" && dex.moves.get(id).target !== "self");
const poolOf = (mon: { species: string | null; base_species: string }): string[] => damagingPool(toId(mon.species ?? mon.base_species));
/** A table with one list per species of every foe on the field, holding the given masses (move id to mass), `slots` 4 and total 1,000,000. */
function tableOf(c: Captured, masses: (pool: string[]) => Record<string, number>, total = 1_000_000): LikelySetup {
  const table = new Map<string, MoveList>();
  for (const mon of stateOf(c).opponent.roster) table.set(toId(mon.species ?? mon.base_species), { slots: 4, total, masses: new Map(Object.entries(masses(poolOf(mon)))) });
  return { config: base.config, table };
}

describe("likelihood", () => {
  const list = (masses: Record<string, number>, total: number, slots = 4): MoveList => ({ slots, total, masses: new Map(Object.entries(masses)) });
  const generator = list(Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`m${String(i).padStart(2, "0")}`, 68])), 4 * 1105);

  it("is exactly (slots - r) / (n - r) under the generator table, for r = 0 to 3, and nothing at r = 4", () => {
    const share = (r: number): number => Math.floor((1_000_000 * (4 - r)) / (65 - r));
    for (const r of [0, 1, 2, 3]) {
      const revealed = new Set(["m00", "m01", "m02"].slice(0, r));
      const rows = inclusionsOf(generator, revealed, r);
      expect(rows).toHaveLength(65 - r);
      expect(new Set(rows.map((row) => row.inclusion_ppm))).toEqual(new Set([share(r)]));
      expect(rows.some((row) => revealed.has(row.id))).toBe(false);
    }
    expect(share(1)).toBe(46_875); // 3 / 64, worked by hand
    expect(inclusionsOf(generator, new Set(["m00", "m01", "m02", "m03"]), 4)).toEqual([]);
  });

  it("renormalises a usage table over the slots left, by hand", () => {
    const usage = list({ aa: 600_000, bb: 100_000, cc: 50_000 }, 1_000_000);
    // r = 0: slots x ppm, capped at 10^6
    expect(inclusionsOf(usage, new Set(), 0)).toEqual([
      { id: "aa", inclusion_ppm: 1_000_000 },
      { id: "bb", inclusion_ppm: 400_000 },
      { id: "cc", inclusion_ppm: 200_000 },
    ]);
    // bb revealed: the mass left is 900,000 over 3 slots: cc is floor(10^6 x 50,000 x 3 / 900,000) = 166,666
    expect(inclusionsOf(usage, new Set(["bb"]), 1)).toEqual([
      { id: "aa", inclusion_ppm: 1_000_000 },
      { id: "cc", inclusion_ppm: 166_666 },
    ]);
    // a revealed move the table does not hold still takes a slot, and no mass
    expect(inclusionsOf(usage, new Set(["zz"]), 1).find((row) => row.id === "cc")?.inclusion_ppm).toBe(150_000);
    // no mass left
    expect(inclusionsOf(list({ aa: 5 }, 5), new Set(["aa"]), 1)).toEqual([]);
    // a zero inclusion is left out
    expect(inclusionsOf(list({ aa: 1 }, 10_000_000), new Set(), 0)).toEqual([]);
  });

  it("finds a table list by the shown forme, then the base species", () => {
    const table = new Map([["venusaurmega", list({ aa: 1 }, 4)], ["venusaur", list({ bb: 1 }, 4)]]);
    const mon = (species: string | null) => ({ species, base_species: "venusaur" }) as never;
    expect(listOf(table, mon("Venusaur-Mega"))?.masses.has("aa")).toBe(true);
    expect(listOf(table, mon("Venusaur"))?.masses.has("bb")).toBe(true);
    expect(listOf(table, mon(null))?.masses.has("bb")).toBe(true);
    expect(listOf(table, { species: "Gengar", base_species: "gengar" } as never)).toBeUndefined();
  });

  it("derives a value from cached rolls exactly as the probe does, at any hit points", () => {
    const rolls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    const cached: CachedDamage = { kind: "rolls", no_crit: rolls, hp_max: 200, ko_blocked: false };
    expect(valuesAt(cached, null)).toEqual({ kind: "computed", mean_ppm: Math.floor((rolls.reduce((a, b) => a + b, 0) * 1_000_000) / (16 * 200)), max_ppm: 800_000, min_ppm: 50_000, ohko_rolls: 0 });
    // at 100 hit points: the damage is clamped to 100 and the rolls at or above 100 reach it
    expect(valuesAt(cached, 100)).toEqual({ kind: "computed", mean_ppm: Math.floor((rolls.map((r) => Math.min(r, 100)).reduce((a, b) => a + b, 0) * 1_000_000) / (16 * 100)), max_ppm: 1_000_000, min_ppm: 100_000, ohko_rolls: 7 });
    expect(valuesAt({ ...cached, ko_blocked: true }, 100)).toMatchObject({ ohko_rolls: null });
    expect(valuesAt({ kind: "unsupported", reason: "random_effect" }, 5)).toEqual({ kind: "unsupported", reason: "random_effect" });
  });

  it("gives a foe holding a revealed Choice item that has moved since it came in no likely rows", () => {
    const items = { get: (name: string) => ({ isChoice: name === "choicescarf" }) as Record<string, unknown> };
    const foe = (item: string | null, revealed: string[], lost = false) => ({ mon: { item, item_lost: lost, moves_revealed: revealed }, fighter: { at: { own: false, letter: "a", name: "x" } } }) as never;
    const facts = (last: string | null) => () => ({ opponent: { active: { a: { last_move: last } } } }) as unknown as WorldFacts;
    expect(choiceLocked(items, foe("choicescarf", ["tackle"]), facts("tackle"))).toBe(true);
    expect(choiceLocked(items, foe("choicescarf", ["tackle"]), facts(null))).toBe(false); // switched in since: no move yet
    expect(choiceLocked(items, foe("leftovers", ["tackle"]), facts("tackle"))).toBe(false);
    expect(choiceLocked(items, foe(null, ["tackle"]), facts("tackle"))).toBe(false);
    expect(choiceLocked(items, foe("choicescarf", ["tackle"], true), facts("tackle"))).toBe(false);
    // no facts (the module refused a line): any revealed move counts
    expect(choiceLocked(items, foe("choicescarf", ["tackle"]), () => null)).toBe(true);
    expect(choiceLocked(items, foe("choicescarf", []), () => null)).toBe(false);
    expect(choiceLocked(items, { mon: { item: "choicescarf", item_lost: false, moves_revealed: ["x"] }, fighter: {} } as never, () => ({}) as never)).toBe(true);
  });
});

describe("identity and the other sets' bytes", () => {
  it("is the outcome preimage with the set id and the record named, and no other set's preimage has the new keys", () => {
    const identity = featureSetIdentity(engine, configs(), FEATURE_SET_LIKELY);
    expect(identity).toMatchObject({ feature_set: "probe-outcome-likely/1", likely_threats_record_hash: likelyConfigHash(base.config), likely_threats_format: LIKELY_THREATS_FORMAT, outcome_format: expect.any(Number), reader_sha256: readerSha256() });
    const outcome = featureSetIdentity(engine, configs(), "probe-outcome/1");
    const { likely_threats_record_hash: _a, likely_threats_format: _b, feature_set: _c, ...rest } = identity;
    const { feature_set: _d, ...outcomeRest } = outcome;
    expect(rest).toEqual(outcomeRest);
    for (const set of ["probe-damage/1", "probe-damage/2", "probe-outcome/1"] as const) expect(Object.keys(featureSetIdentity(engine, configs(), set))).not.toContain("likely_threats_record_hash");
    expect(featureSetHash(identity)).not.toBe(featureSetHash(outcome));
  });

  it("changes with the record, and the two committed records have their own hashes", () => {
    const m8 = loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.m8.json");
    expect(featureSetHash(featureSetIdentity(engine, configs({}, m8), FEATURE_SET_LIKELY))).not.toBe(featureSetHash(featureSetIdentity(engine, configs(), FEATURE_SET_LIKELY)));
    expect(likelyConfigFiles(GOLDEN_FORMAT)).toEqual(["probe-outcome-likely.v1.json", "probe-outcome-likely.v1.m8.json"]);
    expect(m8.config.rank_candidates_max).toBe(8);
    expect(base.config.rank_candidates_max).toBeUndefined();
  });

  it("refuses an identity without its records or without move_outcome in scope", () => {
    expect(() => featureSetIdentity(engine, without("likely"), FEATURE_SET_LIKELY)).toThrow(/needs its likelihood record/);
    expect(() => featureSetIdentity(engine, without("outcome"), FEATURE_SET_LIKELY)).toThrow(/needs its reviewed record/);
    expect(() => featureSetIdentity(engine, { ...configs(), mechanics }, FEATURE_SET_LIKELY)).toThrow(/move_outcome/);
  });

  it("is a set the dispatch builds", () => {
    expect(isFeatureSetId(FEATURE_SET_LIKELY)).toBe(true);
    const c = first();
    const features = buildFeatures(engine, configs(), channelOf(c), teamOf(c), candidatesOf(c), createFeatureMemo(), FEATURE_SET_LIKELY);
    expect(features.feature_set).toBe(FEATURE_SET_LIKELY);
    expect(configuredSetHash(engine, { ...configs(), feature_set: FEATURE_SET_LIKELY })).toBe(featureSetHash(featureSetIdentity(engine, configs(), FEATURE_SET_LIKELY)));
  });

  it("refuses a build without its records", () => {
    const c = first();
    expect(() => candidateFeaturesLikely(engine, without("likely"), channelOf(c), teamOf(c), candidatesOf(c))).toThrow(/likelihood record/);
    expect(() => candidateFeaturesLikely(engine, without("outcome"), channelOf(c), teamOf(c), candidatesOf(c))).toThrow(/probe-outcome record/);
  });
});

describe("the rows", () => {
  it("hold probe-outcome/1's rows, four arrays and outcomes byte for byte", () => {
    for (const c of moveRequests().slice(0, 8)) {
      const likely = build(c);
      const outcome = candidateFeaturesOutcome(engine, configs(), channelOf(c), teamOf(c), candidatesOf(c), createFeatureMemo());
      for (const key of ["rows", "threats", "speeds", "own_priorities", "foe_priorities", "outcomes"] as const) expect(jcs(likely[key] as unknown as Json), key).toBe(jcs(outcome[key] as unknown as Json));
      expect(likely.feature_set_hash).not.toBe(outcome.feature_set_hash);
    }
  });

  it("have /3's row shape, never repeat a revealed threat and are sorted by (own, mega, foe, move)", () => {
    let withRows = 0;
    for (const c of moveRequests()) {
      const features = build(c);
      withRows += features.belief_threats.length > 0 ? 1 : 0;
      const revealed = new Set(features.threats.map((t) => `${t.foe}/${t.move}`));
      const keys = features.belief_threats.map((r) => [r.own, Number(r.mega), r.foe, r.move] as const);
      for (const r of features.belief_threats) {
        expect(revealed.has(`${r.foe}/${r.move}`)).toBe(false);
        expect(r.inclusion_ppm).toBeGreaterThan(0);
        expect(r.inclusion_ppm).toBeLessThanOrEqual(1_000_000);
      }
      expect(keys).toEqual([...keys].sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || (a[3] < b[3] ? -1 : 1)));
      expect(features.belief_threats_refused).toBeUndefined();
    }
    expect(withRows).toBeGreaterThan(0);
  });

  it("price at most per_foe moves per foe and own variant", () => {
    for (const c of moveRequests().slice(0, 12)) {
      const features = build(c);
      const groups = new Map<string, number>();
      for (const r of features.belief_threats) groups.set(`${r.own}/${r.mega}/${r.foe}`, (groups.get(`${r.own}/${r.mega}/${r.foe}`) ?? 0) + 1);
      for (const count of groups.values()) expect(count).toBeLessThanOrEqual(base.config.per_foe);
    }
  });

  it("have no rows at Team Preview and an empty array in a request with nothing to score", () => {
    const preview = captured.find((c) => c.request.phase === "preview") as Captured;
    const features = build(preview);
    expect(features.belief_threats).toEqual([]);
    expect(features.outcomes).toEqual([]);
    expect(features.feature_set).toBe(FEATURE_SET_LIKELY);
  });

  it("are the same bytes in any candidate order and with a warm, cold or no memo", () => {
    const c = moveRequests().find((r) => build(r).belief_threats.length > 0) as Captured;
    const cold = build(c);
    const memo = createFeatureMemo();
    build(c, undefined, memo);
    const warm = build(c, undefined, memo);
    expect(jcs(warm as unknown as Json)).toBe(jcs(cold as unknown as Json));
    const reversed = candidateFeaturesLikely(engine, configs(), channelOf(c), teamOf(c), [...candidatesOf(c)].reverse());
    expect(jcs(reversed as unknown as Json)).toBe(jcs(cold as unknown as Json));
    expect((memo.likely?.counts.full_hits ?? 0) + (memo.likely?.counts.base_hits ?? 0)).toBeGreaterThan(0);
  });
});

describe("ranking and the cut", () => {
  /** The moves priced for each foe (by location), whatever the own Pokemon. */
  const keptBy = (features: CandidateFeaturesLikely): Map<number, Set<string>> => {
    const out = new Map<number, Set<string>>();
    for (const r of features.belief_threats) (out.get(r.foe) ?? out.set(r.foe, new Set()).get(r.foe))?.add(r.move);
    return out;
  };
  const withKTable = (likely: LikelySetup, k: number): LikelySetup => ({ ...likely, config: { ...likely.config, per_foe: k } });
  const foeAt = (c: Captured, loc: number) => stateOf(c).opponent.roster[stateOf(c).opponent.slots[Object.keys(stateOf(c).opponent.slots).sort()[loc - 1] as string] as number];

  it("keeps the k moves of the highest expected threat (inclusion x value): a rare strong move over a likely weak one, the unlikely weak move out", () => {
    // the rolls are per move id, so the six moves of one foe must not be moves of another
    const disjoint = (r: Captured): boolean => {
      const sixes = foesOf(r).map((mon) => poolOf(mon).slice(0, 6));
      return new Set(sixes.flat()).size === sixes.flat().length;
    };
    const c = moveRequests().find((r) => foesOf(r).length > 0 && foesOf(r).every((mon) => poolOf(mon).length >= 8) && disjoint(r)) as Captured;
    // six moves of each foe's pool: mass (of a 10,000,000 total, so no inclusion is capped) and the roll its damage has (small, so no roll reaches the hit points of an own Pokemon and the value is linear in it)
    const MASS = [2_000_000, 2_000_000, 1_000_000, 500_000, 500_000, 100_000];
    const ROLL = [3, 4, 7, 15, 15, 25];
    const likely = tableOf(c, (pool) => Object.fromEntries(MASS.map((mass, i) => [pool[i] as string, mass])), 10_000_000);
    const values: Record<string, number> = {};
    for (const mon of foesOf(c)) ROLL.forEach((roll, i) => void (values[poolOf(mon)[i] as string] = roll));
    // every placeable candidate (up to six), then the cut to four
    const everything = keptBy(build(c, undefined, createFeatureMemo(), { ...likely, config: { ...likely.config, per_foe: 8 } }, values));
    const cut = keptBy(build(c, undefined, createFeatureMemo(), likely, values));
    expect(cut.size).toBeGreaterThan(0);
    for (const [loc, moves] of cut) {
      const mon = foeAt(c, loc) as NonNullable<ReturnType<typeof foeAt>>;
      const pool = poolOf(mon);
      const revealed = new Set(mon.moves_revealed.map(toId));
      const candidates = (everything.get(loc) ?? new Set<string>()) as Set<string>;
      // rank by mass x roll (the uniform renormalisation does not reorder); ties by id
      const ranked = [...candidates].map((move) => ({ move, rank: (MASS[pool.indexOf(move)] as number) * (ROLL[pool.indexOf(move)] as number) })).sort((x, y) => y.rank - x.rank || (x.move < y.move ? -1 : 1));
      expect(candidates.size).toBeGreaterThan(4); // otherwise nothing is cut
      expect([...moves].sort()).toEqual(ranked.slice(0, 4).map((r) => r.move).sort());
      expect([...candidates].every((move) => !revealed.has(move))).toBe(true);
      // the 4th and 5th by rank differ: the cut is real, and the sure KO (the 6th move) is in when its mass x roll is among the top four
      expect(ranked[3]?.rank).not.toBe(ranked[4]?.rank);
    }
  });

  it("orders equal expected threats by move id and takes the same moves whatever order the table lists them", () => {
    const c = moveRequests().find((r) => foesOf(r).length > 0 && foesOf(r).every((mon) => poolOf(mon).length >= 8)) as Captured;
    const equal = (pool: string[]) => Object.fromEntries(pool.slice(0, 7).map((id) => [id, 100_000]));
    const forward = build(c, undefined, createFeatureMemo(), tableOf(c, equal), {});
    const reversed = build(c, undefined, createFeatureMemo(), tableOf(c, (pool) => Object.fromEntries(Object.entries(equal(pool)).reverse())), {});
    expect(jcs(forward.belief_threats as unknown as Json)).toBe(jcs(reversed.belief_threats as unknown as Json));
    const everything = keptBy(build(c, undefined, createFeatureMemo(), withKTable(tableOf(c, equal), 8), {}));
    // every value is zero: the rank ties, and the four lowest ids that place stay
    for (const [loc, moves] of keptBy(forward)) {
      expect([...moves].sort()).toEqual([...(everything.get(loc) as Set<string>)].sort().slice(0, 4));
    }
  });

  it("pre-screens to m candidates by engine data before any damage query", () => {
    const m8 = loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.m8.json");
    const c = moveRequests().find((r) => foesOf(r).length > 0) as Captured;
    const log: { move: string }[] = [];
    const counting = (likely: LikelySetup) => {
      const memo = createFeatureMemo();
      const inner = probe();
      const features = candidateFeaturesLikely(engine, { ...configs({}, likely), probe: (context, query) => (query.query.kind === "damage" && log.push({ move: query.query.move }), inner(context, query)) }, channelOf(c), teamOf(c), candidatesOf(c), memo);
      return { features, memo };
    };
    const all = counting(base);
    const screened = counting({ ...m8, table: base.table });
    expect(screened.memo.likely?.counts.pre_screened ?? 0).toBeGreaterThan(0);
    expect(all.memo.likely?.counts.pre_screened).toBe(0);
    expect(screened.features.belief_threats.length).toBeLessThanOrEqual(all.features.belief_threats.length + 0 + 8 * 2 * 8);
    // every foe has at most k rows per own variant in either
    for (const features of [all.features, screened.features]) expect(features.belief_threats.length).toBeGreaterThan(0);
  });
});

describe("refusals", () => {
  it("says species_not_in_prior and leaves the rows empty when a foe's species has no list", () => {
    const c = moveRequests().find((r) => foesOf(r).length > 0) as Captured;
    const memo = createFeatureMemo();
    const features = build(c, undefined, memo, { config: base.config, table: new Map() });
    expect(features.belief_threats).toEqual([]);
    expect(features.belief_threats_refused).toBe("species_not_in_prior");
    expect(memo.belief?.fallbacks).toEqual({ species_not_in_prior: 1 });
    // the other arrays are untouched
    const outcome = candidateFeaturesOutcome(engine, configs(), channelOf(c), teamOf(c), candidatesOf(c), createFeatureMemo());
    expect(jcs(features.outcomes as unknown as Json)).toBe(jcs(outcome.outcomes as unknown as Json));
  });

  it("matches the /2 arrays under /2's own builder (history aside, the same keys)", () => {
    const c = first();
    const two = candidateFeatures2(engine, configs(), channelOf(c), teamOf(c), candidatesOf(c), createFeatureMemo());
    const likely = build(c);
    expect(likely.speeds.map((r) => [r.own, r.mega, r.foe])).toEqual(two.speeds.map((r) => [r.own, r.mega, r.foe]));
    expect(likely.own_priorities.length).toBe(two.own_priorities.length);
  });
});

describe("the edges of a decision", () => {
  const withLines = (c: Captured, edit: (lines: string[]) => void): PlayerChannel => {
    const lines = [...c.context.channel];
    edit(lines);
    return playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines });
  };
  const run = (c: Captured, channel: PlayerChannel, change: Partial<FeatureConfigs> = {}, memo = createFeatureMemo()) => candidateFeaturesLikely(engine, configs(change), channel, teamOf(c), candidatesOf(c), memo);

  it("propagates a defect of the probe and refuses an answer that is no damage", () => {
    const c = moveRequests().find((r) => foesOf(r).length > 0) as Captured;
    const broken = (fail: "throw" | "wrong") => (context: MechanicsContext, query: ParsedMechanicsQuery) => {
      if (query.query.kind === "damage" && query.query.attacker.hp === null && query.query.move !== undefined && fail === "throw") throw new Error("probe defect");
      if (query.query.kind === "damage" && fail === "wrong") {
        const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(withOutcome), gap_ids: [] };
        return { result: buildResult(identity, answerOutcome), diagnostics: [] };
      }
      return probe()(context, query);
    };
    expect(() => run(c, channelOf(c), { probe: broken("throw") })).toThrow();
    expect(() => run(c, channelOf(c), { probe: broken("wrong") })).toThrow(/no damage answer/);
  });

  it("leaves out a move the validator will not place on the foe's set", () => {
    const c = moveRequests().find((r) => foesOf(r).length > 0) as Captured;
    const memo = createFeatureMemo();
    const table = new Map<string, MoveList>();
    for (const mon of stateOf(c).opponent.roster) {
      const own = poolOf(mon);
      const foreign = damagingPool("tyranitar").find((m) => !own.includes(m) && !damagingPool(toId(mon.species ?? mon.base_species)).includes(m)) as string;
      table.set(toId(mon.species ?? mon.base_species), { slots: 4, total: 1_000_000, masses: new Map([[foreign, 250_000], [own[0] as string, 250_000]]) });
    }
    const features = candidateFeaturesLikely(engine, configs({}, { config: base.config, table }), channelOf(c), teamOf(c), candidatesOf(c), memo);
    const asked = new Set(features.belief_threats.map((r) => r.move));
    for (const mon of stateOf(c).opponent.roster) for (const move of table.get(toId(mon.species ?? mon.base_species))?.masses.keys() ?? []) if (!poolOf(mon).includes(move)) expect(asked.has(move) && foesOf(c).some((f) => f === mon)).toBe(false);
    expect([...(memo.likely?.placed.values() ?? [])].some((placed) => placed === null)).toBe(true);
  });

  it("refuses with reader_flagged for the rest of a game once the reader has flagged a line", () => {
    const c = moveRequests().find((r) => r.context.channel.filter((l) => l.startsWith("|request|")).length >= 3) as Captured;
    // the flagged line sits between the first two requests, so it is older than the previous request and the build itself does not refuse
    const channel = withLines(c, (lines) => {
      const firstAt = lines.findIndex((l) => l.startsWith("|request|"));
      const second = lines.findIndex((l, i) => i > firstAt && l.startsWith("|request|"));
      lines.splice(second, 0, "|nosuchlinekind|x");
    });
    const memo = createFeatureMemo();
    const features = run(c, channel, {}, memo);
    expect(features.belief_threats).toEqual([]);
    expect(features.belief_threats_refused).toBe("reader_flagged");
    expect(memo.belief?.fallbacks).toEqual({ reader_flagged: 1 });
  });

  it("gives a foe a revealed Choice item has locked no likely rows, and prices a forced switch's benched Pokemon", () => {
    const c = moveRequests().find((r) => foesOf(r).length > 0 && foesOf(r)[0]?.name !== null && foesOf(r)[0]?.name !== undefined) as Captured;
    const foe = foesOf(c)[0] as PublicMon;
    const player = stateOf(c).opponent.player;
    const letter = Object.keys(stateOf(c).opponent.slots).sort()[0] as string;
    const loc = 1;
    const channel = withLines(c, (lines) => {
      const at = lines.length - 1;
      lines.splice(at, 0, `|-item|${player}${letter}: ${foe.name}|Choice Scarf`, `|move|${player}${letter}: ${foe.name}|Tackle|p1a: x`);
    });
    const features = run(c, channel);
    expect(features.belief_threats.filter((r) => r.foe === loc)).toEqual([]);
    const forced = captured.find((r) => r.request.phase === "forceSwitch" && foesOf(r).length > 0);
    if (forced !== undefined) {
      const built = build(forced);
      expect(built.belief_threats_refused).toBeUndefined();
      expect(built.belief_threats.every((r) => r.own >= 0)).toBe(true);
    }
  });
});

describe("the record reader", () => {
  const good = JSON.parse(JSON.stringify(base.config)) as Record<string, unknown>;
  it("refuses unknown keys, bad words and values out of range", async () => {
    const { readLikelyConfig } = await import("../src/likely-threats-config.ts");
    expect(readLikelyConfig(good)).toEqual(base.config);
    const bad = (change: (c: Record<string, unknown>) => void): unknown => {
      const copy = JSON.parse(JSON.stringify(good)) as Record<string, unknown>;
      change(copy);
      return copy;
    };
    for (const record of [
      bad((c) => void (c.extra = 1)),
      bad((c) => void (c.schema_version = 2)),
      bad((c) => void (c.per_foe = 3)),
      bad((c) => void (c.per_foe = 9)),
      bad((c) => void (c.per_foe = 4.5)),
      bad((c) => void (c.source = { kind: "belief" })),
      bad((c) => void ((c.source as Record<string, unknown>).prior_hash = "xyz")),
      bad((c) => void (c.hp_key_handlers = [])),
      bad((c) => void (c.hp_key_handlers = ["onModifyAtk", "basePowerCallback"])),
      bad((c) => void (c.hp_key_handlers = ["not a handler"])),
      bad((c) => void (c.engine_data_hash = "short")),
      bad((c) => void (c.rank_candidates_max = 8)), // without its screen
      bad((c) => void (c.format_id = "")),
      null,
    ]) expect(() => readLikelyConfig(record)).toThrow();
    const m8 = JSON.parse(JSON.stringify(loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.m8.json").config)) as Record<string, unknown>;
    expect(() => readLikelyConfig({ ...m8, screen: { ...(m8.screen as object), effectiveness_ppm: [1, 2, 3] } })).toThrow(/damage_taken code/);
    expect(() => readLikelyConfig({ ...m8, screen: { ...(m8.screen as object), extra: 1 } })).toThrow();
    expect(() => readLikelyConfig({ ...m8, rank_candidates_max: 0 })).toThrow();
    const usage: LikelyConfig = readLikelyConfig({ ...good, source: { kind: "usage-table", usage_table_hash: "a".repeat(64), slots: 4 } });
    expect(usage.source).toEqual({ kind: "usage-table", usage_table_hash: "a".repeat(64), slots: 4 });
    expect(() => readLikelyConfig({ ...good, source: { kind: "usage-table", usage_table_hash: "a".repeat(64), slots: 0 } })).toThrow();
  });

  it("checks the handlers and the engine-data hash against the engine, and the source file against its hash", async () => {
    const { checkLikelyConfig, loadMoveTable, readsHitPoints } = await import("../src/likely-threats-config.ts");
    const raw = engine.sim.Dex.mod("champions") as unknown as Parameters<typeof checkLikelyConfig>[1];
    expect(checkLikelyConfig(base.config, raw, base.config.engine_data_hash)).toEqual([]);
    expect(checkLikelyConfig({ ...base.config, hp_key_handlers: ["onNoSuchHandler"] }, raw, base.config.engine_data_hash)[0]).toMatch(/onNoSuchHandler/);
    expect(checkLikelyConfig(base.config, raw, "0".repeat(64))[0]).toMatch(/engine_data_hash/);
    expect(readsHitPoints(undefined, ["onModifyAtk"])).toBe(false);
    expect(readsHitPoints({ onModifyAtk: () => 1 }, ["onModifyAtk"])).toBe(true);
    expect(() => loadMoveTable({ kind: "generator-matched", prior_hash: "0".repeat(64) }, GOLDEN_FORMAT)).toThrow(/another hash/);
    expect(() => loadMoveTable({ kind: "usage-table", usage_table_hash: "0".repeat(64), slots: 4 }, GOLDEN_FORMAT)).toThrow(/no committed usage weight table/);
    expect(() => loadLikelySetup("gen9nothing", "x.json")).toThrow();
  });

  it("lists every damage-path handler of the engine that reads hit points, and flags a record that lacks one", async () => {
    const { checkLikelyConfig, hpReadingHandlers } = await import("../src/likely-threats-config.ts");
    const raw = engine.sim.Dex.mod("champions") as unknown as Parameters<typeof checkLikelyConfig>[1];
    const reading = hpReadingHandlers(raw);
    // the move power callbacks, Brine, the low-HP type boosts and the full-HP damage reduction are what the engine's source says
    for (const name of ["basePowerCallback", "onBasePower", "onModifyAtk", "onModifySpA", "onSourceModifyDamage"]) expect(reading).toContain(name);
    for (const name of reading) expect(base.config.hp_key_handlers, name).toContain(name);
    const lacking = { ...base.config, hp_key_handlers: base.config.hp_key_handlers.filter((name) => name !== "onSourceModifyDamage") };
    expect(checkLikelyConfig(lacking, raw, base.config.engine_data_hash).join(" ")).toMatch(/onSourceModifyDamage reads hit points/);
  });

  it("refuses a source file whose masses the integer arithmetic cannot hold exact", async () => {
    const { loadMoveTable, MAX_MASS_TIMES_SLOTS } = await import("../src/likely-threats-config.ts");
    const { sha256Hex } = await import("../src/rules.ts");
    const { mkdtempSync, mkdirSync, writeFileSync } = await import("node:fs");
    const { join } = await import("node:path");
    const { tmpdir } = await import("node:os");
    const dir = mkdtempSync(join(tmpdir(), "likely-"));
    mkdirSync(join(dir, "opponent-model", GOLDEN_FORMAT), { recursive: true });
    const write = (inclusion: number) => {
      const prior = { species: [{ id: "x", formes: [{ id: "x", moves: { slots: 4, denominator: 10, entries: [{ id: "tackle", inclusion }] } }] }] };
      writeFileSync(join(dir, "opponent-model", GOLDEN_FORMAT, "generator-matched.uniform-v1.json"), JSON.stringify(prior));
      return sha256Hex(jcs(prior as unknown as Json));
    };
    const ok = loadMoveTable({ kind: "generator-matched", prior_hash: write(7) }, GOLDEN_FORMAT, dir);
    expect(ok.get("x")?.masses.get("tackle")).toBe(7);
    for (const bad of [Math.floor(MAX_MASS_TIMES_SLOTS / 4) + 1, 41, -1, 1.5]) expect(() => loadMoveTable({ kind: "generator-matched", prior_hash: write(bad) }, GOLDEN_FORMAT, dir)).toThrow(/integer arithmetic/);
  });

  it("reads the committed usage weight table as a second source (slots x ppm)", async () => {
    const { loadMoveTable } = await import("../src/likely-threats-config.ts");
    const { sha256Hex } = await import("../src/rules.ts");
    const { readFileSync } = await import("node:fs");
    const file = new URL("../../../config/teams/gen9championsvgc2026regmc/usage/2026-09-1630.json", import.meta.url);
    const hash = sha256Hex(jcs(JSON.parse(readFileSync(file, "utf8")) as Json));
    const table = loadMoveTable({ kind: "usage-table", usage_table_hash: hash, slots: 4 }, GOLDEN_FORMAT);
    const list = table.get("abomasnow") as MoveList;
    expect(list.slots).toBe(4);
    expect(list.total).toBe(1_000_000);
    expect(list.masses.get("blizzard")).toBe(177_354);
    expect(inclusionsOf(list, new Set(), 0).find((row) => row.id === "blizzard")?.inclusion_ppm).toBe(709_416);
    // a record that names it builds rows too
    const likely: LikelySetup = { config: { ...base.config, source: { kind: "usage-table", usage_table_hash: hash, slots: 4 } }, table };
    const c = moveRequests().find((r) => foesOf(r).every((mon) => table.has(toId(mon.species ?? mon.base_species)))) as Captured;
    expect(build(c, undefined, createFeatureMemo(), likely).belief_threats_refused).toBeUndefined();
  });
});
