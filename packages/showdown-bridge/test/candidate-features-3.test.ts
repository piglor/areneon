import { beforeAll, describe, expect, it } from "vitest";
import { type BeliefMoveInclusion, type BeliefThreatsPort } from "../src/belief-features.ts";
import { type CandidateFeatures3, candidateFeatures3, FEATURE_SET_3 } from "../src/candidate-features-3.ts";
import { candidateFeatures2 } from "../src/candidate-features-2.ts";
import { BELIEF_THREATS_FORMAT, BELIEF_THREATS_PER_FOE, candidateFeatures, type FeatureConfigs, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import { buildFeatures, configuredSetHash, isFeatureSetId } from "../src/feature-build.ts";
import { featureCoverageProblem, featureRowsProblem } from "../src/feature-rows.ts";
import { DamageAwareError } from "../src/damage-aware-view.ts";
import { assumedWithMove, buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import { playerChannel } from "../src/player-channel.ts";
import { mechanicsConfigHash } from "../src/mechanics-config.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { buildResult } from "../src/mechanics-result.ts";
import { resolveFormat } from "../src/formats.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stateOf, stubProbe, vocabulary } from "./damage-aware-support.ts";

// ADR-022, `probe-damage/3`: `/2`'s rows and arrays byte for byte, plus `belief_threats`: the moves the belief says a foe may hold
// (damaging, reaching an opposing Pokemon, placeable on the foe's assumed set, never a revealed one), cut to `per_foe` AFTER the
// filter, each with its inclusion and the damage the probe gives on the assumed set plus that move. The opponent model is a stand-in
// port here (the real model is in packages/team-dataset/test).

const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const dex = engine.sim.Dex.mod("champions") as unknown as { moves: { get(id: string): { exists: boolean; id: string; name: string; category: string; target: string } } };

interface Spy extends BeliefThreatsPort {
  calls: { opponent: unknown; flags: number }[];
}
/** A port whose inclusions come from `by(roster position, species id)`; `null` is a refusal. */
function portOf(by: (position: number, species: string) => BeliefMoveInclusion[] | null, change: Partial<BeliefThreatsPort> = {}): Spy {
  const calls: Spy["calls"] = [];
  return {
    identity_hash: "c".repeat(64),
    calls,
    moveInclusions: ({ opponent, flags }) => {
      calls.push({ opponent, flags: flags.length });
      const rows = opponent.roster.map((mon, position) => by(position, mon.base_species));
      return rows.some((r) => r === null) ? null : (rows as BeliefMoveInclusion[][]);
    },
    ...change,
  };
}
/** The damage stand-in of the support module, with speed orders that put every action in a group of its own (own side first). */
function orderProbe(damage: Parameters<typeof stubProbe>[0] = {}): NonNullable<FeatureConfigs["probe"]> {
  const base = stubProbe(damage);
  return (context, query) => {
    if (query.query.kind !== "speed_order") return base(context, query);
    const identity = { data_class: query.data_class, query_hash: mechanicsQueryHash(query), manifest, mechanics_config_hash: mechanicsConfigHash(mechanics), gap_ids: [] };
    const actions = query.query.actions.map((a, i) => ({ ...a, order: a.move === null ? 103 : 200, priority: 0, speed: 1000 - i }));
    const groups = actions.map((_, i) => [i]);
    return { result: buildResult(identity, { kind: "computed", answer: { kind: "speed_order", order_basis: "turn_start", actions, groups } as never }), diagnostics: [] };
  };
}
const configs = (port: BeliefThreatsPort | null = portOf(() => []), change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({
  vocabulary,
  config,
  mechanics,
  snapshot,
  data_class: "synthetic",
  probe: orderProbe(),
  ...(port === null ? {} : { belief_threats: { port, per_foe: BELIEF_THREATS_PER_FOE } }),
  ...change,
});

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const channelOf = (c: Captured) => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const build = (c: Captured, cfg: FeatureConfigs, memo?: Parameters<typeof candidateFeatures3>[5]): CandidateFeatures3 => candidateFeatures3(engine, cfg, channelOf(c), c.context.team as never, candidatesOf(c), memo);
const bytes = (value: unknown): string => JSON.stringify(value);

/** Damaging pool moves of a species (the roster entry's), in the pool's order, none of `not`. */
function damagingPool(species: string, not: string[] = []): string[] {
  const entry = vocabulary.roster.find((r) => r.id === species) ?? vocabulary.roster.find((r) => species.startsWith(r.id));
  return (entry?.moves ?? []).filter((id) => !not.includes(id) && dex.moves.get(id).category !== "Status" && dex.moves.get(id).target !== "self");
}
const statusPool = (species: string): string[] => (vocabulary.roster.find((r) => r.id === species)?.moves ?? []).filter((id) => dex.moves.get(id).category === "Status");
const foeCount = (c: Captured): number => Object.values(stateOf(c).opponent.slots).filter((i) => i !== null && i !== undefined).length;

describe("probe-damage/3 holds probe-damage/2 byte for byte", () => {
  it("gives /2's rows and four arrays unchanged, whatever the belief says", () => {
    let compared = 0;
    for (const c of moveRequests().slice(0, 12)) {
      const port = portOf((_, species) => damagingPool(species).slice(0, 3).map((id, i) => ({ id, inclusion_ppm: 500000 - 100000 * i })));
      const three = build(c, configs(port));
      const two = candidateFeatures2(engine, configs(null), channelOf(c), c.context.team as never, candidatesOf(c));
      const { belief_threats: _belief, feature_set: _set, feature_set_hash: _hash, ...rest } = three;
      const { feature_set: _s2, feature_set_hash: _h2, ...base } = two;
      expect(bytes(rest)).toBe(bytes(base));
      compared += 1;
    }
    expect(compared).toBeGreaterThan(5);
  });

  it("has its own id and a hash that names the belief, the cut and the builder format", () => {
    const c = moveRequests()[0] as Captured;
    const cfg = configs();
    const features = build(c, cfg);
    expect(features.feature_set).toBe(FEATURE_SET_3);
    const identity = featureSetIdentity(engine, cfg, FEATURE_SET_3);
    expect(identity).toMatchObject({ belief_threats_identity_hash: "c".repeat(64), belief_threats_per_foe: BELIEF_THREATS_PER_FOE, belief_threats_format: BELIEF_THREATS_FORMAT });
    expect(features.feature_set_hash).toBe(featureSetHash(identity));
    expect(configuredSetHash(engine, { ...cfg, feature_set: FEATURE_SET_3 })).toBe(features.feature_set_hash);
    // every part of the identity moves the hash
    const other = (change: Partial<FeatureConfigs>) => featureSetHash(featureSetIdentity(engine, { ...cfg, ...change }, FEATURE_SET_3));
    expect(other({ belief_threats: { port: portOf(() => [], { identity_hash: "d".repeat(64) }), per_foe: BELIEF_THREATS_PER_FOE } })).not.toBe(features.feature_set_hash);
    expect(other({ belief_threats: { port: (cfg.belief_threats as NonNullable<FeatureConfigs["belief_threats"]>).port, per_foe: 3 } })).not.toBe(features.feature_set_hash);
    // /1 and /2 preimages and hashes keep their bytes: no belief key
    for (const set of ["probe-damage/1", "probe-damage/2"] as const) expect(Object.keys(featureSetIdentity(engine, cfg, set)).some((k) => k.startsWith("belief"))).toBe(false);
    expect(featureSetHash(featureSetIdentity(engine, configs(null), "probe-damage/2"))).toBe(featureSetHash(featureSetIdentity(engine, cfg, "probe-damage/2")));
    expect(features.feature_set_hash).not.toBe(featureSetHash(featureSetIdentity(engine, cfg, "probe-damage/2")));
  });

  it("is refused without the belief port, and dispatched by feature-build", () => {
    const c = moveRequests()[0] as Captured;
    expect(() => featureSetIdentity(engine, configs(null), FEATURE_SET_3)).toThrow(DamageAwareError);
    expect(() => candidateFeatures3(engine, configs(null), channelOf(c), c.context.team as never, candidatesOf(c))).toThrow(/belief threats port/);
    expect(isFeatureSetId(FEATURE_SET_3)).toBe(true);
    const cfg = configs();
    expect(bytes(buildFeatures(engine, cfg, channelOf(c), c.context.team as never, candidatesOf(c), { results: new Map(), counts: { queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 } }, FEATURE_SET_3))).toBe(bytes(build(c, cfg)));
  });
});

describe("belief_threats", () => {
  const withFoes = (): Captured[] => moveRequests().filter((c) => foeCount(c) >= 1);

  it("lists, per own Pokemon and foe, the first per_foe damaging moves of the belief, with their inclusion, sorted", () => {
    let rows = 0;
    for (const c of withFoes().slice(0, 10)) {
      // A status move first, then six damaging moves with falling inclusion: the cut is four DAMAGING moves, not four moves.
      const port = portOf((_, species) => [...statusPool(species).slice(0, 1), ...damagingPool(species).slice(0, 6)].map((id, i) => ({ id, inclusion_ppm: 900000 - 100000 * i })));
      const features = build(c, configs(port));
      const perFoe = new Map<number, Set<string>>();
      for (const r of features.belief_threats) {
        expect(dex.moves.get(r.move).category).not.toBe("Status");
        perFoe.set(r.foe, (perFoe.get(r.foe) ?? new Set()).add(r.move));
        expect(r.inclusion_ppm).toBeGreaterThan(0);
        rows += 1;
      }
      for (const moves of perFoe.values()) expect(moves.size).toBeLessThanOrEqual(BELIEF_THREATS_PER_FOE);
      expect(featureRowsProblem(features as never)).toBeNull();
      expect(featureCoverageProblem({ observation: { log: c.context.channel }, candidates: candidatesOf(c) }, features as never, (id) => dex.moves.get(id).category !== "Status")).toBeNull();
    }
    expect(rows).toBeGreaterThan(10);
  });

  it("keeps the belief's own order within a foe (inclusion, then id) and the first four damaging moves", () => {
    const c = withFoes()[0] as Captured;
    const state = stateOf(c);
    const species = state.opponent.roster[Object.values(state.opponent.slots).find((i) => i !== null && i !== undefined) as number]?.base_species as string;
    const pool = damagingPool(species);
    const port = portOf((position) => (state.opponent.roster[position]?.base_species === species ? pool.slice(0, 6).map((id, i) => ({ id, inclusion_ppm: 800000 - 10000 * i })) : []));
    const features = build(c, configs(port));
    const ids = new Set(features.belief_threats.map((r) => r.move));
    // moves the assumed set can hold, in order: a revealed one is skipped (it is a /2 threat), the rest are cut at four
    const first = state.opponent.roster.filter((m) => m.base_species === species).flatMap((m) => m.moves_revealed);
    const expected = pool.slice(0, 6).filter((id) => !first.includes(id) && ids.has(id));
    expect(expected.length).toBeLessThanOrEqual(BELIEF_THREATS_PER_FOE);
    expect(ids.size).toBeLessThanOrEqual(BELIEF_THREATS_PER_FOE * 2);
  });

  it("never lists a revealed move: it is a /2 threat already", () => {
    let checked = 0;
    for (const c of withFoes()) {
      const revealed = [...new Set(stateOf(c).opponent.roster.flatMap((m) => m.moves_revealed))];
      if (revealed.length === 0) continue;
      const port = portOf(() => revealed.map((id, i) => ({ id, inclusion_ppm: 900000 - i })));
      const features = build(c, configs(port));
      const threats = new Set(features.threats.map((r) => `${r.foe}/${r.move}`));
      for (const r of features.belief_threats) expect(threats.has(`${r.foe}/${r.move}`)).toBe(false);
      checked += 1;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("values a belief move on the assumed set plus that move, at the own Pokemon's exact HP", () => {
    const c = withFoes()[0] as Captured;
    const log: ParsedMechanicsQuery[] = [];
    const pool = (species: string) => damagingPool(species).slice(0, 1);
    const port = portOf((_, species) => pool(species).map((id) => ({ id, inclusion_ppm: 250000 })));
    const features = build(c, configs(port, { probe: orderProbe({ log, values: Object.fromEntries(vocabulary.roster.flatMap((r) => r.moves.map((m) => [m, 12]))) }) }));
    expect(features.belief_threats.length).toBeGreaterThan(0);
    const row = features.belief_threats[0] as CandidateFeatures3["belief_threats"][number];
    expect(row.inclusion_ppm).toBe(250000);
    const asked = log.filter((q) => q.query.kind === "damage" && q.query.move === row.move);
    expect(asked.length).toBeGreaterThan(0);
    const attacker = (asked[0]?.query as { attacker: { set: string } }).attacker.set;
    expect(attacker).toContain(`- ${dex.moves.get(row.move).name}`);
    expect(row.kind).toBe("computed");
  });

  it("is empty for a refused belief, and /2's arrays are still there", () => {
    const c = withFoes()[0] as Captured;
    const features = build(c, configs(portOf(() => null)));
    expect(features.belief_threats).toEqual([]);
    expect(features.speeds.length).toBeGreaterThan(0);
  });

  it("asks the belief for the reader's opponent half and every flag, once per request", () => {
    const c = withFoes()[0] as Captured;
    const port = portOf(() => []);
    build(c, configs(port));
    expect(port.calls).toHaveLength(1);
    expect(port.calls[0]?.flags).toBe(0);
  });

  it("has no rows for Team Preview, a wait or a request with nothing to score", () => {
    const others = captured.filter((c) => c.request.phase !== "move");
    expect(others.length).toBeGreaterThan(0);
    for (const c of others) {
      const features = build(c, configs(portOf(() => [{ id: "tackle", inclusion_ppm: 1 }])));
      expect(features.belief_threats).toEqual([]);
    }
  });

  it("is the same bytes with or without the memo, in any candidate order, and run twice", () => {
    for (const c of withFoes().slice(0, 6)) {
      const port = () => portOf((_, species) => damagingPool(species).slice(0, 5).map((id, i) => ({ id, inclusion_ppm: 700000 - 50000 * i })));
      const plain = bytes(build(c, configs(port())));
      expect(bytes(build(c, configs(port())))).toBe(plain);
      expect(bytes(build(c, configs(port()), { results: new Map(), counts: { queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 } }))).toBe(plain);
      const shuffled = [...candidatesOf(c)].reverse();
      expect(bytes(candidateFeatures3(engine, configs(port()), channelOf(c), c.context.team as never, shuffled))).toBe(plain);
    }
  });

  it("a /1 or /2 builder never calls the belief port", () => {
    const c = withFoes()[0] as Captured;
    const port = portOf(() => []);
    candidateFeatures(engine, configs(port), channelOf(c), c.context.team as never, candidatesOf(c));
    candidateFeatures2(engine, configs(port), channelOf(c), c.context.team as never, candidatesOf(c));
    expect(port.calls).toHaveLength(0);
  });
});

describe("assumedWithMove", () => {
  const c = (): Captured => moveRequests().find((x) => foeCount(x) >= 1) as Captured;
  const assumed = () => {
    const state = stateOf(c());
    const mon = state.opponent.roster[Object.values(state.opponent.slots).find((i) => i !== null && i !== undefined) as number] as never;
    return buildAssumedSet({ engine, vocabulary, config }, mon);
  };
  const context = { engine, format_id: config.format_id };

  it("adds the move after the revealed ones (a filler move is dropped) and gives the engine's canonical paste", () => {
    const base = assumed();
    const move = damagingPool(base.set.species.toLowerCase().replace(/[^a-z0-9]/g, ""), base.revealed_moves)[0];
    if (move === undefined) throw new Error("no pool move");
    const placed = assumedWithMove(context, base, move);
    expect(placed).not.toBeNull();
    expect((placed?.set.moves ?? []).map((m) => m.toLowerCase().replace(/[^a-z0-9]/g, ""))).toEqual([...base.revealed_moves, move]);
    expect(placed?.paste).toContain(`- ${dex.moves.get(move).name}`);
  });

  it("is null for a revealed move, a fifth move, an unknown move and a move outside the species' pool", () => {
    const base = assumed();
    expect(assumedWithMove(context, base, base.revealed_moves[0] ?? "nothing")).toBeNull();
    expect(assumedWithMove(context, base, "notamove")).toBeNull();
    const full = { ...base, revealed_moves: ["a", "b", "c", "d"] };
    expect(assumedWithMove(context, full, "tackle")).toBeNull();
    const outside = Object.keys(Object.fromEntries(vocabulary.roster.flatMap((r) => r.moves.map((m) => [m, 1])))).find((m) => !(vocabulary.roster.find((r) => r.id === base.set.species.toLowerCase().replace(/[^a-z0-9]/g, ""))?.moves ?? ([] as string[])).includes(m) && dex.moves.get(m).category !== "Status");
    expect(assumedWithMove(context, base, outside as string)).toBeNull();
  });
});

