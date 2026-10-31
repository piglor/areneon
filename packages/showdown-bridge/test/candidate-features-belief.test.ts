import { beforeAll, describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { type BeliefFeatureSetup, candidateFeatures, createFeatureMemo, FEATURE_SET, FEATURE_SET_BELIEF, FeatureBuildError, type FeatureConfigs, type FeatureRow, featureSetHash, featureSetIdentity } from "../src/candidate-features.ts";
import { type CandidateFeaturesBelief, candidateFeaturesBelief, featureWorldsStream, meanRows } from "../src/candidate-features-belief.ts";
import { jcs } from "../src/canonical-json.ts";
import { buildFeatures, configuredSetHash, isFeatureSetId } from "../src/feature-build.ts";
import { DamageAwareError } from "../src/damage-aware-view.ts";
import { worldsStream } from "../src/damage-aware-worlds.ts";
import type { BattleRecord, HiddenVerdicts, OmniscientLog } from "../src/index.ts";
import type { ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { decodeRequest } from "../src/request.ts";
import { playerChannel } from "../src/player-channel.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, stubProbe, vocabulary } from "./damage-aware-support.ts";
import { configV2, fakePort, worldOf } from "./damage-aware-world-support.ts";

const refusal = (code: string): Error => Object.assign(new Error("refused"), { name: "BeliefError", code });
const setup = (port = fakePort(), worlds = 8): BeliefFeatureSetup => ({ port, worlds });
const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe: stubProbe(), belief: setup(), feature_set: FEATURE_SET_BELIEF, ...change });

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never))];
}, 120_000); // seeded real battles on the pinned engine: seconds on a loaded machine

const channelOf = (c: Captured) => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const moveRequests = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
const build = (c: Captured, change: Partial<FeatureConfigs> = {}, memo = createFeatureMemo()): CandidateFeaturesBelief => candidateFeaturesBelief(engine, configs(change), channelOf(c), c.context.team as never, candidatesOf(c), memo);
const keyOf = (row: FeatureRow): string => `${row.slot}/${row.move}/${row.target}/${row.mega}`;

describe("identity (ADR-022 erratum E-22.2)", () => {
  const base = () => featureSetIdentity(engine, configs(), FEATURE_SET);
  const belief = () => featureSetIdentity(engine, configs(), FEATURE_SET_BELIEF);

  it("leaves probe-damage/1 and /2 preimages and hashes as they were: no belief key in either", () => {
    expect(Object.keys(base())).toEqual(["feature_set", "mechanics_config_hash", "assumed_set_config_hash", "adapter_format", "reader_sha256", "version_manifest"]);
    expect(featureSetHash(base())).toBe(createHash("sha256").update(jcs(base() as never)).digest("hex"));
    expect(featureSetIdentity(engine, configs(), "probe-damage/2")).not.toHaveProperty("belief_identity_hash");
    // the /1 hash does not depend on the belief at all
    expect(featureSetHash(featureSetIdentity(engine, configs({ belief: setup(fakePort(undefined, { identity_hash: "c".repeat(64) }), 3) }), FEATURE_SET))).toBe(featureSetHash(base()));
  });

  it("adds the belief identity, the world count and the aggregation format, and every one changes the hash", () => {
    expect(belief()).toMatchObject({ feature_set: FEATURE_SET_BELIEF, belief_identity_hash: "b".repeat(64), worlds: 8, belief_features_format: 1 });
    const hash = featureSetHash(belief());
    expect(hash).not.toBe(featureSetHash(base()));
    expect(featureSetHash({ ...belief(), belief_identity_hash: "c".repeat(64) })).not.toBe(hash);
    expect(featureSetHash({ ...belief(), worlds: 4 })).not.toBe(hash);
    expect(featureSetHash({ ...belief(), belief_features_format: 2 })).not.toBe(hash);
    expect(featureSetHash({ ...belief(), reader_sha256: "0".repeat(64) })).not.toBe(hash);
    expect(configuredSetHash(engine, configs())).toBe(hash);
  });

  it("needs the belief port, and the dispatch knows the set", () => {
    expect(() => featureSetIdentity(engine, configs({ belief: undefined as never }), FEATURE_SET_BELIEF)).toThrow(DamageAwareError);
    expect(isFeatureSetId(FEATURE_SET_BELIEF)).toBe(true);
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const viaDispatch = buildFeatures(engine, configs(), channelOf(c), c.context.team as never, candidatesOf(c), memo);
    expect(viaDispatch.feature_set).toBe(FEATURE_SET_BELIEF);
    expect(JSON.stringify(viaDispatch)).toBe(JSON.stringify(build(c)));
  });
});

describe("rows over sampled worlds", () => {
  it("asks the belief once per request with the configured count on the stream of the player's own request", () => {
    const port = fakePort();
    const c = moveRequests()[0] as Captured;
    build(c, { belief: setup(port, 5) });
    expect(port.calls).toHaveLength(1);
    expect(port.calls[0]?.n).toBe(5);
    const hash = decodeRequest((c.context.channel[c.context.channel.length - 1] as string).slice("|request|".length)).request_hash;
    const seq = c.context.channel.filter((line) => line.startsWith("|request|")).length;
    const stream = featureWorldsStream({ request_hash: hash, request_seq: seq, player: c.context.player });
    expect(Buffer.from(port.calls[0]?.stream as Uint8Array).toString("hex")).toBe(Buffer.from(stream).toString("hex"));
    // its own purpose: never the stream the damage-aware baseline draws
    expect(Buffer.from(stream).toString("hex")).not.toBe(Buffer.from(worldsStream({ request_hash: hash, request_seq: seq, player: c.context.player })).toString("hex"));
  });

  it("has exactly the row keys of probe-damage/1 and a request with no move to score has none", () => {
    let checked = 0;
    for (const c of captured) {
      const features = build(c);
      expect(features.feature_set).toBe(FEATURE_SET_BELIEF);
      if (c.request.phase !== "move") {
        expect(features.rows).toEqual([]);
        continue;
      }
      const one = candidateFeatures(engine, configs(), channelOf(c), c.context.team as never, candidatesOf(c));
      expect(features.rows.map(keyOf)).toEqual(one.rows.map(keyOf));
      checked += 1;
    }
    expect(checked).toBeGreaterThan(10);
  });

  it("is the integer mean of the worlds' rows: 3 worlds at 80 and 5 at 20 hit points of damage give 425000 ppm", () => {
    const port = fakePort((index, opponent) => worldOf(opponent, { nature: index < 3 ? "adamant" : "hardy" }));
    const probe = (_context: unknown, query: ParsedMechanicsQuery) => {
      const q = query.query;
      const roll = q.kind === "damage" ? (q.defender.set.includes("Adamant") ? 80 : 20) : 0;
      return stubProbe({ values: q.kind === "damage" ? { [q.move]: roll } : {} })(null as never, query);
    };
    let seen = 0;
    for (const c of moveRequests()) {
      const rows = build(c, { belief: setup(port), probe: probe as never }).rows;
      for (const row of rows) {
        if (row.kind !== "computed") continue;
        expect(row.mean_ppm).toBe(425000);
        expect(row.max_ppm).toBe(425000);
        expect(row.min_ppm).toBe(425000);
        seen += 1;
      }
    }
    expect(seen).toBeGreaterThan(0);
  });

  it("gives the same bytes with and without the memo, on a repeat, and counts the worlds", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const first = JSON.stringify(build(c, {}, memo));
    expect(JSON.stringify(build(c))).toBe(first);
    expect(JSON.stringify(build(c, {}, memo))).toBe(first);
    expect(memo.belief).toMatchObject({ decisions: 2, worlds_sampled: 16, contradicted_worlds: 0, fallbacks: {} });
  });
});

describe("meanRows", () => {
  const row = (target: number, mean: number, ohko: number | null): FeatureRow => ({ slot: 0, move: 1, target, mega: false, kind: "computed", mean_ppm: mean, max_ppm: mean + 10, min_ppm: mean - 10, ohko_rolls: ohko });
  it("floors the integer means of every field, in canonical order", () => {
    const out = meanRows([[row(2, 100, 1), row(1, 7, 0)], [row(2, 201, 2), row(1, 8, 0)]]);
    expect(out.map(keyOf)).toEqual(["0/1/1/false", "0/1/2/false"]);
    expect(out[1]).toMatchObject({ kind: "computed", mean_ppm: 150, max_ppm: 160, min_ppm: 140, ohko_rolls: 1 });
    expect(out[0]).toMatchObject({ mean_ppm: 7, ohko_rolls: 0 });
  });
  it("is null for ohko_rolls when any world has none, and unsupported (the first world's reason) when any world is", () => {
    expect(meanRows([[row(1, 5, 3)], [row(1, 5, null)]])[0]).toMatchObject({ kind: "computed", ohko_rolls: null });
    const refused = (reason: "query_kind" | "random_effect"): FeatureRow => ({ slot: 0, move: 1, target: 1, mega: false, kind: "unsupported", reason });
    expect(meanRows([[row(1, 5, 0)], [refused("random_effect")], [refused("query_kind")]])).toEqual([refused("random_effect")]);
  });
});

describe("fallback to ADR-018's assumed sets", () => {
  it("answers a belief refusal with probe-damage/1's rows under this set's hash and counts it by code", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const refused = fakePort();
    refused.sample = () => {
      throw refusal("evidence_has_no_support");
    };
    const features = build(c, { belief: setup(refused) }, memo);
    const one = candidateFeatures(engine, configs(), channelOf(c), c.context.team as never, candidatesOf(c));
    expect(JSON.stringify(features.rows)).toBe(JSON.stringify(one.rows));
    expect(features.feature_set_hash).toBe(featureSetHash(featureSetIdentity(engine, configs(), FEATURE_SET_BELIEF)));
    expect(memo.belief).toMatchObject({ decisions: 1, worlds_sampled: 0, fallbacks: { evidence_has_no_support: 1 } });
  });

  it("falls back when every world contradicts a public fact, and counts the dropped worlds", () => {
    const c = moveRequests()[0] as Captured;
    const memo = createFeatureMemo();
    const none = fakePort((_index, opponent) => ({ slots: worldOf(opponent).slots.map((slot) => ({ ...slot, brought: false, set: null })) }));
    const features = build(c, { belief: setup(none, 3) }, memo);
    const one = candidateFeatures(engine, configs(), channelOf(c), c.context.team as never, candidatesOf(c));
    expect(JSON.stringify(features.rows)).toBe(JSON.stringify(one.rows));
    expect(memo.belief).toMatchObject({ worlds_sampled: 3, contradicted_worlds: 3, fallbacks: { no_consistent_world: 1 } });
  });

  it("does not hide a run-setup defect, a prior of another format or a missing port", () => {
    const c = moveRequests()[0] as Captured;
    const defect = fakePort();
    defect.sample = () => {
      throw refusal("format_mismatch");
    };
    expect(() => build(c, { belief: setup(defect) })).toThrow("refused");
    expect(() => build(c, { belief: setup(fakePort(undefined, { prior_format_id: "gen9championsvgc2026regmd" })) })).toThrow("another format");
    expect(() => build(c, { belief: undefined as never })).toThrow(DamageAwareError);
    expect(() => build(c, { belief: setup(fakePort(), 0) })).toThrow("positive integer");
    expect(configV2.format_id).toBe(config.format_id);
  });

  it("is still a FeatureBuildError for a request it cannot feature", () => {
    const c = moveRequests()[0] as Captured;
    const bad = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel.slice(0, -1) });
    expect(() => candidateFeaturesBelief(engine, configs(), bad, c.context.team as never, candidatesOf(c))).toThrow(FeatureBuildError);
  });
});

describe("the type rule", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (log: OmniscientLog, verdicts: HiddenVerdicts[], record: BattleRecord, seed: string): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeaturesBelief(engine, configs(), log, team, []);
      // @ts-expect-error: nor hidden verdicts
      void candidateFeaturesBelief(engine, configs(), verdicts, team, []);
      // @ts-expect-error: nor a battle record
      void candidateFeaturesBelief(engine, configs(), record, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeaturesBelief(engine, configs(), seed, team, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });
});
