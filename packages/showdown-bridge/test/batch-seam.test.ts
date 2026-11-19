// The batch seam (ADR-030 step 1, #1455): its primitives, and the claim that matters: building decisions through the wave's batched
// mechanics call gives the bytes the inline builders give, and a generation played with the seam writes the same stream, rows and counts.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { type Json, jcs } from "../src/canonical-json.ts";
import { batchRunnerOf, createFeatureMemo, type FeatureConfigs, inlineProbe } from "../src/candidate-features.ts";
import { FEATURE_SET_OUTCOME } from "../src/candidate-features-outcome.ts";
import { buildFeatures, buildFeaturesG } from "../src/feature-build.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { featureSetup, playEntries, playGeneration, trainingTeamPool } from "../src/league-generation.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { MechanicsProbeError } from "../src/mechanics-probe.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import type { MechanicsResult } from "../src/mechanics-result.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { PolicyClient } from "../src/policy-worker.ts";
import type { PolicyQueryInput } from "../src/policy-protocol.ts";
import { batchOf, driveSync, emptySeamStats, type Gen, SEAM_HOLD_MS, pure, SeamError, sumSeamStats, WaveSeam } from "../src/probe-batch.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";
import type { PokemonSet } from "../src/engine.ts";

// ---------------------------------------------------------------- primitives

const query = (n: number): ParsedMechanicsQuery => readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "stats", set: `Level: ${50 - n}` } });
const result = (n: number): MechanicsResult => ({ kind: "tag", n }) as unknown as MechanicsResult;

/** A generator that asks `n` queries one after another and returns the numbers of its answers (or the error it was handed). */
function* asking(first: number, n: number, trap = false): Gen<unknown[]> {
  const got: unknown[] = [];
  for (let k = 0; k < n; k++) {
    try {
      got.push(((yield query(first + k)) as unknown as { n: number }).n);
    } catch (error) {
      if (!trap) throw error;
      got.push(`caught ${(error as Error).message}`);
    }
  }
  return got;
}

describe("driveSync and batchOf", () => {
  it("answers each yielded query in turn and throws a refusal into the generator, which may catch it", () => {
    const seen: string[] = [];
    const run = (q: ParsedMechanicsQuery): MechanicsResult => {
      seen.push(mechanicsQueryHash(q).slice(0, 4));
      if (seen.length === 2) throw new MechanicsProbeError("refused");
      return result(seen.length);
    };
    expect(driveSync(asking(0, 3, true), run)).toEqual([1, "caught refused", 3]);
    expect(seen.length).toBe(3);
    expect(() => driveSync(asking(0, 3, false), (q) => (mechanicsQueryHash(q) === mechanicsQueryHash(query(1)) ? (() => { throw new Error("boom"); })() : result(0)))).toThrow("boom");
  });

  it("runs a batch in query order, once per distinct query, and keeps a refusal as the query's outcome", () => {
    let runs = 0;
    const batch = batchOf((q) => {
      runs += 1;
      if (mechanicsQueryHash(q) === mechanicsQueryHash(query(1))) throw new MechanicsProbeError("no");
      return result(runs);
    });
    const out = batch([query(0), query(1), query(0), query(2)]);
    expect(runs).toBe(3);
    expect(out.map((o) => (o.ok ? "ok" : "refused"))).toEqual(["ok", "refused", "ok", "ok"]);
    expect(out[0]).toBe(out[2]);
    expect((out[1] as { error: Error }).error.message).toBe("no");
  });
});

/** A seam whose flush is run by hand and whose clock is a number. */
function manual(options: { batch?: Parameters<typeof batchOf>[0]; hold_ms?: number } = {}) {
  const flushes: (() => void)[] = [];
  const clock = { t: 0 };
  const seam = new WaveSeam({ ...(options.batch === undefined ? {} : { batch: batchOf(options.batch) }), ...(options.hold_ms === undefined ? {} : { hold_ms: options.hold_ms }), now: () => clock.t, defer: (callback) => void flushes.push(callback) });
  /** Run the scheduled flush (and what it schedules) until nothing is left, letting promises settle between. */
  const settle = async (): Promise<void> => {
    for (let guard = 0; guard < 50; guard++) {
      await Promise.resolve();
      await Promise.resolve();
      const next = flushes.shift();
      if (next === undefined) return;
      next();
    }
  };
  return { seam, flushes, clock, settle };
}

describe("WaveSeam builds", () => {
  it("answers the queries of every pending build through one batch per round, in query order", async () => {
    const calls: number[][] = [];
    const run = (q: ParsedMechanicsQuery): MechanicsResult => result(Number(/Level: (\d+)/.exec(q.query.kind === "stats" ? q.query.set : "")?.[1]));
    const { seam, settle } = manual({
      batch: (q) => {
        calls.push([Number(/Level: (\d+)/.exec((q.query as { set: string }).set)?.[1])]);
        return run(q);
      },
    });
    const a = seam.build(asking(0, 3));
    const b = seam.build(asking(10, 1));
    const c = seam.build(asking(20, 2));
    await settle();
    expect(await a).toEqual([50, 49, 48]);
    expect(await b).toEqual([40]);
    expect(await c).toEqual([30, 29]);
    const stats = seam.stats();
    // Three builds: round 1 holds the first query of each, round 2 the second of a and c, round 3 the third of a.
    expect(stats.builds).toBe(3);
    expect(stats.rounds).toBe(3);
    expect(stats.queries).toBe(6);
    expect(stats.queries_per_round).toEqual({ "1": 1, "2": 1, "3": 1 });
    expect(stats.builds_per_flush).toEqual({ "3": 1 });
    expect(calls.length).toBe(6);
  });

  it("hands a refusal back to the build that asked and fails one build without failing the others", async () => {
    const { seam, settle } = manual({
      batch: (q) => {
        if (/Level: 49/.test((q.query as { set: string }).set)) throw new MechanicsProbeError("nope");
        return result(1);
      },
    });
    const caught = seam.build(asking(0, 2, true));
    const failing = seam.build(asking(0, 2, false));
    const fine = seam.build(asking(10, 1));
    const failed = failing.then(() => "no error", (error: Error) => error.message);
    await settle();
    expect(await caught).toEqual([1, "caught nope"]);
    expect(await failed).toBe("nope");
    expect(await fine).toEqual([1]);
  });

  it("fails every build that waits on a batch that itself fails, and refuses a build with no batch", async () => {
    const dead = new WaveSeam({ batch: () => { throw new Error("batch down"); }, defer: (callback) => void callback() });
    const outcome = await Promise.allSettled([dead.build(asking(0, 1)), dead.build(asking(5, 1))]);
    expect(outcome.map((o) => (o.status === "rejected" ? (o.reason as Error).message : "ok"))).toEqual(["batch down", "batch down"]);
    expect(() => new WaveSeam().build(asking(0, 1))).toThrow(SeamError);
  });

  it("a build whose generator throws before its first query rejects; one with no query resolves", async () => {
    const { seam, settle } = manual({ batch: () => result(0) });
    const boom = seam.build((function* (): Gen<number> {
      yield* pure(0);
      throw new Error("early");
    })());
    const none = seam.build(pure(7));
    const early = boom.then(() => "ok", (error: Error) => error.message);
    await settle();
    expect(await early).toBe("early");
    expect(await none).toBe(7);
  });
});

describe("WaveSeam policy queries", () => {
  it("issues the queries of a wave together and returns each its own answer", async () => {
    const { seam, settle } = manual();
    const issued: string[] = [];
    const ask = (name: string, value: number) => seam.infer(() => { issued.push(name); return Promise.resolve(value); });
    const all = Promise.all([ask("a", 1), ask("b", 2), ask("c", 3)]);
    expect(issued).toEqual([]);
    await settle();
    expect(await all).toEqual([1, 2, 3]);
    expect(issued).toEqual(["a", "b", "c"]);
    expect(seam.stats().asks_per_issue).toEqual({ "3": 1 });
  });

  it("passes on a refused or throwing query to its game only", async () => {
    const { seam, settle } = manual();
    const bad = seam.infer(() => Promise.reject(new Error("worker gone")));
    const throws = seam.infer(() => { throw new Error("sync"); });
    const good = seam.infer(() => Promise.resolve("fine"));
    const reasons = Promise.all([bad.catch((e: Error) => e.message), throws.catch((e: Error) => e.message)]);
    await settle();
    expect(await reasons).toEqual(["worker gone", "sync"]);
    expect(await good).toBe("fine");
  });

  it("holds the builds while a policy query is outstanding (the lockstep) and lets them go once it is answered", async () => {
    const { seam, settle } = manual({ batch: () => result(0) });
    let answer: (value: string) => void = () => undefined;
    const slow = seam.infer(() => new Promise<string>((resolve) => (answer = resolve)));
    await settle(); // issued
    const built = seam.build(asking(0, 1));
    let done = false;
    void built.then(() => (done = true));
    await settle();
    expect(done).toBe(false); // waits for the policy query
    answer("ok");
    await slow;
    await settle();
    await built;
    expect(done).toBe(true);
    expect(seam.stats().held_forced).toBe(0);
  });

  it("goes ahead of a policy query that has outlasted the hold, and counts it", async () => {
    const { seam, settle, clock } = manual({ batch: () => result(0), hold_ms: 100 });
    void seam.infer(() => new Promise<string>(() => undefined)).catch(() => undefined); // never answered: a game gave up on it
    await settle();
    const built = seam.build(asking(0, 1));
    await settle();
    clock.t = 150;
    // The hold timer is real time: let it fire.
    await new Promise((resolve) => setTimeout(resolve, 200));
    await settle();
    await built;
    expect(seam.stats().held_forced).toBe(1);
  });

  it("uses setImmediate and the wall clock by default, and its hold is five seconds", async () => {
    const seam = new WaveSeam({ batch: batchOf(() => result(3)) });
    expect(SEAM_HOLD_MS).toBe(5000);
    expect(await Promise.all([seam.build(asking(0, 2)), seam.infer(() => Promise.resolve("x"))])).toEqual([[3, 3], "x"]);
  });
});

describe("seam statistics", () => {
  it("add up, size tables merged", () => {
    const a = emptySeamStats();
    a.builds = 2;
    a.rounds = 1;
    a.queries = 5;
    a.queries_per_round = { "5": 1 };
    a.builds_per_flush = { "2": 1 };
    a.asks_per_issue = { "2": 1 };
    const b = { ...emptySeamStats(), builds: 1, rounds: 2, queries: 3, held_forced: 1, queries_per_round: { "5": 1, "1": 2 }, builds_per_flush: { "1": 1 }, asks_per_issue: { "3": 1 } };
    expect(sumSeamStats([a, b])).toEqual({ builds: 3, rounds: 3, queries: 8, held_forced: 1, queries_per_round: { "5": 2, "1": 2 }, builds_per_flush: { "2": 1, "1": 1 }, asks_per_issue: { "2": 1, "3": 1 } });
    expect(sumSeamStats([])).toEqual(emptySeamStats());
  });
});

// ---------------------------------------------------------------- the builders: the wave's bytes are the inline bytes

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const outcomeRecord = loadOutcomeConfig(GOLDEN_FORMAT);
const configsFor = (set: string): FeatureConfigs => ({
  vocabulary,
  config,
  mechanics: set === FEATURE_SET_OUTCOME ? withMoveOutcome(mechanics) : mechanics,
  ...(set === FEATURE_SET_OUTCOME ? { outcome: { config: outcomeRecord } } : {}),
  snapshot,
  data_class: "synthetic",
});
const candidatesOf = (c: Captured) => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  return [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
};
const channelOf = (c: Captured) => playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
const moveDecisions = (): Captured[] => captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2).slice(0, 12);

describe("the feature builders through the wave", () => {
  for (const set of ["probe-damage/1", "probe-damage/2", FEATURE_SET_OUTCOME] as const) {
    it(`${set}: every decision built in one wave equals the inline build, with the same probe counts`, async () => {
      const decisions = [...captured.filter((c) => c.request.phase === "preview").slice(0, 1), ...moveDecisions()];
      expect(decisions.length).toBeGreaterThan(4);
      const cfg = configsFor(set);
      const inline = decisions.map((c) => {
        const memo = createFeatureMemo();
        const features = buildFeatures(engine, cfg, channelOf(c), c.context.team as PokemonSet[], candidatesOf(c), memo, set);
        return { features, counts: { ...memo.counts } };
      });
      const { configs: wave, batch } = batchRunnerOf(engine, cfg);
      const seam = new WaveSeam({ batch });
      const memos = decisions.map(() => createFeatureMemo());
      const built = await Promise.all(decisions.map((c, i) => seam.build(buildFeaturesG(engine, wave, channelOf(c), c.context.team as PokemonSet[], candidatesOf(c), memos[i] as never, set))));
      built.forEach((features, i) => {
        expect(jcs(features as unknown as Json)).toBe(jcs((inline[i] as { features: unknown }).features as Json));
        expect(memos[i]?.counts).toEqual((inline[i] as { counts: unknown }).counts);
      });
      const stats = seam.stats();
      expect(stats.builds).toBe(decisions.length);
      // The wave batches: fewer batched calls than queries, and some round holds more than one.
      expect(stats.rounds).toBeLessThan(stats.queries);
      expect(Object.keys(stats.queries_per_round).some((size) => Number(size) > 1)).toBe(true);
    }, 600_000);
  }

  it("a request the builder refuses is refused with the same code from the wave as inline", async () => {
    const c = captured.find((x) => x.request.phase === "move") as Captured;
    // A channel that does not end in the request being answered: `no_request`.
    const bad = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel.slice(0, -1) });
    const cfg = configsFor("probe-damage/1");
    const inline = inlineProbe(engine, cfg);
    const code = (run: () => unknown): string => {
      try {
        run();
      } catch (error) {
        return (error as { code?: string }).code ?? "other";
      }
      return "none";
    };
    expect(code(() => buildFeatures(engine, inline.configs, bad, c.context.team as PokemonSet[], candidatesOf(c), createFeatureMemo(), "probe-damage/1"))).toBe("no_request");
    const { configs: wave, batch } = batchRunnerOf(engine, cfg);
    const seam = new WaveSeam({ batch });
    const waved = await seam.build(buildFeaturesG(engine, wave, bad, c.context.team as PokemonSet[], candidatesOf(c), createFeatureMemo(), "probe-damage/1")).then(() => "none", (error: { code?: string }) => error.code ?? "other");
    expect(waved).toBe("no_request");
  });
});

// ---------------------------------------------------------------- a generation with and without the seam

const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const withFeatures = (hash: string, set = "probe-damage/1"): TrainingRunConfig => {
  const raw = structuredClone(tiny) as { model: { encoder: { options?: unknown } } };
  raw.model.encoder.options = { candidate_features: { feature_set: set, feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
};
const POLICY = `policy:${"c".repeat(64)}`;
const hashOf = (set = "probe-damage/1"): string => {
  try {
    featureSetup(engine, withFeatures("0".repeat(64), set));
  } catch (error) {
    return /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
  }
  return "";
};

/** A stand-in worker that plays the actor rule over each query's candidates, answering after a delay that differs by battle (replies come back in another order than they went out). */
function standIn(statement: { feature_set: string; feature_set_hash: string } | null): PolicyClient & { beginGame: () => Promise<void>; queries: PolicyQueryInput[] } {
  const queries: PolicyQueryInput[] = [];
  return {
    queries,
    featureSet: statement,
    beginGame: () => Promise.resolve(),
    ask(battle, query) {
      queries.push(query);
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      const delay = battle.endsWith("p1") ? 3 : 1;
      return new Promise((resolve) => setTimeout(() => resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" }), delay));
    },
  };
}

describe("a generation played with the seam", () => {
  for (const set of ["probe-damage/1", "probe-damage/2"] as const) {
    it(`${set}: the stream, the game rows and the queries are the bytes of a generation played without it`, async () => {
      const hash = hashOf(set);
      const cfg = withFeatures(hash, set);
      const setup = featureSetup(engine, cfg) as NonNullable<ReturnType<typeof featureSetup>>;
      const pool = trainingTeamPool(engine, cfg.generation.excluded_abilities);
      const plainClient = standIn(setup.statement);
      const plain = await playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client: plainClient, pool, features: setup.configs });
      const seamClient = standIn(setup.statement);
      const waves: unknown[] = [];
      const seamed = await playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client: seamClient, pool, features: setup.configs, batch_seam: true, on_seam_stats: (stats) => void waves.push(stats) });
      expect(seamed.stream).toBe(plain.stream);
      expect(jcs(seamed.games as unknown as Json)).toBe(jcs(plain.games as unknown as Json));
      expect(seamed.training_set_hash).toBe(plain.training_set_hash);
      // The queries sent are the same set of bytes (the order across games differs).
      const bytes = (list: PolicyQueryInput[]): string[] => list.map((q) => jcs(q as unknown as Json)).sort();
      expect(bytes(seamClient.queries)).toEqual(bytes(plainClient.queries));
      expect(waves.length).toBeGreaterThan(0);
    }, 900_000);
  }

  it("without a feature set only the policy queries are batched, and the stream is the same", async () => {
    const cfg = readTrainingRunConfig(tiny);
    const pool = trainingTeamPool(engine, cfg.generation.excluded_abilities);
    const a = await playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client: standIn(null), pool });
    const stats: { asks_per_issue: Record<string, number>; queries: number }[] = [];
    const b = await playGeneration({ engine, config: cfg, generation: 1, policy_id: POLICY, client: standIn(null), pool, batch_seam: true, on_seam_stats: (s) => void stats.push(s) });
    expect(b.stream).toBe(a.stream);
    expect(jcs(b.games as unknown as Json)).toBe(jcs(a.games as unknown as Json));
    expect(stats.every((s) => s.queries === 0)).toBe(true);
    expect(stats.some((s) => Object.keys(s.asks_per_issue).some((size) => Number(size) >= 1))).toBe(true);
  }, 600_000);

  it("playEntries with the seam off reports no statistics", async () => {
    const cfg = readTrainingRunConfig(tiny);
    const pool = trainingTeamPool(engine, cfg.generation.excluded_abilities);
    let called = false;
    await playEntries({ engine, config: cfg, generation: 1, policy_id: POLICY, client: standIn(null), pool, on_seam_stats: () => (called = true) }, { index: 0, of: 8 });
    expect(called).toBe(false);
  }, 300_000);
});
