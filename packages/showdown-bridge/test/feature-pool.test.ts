import { beforeAll, describe, expect, it } from "vitest";
import { candidateFeatures, createFeatureMemo, FEATURE_SET_BELIEF } from "../src/candidate-features.ts";
import { statsOf } from "../src/feature-bench.ts";
import { fakePort } from "./damage-aware-world-support.ts";
import { answerMessage, createFeatureService, type FeatureResult, type FeatureServiceSpec, type FeatureTask } from "../src/feature-service.ts";
import { FeaturePool, FeaturePoolError } from "../src/feature-pool.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { playerChannel } from "../src/player-channel.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";
import type { Candidate } from "../src/choices.ts";

const spec: FeatureServiceSpec = { format_id: GOLDEN_FORMAT, data_class: "synthetic" };

const taskOf = (c: Captured, game: string, final = false): FeatureTask => {
  const legal = c.request.legal as NonNullable<typeof c.request.legal>;
  const candidates: Pick<Candidate, "choice" | "pending">[] = [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
  return { game, player: c.context.player, lines: [...c.context.channel], team: structuredClone(c.context.team as never), candidates, ...(final ? { final } : {}) };
};

let tasks: FeatureTask[] = [];
beforeAll(async () => {
  // Two seeded games, a handful of move requests each, real probe on the pinned engine.
  const pick = async (index: number, game: string): Promise<FeatureTask[]> => {
    const moves = (await capture(corpus[index] as never, "p1", 6)).filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2).slice(0, 3);
    return moves.map((c, k) => taskOf(c, game, k === moves.length - 1));
  };
  tasks = [...(await pick(0, "game-a")), ...(await pick(1, "game-b"))];
  expect(tasks.length).toBeGreaterThanOrEqual(4);
}, 120_000); // seeded battles on the pinned engine

const outcomes = (results: FeatureResult[]): string => JSON.stringify(results.map((r) => r.outcome));

describe("feature pool (ADR-022 mitigation 3)", () => {
  it("gives the builder's own rows at every pool size, in task order", async () => {
    const reference = tasks.map((t) =>
      candidateFeatures(engine, { vocabulary, config, mechanics, snapshot, data_class: "synthetic" }, playerChannel({ schema_version: 1, data_class: "synthetic", player: t.player, lines: t.lines }), t.team, t.candidates, createFeatureMemo()),
    );
    const expected = JSON.stringify(reference.map((features) => ({ kind: "features", features })));
    for (const size of [0, 1, 3]) {
      const pool = new FeaturePool(size, spec);
      try {
        const results = await pool.run(tasks);
        expect(outcomes(results), `pool size ${size}`).toBe(expected);
        expect(results.every((r) => r.ms >= 0)).toBe(true);
        // A second batch on the same pool (new memo state in the threads) gives the same bytes.
        expect(outcomes(await pool.run(tasks.map((t) => ({ ...t, game: `${t.game}-again` }))))).toBe(expected);
      } finally {
        await pool.close();
      }
    }
  }, 240_000); // three pools, each thread loads the engine; real probe

  it("keeps a game's memo in one thread and drops it after the final task", () => {
    const service = createFeatureService(spec, engine);
    const first = tasks[0] as FeatureTask;
    const a = service.run({ ...first, game: "memo" });
    const b = service.run({ ...first, game: "memo", final: true });
    expect(a.counts.probe_runs).toBeGreaterThan(0);
    expect(b.counts.probe_runs).toBe(0);
    expect(b.counts.queries).toBe(a.counts.queries);
    expect(JSON.stringify(b.outcome)).toBe(JSON.stringify(a.outcome));
    expect(service.run({ ...first, game: "memo" }).counts.probe_runs).toBe(a.counts.probe_runs);
  }, 60_000);

  it("reports a request that cannot be featured as a code, and a defect as a failure", async () => {
    const first = tasks[0] as FeatureTask;
    const service = createFeatureService(spec, engine);
    expect(service.run({ ...first, lines: first.lines.slice(0, -1) }).outcome).toEqual({ kind: "error", code: "no_request" });
    const bad: FeatureTask = { ...first, lines: [...first.lines.slice(0, -1), "|split|p2", first.lines[first.lines.length - 1] as string] };
    expect(() => service.run(bad)).toThrow();
    expect(answerMessage(service, { id: 7, task: bad })).toMatchObject({ id: 7, defect: expect.any(String) });
    expect(answerMessage({ run: () => { throw "x"; } }, { id: 8, task: first })).toEqual({ id: 8, defect: "a defect of the feature builder" });
    const pool = new FeaturePool(1, spec);
    try {
      await expect(pool.run([bad])).rejects.toBeInstanceOf(FeaturePoolError);
    } finally {
      await pool.close();
    }
  }, 120_000);

  it("serves probe-damage-belief/1 on the calling thread with the world counts of each task, and refuses the port anywhere else", async () => {
    const beliefSpec: FeatureServiceSpec = { ...spec, feature_set: FEATURE_SET_BELIEF };
    const port = fakePort();
    const setup = { port, worlds: 3 };
    const service = createFeatureService(beliefSpec, engine, { belief: setup });
    const first = tasks[0] as FeatureTask;
    const a = service.run({ ...first, game: "belief" });
    expect(a.outcome.kind).toBe("features");
    expect(a.belief).toEqual({ decisions: 1, worlds_sampled: 3, redraws: 0, contradicted_worlds: 0, fallbacks: {} });
    // a refusal is a fallback counted by cause in this task's delta
    port.sample = () => {
      throw Object.assign(new Error("refused"), { name: "BeliefError", code: "reader_flagged" });
    };
    const b = service.run({ ...first, game: "belief" });
    expect(b.belief).toEqual({ decisions: 1, worlds_sampled: 0, redraws: 0, contradicted_worlds: 0, fallbacks: { reader_flagged: 1 } });
    // the port goes with the belief set and with no other, and not to a thread
    expect(() => createFeatureService(beliefSpec, engine)).toThrow("belief port");
    expect(() => createFeatureService(spec, engine, { belief: setup })).toThrow("belief port");
    expect(() => new FeaturePool(1, beliefSpec, { belief: setup })).toThrow(FeaturePoolError);
    const pool = new FeaturePool(0, beliefSpec, { belief: setup });
    expect(((await pool.run([{ ...first, game: "pooled" }]))[0] as FeatureResult).belief?.decisions).toBe(1);
    await pool.close();
    // the bench sums them
    expect(statsOf([a, b]).belief).toEqual({ decisions: 2, worlds_sampled: 3, redraws: 0, contradicted_worlds: 0, fallbacks: { reader_flagged: 1 } });
  }, 120_000);

  it("refuses a bad size, a thread that fails and a closed pool", async () => {
    expect(() => new FeaturePool(-1, spec)).toThrow(FeaturePoolError);
    expect(() => createFeatureService({ ...spec, format_id: "other" })).toThrow();
    const broken = new FeaturePool(1, { ...spec, format_id: "other" });
    await expect(broken.run([tasks[0] as FeatureTask])).rejects.toBeInstanceOf(FeaturePoolError);
    await expect(broken.run([tasks[0] as FeatureTask])).rejects.toThrow(/failed|ended|closed/);
    await broken.close();
    const pool = new FeaturePool(0, spec);
    await pool.close();
    await expect(pool.run([])).rejects.toThrow("closed");
  }, 60_000);
});
