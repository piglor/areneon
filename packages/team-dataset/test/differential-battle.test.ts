// The battle-level differential's gates (ADR-030 groundwork G2, ADR-028 section 5, #1462), the parts that join the core and the
// engine side. Gate 1: every committed battle (golden, random-play corpus, unavailable/, the singles) recorded on the unmodified
// engine and replayed on a fresh engine object with its draws forced gives no difference at any step of either seat (the
// search-fidelity corpus is `differential-battle-targeted.test.ts`). Gate 2: broken backends, each caught with the right kind,
// step and seat. Gate 4 in small: a fresh corpus on generator teams across the three mixes, zero differences, with entity and
// interaction-class counts (the 2,000-battle size check is `scripts/differential-battle.ts`).
import { join } from "node:path";
import type { Settled } from "@areneon/differential";
import {
  type BattleBackend,
  type BattleCorpus,
  type BattleStepOutput,
  type BattleTrace,
  battleCorpusSummary,
  type BattleDifference,
  compareBattles,
  type Draw,
} from "@areneon/differential/battle";
import { type CommittedBattle, committedBattles, decodeRequest, loadEngine, readTrainingRunConfig } from "@areneon/showdown-bridge";
import { beforeAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import {
  battleBackend,
  battleFreshSeed,
  compareTraces,
  FRESH_MIXES,
  freshSpec,
  loadInteractionClasses,
  newBattleCorpus,
  poolHash,
  traceOf,
} from "../src/differential-battle.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { engine, FORMAT } from "./support.ts";
import tinyConfig from "../../../fixtures/contracts/v1/training-run-config/valid/tiny.json" with { type: "json" };

const TINY = tinyConfig as unknown as Record<string, unknown>;

const classes = loadInteractionClasses();
const SPEC = { id: "committed", kind: "regression" as const, seed: "0".repeat(64), pool: "1".repeat(64) };
/** The replay runs on a second engine object: a fresh instance of the pinned engine (its own tables), not the recorder's. */
const replayEngine = loadEngine(process.env.SHOWDOWN_DIR || undefined);

const traces: BattleTrace[] = [];
const byCorpus = new Map<string, BattleTrace[]>();
beforeAll(async () => {
  for (const { corpus, spec } of committedBattles(["golden", "random", "unavailable", "singles"])) {
    const trace = await traceOf(engine, spec);
    traces.push(trace);
    byCorpus.set(corpus, [...(byCorpus.get(corpus) ?? []), trace]);
  }
}, 300_000);

const run = (backend: BattleBackend, list: readonly BattleTrace[] = traces): BattleCorpus => compareTraces(classes, list, newBattleCorpus(SPEC, classes), backend, 8);

describe("gate 1: every committed battle replays with its draws forced, with no difference", () => {
  it("records every corpus and compares every step of both seats equal", () => {
    expect(new Set(byCorpus.keys())).toEqual(new Set(["golden", "random", "unavailable", "baton-pass", "baton-pass-volatiles", "fling", "illusion", "revival", "side-guard"]));
    const corpus = run(battleBackend(replayEngine));
    const summary = battleCorpusSummary(corpus) as Record<string, unknown>;
    expect(corpus.differences).toEqual([]);
    expect(corpus.unreached).toBe(0);
    expect(corpus.tally.items).toBe(corpus.steps);
    expect(corpus.battles).toBe(traces.length);
    expect(corpus.steps).toBeGreaterThan(1000);
    // Every entity kind of the rule is tallied, and the classes the corpus can reach are tagged.
    expect(Object.keys(summary.entity_kinds as object).sort()).toEqual(expect.arrayContaining(["ability", "item", "move", "side", "species", "status", "volatile", "weather"]));
    const tagged = new Map((summary.classes as { id: string; tagged: number; equal: number }[]).map((c) => [c.id, c]));
    for (const id of ["faint_replacement", "residual_order", "speed_order_tie", "switch_in_trigger"]) expect(tagged.get(id)?.tagged ?? 0, id).toBeGreaterThan(0);
    for (const c of tagged.values()) expect(c.equal).toBe(c.tagged);
  });

  it("the unavailable corpus holds pending candidates, refused choices and their hidden verdicts", () => {
    const steps = (byCorpus.get("unavailable") as BattleTrace[]).flatMap((t) => t.steps);
    const verdicts = new Set(steps.flatMap((s) => [...s.requests.p1, ...s.requests.p2]).flatMap((r) => (r.mask ?? []).map((m) => m.verdict)));
    expect([...verdicts]).toEqual(expect.arrayContaining(["legal", "pending_illegal"]));
    expect(steps.some((s) => [...s.channels.p1, ...s.channels.p2].some((l) => l.startsWith("|error|[Unavailable choice]")))).toBe(true);
  });

  it("is deterministic: the same specs record the same bytes", async () => {
    const spec = committedBattles(["random"])[0] as CommittedBattle;
    const again = await traceOf(engine, spec.spec);
    expect(JSON.stringify(again)).toBe(JSON.stringify(traces.find((t) => t.battle_id === spec.spec.battle_id)));
  });
});

// ---- gate 2: broken backends ----

type OutputChange = (output: BattleStepOutput, step: number) => BattleStepOutput;
type DrawChange = (draws: Draw[], step: number) => Draw[];

/** The real backend with its outputs (or the draws it is forced with) changed: one fault per backend. */
function broken(inner: BattleBackend, change: { output?: OutputChange; draws?: DrawChange; drop?: boolean }): BattleBackend {
  const steps = new Map<string, number>();
  const out = (o: BattleStepOutput, step: number): BattleStepOutput => (change.output === undefined ? o : change.output(structuredClone(o), step));
  return {
    id: "broken",
    backend_format: 1,
    backend_commit: inner.backend_commit,
    start: (batch) =>
      inner.start(batch.map((b) => ({ ...b, draws: change.draws?.(structuredClone(b.draws), 0) ?? b.draws }))).map((s) => {
        if (!s.ok) return s;
        steps.set(s.result.handle, 0);
        return { ok: true, result: { handle: s.result.handle, output: out(s.result.output, 0) } };
      }),
    step: (batch) => {
      const results = inner
        .step(
          batch.map((b) => {
            const step = (steps.get(b.handle) as number) + 1;
            steps.set(b.handle, step);
            return { ...b, draws: change.draws?.(structuredClone(b.draws), step) ?? b.draws };
          }),
        )
        .map((s, i): Settled<BattleStepOutput> => (s.ok ? { ok: true, result: out(s.result, steps.get((batch[i] as { handle: string }).handle) as number) } : s));
      return change.drop === true ? results.slice(1) : results;
    },
    release: (handles) => inner.release(handles),
  };
}

const firstStep = (trace: BattleTrace, has: (s: BattleTrace["steps"][number]) => boolean): number => {
  const k = trace.steps.findIndex(has);
  expect(k, "the trace has such a step").toBeGreaterThanOrEqual(0);
  return k;
};
const differenceOf = (backend: BattleBackend, trace: BattleTrace): BattleDifference | null => (compareBattles(backend, [trace])[0] as { difference: BattleDifference | null }).difference;
/** Rewrite one request line of a seat; `rehash` recomputes its request_hash (a body the engine shape refuses keeps the old one). */
const rewriteRequest = (o: BattleStepOutput, seat: "p1" | "p2", edit: (body: Record<string, unknown>) => void, rehash = true): BattleStepOutput => {
  const i = o.channels[seat].findIndex((l) => l.startsWith("|request|") && !l.includes('"wait":true'));
  const body = JSON.parse((o.channels[seat][i] as string).slice("|request|".length)) as Record<string, unknown>;
  edit(body);
  const text = JSON.stringify(body);
  o.channels[seat][i] = `|request|${text}`;
  const j = o.requests[seat].length - 1;
  if (rehash) (o.requests[seat][j] as { request_hash: string }).request_hash = decodeRequest(text).request_hash;
  return o;
};

describe("gate 2: broken battle backends are caught with the right kind, step and seat", () => {
  const random = (): BattleTrace => (byCorpus.get("random") as BattleTrace[])[0] as BattleTrace;

  it("a wrong damage roll (one forced random(16) off by one) is a channel difference at its step", () => {
    const trace = random();
    let caught: BattleDifference | null = null;
    // The first roll whose change shows: a one-point change of a small hit can round to the same damage.
    for (const [k, step] of trace.steps.entries()) {
      const at = step.draws.findIndex((d) => d.method === "random" && d.args.join() === "16");
      if (at < 0) continue;
      caught = differenceOf(broken(battleBackend(replayEngine), { draws: (draws, s) => (s === k ? draws.map((d, i) => (i === at ? { ...d, result: [String((Number(d.result[0]) + 1) % 16)] } : d)) : draws) }), trace);
      if (caught !== null) {
        expect(caught).toMatchObject({ kind: "channel", step: k });
        break;
      }
    }
    expect(caught).not.toBeNull();
  });

  it("a flipped forced chance draw is a channel difference at its step", () => {
    const trace = random();
    let caught: BattleDifference | null = null;
    for (const [k, step] of trace.steps.entries()) {
      const at = step.draws.findIndex((d) => d.method === "randomChance");
      if (at < 0) continue;
      caught = differenceOf(broken(battleBackend(replayEngine), { draws: (draws, s) => (s === k ? draws.map((d, i) => (i === at ? { ...d, result: [d.result[0] === "true" ? "false" : "true"] } : d)) : draws) }), trace);
      if (caught?.kind === "channel") {
        expect(caught.step).toBe(k);
        break;
      }
    }
    expect(caught?.kind).toBe("channel");
  });

  it("two draws reported in the wrong order (same answers, same lines) is a draws difference", () => {
    const trace = random();
    const key = (d: Draw): string => `${d.method}(${d.args.join()})`;
    const k = firstStep(trace, (s) => s.draws.length >= 2 && key(s.draws[0] as Draw) !== key(s.draws[1] as Draw));
    const backend = broken(battleBackend(replayEngine), { output: (o, s) => (s === k ? { ...o, draws: [o.draws[1] as Draw, o.draws[0] as Draw, ...o.draws.slice(2)] } : o) });
    expect(differenceOf(backend, trace)).toEqual({ kind: "draws", step: k, seat: null, where: "draw request 0 of the step differs" });
  });

  it("a dropped -damage line in one seat's channel is a channel difference for that seat", () => {
    const trace = random();
    const k = firstStep(trace, (s) => s.channels.p2.some((l) => l.startsWith("|-damage|")));
    const backend = broken(battleBackend(replayEngine), { output: (o, s) => (s === k ? { ...o, channels: { ...o.channels, p2: o.channels.p2.filter((l, i) => i !== o.channels.p2.findIndex((x) => x.startsWith("|-damage|"))) } } : o) });
    expect(differenceOf(backend, trace)).toMatchObject({ kind: "channel", step: k, seat: "p2" });
  });

  it("a skipped residual (no upkeep) is a channel difference", () => {
    const trace = random();
    const k = firstStep(trace, (s) => s.channels.p1.includes("|upkeep"));
    const drop = (lines: string[]): string[] => lines.filter((l) => l !== "|upkeep");
    const backend = broken(battleBackend(replayEngine), { output: (o, s) => (s === k ? { ...o, channels: { p1: drop(o.channels.p1), p2: drop(o.channels.p2) } } : o) });
    expect(differenceOf(backend, trace)).toMatchObject({ kind: "channel", step: k, seat: "p1" });
  });

  it("a changed request body (rqid-free, hash recomputed) is a request difference", () => {
    const trace = random();
    const k = firstStep(trace, (s) => s.channels.p2.some((l) => l.startsWith("|request|") && l.includes('"active"')));
    const backend = broken(battleBackend(replayEngine), {
      output: (o, s) => (s === k ? rewriteRequest(o, "p2", (b) => {
        ((b.side as { pokemon: { condition: string }[] }).pokemon[0] as { condition: string }).condition = "1/1";
      }) : o),
    });
    expect(differenceOf(backend, trace)).toMatchObject({ kind: "request", step: k, seat: "p2" });
  });

  it("a request with a missing field is a request difference", () => {
    const trace = random();
    const backend = broken(battleBackend(replayEngine), {
      output: (o, s) => (s === 0 ? rewriteRequest(o, "p1", (b) => {
        delete ((b.side as { pokemon: Record<string, unknown>[] }).pokemon[0] as Record<string, unknown>).stats;
      }, false) : o),
    });
    expect(differenceOf(backend, trace)).toMatchObject({ kind: "request", step: 0, seat: "p1", where: "request line 0 of the step differs" });
  });

  it("a pending candidate made legal is a mask difference", () => {
    const trace = (byCorpus.get("unavailable") as BattleTrace[])[0] as BattleTrace;
    const pendingIn = (s: BattleTrace["steps"][number], seat: "p1" | "p2"): boolean => s.requests[seat].some((r) => (r.mask ?? []).some((m) => m.verdict !== "legal"));
    const k = firstStep(trace, (s) => pendingIn(s, "p1") || pendingIn(s, "p2"));
    const seat = pendingIn(trace.steps[k] as BattleTrace["steps"][number], "p1") ? "p1" : "p2";
    const backend = broken(battleBackend(replayEngine), {
      output: (o, s) => {
        if (s !== k) return o;
        for (const r of o.requests[seat]) for (const m of r.mask ?? []) if (m.verdict !== "legal") m.verdict = "legal";
        return o;
      },
    });
    expect(differenceOf(backend, trace)).toMatchObject({ kind: "mask", step: k, seat });
  });

  it("a wrong winner at the end is an outcome difference at the last step", () => {
    const trace = (traces.find((t) => t.outcome.winner !== null) as BattleTrace);
    const last = trace.steps.length - 1;
    const backend = broken(battleBackend(replayEngine), { output: (o, s) => (s === last ? { ...o, winner: o.winner === "p1" ? "p2" : "p1" } : o) });
    expect(differenceOf(backend, trace)).toEqual({ kind: "outcome", step: last, seat: null, where: "the winner differs" });
  });

  it("a dropped result in a batch is a count difference for every battle of the batch", () => {
    const pair = (byCorpus.get("random") as BattleTrace[]).slice(0, 2);
    const results = compareBattles(broken(battleBackend(replayEngine), { drop: true }), pair, 2);
    for (const r of results) expect(r.difference).toMatchObject({ kind: "count", step: 1 });
  });
});

// ---- gate 4 in small: a fresh corpus ----

describe("a fresh corpus on generator teams", () => {
  it("draws battles from the dev and training pool across the three mixes and replays them with no difference", { timeout: 300_000 }, async () => {
    const source = join(DATASETS_DIR(FORMAT), "fixtures-v1");
    const config = readTrainingRunConfig({ ...structuredClone(TINY), team_dataset: { path: source, dataset_hash: verifiedDatasetHash(engine, source, FORMAT) } });
    const pool = datasetTeamPool(engine, config);
    const seed = battleFreshSeed(engine);
    const fresh: BattleTrace[] = [];
    for (let i = 0; i < 9; i++) fresh.push(await traceOf(engine, freshSpec(seed, pool, i, FORMAT, 30)));
    expect(new Set(fresh.map((t) => t.battle_id.split("/")[1]))).toEqual(new Set(FRESH_MIXES));
    const corpus = compareTraces(classes, fresh, newBattleCorpus({ id: "fresh-small", kind: "fresh", seed, pool: poolHash(pool) }, classes), battleBackend(replayEngine));
    expect(corpus.differences).toEqual([]);
    expect(corpus.tally.items).toBe(corpus.steps);
    // A second draw gives the same battles (the fresh rule is a function of the backend version and the pool).
    expect(JSON.stringify(await traceOf(engine, freshSpec(seed, pool, 4, FORMAT, 30)))).toBe(JSON.stringify(fresh[4]));
  });
});
