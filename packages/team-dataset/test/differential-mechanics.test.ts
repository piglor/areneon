// The differential harness at level 1 against the pinned probe (ADR-030 groundwork G1, #1461): the gate.
// 1. Self-differential: the probe against itself (a second engine object) over every query the feature builders issue on the
//    first battle of each committed corpus gives zero differences, and the committed full report is fresh and says the same.
// 2. Five deliberately broken backends are each caught with the right reason, entity and field.
// 3. A ledger from a fresh corpus drawn by the backend's seed admits, routes and is refused at another pin.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  admissionRouter,
  type Backend,
  buildLedger,
  compareBatch,
  freshCorpus,
  jcs,
  type Json,
  LEVEL_1_RULE,
  readBackendAdmission,
  renderReport,
  runCorpus,
  shadowCheck,
  ShadowDivergence,
  shadowSample,
  shadowSize,
  type DifferentialReport,
} from "@areneon/differential";
import { jcs as bridgeJcs, loadEngine, type MechanicsResult, mechanicsQueryHash, type ParsedMechanicsQuery, probeBackend, RANDOM_DIR, sampleTarget, type Settled } from "@areneon/showdown-bridge";
import { beforeAll, describe, expect, it } from "vitest";
import { collectQueries, corpusBattles, harnessConfigs, mechanicsHarnessLevel, remembering, reportInputs, SELF_REPORT_DIR, SELF_REPORT_TITLE, selfDifferential } from "../src/differential-mechanics.ts";

const engine = loadEngine();
const second = loadEngine();
const configs = harnessConfigs(engine);
const level = mechanicsHarnessLevel(engine);
const oracle = remembering(probeBackend(engine, { config: configs.oracle, snapshot: configs.snapshot })) as Backend<ParsedMechanicsQuery, MechanicsResult>;

describe("gate 1: the probe against itself", () => {
  it("finds no difference over every query the builders issue on the first battle of each committed corpus", () => {
    const run = selfDifferential(engine, second, 1);
    for (const corpus of run.report.corpora) {
      expect(corpus.items, corpus.id).toBeGreaterThan(500);
      expect(corpus.differences, corpus.id).toBe(0);
      expect(corpus.count_mismatches, corpus.id).toBe(0);
    }
    const kinds = run.report.kinds;
    for (const kind of ["ability", "item", "kind", "move", "species", "status"]) expect(kinds[kind], kind).toBeGreaterThan(0);
    for (const kind of ["damage", "move_outcome", "speed_order", "stats"]) expect(run.report.entities.some((e) => e.id === `kind:${kind}`), kind).toBe(true);
    // Every entity a query names is tallied, and an exercised count never exceeds a present one.
    for (const e of run.report.entities) expect(e.exercised).toBeLessThanOrEqual(e.present);
    // Every species of the battles' teams that the builders asked about is tallied by its engine id.
    const species = new Set(run.report.entities.filter((e) => e.id.startsWith("species:")).map((e) => e.id));
    expect(species.size).toBeGreaterThan(8);
  }, 300_000);

  it("the committed full report is fresh: its inputs are this checkout's and its Markdown is its JSON's", () => {
    const report = JSON.parse(readFileSync(join(SELF_REPORT_DIR, "self-report.json"), "utf8")) as DifferentialReport;
    expect(jcs(report.inputs)).toBe(jcs(reportInputs(engine, configs)));
    expect(readFileSync(join(SELF_REPORT_DIR, "self-report.md"), "utf8")).toBe(renderReport(report, SELF_REPORT_TITLE));
    expect(report.corpora.map((c) => c.id)).toEqual(["random", "search-fidelity", "unavailable"]);
    for (const corpus of report.corpora) expect(corpus.differences, corpus.id).toBe(0);
    expect(report.differences).toEqual([]);
  });
});

// ---------------------------------------------------------------- gate 2: the negative controls

let queries: ParsedMechanicsQuery[] = [];
beforeAll(() => {
  queries = collectQueries(engine, configs, corpusBattles(RANDOM_DIR, 1), oracle as never).queries;
}, 120_000);

type Mutate = (batch: readonly ParsedMechanicsQuery[], answers: Settled[]) => Settled[];
const broken = (id: string, mutate: Mutate): Backend<ParsedMechanicsQuery, MechanicsResult> => ({ id, backend_format: 1, backend_commit: "e".repeat(40), run: (batch) => mutate(batch, oracle.run(batch) as Settled[]) as never });
const answerOf = (s: Settled): MechanicsResult | null => (s.ok ? s.result : null);
const damageOf = (s: Settled) => {
  const r = answerOf(s);
  return r?.kind === "computed" && r.answer.kind === "damage" ? r.answer : null;
};
const clone = (s: Settled): Settled => (s.ok ? { ok: true, result: structuredClone(s.result) } : s);
const sample = (pick: (q: ParsedMechanicsQuery, s: Settled) => boolean, n = 40): ParsedMechanicsQuery[] => {
  const out = queries.filter((q) => pick(q, oracle.run([q])[0] as Settled));
  return out.slice(0, n);
};

describe("gate 2: five broken backends, each caught with the right reason, entity and field", () => {
  it("a backend that perturbs one roll of one move", () => {
    const damaging = sample((_, s) => (damageOf(s)?.no_crit[3] ?? 0) > 0, 200);
    const target = damaging[0] as ParsedMechanicsQuery;
    const move = (target.query as { move: string }).move;
    const backend = broken("one-roll", (batch, answers) =>
      answers.map((s, i) => {
        const q = batch[i] as ParsedMechanicsQuery;
        if (q.query.kind !== "damage" || q.query.move !== move || damageOf(s) === null) return s;
        const c = clone(s);
        const rolls = (damageOf(c) as { no_crit: number[] }).no_crit;
        rolls[3] = (rolls[3] as number) + 1;
        return c;
      }),
    );
    const tally = runCorpus(level, oracle, backend, { id: "control", kind: "regression", seed: "0".repeat(64) }, damaging);
    expect(tally.differences.length).toBeGreaterThan(0);
    for (const d of tally.differences) {
      expect(d.reason).toBe("value");
      expect(d.field).toBe("$.answer.no_crit[3]");
      expect(d.entities).toContain(`move:${move}`);
    }
    expect(tally.entities.get(`move:${move}`)?.differing).toBe(tally.differences.length);
    const other = damaging.find((q) => (q.query as { move: string }).move !== move);
    expect(other).toBeDefined();
    expect(tally.differences.every((d) => (damaging[d.position]?.query as { move: string }).move === move)).toBe(true);
  });

  it("a backend that swaps the crit vectors", () => {
    const differ = sample((_, s) => {
      const d = damageOf(s);
      return d !== null && jcs(d.crit) !== jcs(d.no_crit);
    });
    const backend = broken("swap-crit", (_, answers) =>
      answers.map((s) => {
        const c = clone(s);
        const d = damageOf(c);
        if (d !== null) [d.crit, d.no_crit] = [d.no_crit, d.crit];
        return c;
      }),
    );
    const result = compareBatch(level, oracle, backend, differ);
    expect(result.items.every((item) => item.difference?.reason === "value" && /^\$\.answer\.crit\[\d+\]$/.test(item.difference.field))).toBe(true);
    expect(result.items[0]?.difference?.entities).toContain("kind:damage");
  });

  it("a backend that returns unsupported with another reason", () => {
    const unsupported = sample((_, s) => answerOf(s)?.kind === "unsupported");
    expect(unsupported.length).toBeGreaterThan(0);
    const backend = broken("reason", (_, answers) =>
      answers.map((s) => {
        const c = clone(s);
        const r = answerOf(c) as { kind: string; reason: string };
        if (r.kind === "unsupported") r.reason = r.reason === "random_effect" ? "query_kind" : "random_effect";
        return c;
      }),
    );
    const result = compareBatch(level, oracle, backend, unsupported);
    for (const item of result.items) {
      expect(item.difference).toMatchObject({ reason: "value", field: "$.reason" });
      expect(item.exercised.some((e) => e.startsWith("kind:"))).toBe(true);
    }
  });

  it("a backend that answers out of order", () => {
    const batch = queries.slice(0, 32);
    const backend = broken("reversed", (_, answers) => [...answers].reverse());
    const result = compareBatch(level, oracle, backend, batch);
    expect(result.count_mismatch).toBe(false);
    expect(result.items.map((i) => i.difference?.reason)).toEqual(batch.map(() => "answer"));
    expect(result.items[0]?.difference?.entities).toEqual(level.entities(batch[0] as ParsedMechanicsQuery));
  });

  it("a backend that drops a result", () => {
    const batch = queries.slice(0, 16);
    const backend = broken("drop", (_, answers) => answers.filter((_, i) => i !== 5));
    const result = compareBatch(level, oracle, backend, batch);
    expect(result.count_mismatch).toBe(true);
    expect(result.items.slice(0, 5).every((i) => i.difference === null)).toBe(true);
    expect(result.items[5]?.difference).toMatchObject({ reason: "answer", entities: level.entities(batch[5] as ParsedMechanicsQuery) });
    expect(result.items[15]?.difference?.reason).toBe("missing");
  });

  it("the shadow check catches a broken backend and passes the probe", () => {
    const stable = (q: ParsedMechanicsQuery): Json => mechanicsQueryHash(q);
    const backend = broken("bad-first", (batch, answers) => answers.map((s, i) => (i === 0 ? { ok: false, error: new Error(`broken ${batch.length}`) } : s)));
    expect(() => shadowCheck(level, oracle, backend, queries.slice(0, 600), stable, LEVEL_1_RULE)).toThrow(ShadowDivergence);
    expect(shadowCheck(level, oracle, oracle, queries.slice(0, 600), stable, LEVEL_1_RULE)).toHaveLength(256);
  });
});

// ---------------------------------------------------------------- the core agrees with the bridge where it reuses it

describe("the core reuses the bridge's rules", () => {
  it("sizes the level-1 shadow sample as #1195's divergence sample does", () => {
    for (const n of [0, 1, 255, 256, 257, 25_600, 25_601, 99_999, 1_000_000]) expect(shadowSize(n, LEVEL_1_RULE), String(n)).toBe(sampleTarget(n));
  });
  it("writes the same JCS bytes as the bridge on integer values", () => {
    for (const q of queries.slice(0, 50)) expect(jcs(q as unknown as Json)).toBe(bridgeJcs(q as never));
  });
});

// ---------------------------------------------------------------- gate 3, on real data: a fresh corpus, a ledger, routing

describe("a ledger from a fresh corpus drawn by the backend's own seed", () => {
  const backendId = { kind: "probe-second", backend_format: 1, backend_commit: "d".repeat(40) };
  const backend = probeBackend(second, { config: configs.oracle, snapshot: configs.snapshot }) as Backend<ParsedMechanicsQuery, MechanicsResult>;
  it("admits what at least 20 equal exercising items show, routes by it, and is refused at another pin", () => {
    const fresh = freshCorpus(backendId, (seed) => shadowSample(queries.map((q, i) => `${seed}:${i}`), { share_ppm: 500_000, min: 0 }).map((p) => queries[p] as ParsedMechanicsQuery));
    const tally = runCorpus(level, oracle, backend, fresh.spec, fresh.items);
    const ledger = buildLedger({ level: "mechanics-query", backend: backendId, showdown_commit: engine.commit, tallies: [tally] });
    const admitted = ledger.entities.filter((e) => e.admitted).map((e) => e.id);
    expect(admitted).toContain("kind:damage");
    expect(ledger.revoked).toEqual([]);
    const router = admissionRouter(ledger);
    // A query goes to the backend only when every entity it holds is admitted (an unused set move too: ADR-030 rule 2).
    const admittedSet = new Set(admitted);
    for (const q of fresh.items.slice(0, 200)) {
      const entities = level.entities(q);
      const refusal = router.route(entities);
      expect(refusal === null).toBe(entities.every((e) => admittedSet.has(e)) && ledger.pairs.every((p) => p.admitted || !entities.includes(p.a) || !entities.includes(p.b)));
    }
    expect(router.route(["kind:damage"])).toBeNull();
    expect(() => readBackendAdmission(ledger, { showdown_commit: "0".repeat(40) })).toThrow("another pinned engine");
  });
});
