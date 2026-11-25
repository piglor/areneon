// The native probe through the differential harness (#1534): the gate its admission rests on.
// 1. Regression: every query the `probe-damage/2` builder issues on the first battles of each committed corpus, answered by the native
//    backend where it answers, is byte-equal to the pinned probe (this needs the built `ninjask-probe`: ARENEON_NINJASK_PROBE and
//    ARENEON_NINJASK_DATA, `make ninjask`; skipped, naming that, without it, unless ARENEON_REQUIRE_NINJASK=1).
// 2. The sweeps (the move sweep, the speed sweep, the interaction sweep and the targeted corpus) are fully equal or abstained.
// 3. Negative controls: a backend that perturbs one roll, swaps the crit vectors or answers another query is caught.
// 4. The hand-worked fixtures of the repository (12 damage cases, 20 speed cases, the KO cases) are answered natively with the
//    hand-worked numbers.
// 5. Isolation: no evaluation path imports the native backend.
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Backend, freshSeed, MIN_FRESH_ITEMS, readBackendAdmission, compareBatch, newTally, addBatch } from "@areneon/differential";
import { loadEngine, type MechanicsResult, mechanicsQueryHash, nativeAbstained, type ParsedMechanicsQuery, probeBackend, readMechanicsQuery, type Settled } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { COMMITTED_CORPORA, remembering } from "../src/differential-mechanics.ts";
import { datasetHash, readDataset } from "../src/dataset-reader.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";
import { datasetPoolDetail } from "../src/loop-pool.ts";
import { interactionSweep, specQuery, targetedSpecs } from "../src/differential-kernel.ts";
import { freshBattle, moveSweep, NATIVE_KIND, refusedBuild, type NativeHarness, nativeHarness, nativeLedger, nativeQueries, nativeRegression, speedSweep } from "../src/differential-native.ts";

const FORMAT = "gen9championsvgc2026regmc";
const PROBE = process.env.ARENEON_NINJASK_PROBE;
const DATA = process.env.ARENEON_NINJASK_DATA;
const REQUIRED = process.env.ARENEON_REQUIRE_NINJASK === "1";
const BUILT = PROBE !== undefined && DATA !== undefined;
const withBinary = it.skipIf(!BUILT && !REQUIRED);
const NOT_BUILT = " [needs the built ninjask-probe: make ninjask, ARENEON_NINJASK_PROBE and ARENEON_NINJASK_DATA]";
const COMMIT = "a".repeat(40);
const engine = loadEngine();
let h: NativeHarness | null = null;
beforeAll(() => {
  if (BUILT) h = nativeHarness(engine, COMMIT, { bin: PROBE as string, dex: `${DATA as string}/format/${FORMAT}.json` });
});
afterAll(() => h?.native.close());
const harness = (): NativeHarness => {
  if (h === null) throw new Error(`ARENEON_REQUIRE_NINJASK=1, but the native probe is not built${NOT_BUILT}`);
  return h;
};

describe(`1. regression: the native backend equals the probe on every query it answers${BUILT ? "" : NOT_BUILT}`, () => {
  withBinary("on the first battle of each committed corpus: zero differences, almost every query answered", () => {
    let queries = 0;
    let answered = 0;
    for (const corpus of COMMITTED_CORPORA) {
      const run = nativeRegression(harness(), corpus, 1);
      expect(run.tally.differences, corpus.id).toEqual([]);
      expect(run.tally.count_mismatches, corpus.id).toBe(0);
      const total = Object.values(run.queries).reduce((a, b) => a + b, 0);
      const done = Object.values(run.answered).reduce((a, b) => a + b, 0);
      expect(done).toBe(run.tally.items);
      expect(done + [...run.abstained.values()].reduce((a, b) => a + b, 0)).toBe(total);
      queries += total;
      answered += done;
    }
    expect(queries).toBeGreaterThan(300);
    // Measured: 99.5% on the committed random corpus; this floor catches a layer that stopped answering.
    expect(answered * 100).toBeGreaterThan(queries * 90);
  }, 600_000);
});

describe(`2. the sweeps: fully equal or abstained${BUILT ? "" : NOT_BUILT}`, () => {
  withBinary("the targeted corpus, a stride of the interaction sweep, the move sweep and a speed sweep", () => {
    const targeted = targetedSpecs().map((spec) => specQuery(engine, spec, FORMAT));
    const interaction = interactionSweep(engine, FORMAT)
      .filter((_, i) => i % 40 === 0)
      .map((spec) => specQuery(engine, spec, FORMAT));
    const runs = [nativeQueries(harness(), "targeted", targeted), nativeQueries(harness(), "interaction", interaction), nativeQueries(harness(), "moves", moveSweep(engine, FORMAT).filter((_, i) => i % 25 === 0)), nativeQueries(harness(), "speed", speedSweep(engine, FORMAT, 600, 7))];
    for (const run of runs) {
      expect(run.tally.differences, run.tally.spec.id).toEqual([]);
      expect(Object.values(run.answered).reduce((a, b) => a + b, 0), run.tally.spec.id).toBeGreaterThan(5);
    }
  }, 600_000);

  it("the sweeps are functions of the pinned engine and their seed alone", () => {
    const hashes = (qs: ParsedMechanicsQuery[]) => qs.map((q) => mechanicsQueryHash(q));
    expect(hashes(speedSweep(engine, FORMAT, 40, 5))).toEqual(hashes(speedSweep(engine, FORMAT, 40, 5)));
    expect(hashes(speedSweep(engine, FORMAT, 40, 5))).not.toEqual(hashes(speedSweep(engine, FORMAT, 40, 6)));
    const moves = moveSweep(engine, FORMAT);
    expect(moves.length).toBeGreaterThan(10_000);
    expect(new Set(hashes(moves)).size).toBe(moves.length);
    expect(speedSweep(engine, FORMAT, 40, 5).every((q) => q.query.kind === "speed_order")).toBe(true);
    // Enough queries to fill the set pool (more than 400 sets of each kind).
    expect(speedSweep(engine, FORMAT, 500, 9)).toHaveLength(500);
  }, 600_000);
});

describe(`3. negative controls: the harness catches a wrong backend${BUILT ? "" : NOT_BUILT}`, () => {
  const queries = (): ParsedMechanicsQuery[] => nativeAnswered(moveSweep(engine, FORMAT).filter((_, i) => i % 400 === 0)).slice(0, 12);
  const nativeAnswered = (list: ParsedMechanicsQuery[]) => list.filter((q) => !nativeAbstained(harness().native.run([q])[0] as Settled) && (harness().native.run([q])[0] as { ok: boolean; result: MechanicsResult }).result.kind === "computed");
  const oracle = () => remembering(probeBackend(engine, { config: harness().configs.oracle, snapshot: harness().configs.snapshot })) as Backend<ParsedMechanicsQuery, MechanicsResult>;
  const wrap = (change: (result: MechanicsResult, query: ParsedMechanicsQuery, index: number) => MechanicsResult): Backend<ParsedMechanicsQuery, MechanicsResult> => ({
    id: "perturbed",
    backend_format: 1,
    backend_commit: COMMIT,
    run: (batch) => harness().native.run(batch).map((s, i) => (s.ok ? { ok: true, result: change(structuredClone(s.result), batch[i] as ParsedMechanicsQuery, i) } : s)) as never,
  });
  const compare = (backend: Backend<ParsedMechanicsQuery, MechanicsResult>, list: ParsedMechanicsQuery[]) => {
    const tally = newTally({ id: "control", kind: "regression", seed: "a".repeat(64), pool: "b".repeat(64) });
    addBatch(tally, compareBatch(harness().level, oracle(), backend, list, 0));
    return tally;
  };

  withBinary("a perturbed roll, swapped crit vectors and a wrong reason are caught where they are", () => {
    const list = queries();
    expect(list.length).toBeGreaterThan(5);
    expect(compare(harness().native as Backend<ParsedMechanicsQuery, MechanicsResult>, list).differences).toEqual([]);
    const roll = compare(wrap((r) => ((r as { answer: { no_crit: number[] } }).answer.no_crit[3] = ((r as { answer: { no_crit: number[] } }).answer.no_crit[3] as number) + 1, r)), list);
    expect(roll.differences.length).toBe(list.length);
    expect(roll.differences.every((d) => d.field === "$.answer.no_crit[3]")).toBe(true);
    const swapped = compare(wrap((r) => {
      const a = (r as { answer: { no_crit: number[]; crit: number[] } }).answer;
      [a.no_crit, a.crit] = [a.crit, a.no_crit];
      return r;
    }), list);
    expect(swapped.differences.length).toBeGreaterThan(0);
    const reason = compare(wrap((r) => ({ ...r, kind: "unsupported", reason: "random_effect", position: null }) as never), list);
    expect(reason.differences.every((d) => d.reason === "value")).toBe(true);
    expect(reason.differences.length).toBe(list.length);
  });

  withBinary("a backend that answers out of order is caught", () => {
    const list = queries();
    const reordered: Backend<ParsedMechanicsQuery, MechanicsResult> = { id: "reordered", backend_format: 1, backend_commit: COMMIT, run: (batch) => [...harness().native.run(batch)].reverse() as never };
    expect(compare(reordered, list).differences.some((d) => d.reason === "answer")).toBe(true);
  });
});

describe(`4. the repository's hand-worked fixtures${BUILT ? "" : NOT_BUILT}`, () => {
  const root = fileURLToPath(new URL(`../../../fixtures/mechanics/${FORMAT}/`, import.meta.url));
  withBinary("the twelve damage cases: the native backend gives the hand-worked vectors, crit rule, HP and KO counts", () => {
    const fixture = JSON.parse(readFileSync(join(root, "damage/cases.json"), "utf8")) as { cases: { id: string; query: unknown; expected: Record<string, unknown> }[] };
    expect(fixture.cases.length).toBeGreaterThanOrEqual(12);
    for (const c of fixture.cases) {
      const [settled] = harness().native.run([readMechanicsQuery(c.query)]);
      if (!settled?.ok) throw new Error(`${c.id}: ${String((settled as { error: unknown }).error)}`);
      const result = settled.result as unknown as { kind: string; answer: Record<string, unknown> };
      expect(result.kind, c.id).toBe("computed");
      for (const key of Object.keys(c.expected)) expect(result.answer[key], `${c.id}.${key}`).toEqual(c.expected[key]);
    }
  });
});

describe("6. the harness around a stand-in process (always runs: the stand-in answers `stats` only, so differences are expected and counted)", () => {
  const FAKE = fileURLToPath(new URL("../../showdown-bridge/test/support/fake-native-probe.mjs", import.meta.url));
  it("counts what it compares, abstains by entity, builds a ledger and plays a fresh battle", async () => {
    process.env.FAKE_MODE = "plain";
    const fake = nativeHarness(engine, COMMIT, { bin: FAKE, dex: "unused" });
    try {
      const run = nativeRegression(fake, COMMITTED_CORPORA[0] as (typeof COMMITTED_CORPORA)[number], 1);
      const total = Object.values(run.queries).reduce((a, b) => a + b, 0);
      expect(total).toBeGreaterThan(100);
      expect(Object.values(run.answered).reduce((a, b) => a + b, 0)).toBe(run.tally.items);
      expect([...run.abstained.get("move:fake") === undefined ? [] : [1]]).toEqual([1]);
      const some = moveSweep(engine, FORMAT).slice(0, 20);
      // A query given twice is compared once.
      const swept = nativeQueries(fake, "sweep", [...some, ...some, ...speedSweep(engine, FORMAT, 10, 3)]);
      expect(swept.queries.damage).toBe(20);
      expect(() => refusedBuild(new TypeError("a defect"))).toThrow(TypeError);
      const ledger = nativeLedger(fake, [run, swept]);
      expect(ledger.entities.some((e) => e.admitted)).toBe(false);
      const dir = new URL("../../../fixtures/teams/gen9championsvgc2026regmc/datasets/fixtures-v1/", import.meta.url).pathname;
      const pool = datasetPoolDetail(engine, { format_id: FORMAT, team_dataset: { path: dir, dataset_hash: datasetHash(readDataset(dir, loadFixtureInputs(engine, FORMAT, dir))) }, generation: { excluded_abilities: ["Illusion"] } } as never).pool;
      const battle = await freshBattle(fake, pool, 0, 30);
      expect(battle.provenance).toBe("fresh");
      expect(battle.channels.p1.lines.length).toBeGreaterThan(10);
    } finally {
      fake.native.close();
      delete process.env.FAKE_MODE;
    }
  }, 600_000);
});

describe("5. the ledger and the isolation of the native backend", () => {
  withBinary(`a corpus drawn by the backend version's own seed builds a ledger the reader accepts, which admits nothing below the level's minimum${BUILT ? "" : NOT_BUILT}`, () => {
    const run = nativeQueries(harness(), "small", moveSweep(engine, FORMAT).filter((_, i) => i % 300 === 0));
    const ledger = nativeLedger(harness(), [run]);
    expect(ledger.backend.kind).toBe(NATIVE_KIND);
    expect(readBackendAdmission(JSON.parse(JSON.stringify(ledger)), { showdown_commit: engine.commit })).toEqual(ledger);
    expect(ledger.entities.some((e) => e.admitted)).toBe(false);
    expect(MIN_FRESH_ITEMS["mechanics-query"]).toBe(200_000);
    expect(freshSeed(harness().native)).toMatch(/^[0-9a-f]{64}$/);
  });

  const root = fileURLToPath(new URL("../../../", import.meta.url));
  const NAMES = ["nativeProbeBackend", "NativeProbeProcess", "nativeCase", "NativeAbstain", "nativeAbstained"];
  /** Whether a source reaches the native backend: a module specifier of one of its modules or of the composition root, or one of its names off the bridge's index. */
  const reaches = (source: string): boolean => {
    if (/(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)["'][^"']*(native-probe-|differential-native)[^"']*["']/.test(source)) return true;
    for (const m of source.matchAll(/import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']@areneon\/showdown-bridge["']/g)) {
      const names = (m[1] as string).split(",").map((p) => p.replace(/^\s*type\s+/, "").trim().split(/\s+as\s+/)[0]?.trim());
      if (names.some((n) => n !== undefined && NAMES.includes(n))) return true;
    }
    return false;
  };
  const tracked = (): string[] => execFileSync("git", ["ls-files", "-z", "--", "packages", "scripts", "ml", "services", "crates"], { cwd: root, encoding: "utf8" }).split("\0").filter((f) => /\.(ts|js|mjs|cjs)$/.test(f) && !f.includes("node_modules"));
  const ALLOWED = (f: string): boolean => /^packages\/showdown-bridge\/(src|test)\/(native-probe-|index\.ts|support\/fake-native)/.test(f) || /^packages\/team-dataset\/(src\/differential-native|scripts\/(differential-native|native-probe-cases)|test\/differential-native)/.test(f);

  it("no source outside the native backend's own files, its composition root and their tests reaches it", () => {
    const found = tracked().filter((f) => !ALLOWED(f) && statSync(join(root, f), { throwIfNoEntry: false })?.isFile() === true && reaches(readFileSync(join(root, f), "utf8")));
    expect(found).toEqual([]);
  });

  it("the scan flags every way in and passes an unrelated import", () => {
    expect(reaches('import { x } from "../src/native-probe-backend.ts";')).toBe(true);
    expect(reaches('import { nativeProbeBackend } from "@areneon/showdown-bridge";')).toBe(true);
    expect(reaches('import {\n  loadEngine,\n  nativeAbstained as a,\n} from "@areneon/showdown-bridge";')).toBe(true);
    expect(reaches('const m = await import("@areneon/team-dataset/src/differential-native.ts");')).toBe(true);
    expect(reaches('import { loadEngine } from "@areneon/showdown-bridge";')).toBe(false);
    expect(reaches('const kind = "native-probe";')).toBe(false);
  });

  it("is named by no image, compose file or workflow but the `ninjask` job of the CI workflow (a source scan)", () => {
    const files = readdirSync(join(root, ".github/workflows")).map((f) => join(root, ".github/workflows", f));
    // The one job that builds the fork's probe process runs the native backend's tests; every other job and workflow must not name it.
    const withoutNinjaskJob = (text: string): string => text.replace(/\n {2}ninjask:\n[\s\S]*?(?=\n {2}[a-z][a-z-]*:\n)/, "");
    for (const f of files) expect(withoutNinjaskJob(readFileSync(f, "utf8")), f).not.toMatch(/differential-native|native-probe-/);
    const ci = readFileSync(join(root, ".github/workflows/ci.yml"), "utf8");
    expect(ci.length - withoutNinjaskJob(ci).length, "the ninjask job must be found").toBeGreaterThan(500);
    expect(ci).toMatch(/ARENEON_REQUIRE_NINJASK: "1"[\s\S]*native-probe-backend\.test\.ts/);
  });
});
