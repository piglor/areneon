// The damage kernel through the differential harness (ADR-030 step 3, #1457): the gate the kernel's admission rests on.
// 1. Regression: every damage query the feature builders issue on the first battles of each committed corpus, answered by the
//    kernel where it covers it, is byte-equal to the pinned probe; the share it covers is reported.
// 2. Negative controls: a kernel that perturbs one roll, or that answers an abstained query with a guess, is caught.
// 3. The ledger: a fresh corpus drawn by the kernel version's own seed builds a ledger both readers accept, which admits nothing
//    below the level's minimum size; the committed admission ledger (when present) is this pin's and admits only what its fresh
//    corpus showed.
// 4. Isolation: no evaluation path imports the kernel.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { admissionHash, type Backend, freshSeed, MIN_FRESH_ITEMS, readBackendAdmission } from "@areneon/differential";
import { abstained, KernelAbstain, loadEngine, type MechanicsResult, type ParsedMechanicsQuery, type Settled } from "@areneon/showdown-bridge";
import { beforeAll, describe, expect, it } from "vitest";
import { COMMITTED_CORPORA } from "../src/differential-mechanics.ts";
import { freshBattle, KERNEL_KIND, type KernelHarness, kernelFresh, kernelHarness, kernelLedger, kernelRegression, poolHash } from "../src/differential-kernel.ts";
import { datasetPoolDetail } from "../src/loop-pool.ts";
import { datasetHash, readDataset } from "../src/dataset-reader.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";

const COMMIT = "a".repeat(40);
const engine = loadEngine();
let h: KernelHarness;
beforeAll(() => {
  h = kernelHarness(engine, COMMIT);
});

describe("1. regression: the kernel equals the probe on every query it covers", () => {
  it("on the first battles of each committed corpus: zero differences, most damage queries covered", () => {
    let queries = 0;
    let covered = 0;
    for (const corpus of COMMITTED_CORPORA) {
      const run = kernelRegression(h, corpus, 2);
      expect(run.tally.differences, corpus.id).toEqual([]);
      expect(run.tally.count_mismatches, corpus.id).toBe(0);
      expect(run.covered).toBe(run.tally.items);
      expect(run.covered + [...run.abstained.values()].reduce((a, b) => a + b, 0)).toBe(run.damage_queries);
      queries += run.damage_queries;
      covered += run.covered;
    }
    expect(queries).toBeGreaterThan(1000);
    // Measured share on the committed random corpus is about 60%; this floor catches a kernel that stopped covering.
    expect(covered * 100).toBeGreaterThan(queries * 35);
  }, 600_000);
});

describe("2. negative controls", () => {
  const wrap = (mutate: (q: ParsedMechanicsQuery, s: Settled) => Settled): KernelHarness => ({
    ...h,
    kernel: { ...h.kernel, run: (batch: readonly ParsedMechanicsQuery[]) => h.kernel.run(batch).map((s, i) => mutate(batch[i] as ParsedMechanicsQuery, s)) },
  });
  it("a kernel that perturbs one roll is caught on the queries that move names", () => {
    let target: string | null = null;
    const broken = wrap((q, s) => {
      if (!s.ok || s.result.kind !== "computed" || s.result.answer.kind !== "damage") return s;
      const move = (q.query as { move: string }).move;
      target ??= move;
      if (move !== target) return s;
      const result = structuredClone(s.result) as Extract<MechanicsResult, { kind: "computed" }>;
      (result.answer as { no_crit: number[] }).no_crit[3] = ((result.answer as { no_crit: number[] }).no_crit[3] as number) + 1;
      return { ok: true, result };
    });
    const run = kernelRegression(broken, COMMITTED_CORPORA[0] as { id: string; dir: string }, 1);
    expect(run.tally.differences.length).toBeGreaterThan(0);
    for (const d of run.tally.differences) {
      expect(d.field).toBe("$.answer.no_crit[3]");
      expect(d.entities).toContain(`move:${target}`);
    }
  }, 300_000);
  it("an abstention is counted and never compared; a kernel that answered in its place would be caught", () => {
    const honest = kernelRegression(h, COMMITTED_CORPORA[0] as { id: string; dir: string }, 1);
    expect(honest.covered).toBeLessThan(honest.damage_queries);
    const guessing = wrap((_q, s) => (abstained(s) ? { ok: false, error: new Error("a guess") } : s));
    const run = kernelRegression(guessing, COMMITTED_CORPORA[0] as { id: string; dir: string }, 1);
    expect(run.tally.differences.length).toBe(honest.damage_queries - honest.covered);
    // The probe answered (`settled`) or refused differently (`refusal`): either way the guess is a difference.
    expect(run.tally.differences.every((d) => d.reason === "settled" || d.reason === "refusal")).toBe(true);
  }, 300_000);
});

describe("3. the ledger", () => {
  const datasetDir = fileURLToPath(new URL("../../../fixtures/teams/gen9championsvgc2026regmc/datasets/fixtures-v1/", import.meta.url));
  it("a small fresh corpus, drawn by the kernel version's own seed, builds a ledger that admits nothing below the minimum size", async () => {
    const dataset = readDataset(datasetDir, loadFixtureInputs(engine, "gen9championsvgc2026regmc", datasetDir));
    const pool = datasetPoolDetail(engine, { format_id: "gen9championsvgc2026regmc", team_dataset: { path: datasetDir, dataset_hash: datasetHash(dataset) }, generation: { excluded_abilities: ["Illusion"] } } as never).pool;
    const battle = await freshBattle(h, pool, 0);
    expect(battle.channels.p1.lines.length).toBeGreaterThan(0);
    const fresh = await kernelFresh(h, pool, 400, 3);
    expect(fresh.tally.spec.kind).toBe("fresh");
    expect(fresh.tally.spec.seed).toBe(freshSeed(h.kernel));
    expect(fresh.tally.spec.pool).toBe(poolHash(pool));
    expect(fresh.tally.differences).toEqual([]);
    expect(fresh.covered).toBeLessThanOrEqual(400);
    const regression = kernelRegression(h, COMMITTED_CORPORA[2] as { id: string; dir: string }, 1);
    const ledger = kernelLedger(h, [fresh, regression]);
    expect(ledger.backend).toEqual({ kind: KERNEL_KIND, backend_format: h.kernel.backend_format, backend_commit: COMMIT });
    expect(ledger.entities.length).toBeGreaterThan(10);
    expect(ledger.entities.every((e) => !e.admitted)).toBe(true);
    expect(fresh.covered).toBeLessThan(MIN_FRESH_ITEMS["mechanics-query"]);
    expect(readBackendAdmission(JSON.parse(JSON.stringify(ledger)), { showdown_commit: engine.commit })).toEqual(ledger);
  }, 600_000);

  it("the committed admission ledger is this pin's, read by the reader, with every admitted entity backed by its fresh corpus", () => {
    const path = fileURLToPath(new URL("../../../config/backends/gen9championsvgc2026regmc.damage-kernel.admission.json", import.meta.url));
    let text: string;
    try {
      text = readFileSync(path, "utf8");
    } catch {
      return; // not committed yet
    }
    const ledger = readBackendAdmission(JSON.parse(text), { showdown_commit: engine.commit });
    expect(ledger.backend.kind).toBe(KERNEL_KIND);
    expect(admissionHash(ledger)).toMatch(/^[0-9a-f]{64}$/);
    const fresh = ledger.corpora.find((c) => c.kind === "fresh");
    expect(fresh?.distinct_items).toBeGreaterThanOrEqual(MIN_FRESH_ITEMS["mechanics-query"]);
    // The ledger is the current kernel version's, and its fresh corpus found no difference (nothing revoked).
    expect(ledger.backend.backend_format).toBe(h.kernel.backend_format);
    expect(ledger.revoked).toEqual([]);
    expect(ledger.entities.some((e) => e.admitted)).toBe(true);
  });
});

describe("4. isolation: no evaluation path reaches the kernel", () => {
  const root = fileURLToPath(new URL("../../../", import.meta.url));
  const files = (dir: string): string[] =>
    readdirSync(dir).flatMap((name) => {
      const path = join(dir, name);
      if (name === "node_modules" || name.startsWith(".")) return [];
      return statSync(path).isDirectory() ? files(path) : /\.(ts|py)$/.test(name) ? [path] : [];
    });
  it("only the kernel's own modules, the bridge index and the differential composition root and its script name it", () => {
    const allowed = [
      "packages/showdown-bridge/src/damage-kernel.ts",
      "packages/showdown-bridge/src/damage-kernel-backend.ts",
      "packages/showdown-bridge/src/index.ts",
      "packages/team-dataset/src/differential-kernel.ts",
      "packages/team-dataset/scripts/kernel-admission.ts",
    ];
    const sources = [...["packages"].flatMap((p) => files(join(root, p))).filter((f) => /\/(src|scripts)\//.test(f)), ...files(join(root, "ml")), ...files(join(root, "scripts"))];
    const naming = sources.filter((f) => /["'][^"']*(damage-kernel|differential-kernel)[^"']*\.ts["']|\bkernelBackend\b|\bKernelAbstain\b/.test(readFileSync(f, "utf8"))).map((f) => f.slice(root.length));
    expect(naming.sort()).toEqual(allowed.sort());
  });
  it("its abstention is a typed refusal the router can count", () => {
    const error = new KernelAbstain("ability:x", "hook");
    expect(abstained({ ok: false, error })).toBe(true);
    expect(abstained({ ok: false, error: new Error("other") })).toBe(false);
    expect(error.message).toContain("ability:x");
    const asBackend: Backend<ParsedMechanicsQuery, MechanicsResult> = h.kernel as Backend<ParsedMechanicsQuery, MechanicsResult>;
    expect(asBackend.backend_commit).toBe(COMMIT);
  });
});
