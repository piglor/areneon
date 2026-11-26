// The damage kernel through the differential harness (ADR-030 step 3, #1457): the gate the kernel's admission rests on.
// 1. Regression: every damage query the feature builders issue on the first battles of each committed corpus, answered by the
//    kernel where it covers it, is byte-equal to the pinned probe; the share it covers is reported.
// 2. Negative controls: a kernel that perturbs one roll, or that answers an abstained query with a guess, is caught.
// 3. The ledger: a fresh corpus drawn by the kernel version's own seed builds a ledger both readers accept, which admits nothing
//    below the level's minimum size; the committed admission ledger (when present) is this pin's and admits only what its fresh
//    corpus showed.
// 4. Isolation: no evaluation path imports the kernel.
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { admissionHash, admissionLines, type Backend, freshSeed, MIN_FRESH_ITEMS, readAdmissionLines, readBackendAdmission } from "@areneon/differential";
import { abstained, goldenTeamVocabulary, KernelAbstain, loadEngine, loadKernelEffects, type MechanicsResult, type ParsedMechanicsQuery, type Settled } from "@areneon/showdown-bridge";
import { beforeAll, describe, expect, it } from "vitest";
import { COMMITTED_CORPORA } from "../src/differential-mechanics.ts";
import { freshBattle, importsKernel, interactionSweep, kernelSpecs, targetedSpecs, KERNEL_ADMISSION_HASH, KERNEL_INPUTS, KERNEL_KIND, KERNEL_LEDGER, kernelExportNames, type KernelHarness, kernelFresh, kernelHarness, kernelLedger, kernelRegression, poolHash } from "../src/differential-kernel.ts";
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

describe("1b. the targeted corpus and the interaction sweep: fully equal or abstained", () => {
  it("every committed targeted case (each with its reason) is equal to the probe or abstained", () => {
    const specs = targetedSpecs();
    expect(specs.length).toBeGreaterThan(10);
    for (const spec of specs) expect(spec.reason.length, spec.id).toBeGreaterThan(20);
    expect(new Set(specs.map((s) => s.id)).size).toBe(specs.length);
    const run = kernelSpecs(h, "targeted", specs);
    expect(run.tally.differences).toEqual([]);
    expect(run.damage_queries).toBe(specs.length);
    // The format-1 classes stay abstained.
    expect(run.abstained.get("ability:magician")).toBe(1);
    expect(run.abstained.get("ability:electricsurge")).toBe(1);
    expect(run.abstained.get("move:shellsidearm")).toBe(1);
  }, 300_000);

  it("the interaction sweep is derived from the record and the engine's data, and a sample of it is fully equal or abstained", () => {
    const sweep = interactionSweep(engine, "gen9championsvgc2026regmc");
    expect(sweep.length).toBeGreaterThan(10_000);
    expect(new Set(sweep.map((s) => s.id)).size).toBe(sweep.length);
    // Every ability, item and condition of the record appears; the unusual moves are there (Body Press, Foul Play, Psyshock).
    const effects = loadKernelEffects("gen9championsvgc2026regmc", engine.commit);
    // An ability only a battle-only forme holds (a Mega's) has no holder a set can name: it cannot be swept.
    const holdable = new Set(goldenTeamVocabulary().roster.filter((r) => r.required_item === undefined).flatMap((r) => r.abilities));
    for (const id of Object.keys(effects.abilities)) if (holdable.has(id)) expect(sweep.some((s) => s.id.includes(`:${id}:`)), id).toBe(true);
    for (const id of Object.keys(effects.items)) expect(sweep.some((s) => s.id.startsWith(`ai:${id}:`)), id).toBe(true);
    for (const id of Object.keys(effects.conditions)) expect(sweep.some((s) => s.id.startsWith(`c:${id}:`)), id).toBe(true);
    for (const move of ["bodypress", "foulplay", "psyshock"]) expect(sweep.some((s) => s.move === move), move).toBe(true);
    // Condition x entity crossings: every weather, terrain and screen meets the stat- and damage-modifying entities on both sides.
    const crossed = Object.entries(effects.conditions).filter(([, c]) => c.kind === "weather_damage" || c.kind === "weather_stat" || c.kind === "terrain" || c.kind === "screen");
    for (const [id] of crossed) expect(sweep.some((s) => s.id.startsWith(`x:${id}:`)), id).toBe(true);
    const crossings = sweep.filter((s) => s.id.startsWith("x:"));
    expect(crossings.length).toBeGreaterThan(1_000);
    for (const s of crossings) expect(s.state.weather ?? s.state.terrain ?? s.state.sides, s.id).toBeTruthy();
    expect(crossings.some((s) => s.id.startsWith("x:snowscape:") && s.defender.ability === "furcoat") || crossings.some((s) => s.id.startsWith("x:snowscape:d:"))).toBe(true);
    expect(crossings.some((s) => s.id.startsWith("x:sunnyday:a:") && s.attacker.item !== undefined)).toBe(true);
    expect(crossings.some((s) => s.id.startsWith("x:reflect:d:") && s.defender.item !== undefined)).toBe(true);
    expect(crossings.some((s) => s.id.startsWith("x:electricterrain:a:"))).toBe(true);
    // The whole sweep runs in the admission flow; CI runs every 20th spec.
    const run = kernelSpecs(h, "interaction-sweep", sweep.filter((_, i) => i % 20 === 0));
    expect(run.tally.differences).toEqual([]);
    expect(run.covered).toBeGreaterThan(run.damage_queries / 2);
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

  it("the committed admission ledger is this kernel version's, frozen at a commit whose kernel inputs equal this checkout's", () => {
    // The committed lines form; no silent pass when it is missing (readFileSync throws).
    const text = readFileSync(fileURLToPath(new URL(`../../../${KERNEL_LEDGER}`, import.meta.url)), "utf8");
    const ledger = readAdmissionLines(text, { showdown_commit: engine.commit });
    expect(admissionLines(ledger)).toBe(text); // the committed bytes are the canonical lines
    expect(admissionHash(ledger)).toBe(KERNEL_ADMISSION_HASH);
    expect(ledger.backend).toEqual({ kind: KERNEL_KIND, backend_format: h.kernel.backend_format, backend_commit: ledger.backend.backend_commit });
    const fresh = ledger.corpora.find((c) => c.kind === "fresh");
    expect(fresh?.seed).toBe(freshSeed(ledger.backend));
    expect(fresh?.distinct_items).toBeGreaterThanOrEqual(MIN_FRESH_ITEMS["mechanics-query"]);
    expect(ledger.revoked).toEqual([]);
    expect(ledger.entities.some((e) => e.admitted)).toBe(true);
    // Every input that changes a kernel answer is as it was at the frozen commit (the typescript CI job has the full history).
    const root = fileURLToPath(new URL("../../../", import.meta.url));
    expect(() => execFileSync("git", ["diff", "--quiet", ledger.backend.backend_commit, "--", ...KERNEL_INPUTS], { cwd: root })).not.toThrow();
  });

  it("names every input that changes a kernel answer", () => {
    for (const path of [
      "packages/showdown-bridge/src/damage-kernel.ts",
      "packages/showdown-bridge/src/mechanics-probe.ts",
      "packages/showdown-bridge/src/mechanics-ko.ts",
      "packages/showdown-bridge/src/mechanics-scenario.ts",
      "packages/showdown-bridge/src/mechanics-query.ts",
      "packages/showdown-bridge/src/mechanics-result.ts",
      "packages/showdown-bridge/src/canonical-json.ts",
      "config/mechanics/gen9championsvgc2026regmc.json",
      "fixtures/showdown/gen9championsvgc2026regmc/mechanics-inventory.json",
      "config/backends/gen9championsvgc2026regmc.damage-kernel.effects.json",
    ]) {
      expect(KERNEL_INPUTS).toContain(path);
    }
  });
});

describe("4. isolation: no evaluation path reaches the kernel", () => {
  const root = fileURLToPath(new URL("../../../", import.meta.url));
  const files = (dir: string): string[] =>
    readdirSync(dir).flatMap((name) => {
      const path = join(dir, name);
      if (name === "node_modules" || name.startsWith(".")) return [];
      return statSync(path).isDirectory() ? files(path) : /\.(ts|py|cjs|js)$/.test(name) ? [path] : [];
    });
  const ALLOWED = [
    "packages/showdown-bridge/src/damage-kernel-backend.ts",
    "packages/showdown-bridge/src/damage-kernel.ts",
    "packages/showdown-bridge/src/damage-kernel-effects.ts",
    "packages/showdown-bridge/src/index.ts",
    "packages/team-dataset/scripts/differential-kernel-admission.ts",
    // Reviewed (#1534): the native probe's admission run and its fork fixture generator reuse the kernel's interaction and
    // targeted sweeps as QUERY generators only (`interactionSweep`, `specQuery`, `targetedSpecs`); they never call the kernel
    // to answer a query, and they are differential tooling, not an evaluation path.
    "packages/team-dataset/scripts/differential-native-admission.ts",
    "packages/team-dataset/scripts/native-probe-cases.ts",
    "packages/team-dataset/src/differential-kernel.ts",
  ];
  const sources = (): string[] => [...files(join(root, "packages")).filter((f) => /\/(src|scripts|web)\//.test(f)), ...files(join(root, "ml")), ...files(join(root, "scripts"))];
  it("only the kernel's modules, the bridge index and the differential composition root and its script import it", () => {
    const names = kernelExportNames(readFileSync(join(root, "packages/showdown-bridge/src/index.ts"), "utf8"));
    expect(names).toEqual(expect.arrayContaining(["prepareQuery", "rollOutcomes", "computeBatch", "kernelData", "loadKernelEffects", "abstained", "kernelBackend", "KernelAbstain"]));
    const found = sources().filter((f) => importsKernel(readFileSync(f, "utf8"), names)).map((f) => f.slice(root.length));
    expect(found.sort()).toEqual([...ALLOWED].sort());
  });
  it("negative control: a planted import, by module or by name through the package index, is flagged", () => {
    const names = kernelExportNames(readFileSync(join(root, "packages/showdown-bridge/src/index.ts"), "utf8"));
    expect(importsKernel('import { computeBatch } from "@areneon/showdown-bridge";', names)).toBe(true);
    expect(importsKernel('import {\n  loadEngine,\n  kernelData as d,\n} from "@areneon/showdown-bridge";', names)).toBe(true);
    expect(importsKernel('import { x } from "../src/damage-kernel-data.ts";', names)).toBe(true);
    expect(importsKernel('const m = await import("@areneon/team-dataset/src/differential-kernel.ts");', names)).toBe(true);
    expect(importsKernel('export { a } from "./damage-kernel.ts";', names)).toBe(true);
    expect(importsKernel('const m = require("../src/damage-kernel-backend.ts");', names)).toBe(true);
    expect(importsKernel('import { loadEngine } from "@areneon/showdown-bridge";', names)).toBe(false);
    expect(importsKernel('const kind = "damage-kernel";', names)).toBe(false);
    // A namespace or whole-module import of the package index: flagged when it reaches a kernel name, through any binding form.
    expect(importsKernel('import * as bridge from "@areneon/showdown-bridge";\nbridge.computeBatch(c);', names)).toBe(true);
    expect(importsKernel('import * as b from "@areneon/showdown-bridge";\nconst k = b["kernelBackend"];', names)).toBe(true);
    expect(importsKernel('import * as b from "@areneon/showdown-bridge";\nb?.prepareQuery(x);', names)).toBe(true);
    expect(importsKernel('const b = await import("@areneon/showdown-bridge");\nb.kernelData(e, f);', names)).toBe(true);
    expect(importsKernel('const b = require("@areneon/showdown-bridge");\nb.abstained(s);', names)).toBe(true);
    expect(importsKernel('const { loadEngine, computeBatch: run } = await import("@areneon/showdown-bridge");', names)).toBe(true);
    expect(importsKernel('export * from "@areneon/showdown-bridge";', names)).toBe(true);
    expect(importsKernel('import * as $b from "@areneon/showdown-bridge";\n$b.computeBatch(c);', names)).toBe(true);
    expect(importsKernel('export * as bridge from "@areneon/showdown-bridge";', names)).toBe(true);
    // Negative controls: a namespace import that reaches no kernel name, and a kernel name on another object.
    expect(importsKernel('import * as bridge from "@areneon/showdown-bridge";\nbridge.loadEngine();', names)).toBe(false);
    expect(importsKernel('import * as other from "./elsewhere.ts";\nother.computeBatch(c);', names)).toBe(false);
    expect(importsKernel('const { loadEngine } = require("@areneon/showdown-bridge");', names)).toBe(false);
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
