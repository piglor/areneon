// The damage kernel's admission run and its speed measurement (ADR-030 step 3, #1457). Read-only on the repository but for
// the files named by --out; never part of a training run.
//
//   node packages/team-dataset/scripts/differential-kernel-admission.ts regression [--limit N] [--json FILE]
//       every committed corpus as a regression corpus: coverage, differences, abstentions by entity.
//   node packages/team-dataset/scripts/differential-kernel-admission.ts fresh --commit SHA --shard K/S --out SHARD.jsonl [--size N]
//       one shard of the fresh corpus: battles K, K+S, ... of the kernel version's fresh draw on the generator pool, each
//       distinct covered damage query with the probe's answer (the shard stops past N/S covered queries).
//   node packages/team-dataset/scripts/differential-kernel-admission.ts ledger --commit SHA --shards A.jsonl,B.jsonl [--size N] --out LEDGER.json --report REPORT.json
//       the fresh corpus from the shards in battle order (its first N distinct covered queries, from battles every shard
//       played), compared kernel against the recorded probe answers; plus the committed corpora as regression corpora; writes
//       the BackendAdmission ledger and a counts report.
//   node packages/team-dataset/scripts/differential-kernel-admission.ts bench --queries FILE [--n N]
//       the kernel and the probe on the same damage queries, one at a time: per-query median and p99.
import { execFileSync } from "node:child_process";
import { closeSync, openSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { addBatch, admissionHash, type Backend, compareBatch, freshCorpusId, freshSeed, newTally } from "@areneon/differential";
import { abstained, jcs, type Json, loadEngine, type MechanicsResult, mechanicsQueryHash, type ParsedMechanicsQuery, probeBackend, type Settled } from "@areneon/showdown-bridge";
import { collectQueries, COMMITTED_CORPORA, remembering } from "../src/differential-mechanics.ts";
import { freshBattle, kernelHarness, kernelLedger, kernelRegression, poolHash } from "../src/differential-kernel.ts";
import { datasetHash, readDataset } from "../src/dataset-reader.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";
import { datasetPoolDetail } from "../src/loop-pool.ts";

const FORMAT = "gen9championsvgc2026regmc";
const KERNEL_FILES = ["packages/showdown-bridge/src/damage-kernel.ts", "packages/showdown-bridge/src/damage-kernel-data.ts", "packages/showdown-bridge/src/damage-kernel-effects.ts", "packages/showdown-bridge/src/damage-kernel-backend.ts", "config/backends/gen9championsvgc2026regmc.damage-kernel.effects.json"];
const args = process.argv.slice(2);
const mode = args[0];
const flag = (name: string, fallback?: string): string => {
  const i = args.indexOf(`--${name}`);
  const value = i < 0 ? fallback : args[i + 1];
  if (value === undefined) {
    console.error(`differential-kernel-admission: --${name} is required`);
    process.exit(2);
  }
  return value;
};
const engine = loadEngine();

/** The kernel's code must be the frozen commit's: the fresh corpus belongs to that version alone (ADR-030 rule 7). */
function frozen(commit: string): string {
  if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error("--commit is a 40-hex commit");
  execFileSync("git", ["diff", "--quiet", commit, "--", ...KERNEL_FILES]);
  return commit;
}

function generatorPool() {
  const dir = new URL("../../../fixtures/teams/gen9championsvgc2026regmc/datasets/fixtures-v1/", import.meta.url).pathname;
  const dataset = readDataset(dir, loadFixtureInputs(engine, FORMAT, dir));
  return datasetPoolDetail(engine, { format_id: FORMAT, team_dataset: { path: dir, dataset_hash: datasetHash(dataset) }, generation: { excluded_abilities: ["Illusion"] } } as never).pool;
}

const settledJson = (s: Settled): unknown => (s.ok ? { ok: true, result: s.result } : { ok: false, error: { name: (s.error as Error).name, message: (s.error as Error).message } });

if (mode === "regression") {
  const h = kernelHarness(engine, "0".repeat(40));
  const limit = args.includes("--limit") ? Number(flag("limit")) : undefined;
  const out: Record<string, unknown> = {};
  for (const corpus of COMMITTED_CORPORA) {
    const run = kernelRegression(h, corpus, limit);
    const top = [...run.abstained].sort((a, b) => b[1] - a[1]).slice(0, 40);
    out[corpus.id] = { battles: run.battles, damage_queries: run.damage_queries, covered: run.covered, differences: run.tally.differences.length, abstained_top: Object.fromEntries(top) };
    console.log(corpus.id, JSON.stringify(out[corpus.id]));
  }
  if (args.includes("--json")) writeFileSync(flag("json"), `${JSON.stringify(out, null, 2)}\n`);
} else if (mode === "fresh") {
  const h = kernelHarness(engine, frozen(flag("commit")));
  const [k, s] = flag("shard").split("/").map(Number) as [number, number];
  const target = Math.ceil((Number(flag("size", "200000")) / s) * 1.05) + 2000;
  const pool = generatorPool();
  const fd = openSync(flag("out"), "w");
  const seen = new Set<string>();
  let covered = 0;
  for (let index = k; covered < target; index += s) {
    const battle = await freshBattle(h, pool, index);
    let damage = 0;
    let mine = 0;
    if (battle !== null) {
      const oracle = remembering(probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));
      for (const query of collectQueries(engine, h.configs, [battle], oracle).queries) {
        if (query.query.kind !== "damage") continue;
        const key = mechanicsQueryHash(query);
        if (seen.has(key)) continue;
        seen.add(key);
        damage++;
        const kernel = h.kernel.run([query])[0] as Settled;
        if (abstained(kernel)) continue;
        writeSync(fd, `${JSON.stringify({ battle: index, query, oracle: settledJson(oracle.run([query])[0] as Settled) })}\n`);
        covered++;
        mine++;
      }
    }
    writeSync(fd, `${JSON.stringify({ battle: index, done: true, damage, covered: mine })}\n`);
    console.error(`shard ${k}/${s} battle ${index}: ${covered} covered`);
  }
  closeSync(fd);
} else if (mode === "ledger") {
  const h = kernelHarness(engine, frozen(flag("commit")));
  const size = Number(flag("size", "200000"));
  const pool = generatorPool();
  type Line = { battle: number; query?: ParsedMechanicsQuery; oracle?: { ok: boolean; result?: MechanicsResult; error?: { name: string; message: string } }; done?: boolean; damage?: number; covered?: number };
  const shards = flag("shards").split(",").map((file) => readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Line));
  // Only battles below the first battle a shard did not finish: the merged corpus is a prefix of the sequential draw.
  const last = Math.min(...shards.map((lines) => Math.max(...lines.filter((l) => l.done).map((l) => l.battle))));
  const lines = shards.flat().filter((l) => l.battle <= last).sort((a, b) => a.battle - b.battle);
  const answers = new Map<string, Settled>();
  const items: ParsedMechanicsQuery[] = [];
  const seen = new Set<string>();
  let damage = 0;
  let battles = 0;
  for (const l of lines) {
    if (l.done) {
      damage += l.damage ?? 0;
      battles += 1;
      continue;
    }
    const q = l.query as ParsedMechanicsQuery;
    const key = mechanicsQueryHash(q);
    if (seen.has(key) || items.length >= size) continue;
    seen.add(key);
    items.push(q);
    const o = l.oracle as NonNullable<Line["oracle"]>;
    answers.set(key, o.ok ? { ok: true, result: o.result as MechanicsResult } : { ok: false, error: Object.assign(new Error(o.error?.message), { name: o.error?.name }) });
  }
  if (items.length < size) throw new Error(`the shards hold ${items.length} distinct covered queries, below ${size}: run them longer`);
  const recorded: Backend<ParsedMechanicsQuery, MechanicsResult> = { id: "showdown-probe-recorded", backend_format: 2, backend_commit: engine.commit, run: (batch) => batch.map((q) => answers.get(mechanicsQueryHash(q)) as Settled) as never };
  const tally = newTally({ id: freshCorpusId(h.kernel), kind: "fresh", seed: freshSeed(h.kernel), pool: poolHash(pool) });
  for (let first = 0; first < items.length; first += 256) addBatch(tally, compareBatch(h.level, recorded, h.kernel as Backend<ParsedMechanicsQuery, MechanicsResult>, items.slice(first, first + 256), first));
  const regressions = COMMITTED_CORPORA.map((c) => kernelRegression(h, c));
  const ledger = kernelLedger(h, [{ tally, damage_queries: damage, covered: items.length, abstained: new Map(), battles }, ...regressions]);
  writeFileSync(flag("out"), `${JSON.stringify(ledger, null, 2)}\n`);
  const admitted = ledger.entities.filter((e) => e.admitted);
  const report = {
    admission_hash: admissionHash(ledger),
    fresh: { battles, damage_queries: damage, covered: items.length, differences: tally.differences.length, count_mismatches: tally.count_mismatches },
    regression: Object.fromEntries(regressions.map((r) => [r.tally.spec.id, { damage_queries: r.damage_queries, covered: r.covered, differences: r.tally.differences.length }])),
    entities: { listed: ledger.entities.length, admitted: admitted.length, by_kind: Object.fromEntries([...new Set(admitted.map((e) => e.id.split(":")[0] as string))].map((k) => [k, admitted.filter((e) => e.id.startsWith(`${k}:`)).length])) },
    pairs: { listed: ledger.pairs.length, admitted: ledger.pairs.filter((p) => p.admitted).length },
    revoked: ledger.revoked.length,
  };
  writeFileSync(flag("report"), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report));
} else if (mode === "bench") {
  const h = kernelHarness(engine, "0".repeat(40));
  const n = Number(flag("n", "3000"));
  const lines = readFileSync(flag("queries"), "utf8").split("\n").filter(Boolean).slice(0, n);
  const every = lines.map((l) => (JSON.parse(l) as { query: ParsedMechanicsQuery }).query);
  const queries = every.filter((q) => !abstained(h.kernel.run([q])[0] as Settled));
  const probe = probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot });
  const time = (list: readonly ParsedMechanicsQuery[], run: (q: ParsedMechanicsQuery) => unknown): number[] =>
    list.map((q) => {
      const t = performance.now();
      run(q);
      return performance.now() - t;
    });
  // One warm pass each (caches, JIT), then the measured pass; every answer is also checked equal.
  for (const q of every.slice(0, 200)) {
    probe.run([q]);
    h.kernel.run([q]);
  }
  const k = time(queries, (q) => h.kernel.run([q]));
  const p = time(queries, (q) => probe.run([q]));
  // The probe's time on every damage query, and the share of it spent on the queries the kernel covers (time-weighted coverage).
  const all = time(every, (q) => probe.run([q]));
  const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
  let differing = 0;
  for (const q of queries) if (jcs((h.kernel.run([q])[0] as unknown as { result: Json }).result) !== jcs((probe.run([q])[0] as unknown as { result: Json }).result)) differing++;
  const stats = (xs: number[]) => {
    const s = [...xs].sort((a, b) => a - b);
    const at = (f: number) => s[Math.min(s.length - 1, Math.floor(f * s.length))] as number;
    return { median_ms: Number(at(0.5).toFixed(4)), p99_ms: Number(at(0.99).toFixed(4)), mean_ms: Number((xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(4)) };
  };
  console.log(JSON.stringify({ damage_queries: every.length, covered: queries.length, differing, kernel: stats(k), probe: stats(p), probe_time_covered_share_ppm: Math.round((sum(p) / sum(all)) * 1e6), speedup_mean: Number((sum(p) / sum(k)).toFixed(2)) }));
} else {
  console.error("differential-kernel-admission: regression | fresh | ledger | bench");
  process.exit(2);
}
