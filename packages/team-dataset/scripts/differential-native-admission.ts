// The native probe's admission run and its speed measurement (#1534, the Ninjask track of #1530). Read-only on the repository but for the
// files named by --out; never part of a training run. The binary and the format export come from the environment
// (ARENEON_NINJASK_PROBE, ARENEON_NINJASK_DATA: the fork's `ninjask-probe` and the export `make ninjask` writes) or --bin and --dex, so
// no path of the vendored fork is named here.
//
//   node packages/team-dataset/scripts/differential-native-admission.ts regression [--limit N] [--json FILE]
//       every committed corpus as a regression corpus: coverage, differences, abstentions by entity.
//   node packages/team-dataset/scripts/differential-native-admission.ts sweeps [--speed N] [--seed S] [--json FILE]
//       the move sweep, the speed sweep, the interaction sweep and the targeted corpus.
//   node packages/team-dataset/scripts/differential-native-admission.ts fresh --fork-commit SHA --shard K/S --out SHARD.jsonl [--size N]
//       one shard of the fresh corpus: battles K, K+S, ... of the backend version's fresh draw on the generator pool, each distinct
//       answered query with the probe's answer.
//   node packages/team-dataset/scripts/differential-native-admission.ts ledger --fork-commit SHA --shards A.jsonl,B.jsonl [--size N] --out LEDGER.jsonl --report REPORT.json
//       the fresh corpus from the shards in battle order, the native backend compared with the recorded probe answers; plus the
//       committed corpora and sweeps as regression corpora; writes the BackendAdmission ledger (lines form) and a counts report.
//   node packages/team-dataset/scripts/differential-native-admission.ts bench [--n N]
//       per-query microseconds: the probe, the native backend end to end and the native layer alone, on the same queries.
import { closeSync, openSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { addBatch, admissionHash, admissionLines, type Backend, compareBatch, newTally } from "@areneon/differential";
import { jcs, type Json, loadEngine, type MechanicsResult, mechanicsQueryHash, nativeAbstained, type ParsedMechanicsQuery, probeBackend, readMechanicsQuery, type Settled } from "@areneon/showdown-bridge";
import { datasetHash, readDataset } from "../src/dataset-reader.ts";
import { freshBattle, freshCorpusId, freshSeed, moveSweep, nativeHarness, nativeLedger, type NativeCorpusRun, nativeQueries, nativeRegression, builderQueries, poolHash, speedSweep, COMMITTED_CORPORA } from "../src/differential-native.ts";
import { interactionSweep, specQuery, targetedSpecs } from "../src/differential-kernel.ts";
import { remembering } from "../src/differential-mechanics.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";
import { datasetPoolDetail } from "../src/loop-pool.ts";

const FORMAT = "gen9championsvgc2026regmc";
const args = process.argv.slice(2);
const mode = args[0];
const flag = (name: string, fallback?: string): string => {
  const i = args.indexOf(`--${name}`);
  const value = i < 0 ? fallback : args[i + 1];
  if (value === undefined) {
    console.error(`differential-native-admission: --${name} is required`);
    process.exit(2);
  }
  return value;
};
const command = { bin: flag("bin", process.env.ARENEON_NINJASK_PROBE), dex: flag("dex", process.env.ARENEON_NINJASK_DATA === undefined ? undefined : `${process.env.ARENEON_NINJASK_DATA}/format/${FORMAT}.json`) };
const engine = loadEngine();

function generatorPool() {
  const dir = new URL("../../../fixtures/teams/gen9championsvgc2026regmc/datasets/fixtures-v1/", import.meta.url).pathname;
  const dataset = readDataset(dir, loadFixtureInputs(engine, FORMAT, dir));
  return datasetPoolDetail(engine, { format_id: FORMAT, team_dataset: { path: dir, dataset_hash: datasetHash(dataset) }, generation: { excluded_abilities: ["Illusion"] } } as never).pool;
}

const settledJson = (s: Settled): unknown => (s.ok ? { ok: true, result: s.result } : { ok: false, error: { name: (s.error as Error).name, message: (s.error as Error).message } });
const summary = (r: NativeCorpusRun) => ({ battles: r.battles, queries: r.queries, answered: r.answered, differences: r.tally.differences.length, abstained_top: Object.fromEntries([...r.abstained].sort((a, b) => b[1] - a[1]).slice(0, 25)) });

function sweeps(h: ReturnType<typeof nativeHarness>, speed: number, seed: number): NativeCorpusRun[] {
  const targeted = targetedSpecs().map((spec) => specQuery(engine, spec, FORMAT));
  const interaction = interactionSweep(engine, FORMAT).flatMap((spec) => {
    try {
      return [specQuery(engine, spec, FORMAT)];
    } catch {
      return [];
    }
  });
  return [nativeQueries(h, "targeted", targeted), nativeQueries(h, "interaction-sweep", interaction), nativeQueries(h, "move-sweep", moveSweep(engine, FORMAT)), nativeQueries(h, `speed-sweep-${seed}`, speedSweep(engine, FORMAT, speed, seed))];
}

if (mode === "regression") {
  const h = nativeHarness(engine, "0".repeat(40), command);
  const limit = args.includes("--limit") ? Number(flag("limit")) : undefined;
  const out: Record<string, unknown> = {};
  for (const corpus of COMMITTED_CORPORA) {
    const run = nativeRegression(h, corpus, limit);
    out[corpus.id] = summary(run);
    console.log(corpus.id, JSON.stringify(out[corpus.id]));
  }
  if (args.includes("--json")) writeFileSync(flag("json"), `${JSON.stringify(out, null, 2)}\n`);
  h.native.close();
} else if (mode === "sweeps") {
  const h = nativeHarness(engine, "0".repeat(40), command);
  const out: Record<string, unknown> = {};
  for (const run of sweeps(h, Number(flag("speed", "20000")), Number(flag("seed", "20261125")))) {
    out[run.tally.spec.id] = summary(run);
    console.log(run.tally.spec.id, JSON.stringify(out[run.tally.spec.id]));
  }
  if (args.includes("--json")) writeFileSync(flag("json"), `${JSON.stringify(out, null, 2)}\n`);
  h.native.close();
} else if (mode === "fresh") {
  const h = nativeHarness(engine, flag("fork-commit"), command);
  const [k, s] = flag("shard").split("/").map(Number) as [number, number];
  const target = Math.ceil((Number(flag("size", "200000")) / s) * 1.05) + 2000;
  const pool = generatorPool();
  const fd = openSync(flag("out"), "w");
  const seen = new Set<string>();
  let covered = 0;
  for (let index = k; covered < target; index += s) {
    const battle = await freshBattle(h, pool, index);
    let issued = 0;
    let mine = 0;
    const oracle = remembering(probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));
    for (const query of builderQueries(h, [battle], oracle)) {
      const key = mechanicsQueryHash(query);
      if (seen.has(key)) continue;
      seen.add(key);
      issued++;
      if (nativeAbstained(h.native.run([query])[0] as Settled)) continue;
      writeSync(fd, `${JSON.stringify({ battle: index, query, oracle: settledJson(oracle.run([query])[0] as Settled) })}\n`);
      covered++;
      mine++;
    }
    writeSync(fd, `${JSON.stringify({ battle: index, done: true, queries: issued, covered: mine })}\n`);
    console.error(`shard ${k}/${s} battle ${index}: ${covered} covered`);
  }
  closeSync(fd);
  h.native.close();
} else if (mode === "ledger") {
  const h = nativeHarness(engine, flag("fork-commit"), command);
  const size = Number(flag("size", "200000"));
  const pool = generatorPool();
  type Line = { battle: number; query?: ParsedMechanicsQuery; oracle?: { ok: boolean; result?: MechanicsResult; error?: { name: string; message: string } }; done?: boolean; queries?: number; covered?: number };
  const shards = flag("shards").split(",").map((file) => readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as Line));
  const last = Math.min(...shards.map((lines) => Math.max(...lines.filter((l) => l.done).map((l) => l.battle))));
  const lines = shards.flat().filter((l) => l.battle <= last).sort((a, b) => a.battle - b.battle);
  const answers = new Map<string, Settled>();
  const items: ParsedMechanicsQuery[] = [];
  const seen = new Set<string>();
  let issued = 0;
  let battles = 0;
  for (const l of lines) {
    if (l.done) {
      issued += l.queries ?? 0;
      battles += 1;
      continue;
    }
    const q = readMechanicsQuery(l.query);
    const key = mechanicsQueryHash(q);
    if (seen.has(key) || items.length >= size) continue;
    seen.add(key);
    items.push(q);
    const o = l.oracle as NonNullable<Line["oracle"]>;
    answers.set(key, o.ok ? { ok: true, result: o.result as MechanicsResult } : { ok: false, error: Object.assign(new Error(o.error?.message), { name: o.error?.name }) });
  }
  if (items.length < size) throw new Error(`the shards hold ${items.length} distinct answered queries, below ${size}: run them longer`);
  const recorded: Backend<ParsedMechanicsQuery, MechanicsResult> = { id: "showdown-probe-recorded", backend_format: 2, backend_commit: engine.commit, run: (batch) => batch.map((q) => answers.get(mechanicsQueryHash(q)) as Settled) as never };
  const tally = newTally({ id: freshCorpusId(h.native), kind: "fresh", seed: freshSeed(h.native), pool: poolHash(pool) });
  for (let first = 0; first < items.length; first += 256) addBatch(tally, compareBatch(h.level, recorded, h.native as Backend<ParsedMechanicsQuery, MechanicsResult>, items.slice(first, first + 256), first));
  const regressions = [...COMMITTED_CORPORA.map((c) => nativeRegression(h, c)), ...sweeps(h, Number(flag("speed", "20000")), Number(flag("seed", "20261125")))];
  for (const run of regressions) if (run.tally.differences.length > 0) throw new Error(`${run.tally.spec.id}: ${run.tally.differences.length} differences; fix the layer and add a regression test`);
  const ledger = nativeLedger(h, [{ tally, queries: {}, answered: {}, abstained: new Map(), battles }, ...regressions]);
  writeFileSync(flag("out"), admissionLines(ledger));
  const admitted = ledger.entities.filter((e) => e.admitted);
  const report = {
    admission_hash: admissionHash(ledger),
    fresh: { battles, queries: issued, answered: items.length, differences: tally.differences.length, count_mismatches: tally.count_mismatches },
    regression: Object.fromEntries(regressions.map((r) => [r.tally.spec.id, summary(r)])),
    entities: { listed: ledger.entities.length, admitted: admitted.length, by_kind: Object.fromEntries([...new Set(admitted.map((e) => e.id.split(":")[0] as string))].map((kind) => [kind, admitted.filter((e) => e.id.startsWith(`${kind}:`)).length])) },
    pairs: { listed: ledger.pairs.length, admitted: ledger.pairs.filter((p) => p.admitted).length },
    revoked: ledger.revoked.length,
  };
  writeFileSync(flag("report"), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report));
  h.native.close();
} else if (mode === "bench") {
  const h = nativeHarness(engine, "0".repeat(40), command);
  const n = Number(flag("n", "3000"));
  const battles = (await import("../src/differential-mechanics.ts")).corpusBattles(COMMITTED_CORPORA[0]?.dir as string).slice(0, 8);
  const probe = probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot });
  const every = builderQueries(h, battles, remembering(probe)).filter((q) => q.query.kind !== "stats").slice(0, n);
  const answered = every.filter((q) => !nativeAbstained(h.native.run([q])[0] as Settled));
  const time = (list: readonly ParsedMechanicsQuery[], run: (q: ParsedMechanicsQuery) => unknown): number[] =>
    list.map((q) => {
      const t = performance.now();
      run(q);
      return performance.now() - t;
    });
  for (const q of every.slice(0, 200)) {
    probe.run([q]);
    h.native.run([q]);
  }
  const single = time(answered, (q) => h.native.run([q]));
  const p = time(answered, (q) => probe.run([q]));
  // Batches of 256: the way the batch seam calls it.
  const t0 = performance.now();
  for (let first = 0; first < answered.length; first += 256) h.native.run(answered.slice(first, first + 256));
  const batched = ((performance.now() - t0) / answered.length) * 1000;
  let differing = 0;
  for (const q of answered) if (jcs((h.native.run([q])[0] as unknown as { result: Json }).result) !== jcs((probe.run([q])[0] as unknown as { result: Json }).result)) differing++;
  const stats = (xs: number[]) => {
    const s = [...xs].sort((a, b) => a - b);
    const at = (f: number) => s[Math.min(s.length - 1, Math.floor(f * s.length))] as number;
    return { median_us: Math.round(at(0.5) * 1000), p99_us: Math.round(at(0.99) * 1000), mean_us: Math.round((xs.reduce((a, b) => a + b, 0) / xs.length) * 1000) };
  };
  console.log(JSON.stringify({ queries: every.length, answered: answered.length, differing, native_single_call: stats(single), probe: stats(p), native_batch_of_256_mean_us: Math.round(batched), counts: { ...h.native.counts, by_entity: undefined } }));
  h.native.close();
} else {
  console.error("differential-native-admission: regression | sweeps | fresh | ledger | bench");
  process.exit(2);
}
