// The battle-level differential's size check (ADR-030 groundwork G2, ADR-028 section 5, #1462): a fresh corpus of N seeded
// battles on generator-pool teams (dev and training splits of a verified dataset, never validation) across the uniform-all,
// random-legal and pending-first mixes, each recorded on the unmodified engine and replayed on a second engine object with its
// draws forced, compared at every request of both seats. Prints the corpus summary (differences by kind, entity and
// interaction-class counts) with the wall time and the trace bytes per battle, so the M-D spike can budget its 200,000 battles:
//
//   node packages/team-dataset/scripts/differential-battle.ts --battles 2000 [--start 0] [--chunk 50] [--max-turns 100] [--dataset DIR] [--out FILE]
//
// --max-turns is the turn cap (default 100, the committed GPU run config's `generation.max_turns`).
//
// Nothing is written but --out (the summary). The traces are dropped chunk by chunk.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { type BattleTrace, battleCorpusSummary } from "@areneon/differential/battle";
import { loadEngine, readTrainingRunConfig } from "@areneon/showdown-bridge";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { battleBackend, battleFreshSeed, compareTraces, freshSpec, loadInteractionClasses, newBattleCorpus, poolHash, traceOf } from "../src/differential-battle.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import tiny from "../../../fixtures/contracts/v1/training-run-config/valid/tiny.json" with { type: "json" };

const { values } = parseArgs({ options: { battles: { type: "string" }, start: { type: "string", default: "0" }, chunk: { type: "string", default: "50" }, "max-turns": { type: "string", default: "100" }, dataset: { type: "string" }, out: { type: "string" } } });
const battles = Number(values.battles);
const start = Number(values.start);
const chunk = Number(values.chunk);
const maxTurns = Number(values["max-turns"]);
if (!Number.isSafeInteger(battles) || battles < 1 || !Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(chunk) || chunk < 1 || !Number.isSafeInteger(maxTurns) || maxTurns < 1) {
  console.error("differential-battle: --battles N (at least 1), --start S, --chunk C, --max-turns T");
  process.exit(2);
}

const FORMAT = "gen9championsvgc2026regmc";
const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const replay = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const source = values.dataset ?? join(DATASETS_DIR(FORMAT), "fixtures-v1");
const config = readTrainingRunConfig({ ...structuredClone(tiny), team_dataset: { path: source, dataset_hash: verifiedDatasetHash(engine, source, FORMAT) } });
const pool = datasetTeamPool(engine, config);
const seed = battleFreshSeed(engine);
const classes = loadInteractionClasses();
const backend = battleBackend(replay);
const corpus = newBattleCorpus({ id: `fresh-${seed.slice(0, 16)}`, kind: "fresh", seed, pool: poolHash(pool) }, classes);

let recordMs = 0;
let replayMs = 0;
let bytes = 0;
let maxBytes = 0;
let peakRss = 0;
const began = Date.now();
for (let first = start; first < start + battles; first += chunk) {
  const traces: BattleTrace[] = [];
  let t = Date.now();
  for (let i = first; i < Math.min(first + chunk, start + battles); i++) {
    const trace = await traceOf(engine, freshSpec(seed, pool, i, FORMAT, maxTurns));
    const size = Buffer.byteLength(JSON.stringify(trace));
    bytes += size;
    maxBytes = Math.max(maxBytes, size);
    traces.push(trace);
  }
  recordMs += Date.now() - t;
  t = Date.now();
  compareTraces(classes, traces, corpus, backend, chunk);
  replayMs += Date.now() - t;
  peakRss = Math.max(peakRss, process.memoryUsage().rss);
  console.error(`differential-battle: ${corpus.battles} battles, ${corpus.differences.length} differences, ${Math.round((Date.now() - began) / 1000)} s`);
}

const summary = {
  inputs: { showdown_commit: engine.commit, fresh_seed: seed, pool_hash: poolHash(pool), pool_teams: pool.teams.length, start, battles, max_turns: maxTurns },
  cost: {
    wall_ms: Date.now() - began,
    record_ms_per_battle: Math.round(recordMs / battles),
    replay_ms_per_battle: Math.round(replayMs / battles),
    trace_bytes_per_battle: Math.round(bytes / battles),
    trace_bytes_max: maxBytes,
    peak_rss_bytes: peakRss,
  },
  corpus: battleCorpusSummary(corpus),
};
const text = `${JSON.stringify(summary, null, 2)}\n`;
if (values.out !== undefined) writeFileSync(values.out, text);
else process.stdout.write(text);
process.exit(corpus.differences.length === 0 ? 0 : 1);
