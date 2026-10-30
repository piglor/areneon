// Cost of the belief feature set `probe-damage-belief/1` against `probe-damage/1` on the SAME recorded decisions
// (ADR-025 slice 5, ADR-022 section 5 gates). Read-only on run directories; prints aggregate counts and milliseconds only.
//
//   node packages/team-dataset/scripts/feature-bench-belief.ts --run-dir RUN --generation 50 [--episodes 40]
//     [--progress-dir P --progress-games 20] [--worlds 8] [--out FILE]
//
// Single thread (the belief port stays on the calling thread). The generation estimate is the bridge bench's
// (`estimateWallTime`): the builder runs twice (play and stream writing) over every seat decision of the control
// generation; the limit is 1.5 times the control's play plus load.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { join } from "node:path";
import {
  type BenchDecision,
  episodeDecisions,
  estimateWallTime,
  FEATURE_SET_BELIEF,
  loadDamageAwareConfigV2,
  loadEngine,
  loadRulesSnapshot,
  progressDecisions,
  type ProgressGame,
  resolveFormat,
  runBench,
  type Skipped,
} from "@areneon/showdown-bridge";
import { readerSha256 } from "../../public-state/src/index.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { createBeliefWorlds } from "../src/damage-aware-worlds.ts";

const arg = (name: string): string | undefined => (process.argv.includes(`--${name}`) ? process.argv[process.argv.indexOf(`--${name}`) + 1] : undefined);
const num = (name: string, fallback: number): number => {
  const value = arg(name);
  if (value === undefined) return fallback;
  if (!/^[0-9]+$/.test(value)) throw new Error(`--${name} needs a non-negative integer`);
  return Number(value);
};
const json = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));
const runDir = arg("run-dir");
const generation = num("generation", -1);
if (runDir === undefined || generation < 0) throw new Error("--run-dir and --generation are required");
const worlds = num("worlds", 8);
const runConfig = json(join(runDir, "run-config.json")) as { generation: { max_turns: number } };
const gen = `g${String(generation).padStart(6, "0")}`;

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const record = loadDamageAwareConfigV2(vocabulary);
const snapshot = loadRulesSnapshot(engine.commit);
const port = createBeliefWorlds({ engine, vocabulary, config: record, reader_sha256: readerSha256(), manifest: resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest });
const base = { format_id: GOLDEN_FORMAT, data_class: "synthetic" as const, ...(process.env.SHOWDOWN_DIR ? { showdown_dir: process.env.SHOWDOWN_DIR } : {}) };
const beliefSpec = { ...base, feature_set: FEATURE_SET_BELIEF } as const;
const oneSpec = { ...base, feature_set: "probe-damage/1" as const };

const skipped: Skipped = { not_move: 0, unreplayable_games: 0 };
const episodeDir = join(runDir, "generations", gen, "episodes");
const files = readdirSync(episodeDir).sort();
const wanted = Math.min(num("episodes", 40), files.length);
const decisions: BenchDecision[] = [];
for (let k = 0; k < wanted; k++) decisions.push(...episodeDecisions(json(join(episodeDir, files[Math.floor((k * files.length) / wanted)] as string)), files[Math.floor((k * files.length) / wanted)] as string, "episode", skipped));
const progressDir = arg("progress-dir");
const progress: BenchDecision[] = [];
const skippedProgress: Skipped = { not_move: 0, unreplayable_games: 0 };
if (progressDir !== undefined) {
  const root = join(progressDir, gen, "games");
  const games = readdirSync(root).sort();
  const wantedGames = Math.min(num("progress-games", 20), games.length);
  for (let k = 0; k < wantedGames; k++) {
    const file = games[Math.floor((k * games.length) / wantedGames)] as string;
    progress.push(...(await progressDecisions(engine, GOLDEN_FORMAT, runConfig.generation.max_turns, json(join(root, file)) as ProgressGame, file, (policy) => (policy.startsWith("policy:") ? "learned" : "baseline"), skippedProgress)));
  }
}

const report: Record<string, unknown> = {
  machine: { model: cpus()[0]?.model, logical_cpus: cpus().length, loadavg_at_start: loadavg().map((x) => Math.round(x * 100) / 100) },
  pin: engine.commit,
  worlds,
  belief_identity_hash: port.identity_hash,
};
const measure = async (name: string, list: BenchDecision[]): Promise<void> => {
  // Warm-up (engine, prior and tables load once) is a separate small run, so neither set pays it in its figures.
  await runBench(oneSpec, list.slice(0, 2), []);
  await runBench(beliefSpec, list.slice(0, 2), [], { belief: { port, worlds } });
  const one = await runBench(oneSpec, list, []);
  const belief = await runBench(beliefSpec, list, [], { belief: { port, worlds } });
  report[name] = { decisions: list.length, one: one.single_thread, belief: belief.single_thread, mean_ratio_x100: Math.round((belief.single_thread.total_builder_ms * 100) / Math.max(1, one.single_thread.total_builder_ms)) };
};
await measure("episodes", decisions);
if (progress.length > 0) await measure("progress", progress);
report.skipped = { episodes: skipped, progress: skippedProgress };

const metrics = json(join(runDir, "metrics", `${gen}.json`)) as { phase_ms: { play: number; load: number }; counts: { decisions: number } };
const single = (report.episodes as { belief: { decisions: number; total_builder_ms: number } }).belief;
const meanAll = single.total_builder_ms / (single.decisions + skipped.not_move);
report.generation_estimate = {
  control: { generation, play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions },
  ...estimateWallTime({ play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions }, meanAll, [1, 4, 8, 16]),
};
const text = JSON.stringify(report, null, 1);
const out = arg("out");
if (out === undefined) console.log(text);
else writeFileSync(out, text + "\n");
