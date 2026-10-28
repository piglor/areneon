// Candidate feature builder bench (ADR-022 section 5, ticket #1030). Read-only on run directories: it opens
// Episode and progress-game files for reading and writes only the JSON report to --out (or stdout).
//
//   node packages/showdown-bridge/scripts/feature-bench.ts --run-dir RUN --generation 50 [--episodes 40] [--progress-dir P --progress-games 20]
//     [--pools 4,8] [--feature-set probe-damage/1|probe-damage/2] [--out FILE]
//
// The report is aggregate counts and milliseconds only. The machine (model, load, cores) and the engine pin are
// recorded with the figures; a loaded machine makes them pessimistic.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { type BenchDecision, episodeDecisions, estimateWallTime, progressDecisions, type ProgressGame, runBench, type Skipped } from "../src/feature-bench.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { damagePathStats } from "../src/mechanics-damage.ts";

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
if (runDir === undefined || !Number.isInteger(generation) || generation < 0) throw new Error("--run-dir and --generation are required");
const runConfig = json(join(runDir, "run-config.json")) as { generation: { max_turns: number } };
const gen = `g${String(generation).padStart(6, "0")}`;
const pools = (arg("pools") ?? "4,8").split(",").filter((x) => x !== "").map((x) => (/^[1-9][0-9]*$/.test(x) ? Number(x) : Number.NaN));
if (pools.some(Number.isNaN)) throw new Error("--pools needs a comma-separated list of positive integers");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const skipped: Skipped = { not_move: 0, unreplayable_games: 0 };
const skippedProgress: Skipped = { not_move: 0, unreplayable_games: 0 };
const featureArg = arg("feature-set") ?? "probe-damage/1";
if (featureArg !== "probe-damage/1" && featureArg !== "probe-damage/2") throw new Error("--feature-set is probe-damage/1 or probe-damage/2");
const featureSet: "probe-damage/1" | "probe-damage/2" = featureArg;
const spec = { format_id: GOLDEN_FORMAT, data_class: "synthetic" as const, feature_set: featureSet, ...(process.env.SHOWDOWN_DIR ? { showdown_dir: process.env.SHOWDOWN_DIR } : {}) };

// Episodes: an evenly spaced sample of the generation's files, in name order.
const episodeDir = join(runDir, "generations", gen, "episodes");
const files = readdirSync(episodeDir).sort();
const wanted = Math.min(num("episodes", 40), files.length);
const episodes: BenchDecision[] = [];
for (let k = 0; k < wanted; k++) {
  const file = files[Math.floor((k * files.length) / wanted)] as string;
  episodes.push(...episodeDecisions(json(join(episodeDir, file)), file, "episode", skipped));
}

const report: Record<string, unknown> = {
  machine: { model: cpus()[0]?.model, logical_cpus: cpus().length, loadavg_at_start: loadavg().map((x) => Math.round(x * 100) / 100) },
  pin: engine.commit,
  feature_set: featureSet,
  run: { generation, episodes_sampled: wanted, episodes_in_generation: files.length },
};
report.episodes = { decisions: episodes.length, ...(await runBench(spec, episodes, pools)) };

const progressDir = arg("progress-dir");
if (progressDir !== undefined) {
  const root = join(progressDir, gen, "games");
  const games = readdirSync(root).sort();
  const wantedGames = Math.min(num("progress-games", 20), games.length);
  const decisions: BenchDecision[] = [];
  for (let k = 0; k < wantedGames; k++) {
    const file = games[Math.floor((k * games.length) / wantedGames)] as string;
    const record = json(join(root, file)) as ProgressGame;
    decisions.push(...(await progressDecisions(engine, GOLDEN_FORMAT, runConfig.generation.max_turns, record, file, (policy) => (policy.startsWith("policy:") ? "learned" : "baseline"), skippedProgress)));
  }
  report.progress = { games_sampled: wantedGames, games_stored: games.length, decisions: decisions.length, ...(await runBench(spec, decisions, pools)) };
}
report.damage_path_single_thread = { ...damagePathStats };
report.skipped = { episodes: skipped, progress: skippedProgress };

const metrics = json(join(runDir, "metrics", `${gen}.json`)) as { phase_ms: { play: number; load: number }; counts: { decisions: number }; threads: number };
const single = (report.episodes as { single_thread: { decisions: number; total_builder_ms: number } }).single_thread;
// The mean is over every seat decision (non-move requests cost nothing): scale the sample's move decisions by the share of move requests in the sample.
const allSeen = single.decisions + skipped.not_move;
const meanAll = single.total_builder_ms / allSeen;
report.generation_estimate = {
  control: { run: "control run", generation, play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions },
  sample_move_share_ppm: Math.floor((single.decisions * 1_000_000) / allSeen),
  threads_note: "by_threads divides the builder time ideally; measured_speedup gives the pool speed-ups measured on this machine",
  ...estimateWallTime({ play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions }, meanAll, [1, 4, 8, 16]),
};
const est = report.generation_estimate as { control_play_plus_load_ms: number; builder_cpu_ms: number };
report.measured_speedup = (report.episodes as { pools: { size: number; speedup_x100: number }[] }).pools.map((p) => {
  const total = est.control_play_plus_load_ms + Math.round((est.builder_cpu_ms * 100) / p.speedup_x100);
  return { threads: p.size, speedup_x100: p.speedup_x100, total_ms: total, ratio_x100: Math.round((total * 100) / est.control_play_plus_load_ms) };
});
const text = JSON.stringify(report, null, 1);
const out = arg("out");
if (out === undefined) console.log(text);
else writeFileSync(out, text + "\n");
