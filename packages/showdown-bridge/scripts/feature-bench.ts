// Candidate feature builder bench (ADR-022 section 5, ticket #1030). Read-only on run directories: it opens
// Episode and progress-game files for reading and writes only the JSON report to --out (or stdout).
//
//   node packages/showdown-bridge/scripts/feature-bench.ts --run-dir RUN --generation 50 [--episodes 40] [--progress-dir P --progress-games 20]
//     [--pools 4,8] [--out FILE]
//
// The report is aggregate counts and milliseconds only. The machine (model, load, cores) and the engine pin are
// recorded with the figures; a loaded machine makes them pessimistic.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { type BenchDecision, episodeDecisions, estimateWallTime, progressDecisions, type ProgressGame, runBench, type Skipped } from "../src/feature-bench.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";

const arg = (name: string): string | undefined => (process.argv.includes(`--${name}`) ? process.argv[process.argv.indexOf(`--${name}`) + 1] : undefined);
const num = (name: string, fallback: number): number => {
  const value = arg(name);
  return value === undefined ? fallback : Number(value);
};
const runDir = arg("run-dir");
const generation = num("generation", -1);
if (runDir === undefined || !Number.isInteger(generation) || generation < 0) throw new Error("--run-dir and --generation are required");
const gen = `g${String(generation).padStart(6, "0")}`;
const pools = (arg("pools") ?? "4,8").split(",").map(Number);
const json = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const skipped: Skipped = { not_move: 0, unreplayable_games: 0 };
const skippedProgress: Skipped = { not_move: 0, unreplayable_games: 0 };
const spec = { format_id: GOLDEN_FORMAT, data_class: "synthetic" as const, ...(process.env.SHOWDOWN_DIR ? { showdown_dir: process.env.SHOWDOWN_DIR } : {}) };

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
    decisions.push(...(await progressDecisions(engine, GOLDEN_FORMAT, 100, record, file, (policy) => (policy.startsWith("policy:") ? "learned" : "baseline"), skippedProgress)));
  }
  report.progress = { games_sampled: wantedGames, games_stored: games.length, decisions: decisions.length, ...(await runBench(spec, decisions, [])) };
}
report.skipped = { episodes: skipped, progress: skippedProgress };

const metrics = json(join(runDir, "metrics", `${gen}.json`)) as { phase_ms: { play: number; load: number }; counts: { decisions: number }; threads: number };
const single = (report.episodes as { single_thread: { decisions: number; total_builder_ms: number } }).single_thread;
// The mean is over every seat decision (non-move requests cost nothing): scale the sample's move decisions by the share of move requests in the sample.
const allSeen = single.decisions + skipped.not_move;
const meanAll = single.total_builder_ms / allSeen;
report.generation_estimate = {
  control: { run: "control lineage", generation, play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions },
  sample_move_share_ppm: Math.floor((single.decisions * 1_000_000) / allSeen),
  ...estimateWallTime({ play_ms: metrics.phase_ms.play, load_ms: metrics.phase_ms.load, decisions: metrics.counts.decisions }, meanAll, [1, 4, 8, 16]),
};
const text = JSON.stringify(report, null, 1);
const out = arg("out");
if (out === undefined) console.log(text);
else writeFileSync(out, text + "\n");
