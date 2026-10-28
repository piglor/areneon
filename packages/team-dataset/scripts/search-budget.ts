// Measure the cost of one searched decision with the REAL policy worker (ADR-026 section 6, slice R: p99 of the whole decision,
// worker round trips included, at most 10 s; the gate itself is S0's). The searching seat plays seeded A1 games against
// `random-legal`, on teams of the committed `fixtures-v1` dataset (the teams the committed generator-matched belief is derived for,
// not the random-play corpus), with the pair's networks at the SIZE of a real run (`--model-config`, default the `model` section of
// `config/training/public-gpu16.json`), seeded and untrained: latency does not depend on the weights, and the search's cost is the
// number of leaf queries, which the prior only ranks.
//
//   node packages/team-dataset/scripts/search-budget.ts --python /path/to/python [--games 4] [--start 0] [--max-turns 40]
//        [--model-config FILE] [--leaf value-head|hp-ko-differential] [--out FILE]
//
// `--python` is the interpreter of an environment that has `areneon-ml` installed (the worker runs as `python -I -m areneon_ml.worker`).
// It prints one JSON object: per-decision median, p95, p99 and worst for the searched decisions and for all of them, the counts and the
// load. Wall-clock figures are of one machine at one time (a shared one: other work running slows them), not a claim that they repeat.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { cpus, loadavg, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type Actor,
  battleClass,
  engineDataHash,
  goldenWorldFactsData,
  loadEngine,
  loadRulesSnapshot,
  loadSearchConfig,
  PolicyWorkerClient,
  type Player,
  randomLegalActor,
  readEngineData,
  readTrainingRunConfig,
  resolveFormat,
  runBattle,
  searchActor,
  searchPolicy,
  searchSetup,
  stampTeam,
  teamHash,
} from "@areneon/showdown-bridge";
import { readerSha256 } from "../../public-state/src/index.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { createSearchWorlds } from "../src/search-worlds.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const option = (name: string): string | undefined => (process.argv.indexOf(`--${name}`) >= 0 ? process.argv[process.argv.indexOf(`--${name}`) + 1] : undefined);
const flag = (name: string, fallback: number, least = 1): number => {
  const text = option(name);
  if (text === undefined) return fallback;
  const value = Number(text);
  if (!Number.isInteger(value) || value < least) throw new Error(`--${name} needs an integer of at least ${least}`);
  return value;
};
const python = option("python");
if (python === undefined) throw new Error("--python names the interpreter of an environment with areneon-ml installed");
const games = flag("games", 4);
const start = flag("start", 0, 0);
const maxTurns = flag("max-turns", 40);
const leaf = option("leaf") ?? "value-head";
if (leaf !== "value-head" && leaf !== "hp-ko-differential") throw new Error("--leaf is value-head or hp-ko-differential");
const modelFile = option("model-config");
const model = modelFile === undefined ? (JSON.parse(readFileSync(join(ROOT, "config/training/public-gpu16.json"), "utf8")) as { model: unknown }).model : JSON.parse(readFileSync(modelFile, "utf8"));

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const dataFile = join(GOLDEN_DIR, "engine-data.json");

// The pool is the loop's own function over the verified committed dataset (Illusion excluded, as the run configs do).
const raw = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
(raw.generation as Record<string, unknown>).excluded_abilities = ["Illusion"];
const dataset = join(DATASETS_DIR(GOLDEN_FORMAT), "fixtures-v1");
raw.team_dataset = { path: dataset, dataset_hash: verifiedDatasetHash(engine, dataset, GOLDEN_FORMAT) };
const pool = datasetTeamPool(engine, readTrainingRunConfig(raw));

const runDir = mkdtempSync(join(tmpdir(), "areneon-search-budget-"));
const percentile = (sorted: number[], share: number): number => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * share) - 1)] as number;
const summary = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return { n: 0 };
  return {
    n: sorted.length,
    median_ms: Math.round(percentile(sorted, 0.5)),
    mean_ms: Math.round(sorted.reduce((a, b) => a + b, 0) / sorted.length),
    p95_ms: Math.round(percentile(sorted, 0.95)),
    p99_ms: Math.round(percentile(sorted, 0.99)),
    worst_ms: Math.round(sorted[sorted.length - 1] as number),
  };
};

try {
  const modelFileInRun = join(runDir, "model.json");
  writeFileSync(modelFileInRun, JSON.stringify(model));
  const init = (role: string, seed: number): { checkpoint_id: string; encoder_version: string } =>
    JSON.parse(execFileSync(python, ["-I", "-m", "areneon_ml.checkpoint", "init", "--role", role, "--seed", String(seed), "--engine-data", dataFile, "--model-config", modelFileInRun, "--run-dir", runDir], { encoding: "utf8", env: {} }));
  const [preview, battle] = [init("preview", 1), init("battle", 2)];
  const maxLine = 8_000_000;
  const client = new PolicyWorkerClient({
    command: python,
    args: ["-I", "-m", "areneon_ml.worker", "--run-dir", runDir, "--preview", preview.checkpoint_id, "--battle", battle.checkpoint_id, "--engine-data", dataFile, "--max-line-bytes", String(maxLine)],
    env: {},
    max_line_bytes: maxLine,
    n_parallel: 1,
    handshake_timeout_ms: 120_000,
    expect: {
      encoder_version: preview.encoder_version,
      engine_data_hash: engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8")))),
      checkpoints: [{ role: "battle", checkpoint_id: battle.checkpoint_id }, { role: "preview", checkpoint_id: preview.checkpoint_id }],
    },
  });
  await client.start();

  const base = loadSearchConfig(GOLDEN_FORMAT);
  const config = { ...base, leaf: { kind: leaf } } as typeof base;
  const port = createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
  const setup = searchSetup({ engine, format_id: GOLDEN_FORMAT, vocabulary, port, facts_data: goldenWorldFactsData(), data_class: "synthetic", snapshot, config });
  const { policy_id, identity_hash } = searchPolicy(setup, manifest, { preview: preview.checkpoint_id, battle: battle.checkpoint_id }, "argmax");

  const searchedMs: number[] = [];
  const allMs: number[] = [];
  const gameMs: number[] = [];
  const totals: Record<string, number> = {};
  const fallbacks: Record<string, number> = {};
  const refusals: Record<string, number> = {};
  for (let game = start; game < start + games; game += 1) {
    const teams = { p1: pool.teams[(game * 2) % pool.teams.length], p2: pool.teams[(game * 2 + 1) % pool.teams.length] } as Record<Player, never>;
    const seat: Player = game % 2 === 0 ? "p1" : "p2";
    const other: Player = seat === "p1" ? "p2" : "p1";
    const seed = `sodium,${(0x5eed0000 + game).toString(16).padStart(32, "0")}`;
    await client.beginGame();
    const actor = searchActor({
      setup,
      client,
      battle: `game-${game}-${seat}`,
      player: seat,
      team: stampTeam(teams[seat], pool.known_synthetic),
      manifest,
      channel_class: battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, pool.known_synthetic),
      seed,
      game,
      policy_id,
      identity_hash,
      sampling: "argmax",
    });
    let searchedBefore = 0;
    const timed: Actor = async (request, context) => {
      const began = performance.now();
      const choice = await actor(request, context);
      const elapsed = performance.now() - began;
      if (request.phase !== "preview") {
        allMs.push(elapsed);
        const searched = actor.report().counts.searched_decisions;
        if (searched > searchedBefore) searchedMs.push(elapsed);
        searchedBefore = searched;
      }
      return choice;
    };
    const began = performance.now();
    await runBattle({ engine, formatId: GOLDEN_FORMAT, seed, teams, actors: { [seat]: timed, [other]: randomLegalActor(0xabc0 + game) } as Record<Player, Actor>, maxTurns, snapshot });
    gameMs.push(performance.now() - began);
    const { counts } = actor.report();
    for (const [key, value] of Object.entries(counts)) if (typeof value === "number") totals[key] = (totals[key] ?? 0) + value;
    for (const [key, value] of Object.entries(counts.fallbacks)) fallbacks[key] = (fallbacks[key] ?? 0) + value;
    for (const [key, value] of Object.entries(counts.refusals)) refusals[key] = (refusals[key] ?? 0) + value;
  }
  await client.close();

  const result = {
    machine: { node: process.version, cpus: cpus().length, model: cpus()[0]?.model, load_average_at_end: loadavg().map((x) => Number(x.toFixed(2))) },
    config: { search_policy: policy_id, identity_hash, leaf, games, start, max_turns: maxTurns, opponent: "random-legal", teams: "fixtures-v1 dataset", network: model, k: config.top_k, n: config.worlds.n, replies: config.replies },
    per_decision: { searched: summary(searchedMs), all_move_decisions: summary(allMs) },
    per_game: summary(gameMs),
    counts_total: totals,
    fallbacks,
    world_refusals: refusals,
    gate: { p99_limit_ms: 10_000, searched_p99_ms: (summary(searchedMs) as { p99_ms?: number }).p99_ms ?? null },
  };
  const text = JSON.stringify(result, null, 2);
  const out = option("out");
  if (out !== undefined) writeFileSync(out, `${text}\n`);
  console.log(text);
} finally {
  rmSync(runDir, { recursive: true, force: true });
}
