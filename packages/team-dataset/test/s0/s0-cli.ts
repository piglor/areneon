// The S0 measures 3, 4 and 8 with the REAL policy worker (ADR-026 section 8 S0; #1147, #1308). TEST TOOLING (it reads the fork oracle):
//
//   node packages/team-dataset/test/s0/s0-cli.ts --python /path/to/python [--source games|corpus] [--games 100] [--battles 3] [--start 0]
//        [--max-decisions 4] [--stride 1] [--rollouts 8] [--rollout-max-turns 40] [--sampling argmax|sample]
//        [--run-config RUN-CONFIG.json] [--run-dir RUN --preview ID --battle ID --encoder-version V --engine-data FILE] [--out FILE]
//
// `--source games` (default) plays seeded A1 games of the frozen pair against `damage-aware@v1` on the committed dev-split teams
// (`fixtures-v1`), the pair alternating seats, and takes the fork oracle at the pair's sampled own move decisions; every rollout
// asks a fresh `damage-aware@v1` for the opponent. `--source corpus` replays the committed random-corpus battles instead, with
// each battle's own random-legal or pending-first opponent. The frozen pair is either a seeded untrained pair at the size of
// `config/training/public-gpu16.json` (default; the numbers then only show the tool works) or a committed run's checkpoints
// (`--run-dir`, `--preview`, `--battle`, `--encoder-version`, `--engine-data`). `--run-config` names the run's TrainingRunConfig: its
// `model` sizes a seeded pair and its candidate features (`probe-damage/1`) are stated to the worker, built for the root query and
// for the pair's play in games and rollouts, as a features pair needs. The run exits 1 after printing when too many rollouts were
// capped (above `MAX_CAPPED_PPM`). Output is one JSON object of integers and ppm: no team, set or line.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { damageAwareSetup, engineDataHash, featureSetup, goldenWorldFactsData, loadEngine, loadRulesSnapshot, loadSearchConfig, PolicyWorkerClient, readEngineDataAny, readTrainingRunConfig, resolveFormat, searchSetup, syntheticTeamHashes, teamHash } from "@areneon/showdown-bridge";
import { readerSha256 } from "../../../public-state/src/index.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus } from "../../../showdown-bridge/src/random-corpus.ts";
import { verifiedDatasetHash } from "../../src/dataset-hash.ts";
import { DATASETS_DIR } from "../../src/fixtures.ts";
import { datasetTeamPool } from "../../src/loop-pool.ts";
import { createSearchWorlds } from "../../src/search-worlds.ts";
import { pairGameSpec } from "./games.ts";
import { measureBattle, summarise, type BattleMeasure } from "./measure.ts";

const ROOT = fileURLToPath(new URL("../../../..", import.meta.url));
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
const sampling = option("sampling") ?? "argmax";
if (sampling !== "argmax" && sampling !== "sample") throw new Error("--sampling is argmax or sample");
const source = option("source") ?? "games";
if (source !== "games" && source !== "corpus") throw new Error("--source is games or corpus");
const rollouts = flag("rollouts", 8, 2);
if (rollouts % 2 !== 0) throw new Error("--rollouts must be even (the split halves)");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const runConfigFile = option("run-config");
const runConfig = runConfigFile === undefined ? undefined : readTrainingRunConfig(JSON.parse(readFileSync(runConfigFile, "utf8")));
const features = runConfig === undefined ? undefined : featureSetup(engine, runConfig);
const dataFile = option("engine-data") ?? join(GOLDEN_DIR, "engine-data.json");
const runDirOption = option("run-dir");
const runDir = runDirOption ?? mkdtempSync(join(tmpdir(), "areneon-s0-"));
const scratch = runDirOption === undefined;

try {
  let pair: { preview: string; battle: string; encoder_version: string };
  if (runDirOption === undefined) {
    const model = runConfig?.model ?? (JSON.parse(readFileSync(join(ROOT, "config/training/public-gpu16.json"), "utf8")) as { model: unknown }).model;
    const modelFile = join(runDir, "model.json");
    writeFileSync(modelFile, JSON.stringify(model));
    const init = (role: string, seed: number): { checkpoint_id: string; encoder_version: string } =>
      JSON.parse(execFileSync(python, ["-I", "-m", "areneon_ml.checkpoint", "init", "--role", role, "--seed", String(seed), "--engine-data", dataFile, "--model-config", modelFile, "--run-dir", runDir], { encoding: "utf8", env: {} }));
    const [p, b] = [init("preview", 1), init("battle", 2)];
    pair = { preview: p.checkpoint_id, battle: b.checkpoint_id, encoder_version: p.encoder_version };
  } else {
    const [preview, battle, encoder] = [option("preview"), option("battle"), option("encoder-version")];
    if (preview === undefined || battle === undefined || encoder === undefined) throw new Error("--run-dir needs --preview, --battle and --encoder-version");
    pair = { preview, battle, encoder_version: encoder };
  }
  const maxLine = 8_000_000;
  const client = new PolicyWorkerClient({
    command: python,
    args: ["-I", "-m", "areneon_ml.worker", "--run-dir", runDir, "--preview", pair.preview, "--battle", pair.battle, "--engine-data", dataFile, "--max-line-bytes", String(maxLine)],
    env: {},
    max_line_bytes: maxLine,
    n_parallel: 1,
    handshake_timeout_ms: 120_000,
    ...(features === undefined ? {} : { features: features.statement }),
    expect: {
      encoder_version: pair.encoder_version,
      engine_data_hash: engineDataHash(readEngineDataAny(JSON.parse(readFileSync(dataFile, "utf8")))),
      checkpoints: [{ role: "battle", checkpoint_id: pair.battle }, { role: "preview", checkpoint_id: pair.preview }],
    },
  });
  await client.start();

  const base = loadSearchConfig(GOLDEN_FORMAT);
  const hp = { ...base, leaf: { kind: "hp-ko-differential" } } as typeof base;
  const port = createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
  const make = (config: typeof base) => searchSetup({ engine, format_id: GOLDEN_FORMAT, vocabulary, port, facts_data: goldenWorldFactsData(), data_class: "synthetic", snapshot, config });
  const options = {
    engine,
    manifest,
    snapshot,
    setup: make(base),
    setup_hp: make(hp),
    worker: client,
    pair: { preview: pair.preview, battle: pair.battle },
    sampling: sampling as "argmax" | "sample",
    ...(features === undefined ? {} : { features: features.configs }),
    rollouts,
    rollout_max_turns: flag("rollout-max-turns", 40),
    max_decisions: flag("max-decisions", 4),
    stride: flag("stride", 1),
  };
  const measured: BattleMeasure[] = [];
  if (source === "corpus") {
    const start = flag("start", 0, 0);
    const corpus = loadRandomCorpus().slice(start, start + flag("battles", 3));
    for (const [game, battle] of corpus.entries()) {
      await client.beginGame();
      measured.push(await measureBattle(options, battle, game));
    }
  } else {
    // The committed dev-split teams of fixtures-v1 (Illusion excluded, as the run configs do), the pair alternating seats.
    const raw = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
    (raw.generation as Record<string, unknown>).excluded_abilities = ["Illusion"];
    const dataset = join(DATASETS_DIR(GOLDEN_FORMAT), "fixtures-v1");
    raw.team_dataset = { path: dataset, dataset_hash: verifiedDatasetHash(engine, dataset, GOLDEN_FORMAT) };
    const pool = datasetTeamPool(engine, readTrainingRunConfig(raw));
    const damage = damageAwareSetup(engine, GOLDEN_FORMAT);
    const start = flag("start", 0, 0);
    for (let game = start; game < start + flag("games", 3); game += 1) {
      const teams = { p1: pool.teams[(game * 2) % pool.teams.length], p2: pool.teams[(game * 2 + 1) % pool.teams.length] } as never;
      const seat = game % 2 === 0 ? "p1" : "p2";
      const known = syntheticTeamHashes([teamHash((teams as { p1: never }).p1), teamHash((teams as { p2: never }).p2)]);
      await client.beginGame();
      const spec = pairGameSpec({ engine, manifest, client, pair: options.pair, sampling: options.sampling, ...(features === undefined ? {} : { features: features.configs }), damage, id: `g${game}`, seed: `sodium,${(0x5eed0000 + game).toString(16).padStart(32, "0")}`, teams, seat, game, known });
      measured.push(await measureBattle(options, spec, game, seat));
    }
  }
  await client.close();
  const result = {
    config: { source, games: measured.length, sampling, features: features?.statement.feature_set ?? null, rollouts, k: base.top_k, top_k_rule: base.top_k_rule, worlds: base.worlds.n, replies: base.replies, opponent: source === "games" ? "damage-aware@v1" : "the corpus battle's own policy", network: runDirOption === undefined ? "seeded untrained, public-gpu16 size" : "committed run checkpoints" },
    summary: summarise(measured),
  };
  const text = JSON.stringify(result, null, 2);
  const out = option("out");
  if (out !== undefined) writeFileSync(out, `${text}\n`);
  console.log(text);
  if (!result.summary.capped_within_limit) {
    console.error("too many oracle rollouts were capped: the values are not a measurement; raise --rollout-max-turns");
    process.exitCode = 1;
  }
} finally {
  if (scratch) rmSync(runDir, { recursive: true, force: true });
}
