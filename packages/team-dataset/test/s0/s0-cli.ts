// The S0 measures 3, 4 and 8 with the REAL policy worker (ADR-026 section 8 S0; #1147, #1308). TEST TOOLING (it reads the fork oracle):
//
//   node packages/team-dataset/test/s0/s0-cli.ts --python /path/to/python [--battles 3] [--start 0] [--max-decisions 4] [--stride 1]
//        [--rollouts 8] [--rollout-max-turns 40] [--run-dir RUN --preview ID --battle ID --engine-data FILE] [--out FILE]
//
// Battles are the committed seeded A1 battles of the random corpus (`fixtures/showdown/<format>/random/`), replayed from their seeds
// and teams; the fork oracle is taken at each sampled own move decision, with the opponent's real policy of that battle
// (random-legal or pending-first) answering in every rollout. The frozen pair is either a seeded untrained pair at the size of
// `config/training/public-gpu16.json` (default; the numbers then only show the tool works) or a committed run's checkpoints
// (`--run-dir`, `--preview`, `--battle`, `--engine-data`). Output is one JSON object of integers and ppm: no team, set or line.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { engineDataHash, goldenWorldFactsData, loadEngine, loadRulesSnapshot, loadSearchConfig, PolicyWorkerClient, readEngineDataAny, resolveFormat, searchSetup } from "@areneon/showdown-bridge";
import { readerSha256 } from "../../../public-state/src/index.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus } from "../../../showdown-bridge/src/random-corpus.ts";
import { createSearchWorlds } from "../../src/search-worlds.ts";
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
const rollouts = flag("rollouts", 8, 2);
if (rollouts % 2 !== 0) throw new Error("--rollouts must be even (the split halves)");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const dataFile = option("engine-data") ?? join(GOLDEN_DIR, "engine-data.json");
const runDirOption = option("run-dir");
const runDir = runDirOption ?? mkdtempSync(join(tmpdir(), "areneon-s0-"));
const scratch = runDirOption === undefined;

try {
  let pair: { preview: string; battle: string; encoder_version: string };
  if (runDirOption === undefined) {
    const model = (JSON.parse(readFileSync(join(ROOT, "config/training/public-gpu16.json"), "utf8")) as { model: unknown }).model;
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
    sampling: "argmax" as const,
    rollouts,
    rollout_max_turns: flag("rollout-max-turns", 40),
    max_decisions: flag("max-decisions", 4),
    stride: flag("stride", 1),
  };
  const corpus = loadRandomCorpus().slice(flag("start", 0, 0), flag("start", 0, 0) + flag("battles", 3));
  const measured: BattleMeasure[] = [];
  for (const [game, battle] of corpus.entries()) {
    await client.beginGame();
    measured.push(await measureBattle(options, battle, game));
  }
  await client.close();
  const result = {
    config: { battles: corpus.length, rollouts, k: base.top_k, top_k_rule: base.top_k_rule, worlds: base.worlds.n, replies: base.replies, opponent: "the corpus battle's own policy", network: runDirOption === undefined ? "seeded untrained, public-gpu16 size" : "committed run checkpoints" },
    summary: summarise(measured),
  };
  const text = JSON.stringify(result, null, 2);
  const out = option("out");
  if (out !== undefined) writeFileSync(out, `${text}\n`);
  console.log(text);
} finally {
  if (scratch) rmSync(runDir, { recursive: true, force: true });
}
