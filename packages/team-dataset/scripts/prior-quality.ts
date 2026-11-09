// Prediction quality of the opponent model per PRIOR, each on the teams it matches (ADR-025 section 12 slice 3b, #1134;
// evaluation-only, the real teams are read as an oracle exactly as in prediction-quality.ts).
//
//   node packages/team-dataset/scripts/prior-quality.ts --label NAME --prior FILE --dataset DIR \
//        [--generator FILE] [--games 30] [--start 0] [--out FILE]
//
// `--prior` is a committed UsagePrior record under config/opponent-model/<format>/; `--dataset` is a generated team dataset
// whose VALIDATION split supplies the teams (one of `generate`'s outputs, not committed); `--generator` is the generator config
// the prior's "other spread" draws follow (default config/teams/<format>/generator.json). The games are seeded random-legal
// against random-legal, run key `prediction-quality-1`, games `start` to `start + games - 1`. Prints one JSON object of counts and
// integer ppm with every prior, dataset, belief and belief-config identity hash; figures of one machine at one time,
// descriptive only, never a gate and never held-out evidence of strength.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { join } from "node:path";
import { loadEngine, readEngineData, readTeamGeneratorConfig, usagePriorHash } from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { ORACLE_RUN_KEY, accumulator, oracleGames, qualityContextFor, report, scoreGame } from "../src/prediction-quality.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const text = (name: string, fallback?: string): string => {
  const at = process.argv.indexOf(`--${name}`);
  const value = at < 0 ? fallback : process.argv[at + 1];
  if (value === undefined) throw new Error(`--${name} is required`);
  return value;
};
const int = (name: string, fallback: number, least: number): number => {
  const value = Number(text(name, String(fallback)));
  if (!Number.isInteger(value) || value < least) throw new Error(`--${name} needs an integer of at least ${least}`);
  return value;
};
const label = text("label");
const priorFile = text("prior");
const dataset = text("dataset");
const games = int("games", 30, 1);
const start = int("start", 0, 0);
const out = process.argv.includes("--out") ? text("out") : undefined;

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const table = readEngineData(JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT, "engine-data.json"), "utf8")));
const generator = readTeamGeneratorConfig(JSON.parse(readFileSync(text("generator", join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json")), "utf8")), table);
const prior = JSON.parse(readFileSync(priorFile, "utf8"));
const ctx = qualityContextFor(engine, vocabulary, { prior, generator });
const recorded = await oracleGames(engine, GOLDEN_FORMAT, start, games, 40, dataset);
const acc = accumulator();
const began = performance.now();
for (const game of recorded) scoreGame(ctx, game, acc);
const result = {
  label,
  set: { source: `seeded random-legal vs random-legal on the validation split of the dataset below, run key ${ORACLE_RUN_KEY}`, dataset_hash: verifiedDatasetHash(engine, dataset, GOLDEN_FORMAT), start, games, worlds_per_request: ctx.worlds },
  commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  identity: { prior_hash: usagePriorHash(ctx.prior), prior_source: ctx.prior.source.kind, belief_identity_hash: ctx.port.identity_hash, belief_config_hash: ctx.port.belief_config_hash },
  machine: { node: process.version, cpus: cpus().length, loadavg: loadavg(), elapsed_s: Math.round((performance.now() - began) / 1000) },
  ...report(acc),
};
const json = JSON.stringify(result, null, 2);
if (out !== undefined) writeFileSync(out, `${json}\n`);
console.log(json);
