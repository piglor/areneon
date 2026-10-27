// Prediction quality of the opponent model's belief against the opponent's real sets (ADR-025 section 13,
// descriptive measures; evaluation-only: the real teams are read here as an oracle, no actor imports this).
//
//   node packages/team-dataset/scripts/prediction-quality.ts [--games 30] [--start 0] [--out FILE]
//
// The games are seeded A1 battles, random-legal against random-legal, on teams of the committed dev split
// (`fixtures/teams/<format>/datasets/fixtures-v1`, generator-drawn), games `start` to `start + games - 1` of the
// run key `prediction-quality-1`; both teams and both channels are recorded.
// It prints one JSON object of counts and integer ppm (top-1 accuracy, log-loss in micro-nats, a reliability
// table, the joint-consistency rate) with the identity hashes, the commit and the machine's load. Figures of one
// machine at one time; descriptive only, never a gate.
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { loadEngine } from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { ORACLE_RUN_KEY, accumulator, oracleGames, qualityContext, report, scoreGame } from "../src/prediction-quality.ts";

const flag = (name: string, fallback: number, least = 1): number => {
  const at = process.argv.indexOf(`--${name}`);
  if (at < 0) return fallback;
  const value = Number(process.argv[at + 1]);
  if (!Number.isInteger(value) || value < least) throw new Error(`--${name} needs an integer of at least ${least}`);
  return value;
};
const games = flag("games", 30);
const start = flag("start", 0, 0);
const out = process.argv.indexOf("--out") >= 0 ? process.argv[process.argv.indexOf("--out") + 1] : undefined;

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const ctx = qualityContext(engine, goldenTeamVocabulary());
const recorded = await oracleGames(engine, GOLDEN_FORMAT, start, games);
const acc = accumulator();
const began = performance.now();
for (const game of recorded) scoreGame(ctx, game, acc);
const result = {
  set: { source: `seeded random-legal vs random-legal on the dev split of fixtures-v1, run key ${ORACLE_RUN_KEY}`, start, games, worlds_per_request: ctx.worlds },
  commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  identity: { belief_identity_hash: ctx.port.identity_hash, prior_hash: ctx.port.prior_hash, belief_config_hash: ctx.port.belief_config_hash },
  machine: { node: process.version, cpus: cpus().length, model: cpus()[0]?.model, loadavg: loadavg(), elapsed_s: Math.round((performance.now() - began) / 1000) },
  ...report(acc),
};
const text = JSON.stringify(result, null, 2);
if (out !== undefined) writeFileSync(out, `${text}\n`);
console.log(text);
