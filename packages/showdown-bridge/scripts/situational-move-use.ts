// The situational move-use diagnostic (#1283): read-only on a committed progress generation; writes raw integer counts
// (per game and seat) to --out or stdout. `python -m areneon_ml.report.situational RAW.json` derives the rates and pair-level
// intervals.
//
//   node packages/showdown-bridge/scripts/situational-move-use.ts --progress-dir P --generation G [--max-turns T | --run-dir R]
//     [--matches a,b] [--max-games N] [--history] [--out FILE]
//
// `--history` adds the history rows (#1345): a second feature build (`probe-outcome/1`) per decision and a `history` count block per
// seat; without it the output is byte-identical to before.
//
// The input is VERIFIED first, as the Python move-use diagnostic does: the generation's manifest must list the SHA-256 of every
// file, `progress.json` goes through the progress record reader, and each game through the game record reader (schema, choices
// among candidates, team and log hashes). The turn cap is the record's `plan.max_turns`; `--max-turns` or `--run-dir` (its
// `run-config.json`) may state it and must agree with the record.
//
// Label: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure. Run it at `nice -n 19`.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { readVerifiedGeneration } from "../src/situational-input.ts";
import { SITUATIONAL_FORMAT, SITUATIONAL_LABEL, situationalSetup, tallyGame } from "../src/situational-move-use.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";

const arg = (name: string): string | undefined => (process.argv.includes(`--${name}`) ? process.argv[process.argv.indexOf(`--${name}`) + 1] : undefined);
const integer = (name: string): number => {
  const value = arg(name);
  if (value === undefined || !/^[0-9]+$/.test(value)) throw new Error(`--${name} needs a non-negative integer`);
  return Number(value);
};
const dir = arg("progress-dir");
if (dir === undefined) throw new Error("--progress-dir is required");
const generation = integer("generation");
const matches = arg("matches")?.split(",");
const maxGames = arg("max-games") === undefined ? Infinity : integer("max-games");

const { record, games: records } = readVerifiedGeneration(join(dir, `g${String(generation).padStart(6, "0")}`), generation, matches);
const maxTurns = record.plan.max_turns;
const stated = [arg("max-turns") === undefined ? undefined : integer("max-turns"), arg("run-dir") === undefined ? undefined : (JSON.parse(readFileSync(join(arg("run-dir") as string, "run-config.json"), "utf8")) as { generation: { max_turns: number } }).generation.max_turns];
if (stated.some((value) => value !== undefined && value !== maxTurns)) throw new Error(`the stated turn cap differs from the record's (${maxTurns})`);

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const setup = situationalSetup(engine, GOLDEN_FORMAT, maxTurns, goldenWorldFactsData(), process.env.SHOWDOWN_DIR ? { showdown_dir: process.env.SHOWDOWN_DIR } : {}, process.argv.includes("--history"));
const games = [];
let unreplayable = 0;
const taken = new Map<string, number>();
for (const game of records) {
  const n = taken.get(game.match_id) ?? 0;
  if (n >= maxGames) continue;
  taken.set(game.match_id, n + 1);
  const tally = await tallyGame(setup, game);
  if (tally === null) unreplayable++;
  else games.push(tally);
}
const first = records[0];
const out = JSON.stringify({
  format: SITUATIONAL_FORMAT,
  label: SITUATIONAL_LABEL,
  pin: engine.commit,
  generation,
  run_id: record.run_id,
  evaluation_key: first?.evaluation_key ?? null,
  unreplayable_games: unreplayable,
  games,
});
const file = arg("out");
if (file === undefined) console.log(out);
else writeFileSync(file, out + "\n");
