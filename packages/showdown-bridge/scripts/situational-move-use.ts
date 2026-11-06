// The situational move-use diagnostic (#1283): read-only on a committed progress generation's stored games; writes raw integer counts
// (per game and seat) to --out or stdout. `python -m areneon_ml.report.situational RAW.json` derives the rates and pair-level intervals.
//
//   node packages/showdown-bridge/scripts/situational-move-use.ts --progress-dir P --generation G --max-turns T [--matches a,b] [--max-games N] [--out FILE]
//
// Label: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure. Run it at `nice -n 19`.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { SITUATIONAL_FORMAT, SITUATIONAL_LABEL, situationalSetup, type StoredGame, tallyGame } from "../src/situational-move-use.ts";
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
const maxTurns = integer("max-turns");
const matches = arg("matches")?.split(",");
const maxGames = arg("max-games") === undefined ? Infinity : integer("max-games");
const root = join(dir, `g${String(generation).padStart(6, "0")}`, "games");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const setup = situationalSetup(engine, GOLDEN_FORMAT, maxTurns, goldenWorldFactsData(), process.env.SHOWDOWN_DIR ? { showdown_dir: process.env.SHOWDOWN_DIR } : {});
const records = readdirSync(root)
  .filter((f) => f.endsWith(".json"))
  .map((f) => JSON.parse(readFileSync(join(root, f), "utf8")) as StoredGame & { run_id: string; evaluation_key?: string })
  .filter((r) => matches === undefined || matches.includes(r.match_id))
  .sort((a, b) => (a.match_id < b.match_id ? -1 : a.match_id > b.match_id ? 1 : a.game - b.game));
const games = [];
let unreplayable = 0;
const taken = new Map<string, number>();
for (const record of records) {
  const n = taken.get(record.match_id) ?? 0;
  if (n >= maxGames) continue;
  taken.set(record.match_id, n + 1);
  const tally = await tallyGame(setup, record);
  if (tally === null) unreplayable++;
  else games.push(tally);
}
const first = records[0];
const out = JSON.stringify({
  format: SITUATIONAL_FORMAT,
  label: SITUATIONAL_LABEL,
  pin: engine.commit,
  generation,
  run_id: first?.run_id ?? null,
  evaluation_key: first?.evaluation_key ?? null,
  unreplayable_games: unreplayable,
  games,
});
const file = arg("out");
if (file === undefined) console.log(out);
else writeFileSync(file, out + "\n");
