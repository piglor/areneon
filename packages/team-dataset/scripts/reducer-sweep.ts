// The reducer sweep's play side (#919). Plays seeded battles on a run config's
// dataset pool with one actor mix and writes one JSON line per battle:
//   node packages/team-dataset/scripts/reducer-sweep.ts --config RUN_CONFIG.json --mix uniform-all \
//     --battles 1000 [--start 0] [--run-key sweep-1] [--only-ability Illusion] --out OUT.jsonl
// Then: uv run python scripts/reducer_sweep.py OUT.jsonl... (reducer and encoder at every request).
import { createWriteStream } from "node:fs";
import { readFileSync } from "node:fs";
import { once } from "node:events";
import { loadEngine, readTrainingRunConfig } from "@areneon/showdown-bridge";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { parseSweepArgs, playSweepBattle, restrictPool, SweepUsageError } from "../src/reducer-sweep.ts";

let args;
try {
  args = parseSweepArgs(process.argv.slice(2));
} catch (error) {
  if (!(error instanceof SweepUsageError)) throw error;
  console.error(`reducer-sweep: ${error.message}`);
  process.exit(2);
}
const { mix, battles, start, run_key: runKey } = args;

const given = readTrainingRunConfig(JSON.parse(readFileSync(args.config, "utf8")));
// --only-ability plays the teams that hold the ability, so the config's excluded abilities do not apply.
const config = args.only_ability === undefined ? given : { ...given, generation: { ...given.generation, excluded_abilities: [] } };
const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const whole = datasetTeamPool(engine, config);
const pool = args.only_ability === undefined ? whole : restrictPool(whole, args.only_ability);
const out = createWriteStream(args.out);
for (let index = start; index < start + battles; index++) {
  const battle = await playSweepBattle(engine, config.format_id, pool, { run_key: runKey, mix, index, max_turns: config.generation.max_turns });
  if (!out.write(`${JSON.stringify(battle)}\n`)) await once(out, "drain");
}
out.end();
await once(out, "finish");
