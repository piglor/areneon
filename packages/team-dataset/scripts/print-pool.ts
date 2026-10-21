// Print a run's training pool as one canonical JSON line, for the training-data
// inspector (python -m areneon_ml.loop inspect). It rebuilds the pool exactly as
// the loop's play script does and changes nothing:
//   node packages/team-dataset/scripts/print-pool.ts --config RUN_DIR/run-config.json
// A relative `team_dataset.path` in the config is read from the current directory.
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { jcs, type Json, loadEngine, readTrainingRunConfig } from "@areneon/showdown-bridge";
import { listPool } from "../src/pool-listing.ts";

try {
  const { config } = parseArgs({ options: { config: { type: "string" } }, strict: true }).values;
  if (config === undefined) throw new Error("--config is required");
  const run = readTrainingRunConfig(JSON.parse(readFileSync(config, "utf8")));
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  console.log(jcs(listPool(engine, run) as unknown as Json));
} catch (error) {
  console.error(`print-pool: ${(error as Error).message}`);
  process.exitCode = 1;
}
