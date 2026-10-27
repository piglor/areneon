// The scripted head-to-head for the Python driver (ml/areneon_ml/progress/scripted.py): two scripted baselines
// play each other on the validation pairs of the run's team dataset (or of an evaluation-only dataset).
//   node packages/team-dataset/scripts/scripted-h2h.ts --config … --arm-a damage-aware@v1 --arm-b damage-aware@v2 --pairs N --out DIR
import { runScriptedH2HCli, UsageError } from "@areneon/showdown-bridge";
import { beliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";

try {
  process.exitCode = await runScriptedH2HCli(process.argv.slice(2), console.log, datasetValidationPool, beliefWorldsFactory, process.argv[1]);
} catch (error) {
  console.error(`scripted-h2h: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
