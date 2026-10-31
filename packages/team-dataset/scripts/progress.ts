// One generation's progress evaluation for the Python driver (ml/areneon_ml/progress):
// the validation split of the run's team dataset is the only team source.
//   node packages/team-dataset/scripts/progress.ts --config … --generation N …
import { runProgressCli, UsageError } from "@areneon/showdown-bridge";
import { datasetTrainingHashes, datasetValidationPool } from "../src/progress-pool.ts";
import { beliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { progressSearchSetup } from "../src/progress-search-setup.ts";

try {
  process.exitCode = await runProgressCli(process.argv.slice(2), console.log, datasetValidationPool, { search: progressSearchSetup, run_teams: datasetTrainingHashes, belief_worlds: beliefWorldsFactory });
} catch (error) {
  console.error(`progress: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
