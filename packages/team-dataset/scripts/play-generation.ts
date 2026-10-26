// One generation of A1 play for the Python training driver, for a run whose
// config names a team dataset (`team_dataset`); without one it plays the
// interim pool exactly as packages/showdown-bridge/scripts/play-generation.ts.
//   node packages/team-dataset/scripts/play-generation.ts --config … --generation N …
import { interimPoolSource, runGenerationCli, UsageError } from "@areneon/showdown-bridge";
import { datasetPoolSource } from "../src/loop-pool.ts";

try {
  process.exitCode = await runGenerationCli(process.argv.slice(2), console.log, datasetPoolSource(interimPoolSource), process.argv[1]);
} catch (error) {
  console.error(`play-generation: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
// A failed generation ends the process now: a game that was still running when it failed must not keep it (and its workers) alive (#1100).
if (process.exitCode !== 0 && process.exitCode !== undefined) process.exit();
