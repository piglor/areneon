// The game process a CLI test starts (`--game-processes 2`): the team dataset's play-generation script with the committed fixture
// pool in place of a dataset, so a run config that names a dataset path can be played without one. Not a test.
import { runGenerationCli, UsageError } from "@areneon/showdown-bridge";
import { trainingTeamPool } from "../../showdown-bridge/src/league-generation.ts";
import { progressSearchSetup } from "../src/progress-search-setup.ts";

try {
  process.exitCode = await runGenerationCli(process.argv.slice(2), console.log, (engine, config) => trainingTeamPool(engine, config.generation.excluded_abilities), process.argv[1], undefined, undefined, progressSearchSetup);
} catch (error) {
  console.error(`play-generation: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
if (process.exitCode !== 0 && process.exitCode !== undefined) process.exit();
