// The game process a CLI test starts (`--game-processes 2`) for a sheet run: the committed fixture pool and the belief features factory. Not a test.
import { runGenerationCli, UsageError } from "@areneon/showdown-bridge";
import { trainingTeamPool } from "../../showdown-bridge/src/league-generation.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";

try {
  process.exitCode = await runGenerationCli(process.argv.slice(2), console.log, (engine, config) => trainingTeamPool(engine, config.generation.excluded_abilities), process.argv[1], undefined, (context) => createBeliefFeatures(context));
} catch (error) {
  console.error(`play-generation: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
if (process.exitCode !== 0 && process.exitCode !== undefined) process.exit();
