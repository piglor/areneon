// The best-response evaluation for the Python driver (ml/areneon_ml/progress/best_response.py): an evaluation-only responder
// with white-box access to a frozen pair's weights, against the pair, on the validation pairs of a team dataset.
//   node packages/team-dataset/scripts/best-response.ts --config … --mode argmax|sample --pairs N --worlds W --k-own K --out DIR …
import { loadRulesSnapshot, resolveFormat, runBestResponseCli, UsageError } from "@areneon/showdown-bridge";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { createOpponentTeams } from "../src/opponent-teams.ts";
import { datasetTrainingHashes, datasetValidationPool } from "../src/progress-pool.ts";

try {
  process.exitCode = await runBestResponseCli(process.argv.slice(2), console.log, datasetValidationPool, {
    run_teams: datasetTrainingHashes,
    teams: (engine, formatId) =>
      createOpponentTeams({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest: resolveFormat(engine, formatId, loadRulesSnapshot(engine.commit)).manifest, arm: "generator-matched" }),
  });
} catch (error) {
  console.error(`best-response: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
