// One generation of A1 play for the Python training driver (ml/areneon_ml/loop):
//   node packages/showdown-bridge/scripts/play-generation.ts --config … --generation N …
// See src/generation-cli.ts. SHOWDOWN_DIR points at another built checkout.
import { runGenerationCli, UsageError } from "../src/generation-cli.ts";

try {
  process.exitCode = await runGenerationCli(process.argv.slice(2), console.log, undefined, process.argv[1]);
} catch (error) {
  console.error(`play-generation: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
// A failed generation ends the process now: a game that was still running when it failed must not keep it (and its workers) alive (#1100).
if (process.exitCode !== 0 && process.exitCode !== undefined) process.exit();
