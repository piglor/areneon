// The version manifest of a format, for the Python training driver:
//   node packages/showdown-bridge/scripts/format-manifest.ts --format <id>
// See src/generation-cli.ts. SHOWDOWN_DIR points at another built checkout.
import { runManifestCli, UsageError } from "../src/generation-cli.ts";

try {
  process.exitCode = runManifestCli(process.argv.slice(2));
} catch (error) {
  console.error(`format-manifest: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
