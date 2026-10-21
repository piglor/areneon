// Verify a generated team dataset directory and print its dataset hash, the
// value a run config's `team_dataset.dataset_hash` must hold:
//   node packages/team-dataset/scripts/dataset-hash.ts --dir OUT_DIR --format gen9championsvgc2026regmc
import { parseArgs } from "node:util";
import { loadEngine } from "@areneon/showdown-bridge";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";

try {
  const { dir, format } = parseArgs({ options: { dir: { type: "string" }, format: { type: "string" } }, strict: true }).values;
  if (dir === undefined || format === undefined) throw new Error("--dir and --format are required");
  console.log(verifiedDatasetHash(loadEngine(process.env.SHOWDOWN_DIR || undefined), dir, format));
} catch (error) {
  console.error(`dataset-hash: ${(error as Error).message}`);
  process.exitCode = 1;
}
