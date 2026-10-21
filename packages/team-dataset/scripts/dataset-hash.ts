// Verify a generated team dataset directory and print its manifest hash, the
// value a run config's `team_dataset.manifest_hash` must hold:
//   node packages/team-dataset/scripts/dataset-hash.ts --dir OUT_DIR --format gen9championsvgc2026regmc
import { parseArgs } from "node:util";
import { loadEngine } from "@areneon/showdown-bridge";
import { readDataset } from "../src/dataset-reader.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";

try {
  const { dir, format } = parseArgs({ options: { dir: { type: "string" }, format: { type: "string" } }, strict: true }).values;
  if (dir === undefined || format === undefined) throw new Error("--dir and --format are required");
  console.log(readDataset(dir, loadFixtureInputs(loadEngine(process.env.SHOWDOWN_DIR || undefined), format)).manifest_hash);
} catch (error) {
  console.error(`dataset-hash: ${(error as Error).message}`);
  process.exitCode = 1;
}
