// Re-capture the golden request fixtures after a Showdown pin bump:
//   node packages/showdown-bridge/scripts/capture-golden.ts
// Review the diff: a changed fixture is a behaviour change of the engine.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import {
  battleSummary,
  GOLDEN_DIR,
  GOLDEN_SUMMARY,
  goldenBattle,
  goldenFixtures,
} from "../src/golden.ts";

const record = await goldenBattle(loadEngine());
const files: Record<string, unknown> = {
  ...goldenFixtures(record),
  [GOLDEN_SUMMARY]: battleSummary(record),
};
for (const [file, content] of Object.entries(files)) {
  writeFileSync(join(GOLDEN_DIR, file), `${JSON.stringify(content, null, 2)}\n`);
  console.log(`wrote ${file}`);
}
