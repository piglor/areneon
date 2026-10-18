// Re-capture the golden fixtures after a Showdown pin bump:
//   node packages/showdown-bridge/scripts/capture-golden.ts
// It also re-logs the validator verdict for every team fixture.
// Review the diff: a changed fixture is a behaviour change of the engine.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import {
  battleSummary,
  captureProvenance,
  GOLDEN_DIR,
  GOLDEN_PROVENANCE,
  GOLDEN_SUMMARY,
  goldenBattle,
  goldenFixtures,
  TEAM_VERDICTS,
  TEAMS_DIR,
  teamVerdicts,
} from "../src/golden.ts";

const engine = loadEngine();
const record = await goldenBattle(engine);
const files: Record<string, unknown> = {
  ...goldenFixtures(record),
  [GOLDEN_SUMMARY]: battleSummary(record),
  [GOLDEN_PROVENANCE]: captureProvenance(record),
};
for (const [file, content] of Object.entries(files)) {
  writeFileSync(join(GOLDEN_DIR, file), `${JSON.stringify(content, null, 2)}\n`);
  console.log(`wrote ${file}`);
}

writeFileSync(join(TEAMS_DIR, TEAM_VERDICTS), `${JSON.stringify(teamVerdicts(engine), null, 2)}\n`);
console.log(`wrote ${TEAM_VERDICTS}`);
