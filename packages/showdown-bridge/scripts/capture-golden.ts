// Re-capture the golden fixtures after a Showdown pin bump:
//   node packages/showdown-bridge/scripts/capture-golden.ts
// It also re-logs the validator verdict for every team fixture.
// Review the diff: a changed fixture is a behaviour change of the engine.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import {
  battleSummary,
  captureProvenance,
  GOLDEN_DIR,
  GOLDEN_PROVENANCE,
  GOLDEN_SUMMARY,
  goldenBattle,
  goldenChannels,
  goldenFixtures,
  TEAM_VERDICTS,
  TEAMS_DIR,
  syntheticFixtureHashes,
  teamVerdicts,
} from "../src/golden.ts";

const engine = loadEngine();
const record = await goldenBattle(engine);
const files: Record<string, unknown> = {
  ...goldenFixtures(record),
  ...goldenChannels(record, syntheticFixtureHashes(engine)),
  [GOLDEN_SUMMARY]: battleSummary(record),
  [GOLDEN_PROVENANCE]: captureProvenance(record),
};
for (const [file, content] of Object.entries(files)) {
  mkdirSync(dirname(join(GOLDEN_DIR, file)), { recursive: true });
  writeFileSync(join(GOLDEN_DIR, file), `${JSON.stringify(content, null, 2)}\n`);
  console.log(`wrote ${file}`);
}

writeFileSync(join(TEAMS_DIR, TEAM_VERDICTS), `${JSON.stringify(teamVerdicts(engine), null, 2)}\n`);
console.log(`wrote ${TEAM_VERDICTS}`);
