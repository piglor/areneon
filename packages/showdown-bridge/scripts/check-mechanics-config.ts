// Check config/mechanics/<format>.json against the pinned engine and the gap
// ledger (ADR-016 section 3). Part of `make ts`.
import { readdirSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { checkKoBlockers } from "../src/mechanics-ko-candidates.ts";
import { checkMechanicsConfig, loadMechanicsConfig, MECHANICS_CONFIG_DIR } from "../src/mechanics-config.ts";

const engine = loadEngine();
// The committed vocabulary is the pinned engine's: team-vocabulary.test.ts holds it equal to a fresh export.
const vocabulary = goldenTeamVocabulary();
const files = readdirSync(MECHANICS_CONFIG_DIR).filter((file) => file.endsWith(".json"));
let failed = files.length === 0;
for (const file of files) {
  try {
    const config = loadMechanicsConfig(file.replace(/\.json$/, ""));
    for (const problem of [...checkMechanicsConfig(engine, config), ...checkKoBlockers(engine, config, vocabulary)]) {
      console.error(`${file}: ${problem}`);
      failed = true;
    }
  } catch (error) {
    console.error(`${file}: ${(error as Error).message}`);
    failed = true;
  }
}
if (failed) process.exit(1);
console.log("mechanics configs: ok");
