// Check config/baselines/damage-aware.<version>.json against the pinned engine's team vocabulary
// (ADR-018 section 3). Part of `make ts`.
import { readdirSync } from "node:fs";
import { DAMAGE_AWARE_CONFIG_DIR, loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { loadEngine } from "../src/engine.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";

const engine = loadEngine();
// The committed vocabulary is the pinned engine's: team-vocabulary.test.ts holds it equal to a fresh export.
const vocabulary = goldenTeamVocabulary();
const files = readdirSync(DAMAGE_AWARE_CONFIG_DIR).filter((file) => /^damage-aware\.v\d+\.json$/.test(file));
let failed = files.length === 0 || vocabulary.showdown_commit !== engine.commit;
if (vocabulary.showdown_commit !== engine.commit) console.error("the committed team vocabulary is for another engine commit");
for (const file of files) {
  try {
    loadDamageAwareConfig(file.replace(/^damage-aware\.(v\d+)\.json$/, "$1"), vocabulary);
  } catch (error) {
    console.error(`${file}: ${(error as Error).message}`);
    failed = true;
  }
}
if (failed) process.exit(1);
console.log("damage-aware configs: ok");
