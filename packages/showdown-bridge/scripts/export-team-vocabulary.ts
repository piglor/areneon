// Export the pinned engine's team vocabulary (ADR-013 section 2) and print its
// hash:
//   node packages/showdown-bridge/scripts/export-team-vocabulary.ts
// It rewrites the golden vocabulary and its summary under
// fixtures/showdown/<format>/. Run it after a deliberate Showdown pin bump or a
// change to the vocabulary export; review the diff: a changed entry is a
// change of what the pinned validator accepts, and it changes the vocabulary
// hash every team manifest carries.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { TEAM_VOCABULARY_FILE, TEAM_VOCABULARY_SUMMARY, exportTeamVocabulary, teamVocabularySummary } from "../src/team-vocabulary.ts";

const vocabulary = exportTeamVocabulary(loadEngine(), GOLDEN_FORMAT);
mkdirSync(GOLDEN_DIR, { recursive: true });
writeFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), `${JSON.stringify(vocabulary, null, 2)}\n`);
const summary = teamVocabularySummary(vocabulary);
writeFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`wrote ${TEAM_VOCABULARY_FILE} and ${TEAM_VOCABULARY_SUMMARY}`);
console.log(`vocabulary_hash ${summary.vocabulary_hash}`);
