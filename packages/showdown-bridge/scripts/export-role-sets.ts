// Export the pinned engine's Champions doubles role sets (ADR-020 section 3)
// and print their hash:
//   node packages/showdown-bridge/scripts/export-role-sets.ts
// It reads the golden team vocabulary (run export-team-vocabulary.ts first
// after a pin bump) and rewrites the golden role-set table and its summary
// under fixtures/showdown/<format>/. Review the diff: a changed entry is a
// change of the engine's own file, and it changes the role-set table hash.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { exportRoleSets, ROLE_SETS_FILE, ROLE_SETS_SUMMARY, roleSetSummary } from "../src/role-sets.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE } from "../src/team-vocabulary.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const roleSets = exportRoleSets(engine, GOLDEN_FORMAT, vocabulary, table);
mkdirSync(GOLDEN_DIR, { recursive: true });
writeFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), `${JSON.stringify(roleSets, null, 2)}\n`);
const summary = roleSetSummary(roleSets);
writeFileSync(join(GOLDEN_DIR, ROLE_SETS_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`wrote ${ROLE_SETS_FILE} and ${ROLE_SETS_SUMMARY}`);
console.log(`role_set_table_hash ${summary.role_set_table_hash}`);
