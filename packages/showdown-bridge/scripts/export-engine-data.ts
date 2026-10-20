// Export the pinned engine's vocabularies and attributes (ADR-010 section 3)
// and print their hash:
//   node packages/showdown-bridge/scripts/export-engine-data.ts
// It rewrites the golden table and its summary under fixtures/showdown/<format>/.
// Run it after a deliberate Showdown pin bump; review the diff: a changed
// entry is a change of the engine's data, and it changes the encoder version.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ENGINE_DATA_FILE, ENGINE_DATA_SUMMARY, engineDataSummary, exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";

const table = exportEngineData(loadEngine(), GOLDEN_FORMAT);
mkdirSync(GOLDEN_DIR, { recursive: true });
writeFileSync(join(GOLDEN_DIR, ENGINE_DATA_FILE), `${JSON.stringify(table, null, 2)}\n`);
const summary = engineDataSummary(table);
writeFileSync(join(GOLDEN_DIR, ENGINE_DATA_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`wrote ${ENGINE_DATA_FILE} and ${ENGINE_DATA_SUMMARY}`);
console.log(`engine_data_hash ${summary.engine_data_hash}`);
