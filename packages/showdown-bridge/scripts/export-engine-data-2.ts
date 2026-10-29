// Export version 2 of the pinned engine's table (ADR-010 erratum A4 item 7): the version 1
// vocabularies plus the engine's condition ids. Writes the golden table and its summary:
//   node packages/showdown-bridge/scripts/export-engine-data-2.ts
// Run it after a deliberate Showdown pin bump (beside export-engine-data.ts) or a change of the
// export; review the diff. The version 1 table is not touched.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ENGINE_DATA_2_FILE, ENGINE_DATA_2_SUMMARY, engineDataSummary, exportEngineDataV2 } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";

const table = exportEngineDataV2(loadEngine(), GOLDEN_FORMAT);
mkdirSync(GOLDEN_DIR, { recursive: true });
writeFileSync(join(GOLDEN_DIR, ENGINE_DATA_2_FILE), `${JSON.stringify(table, null, 2)}\n`);
const summary = engineDataSummary(table);
writeFileSync(join(GOLDEN_DIR, ENGINE_DATA_2_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`wrote ${ENGINE_DATA_2_FILE} and ${ENGINE_DATA_2_SUMMARY}`);
console.log(`engine_data_hash ${summary.engine_data_hash}`);
