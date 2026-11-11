// Export the pinned engine's history-facts data (ADR-022 E-22.5.2) and print the hash:
//   node packages/showdown-bridge/scripts/export-history-facts-data.ts
// Rewrites ml/areneon_ml/reducer/data/history-facts.<format>.json, which the Python reducer loads for the history
// facts (the stall counter constants, the move-line action sources and the timed slot conditions). Review the diff;
// run after a pin bump or a change of the format's move list.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { exportHistoryFactsData, goldenHistoryFactsData, HISTORY_FACTS_DIR, historyFactsFile } from "../src/history-facts-data.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const file = exportHistoryFactsData(engine.commit, GOLDEN_FORMAT, goldenHistoryFactsData());
mkdirSync(HISTORY_FACTS_DIR, { recursive: true });
writeFileSync(join(HISTORY_FACTS_DIR, historyFactsFile(GOLDEN_FORMAT)), `${JSON.stringify(file, null, 2)}\n`);
console.log(`wrote ${historyFactsFile(GOLDEN_FORMAT)}: ${file.action_sources.length} action sources, ${file.slot_conditions.length} slot conditions`);
console.log(`history_facts_data_hash ${file.history_facts_data_hash}`);
