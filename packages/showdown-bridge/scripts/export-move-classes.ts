// Export the move classes of the pinned engine's move data (#1164) and print
// their hash:
//   node packages/showdown-bridge/scripts/export-move-classes.ts
// It rewrites the golden move-class table and its summary under
// fixtures/showdown/<format>/. Review the diff: a changed entry is a change of
// the engine's own move data (or of a class rule), and it changes the table hash.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { exportMoveClasses, MOVE_CLASSES_FILE, MOVE_CLASSES_SUMMARY, moveClassSummary } from "../src/move-classes.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const classes = exportMoveClasses(engine, GOLDEN_FORMAT, table);
mkdirSync(GOLDEN_DIR, { recursive: true });
writeFileSync(join(GOLDEN_DIR, MOVE_CLASSES_FILE), `${JSON.stringify(classes, null, 2)}\n`);
const summary = moveClassSummary(classes);
writeFileSync(join(GOLDEN_DIR, MOVE_CLASSES_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
console.log(`wrote ${MOVE_CLASSES_FILE} and ${MOVE_CLASSES_SUMMARY}`);
console.log(`move_class_table_hash ${summary.move_class_table_hash} ${JSON.stringify(summary.counts)}`);
