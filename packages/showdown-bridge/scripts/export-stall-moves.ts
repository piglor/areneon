// Export the pinned engine's stalling moves (#1013) and print the hash:
//   node packages/showdown-bridge/scripts/export-stall-moves.ts
// Rewrites ml/areneon_ml/reducer/data/stall-moves.<format>.json, which the Python reducer loads for its
// Protect streak. Review the diff; run after a pin bump or a change of the format's move list.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { exportStallMoves, STALL_MOVES_DIR, stallMovesFile } from "../src/stall-moves.ts";
import type { FactsTable } from "../src/world-facts-data.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const table = JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as FactsTable;
const moves = exportStallMoves(engine, GOLDEN_FORMAT, table, process.env.SHOWDOWN_DIR || undefined);
mkdirSync(STALL_MOVES_DIR, { recursive: true });
writeFileSync(join(STALL_MOVES_DIR, stallMovesFile(GOLDEN_FORMAT)), `${JSON.stringify(moves, null, 2)}\n`);
console.log(`wrote ${stallMovesFile(GOLDEN_FORMAT)}: ${moves.stall_moves.length} stall moves`);
console.log(`stall_moves_hash ${moves.stall_moves_hash}`);
