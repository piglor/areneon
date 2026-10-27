// Export the pinned engine's Baton Pass / Shed Tail volatile-copy rule (#1102) and print its hash:
//   node packages/showdown-bridge/scripts/export-volatile-copy.ts
// Rewrites ml/areneon_ml/reducer/data/volatile-copy.<format>.json, which the Python reducer and
// the shared interpreter both load. Review the diff: a changed entry is a change of the engine's
// own condition data. Run after a pin bump.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { exportVolatileCopy, VOLATILE_COPY_DIR, volatileCopyFile } from "../src/volatile-copy.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const copy = exportVolatileCopy(engine, GOLDEN_FORMAT, process.env.SHOWDOWN_DIR || undefined);
mkdirSync(VOLATILE_COPY_DIR, { recursive: true });
writeFileSync(join(VOLATILE_COPY_DIR, volatileCopyFile(GOLDEN_FORMAT)), `${JSON.stringify(copy, null, 2)}\n`);
console.log(`wrote ${volatileCopyFile(GOLDEN_FORMAT)}: ${copy.not_copied.length} ids not copied`);
console.log(`volatile_copy_hash ${copy.volatile_copy_hash}`);
