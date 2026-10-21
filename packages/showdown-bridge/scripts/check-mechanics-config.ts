// Check config/mechanics/<format>.json against the pinned engine and the gap
// ledger (ADR-016 section 3). Part of `make ts`.
import { readdirSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { checkMechanicsConfig, loadMechanicsConfig, MECHANICS_CONFIG_DIR } from "../src/mechanics-config.ts";

const engine = loadEngine();
const files = readdirSync(MECHANICS_CONFIG_DIR).filter((file) => file.endsWith(".json"));
let failed = files.length === 0;
for (const file of files) {
  try {
    for (const problem of checkMechanicsConfig(engine, loadMechanicsConfig(file.replace(/\.json$/, "")))) {
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
