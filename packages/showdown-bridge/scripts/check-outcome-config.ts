// Check config/features/<format>/probe-outcome.v1.json against the pinned engine (ADR-022 erratum E-22.5.3). Part of `make ts`.
import { readdirSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { checkOutcomeConfig, loadOutcomeConfig, OUTCOME_CONFIG_DIR } from "../src/outcome-config.ts";

const engine = loadEngine();
const formats = readdirSync(OUTCOME_CONFIG_DIR, { withFileTypes: true }).filter((entry) => entry.isDirectory());
let failed = formats.length === 0;
for (const { name } of formats) {
  try {
    const config = loadOutcomeConfig(name);
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(name).mod) as unknown as Parameters<typeof checkOutcomeConfig>[1];
    for (const problem of checkOutcomeConfig(config, dex)) {
      console.error(`${name}: ${problem}`);
      failed = true;
    }
  } catch (error) {
    console.error(`${name}: ${(error as Error).message}`);
    failed = true;
  }
}
if (failed) process.exit(1);
console.log("probe-outcome configs: ok");
