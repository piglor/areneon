// Check config/features/<format>/probe-outcome-likely*.json against the pinned engine and the committed source tables (ADR-022 erratum E-22.7). Part of `make ts`.
import { readdirSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { checkLikelyConfig, LIKELY_CONFIG_DIR, likelyConfigFiles, loadLikelySetup } from "../src/likely-threats-config.ts";

const engine = loadEngine();
const formats = readdirSync(LIKELY_CONFIG_DIR, { withFileTypes: true }).filter((entry) => entry.isDirectory());
let failed = formats.length === 0;
for (const { name } of formats) {
  const files = likelyConfigFiles(name);
  if (files.length === 0) {
    console.error(`${name}: no probe-outcome-likely record`);
    failed = true;
  }
  for (const file of files) {
    try {
      const { config } = loadLikelySetup(name, file);
      const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(name).mod) as unknown as Parameters<typeof checkLikelyConfig>[1];
      for (const problem of checkLikelyConfig(config, dex, goldenTeamVocabulary().engine_data_hash)) {
        console.error(`${name}/${file}: ${problem}`);
        failed = true;
      }
    } catch (error) {
      console.error(`${name}/${file}: ${(error as Error).message}`);
      failed = true;
    }
  }
}
if (failed) process.exit(1);
console.log("probe-outcome-likely configs: ok");
