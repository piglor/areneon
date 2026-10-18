// Re-capture the shared contract sample corpus (ADR-008 §2) after a contract
// or golden-battle change:
//   node packages/showdown-bridge/scripts/capture-samples.ts
// Review the diff: every language must classify these the same way.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { contractSamples, SAMPLES_DIR } from "../src/samples.ts";

const samples = await contractSamples(loadEngine());
rmSync(join(SAMPLES_DIR, "v1"), { recursive: true, force: true });
for (const [file, content] of Object.entries(samples)) {
  mkdirSync(dirname(join(SAMPLES_DIR, file)), { recursive: true });
  writeFileSync(join(SAMPLES_DIR, file), `${JSON.stringify(content, null, 2)}\n`);
}
console.log(`wrote ${Object.keys(samples).length} files under ${SAMPLES_DIR}`);
