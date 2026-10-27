// Regenerate docs/mechanics-parity.md from fixtures/mechanics/, the calc
// divergence list and the gap ledger:
//   node packages/showdown-bridge/scripts/mechanics-parity.ts
// A test fails while the committed report differs from this output.
import { writeFileSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { loadParityInputs, PARITY_DOC, renderMechanicsParity } from "../src/mechanics-parity.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";

const engine = loadEngine();
const inputs = loadParityInputs(loadMechanicsConfig(GOLDEN_FORMAT), loadRulesSnapshot(engine.commit));
writeFileSync(PARITY_DOC, renderMechanicsParity(inputs));
console.log(`wrote ${PARITY_DOC}`);
