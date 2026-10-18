// Regenerate docs/showdown-champions-gaps.md from the gap records:
//   node packages/showdown-bridge/scripts/gaps-doc.ts
// A test fails while the committed doc differs from this output.
import { writeFileSync } from "node:fs";
import { loadEngine } from "../src/engine.ts";
import { formatLabels, GAPS_DOC, renderGapsDoc } from "../src/gaps-doc.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
writeFileSync(GAPS_DOC, renderGapsDoc(snapshot, formatLabels(engine, snapshot)));
console.log(`wrote ${GAPS_DOC}`);
