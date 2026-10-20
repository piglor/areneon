// Capture the A1 and A2 observations of the ADR-005 equality fixtures, with the real candidates
// (enumerateCandidates), for the Python encoder's A1/A2 tensor-equality test (ADR-010 section 2):
//   node packages/bot-client/scripts/capture-encoder-pairs.ts
// A1 = the replayed battle's recordedChannel; A2 = the channel rebuilt from the recorded server frames.
// Writes fixtures/showdown/<format>/a2-pairs.json (CC0, synthetic). Review the diff.
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { encoderPairs } from "../test/encoder-pairs.ts";

const engine = loadEngine();
const out = fileURLToPath(new URL("../../../fixtures/showdown/", import.meta.url));
for (const [format, document] of Object.entries(await encoderPairs(engine, loadRulesSnapshot(engine.commit)))) {
  mkdirSync(`${out}${format}`, { recursive: true });
  writeFileSync(`${out}${format}/a2-pairs.json`, `${JSON.stringify(document)}\n`);
  console.log(`${format}: ${(document as { decisions: unknown[] }).decisions.length} decisions`);
}
