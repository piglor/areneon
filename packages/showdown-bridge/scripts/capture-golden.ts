// Re-capture the golden request fixtures after a Showdown pin bump:
//   node packages/showdown-bridge/scripts/capture-golden.ts
// Review the diff: a changed fixture is a behaviour change of the engine.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, goldenBattle, goldenFixtures } from "../src/golden.ts";

const fixtures = goldenFixtures(await goldenBattle(loadEngine()));
for (const [file, fixture] of Object.entries(fixtures)) {
  writeFileSync(join(GOLDEN_DIR, file), `${JSON.stringify(fixture, null, 2)}\n`);
  console.log(`wrote ${file} (request_seq ${fixture.request_seq})`);
}
