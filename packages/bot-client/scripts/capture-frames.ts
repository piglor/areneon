// Re-capture the protocol reader's frame fixtures after a Showdown pin bump:
//   node packages/bot-client/scripts/capture-frames.ts
// Nothing connects anywhere: the frames come from the pinned in-process sim.
// Review the diff: a changed frame is a behaviour change of the engine.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "@areneon/showdown-bridge";
import { captureFrames, FRAMES_DIR } from "../testing/server-frames.ts";

const fixtures = await captureFrames(loadEngine());
mkdirSync(FRAMES_DIR, { recursive: true });
for (const [file, content] of Object.entries(fixtures)) {
  writeFileSync(join(FRAMES_DIR, file), `${JSON.stringify(content, null, 2)}\n`);
  console.log(`wrote ${join(FRAMES_DIR, file)}`);
}
