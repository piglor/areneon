import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { captureFrames, FRAMES_DIR } from "../testing/server-frames.ts";

describe("frame fixtures (fixtures/bot-client/frames)", () => {
  it("equal a fresh capture from the pinned engine", async () => {
    const fresh = await captureFrames(loadEngine());
    for (const [file, content] of Object.entries(fresh)) {
      const committed = JSON.parse(readFileSync(join(FRAMES_DIR, file), "utf8")) as unknown;
      expect(committed, file).toEqual(JSON.parse(JSON.stringify(content)));
    }
  });
});
