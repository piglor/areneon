import { readFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { captureFrames, FRAMES_DIR, serverFrames } from "../testing/server-frames.ts";

describe("frame fixtures (fixtures/bot-client/frames)", () => {
  it("equal a fresh capture from the pinned engine", async () => {
    const fresh = await captureFrames(loadEngine());
    for (const [file, content] of Object.entries(fresh)) {
      const committed = JSON.parse(readFileSync(join(FRAMES_DIR, file), "utf8")) as unknown;
      expect(committed, file).toEqual(JSON.parse(JSON.stringify(content)));
    }
  });
});

describe("serverFrames (the server's rqid, reconstructed)", () => {
  const request = (body: object): string[] => [`|request|${JSON.stringify(body)}`];
  const rqids = (frames: string[]): number[] =>
    frames.map((f) => (JSON.parse(f.split("\n")[1]?.slice("|request|".length) ?? "") as { rqid: number }).rqid);

  it("counts p2's request of each round after p1's, and an update for p1 alone", () => {
    const frames = serverFrames({
      p1: [request({}), request({ update: true }), request({})],
      p2: [request({}), request({})],
    });
    expect(rqids(frames)).toEqual([1, 3, 4]);
  });

  it("refuses chunks whose rqids would be ambiguous", () => {
    expect(() => serverFrames({ p1: [request({})], p2: [request({ update: true })] })).toThrow(/ambiguous/);
    expect(() => serverFrames({ p1: [request({}), request({})], p2: [request({})] })).toThrow(/ambiguous/);
  });
});
