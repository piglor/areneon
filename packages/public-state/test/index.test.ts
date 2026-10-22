import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createInterpreter, INTERPRETER_PATH, interpretChannel, parseLine, readerSha256, toId } from "../src/index.ts";
import { fixtures, fullDex } from "./support.ts";

describe("the typed wrapper over the shared state reader", () => {
  it("gives the reader's own functions", () => {
    expect(toId("Mr. Mime")).toBe("mrmime");
    expect(parseLine("|move|p1a: A|Protect|p1a: A").kind).toBe("move");
    const interp = createInterpreter({ player: "p1" });
    expect(interp.feed("|gametype|doubles").flagged).toBeNull();
    expect(interp.snapshot().gametype).toBe("doubles");
  });

  it("reads a channel to the same states as the reader's own run, never in omniscient mode", () => {
    const fixture = fixtures().find((f) => f.id === "golden/p1")!;
    const run = interpretChannel(fixture.player, fixture.lines, fullDex());
    expect(run.flags).toEqual([]);
    expect(run.frames.length).toBeGreaterThan(0);
    expect(run.frames.at(-1)!.state.omniscient).toBeUndefined();
    expect(run.frames.at(-1)!.state.opponent.roster.every((mon) => !mon.own)).toBe(true);
  });

  it("names the file the replay pages inline, and hashes its bytes", () => {
    expect(INTERPRETER_PATH.endsWith("packages/public-state/src/interpreter.cjs")).toBe(true);
    expect(readerSha256()).toBe(createHash("sha256").update(readFileSync(INTERPRETER_PATH)).digest("hex"));
    expect(readerSha256()).toMatch(/^[0-9a-f]{64}$/);
  });
});
