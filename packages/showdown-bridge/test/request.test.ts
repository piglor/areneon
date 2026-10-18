import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_DIR, type RequestFixture } from "../src/golden.ts";
import { decodeRequest, RequestDecodeError, requestHash } from "../src/index.ts";

function fixture(file: string): RequestFixture {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as RequestFixture;
}

describe("decodeRequest on the golden fixtures", () => {
  it("decodes Team Preview: bring 6, choose 4, two leads", () => {
    const { request, request_hash } = fixture("preview.request.json");
    const decoded = decodeRequest(JSON.stringify(request));
    expect(decoded.phase).toBe("preview");
    expect(decoded.player).toBe("p1");
    expect(decoded.preview).toEqual({ teamSize: 6, maxChosenTeamSize: 4 });
    // Two leads: an engine fact, read from its first turn request's active
    // slots, never declared in code or rules records (ADR-007).
    const turn = decodeRequest(JSON.stringify(fixture("turn.request.json").request));
    expect(turn.request.active).toHaveLength(2);
    expect(decoded.request_hash).toBe(request_hash);
    expect(decoded.rqid).toBeUndefined();
  });

  it("decodes a turn request with two active slots", () => {
    const { request } = fixture("turn.request.json");
    const decoded = decodeRequest(JSON.stringify(request));
    expect(decoded.phase).toBe("move");
    expect(decoded.request.active).toHaveLength(2);
    expect(decoded.preview).toBeUndefined();
  });

  it("decodes a forced switch", () => {
    const { request } = fixture("force-switch.request.json");
    const decoded = decodeRequest(JSON.stringify(request));
    expect(decoded.phase).toBe("forceSwitch");
    expect(decoded.request.forceSwitch).toEqual([true, false]);
  });

  it("extracts rqid from the server path without changing the hash", () => {
    const { request } = fixture("turn.request.json");
    const decoded = decodeRequest(JSON.stringify({ ...request, rqid: 7 }));
    expect(decoded.rqid).toBe(7);
    expect(decoded.request_hash).toBe(requestHash(request));
  });
});

describe("decodeRequest rejects malformed input", () => {
  const side = { id: "p1", pokemon: [] };
  it.each([
    ["a non-object", "[]"],
    ["null", "null"],
    ["a missing side", JSON.stringify({ wait: true })],
    ["a side without pokemon", JSON.stringify({ wait: true, side: { id: "p1" } })],
    ["no phase", JSON.stringify({ side })],
    ["a preview without a size", JSON.stringify({ teamPreview: true, side })],
  ])("%s", (_label, body) => {
    expect(() => decodeRequest(body)).toThrow(RequestDecodeError);
  });

  it("recognises wait requests", () => {
    expect(decodeRequest(JSON.stringify({ wait: true, side })).phase).toBe("wait");
  });
});
