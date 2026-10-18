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
  const side = { name: "Areneon", id: "p1", pokemon: [] };
  const move = { move: "Protect", id: "protect", pp: 16, maxpp: 16, target: "self", disabled: false };
  it.each([
    ["a non-object", "[]"],
    ["null", "null"],
    ["a missing side", JSON.stringify({ wait: true })],
    ["a side without pokemon", JSON.stringify({ wait: true, side: { id: "p1" } })],
    ["no phase", JSON.stringify({ side })],
    ["a preview without a size", JSON.stringify({ teamPreview: true, side })],
    // The closed engine-mirror schema (ADR-008 §1b), checked before hashing.
    ["an unknown request key", JSON.stringify({ wait: true, side, turn: 3 })],
    ["a multi-battle ally", JSON.stringify({ active: [], side, ally: side })],
    ["a Z-Move offer", JSON.stringify({ active: [{ moves: [move], canZMove: [] }], side })],
    ["an rqid of 0", JSON.stringify({ wait: true, side, rqid: 0 })],
    ["a fractional rqid", JSON.stringify({ wait: true, side, rqid: 1.5 })],
    ["a numeric disabled flag", JSON.stringify({ active: [{ moves: [{ ...move, disabled: 1 }] }], side })],
    ["a side id outside p1-p4", JSON.stringify({ wait: true, side: { ...side, id: "p5" } })],
    // Combinations the engine never sends; the schema alone cannot refuse them.
    ["wait and move at once", JSON.stringify({ wait: true, active: [], side })],
    ["preview and force switch at once", JSON.stringify({ teamPreview: true, maxChosenTeamSize: 4, forceSwitch: [true], side })],
    ["force switch and move at once", JSON.stringify({ forceSwitch: [true], active: [], side })],
  ])("%s", (_label, body) => {
    expect(() => decodeRequest(body)).toThrow(RequestDecodeError);
  });

  it("recognises wait requests", () => {
    expect(decodeRequest(JSON.stringify({ wait: true, side })).phase).toBe("wait");
  });

  it("names every phase a request marks when it marks more than one", () => {
    expect(() => decodeRequest(JSON.stringify({ wait: true, active: [], side }))).toThrow(
      "request marks more than one phase: wait, move",
    );
  });

  it("names what did not match the engine's shape", () => {
    expect(() => decodeRequest(JSON.stringify({ wait: true, side, turn: 3 }))).toThrow(
      /request does not match the engine's request shape: \/ must NOT have additional properties \(turn\)/,
    );
  });

  it("accepts the server's rqid and a disabled reason string", () => {
    const decoded = decodeRequest(
      JSON.stringify({ active: [{ moves: [{ ...move, disabled: "Taunt" }] }], side, rqid: 2 }),
    );
    expect(decoded).toMatchObject({ phase: "move", rqid: 2 });
  });
});
