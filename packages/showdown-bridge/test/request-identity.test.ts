import { describe, expect, it } from "vitest";
import { canonicalJson, requestHash } from "../src/index.ts";

describe("canonicalJson", () => {
  it("sorts keys recursively and drops whitespace", () => {
    expect(canonicalJson({ b: 1, a: { d: [true, null], c: "x" } })).toBe(
      '{"a":{"c":"x","d":[true,null]},"b":1}',
    );
  });

  it("serialises scalars like JSON.stringify", () => {
    expect(canonicalJson(null)).toBe("null");
    expect(canonicalJson("é")).toBe('"é"');
    expect(canonicalJson(1.5)).toBe("1.5");
  });

  it("rejects non-finite numbers instead of collapsing them to null", () => {
    expect(() => canonicalJson(Number.NaN)).toThrow(RangeError);
    expect(() => canonicalJson({ a: [Infinity] })).toThrow(RangeError);
  });
});

describe("requestHash", () => {
  const request = { teamPreview: true, side: { id: "p1", pokemon: [] }, maxChosenTeamSize: 4 };

  it("is independent of key order", () => {
    const reordered = { maxChosenTeamSize: 4, side: { pokemon: [], id: "p1" }, teamPreview: true };
    expect(requestHash(reordered)).toBe(requestHash(request));
  });

  it("ignores the server-only top-level rqid", () => {
    expect(requestHash({ ...request, rqid: 7 })).toBe(requestHash(request));
  });

  it("keeps a nested rqid-named field, which is request content", () => {
    expect(requestHash({ ...request, side: { id: "p1", pokemon: [], rqid: 1 } })).not.toBe(
      requestHash(request),
    );
  });

  it("hashes non-ASCII content stably", () => {
    const named = { ...request, side: { id: "p1", name: "Flabébé", pokemon: [] } };
    expect(requestHash(named)).toBe(requestHash({ ...named }));
    expect(requestHash(named)).not.toBe(requestHash(request));
  });

  it("changes when content changes", () => {
    expect(requestHash({ ...request, maxChosenTeamSize: 3 })).not.toBe(requestHash(request));
  });

  it("is a sha256 hex digest", () => {
    expect(requestHash(request)).toMatch(/^[0-9a-f]{64}$/);
  });
});
