import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { canonicalJson, jcs } from "../src/canonical-json.ts";
import * as requestIdentity from "../src/request-identity.ts";
import * as rules from "../src/rules.ts";

const SOURCE = new URL("../src/canonical-json.ts", import.meta.url);
const VECTORS = JSON.parse(
  readFileSync(new URL("../../../fixtures/contracts/jcs-vectors.json", import.meta.url), "utf8"),
) as { vectors: { name: string; json: string; jcs: string | null }[] };

describe("canonical-json", () => {
  it("imports nothing, so the bundled builder and archived rules-hash can depend on it", () => {
    const source = readFileSync(SOURCE, "utf8");
    // `export … from` loads a module as surely as `import` does. The file has
    // no string literals, so any `from "…"` is a module specifier.
    expect(source).not.toMatch(/^\s*import\b/m);
    expect(source).not.toMatch(/\bfrom\s*["']/);
    expect(source).not.toMatch(/\brequire\s*\(|\bimport\s*\(|\bimport\.meta\b/);
  });

  it("is the one implementation that request-identity.ts and rules.ts re-export", () => {
    expect(requestIdentity.canonicalJson).toBe(canonicalJson);
    expect(rules.jcs).toBe(jcs);
  });

  it.each(VECTORS.vectors)("reproduces the shared JCS vector $name", ({ json, jcs: expected }) => {
    const value = JSON.parse(json) as Parameters<typeof jcs>[0];
    if (expected === null) expect(() => jcs(value)).toThrow(RangeError);
    else expect(jcs(value)).toBe(expected);
  });

  it("keeps its bytes: sorted keys, no whitespace, I-JSON checks only in jcs", () => {
    expect(canonicalJson({ b: [1, "x"], a: null })).toBe('{"a":null,"b":[1,"x"]}');
    expect(canonicalJson("\uD800")).toBe('"\\ud800"');
    expect(() => jcs("\uD800")).toThrow(/lone surrogate/);
    expect(() => jcs({ ["\uDC00"]: 1 })).toThrow(/lone surrogate/);
    expect(() => canonicalJson(Number.NaN)).toThrow(/non-finite/);
    // Large doubles keep the ECMAScript form RFC 8785 specifies.
    expect(jcs(2 ** 60)).toBe("1152921504606847000");
  });
});
