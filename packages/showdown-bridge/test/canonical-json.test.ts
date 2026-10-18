import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { canonicalJson, jcs } from "../src/canonical-json.ts";
import * as requestIdentity from "../src/request-identity.ts";
import * as rules from "../src/rules.ts";

const SOURCE = new URL("../src/canonical-json.ts", import.meta.url);

describe("canonical-json", () => {
  it("imports nothing, so the bundled builder and archived rules-hash can depend on it", () => {
    const source = readFileSync(SOURCE, "utf8");
    expect(source).not.toMatch(/^\s*import\b/m);
    expect(source).not.toMatch(/\brequire\s*\(|\bimport\s*\(/);
  });

  it("is the one implementation that request-identity.ts and rules.ts re-export", () => {
    expect(requestIdentity.canonicalJson).toBe(canonicalJson);
    expect(rules.jcs).toBe(jcs);
  });

  it("keeps its bytes: sorted keys, no whitespace, I-JSON checks only in jcs", () => {
    expect(canonicalJson({ b: [1, "x"], a: null })).toBe('{"a":null,"b":[1,"x"]}');
    expect(canonicalJson("\uD800")).toBe('"\\ud800"');
    expect(() => jcs("\uD800")).toThrow(/lone surrogate/);
    expect(() => jcs({ ["\uDC00"]: 1 })).toThrow(/lone surrogate/);
    expect(() => canonicalJson(Number.NaN)).toThrow(/non-finite/);
  });
});
