import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// The network lives in the shared client (packages/object-store, whose own
// test holds the rule). Nothing in this package calls fetch, imports a network
// module or imports the signing library itself.

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const FILES = readdirSync(SRC).filter((name) => name.endsWith(".ts"));
const code = (name: string): string =>
  readFileSync(join(SRC, name), "utf8")
    .split("\n")
    .filter((line) => !/^\s*(\/\/|\/\*|\*)/.test(line))
    .join("\n");

describe("network isolation of the private store", () => {
  it("finds the sources it scans", () => {
    expect(FILES).toEqual(expect.arrayContaining(["r2-client.ts", "r2-settings.ts", "publish.ts", "memory-store.ts"]));
  });

  it("calls fetch and touches a network module in no file", () => {
    const network = /(?<![\w.])fetch\(|\bglobalThis\.fetch\b|node:(net|http|https|http2|dns|tls|dgram)\b|\bWebSocket\b|\bXMLHttpRequest\b/;
    expect(FILES.filter((name) => network.test(code(name)))).toEqual([]);
  });

  it("imports the signing library in no file", () => {
    expect(FILES.filter((name) => /from "aws4fetch"/.test(code(name)))).toEqual([]);
  });
});
