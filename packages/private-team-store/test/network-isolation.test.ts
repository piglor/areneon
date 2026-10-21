import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Network input and output happen in the R2 client only (ADR-015 section 1,
// ticket "no CI test touches the network"): no other module of the package
// calls fetch or imports a network module, and the signing library is
// imported by the settings module alone.

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

  it("lets only r2-client.ts call fetch or touch a network module", () => {
    const network = /(?<![\w.])fetch\(|\bglobalThis\.fetch\b|node:(net|http|https|http2|dns|tls|dgram)\b|\bWebSocket\b|\bXMLHttpRequest\b/;
    expect(FILES.filter((name) => network.test(code(name)))).toEqual(["r2-client.ts"]);
  });

  it("lets only r2-settings.ts import the signing library", () => {
    expect(FILES.filter((name) => /from "aws4fetch"/.test(code(name)))).toEqual(["r2-settings.ts"]);
  });

  it("reads credentials from no file and writes none: neither settings nor client touch the file system or process.env", () => {
    for (const name of ["r2-settings.ts", "r2-client.ts"]) {
      expect(code(name), name).not.toMatch(/node:fs|process\.env|readFile|writeFile|homedir/);
    }
  });
});
