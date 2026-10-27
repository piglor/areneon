import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const PACKAGES = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SRC = join(PACKAGES, "opponent-model", "src");
const sources = readdirSync(SRC).filter((f) => f.endsWith(".ts"));
const pkg = (name: string): { dependencies?: Record<string, string>; devDependencies?: Record<string, string> } => JSON.parse(readFileSync(join(PACKAGES, name, "package.json"), "utf8"));

describe("the package is engine-free and sits between the bridge and the reader (ADR-025 section 1)", () => {
  it("depends on the reader and the schema types only", () => {
    expect(Object.keys(pkg("opponent-model").dependencies ?? {}).sort()).toEqual(["@areneon/public-state", "@areneon/schema"]);
    expect(pkg("opponent-model").devDependencies).toBeUndefined();
  });

  it("imports no engine, bridge, Node built-in or network module", () => {
    for (const file of sources) {
      const code = readFileSync(join(SRC, file), "utf8");
      for (const m of code.matchAll(/^(?:import|export)\s[^;]*?from\s+"([^"]+)";/gms)) {
        const spec = m[1] as string;
        const type = /^(?:import|export)\s+type\b/.test(m[0]);
        const allowed = spec.startsWith("./") || (type && (spec === "@areneon/public-state" || spec.startsWith("@areneon/schema/types/")));
        expect(allowed, `${file} imports ${spec}`).toBe(true);
      }
      expect(code, file).not.toMatch(/\brequire\(|\bimport\(|\bfetch\(|process\.|Math\.random/);
    }
  });

  it("no package the belief depends on depends on it: no cycle", () => {
    for (const name of ["public-state", "schema"]) {
      const p = pkg(name);
      expect(Object.keys({ ...p.dependencies, ...p.devDependencies })).not.toContain("@areneon/opponent-model");
    }
  });

  it("the arithmetic is exact: no floating-point conversion in the source", () => {
    for (const file of sources) expect(readFileSync(join(SRC, file), "utf8"), file).not.toMatch(/parseFloat|toFixed|Math\.(floor|round|ceil|pow|sqrt|log)/);
  });
});
