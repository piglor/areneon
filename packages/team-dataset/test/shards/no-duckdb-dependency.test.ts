import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { REPO } from "./paths.ts";

// DuckDB is a tool the operator installs (ADR-013 section 7), never a
// dependency of shipped code: it is in no manifest, lockfile or image recipe.
const MANIFESTS = /(^|\/)(package\.json|pyproject\.toml|Cargo\.toml|go\.mod|requirements[^/]*\.txt|Dockerfile[^/]*|[^/]*\.Dockerfile|pnpm-lock\.yaml|uv\.lock|Cargo\.lock|compose[^/]*\.ya?ml|docker-compose[^/]*\.ya?ml)$/;

// A bare module specifier naming DuckDB (`@duckdb/node-api`, `duckdb-wasm`, ...), or a Python import of it.
const library = /(?:\bfrom\s+|\bimport\s*\(?\s*|\brequire\s*\(\s*)["'](?![./])[^"']*duckdb[^"']*["']|^\s*(?:import|from)\s+duckdb\b/im;

const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: REPO, encoding: "utf8", maxBuffer: 1 << 26 })
  .split("\0")
  .filter((f) => f !== "" && !f.startsWith("vendor/"));

describe("DuckDB stays out of everything shipped", () => {
  it("is not named in any manifest, lockfile or Dockerfile", () => {
    const manifests = tracked.filter((f) => MANIFESTS.test(f));
    expect(manifests.length).toBeGreaterThan(5);
    expect(manifests).toContain("package.json");
    expect(manifests).toContain("pnpm-lock.yaml");
    expect(manifests).toContain("uv.lock");
    expect(manifests.some((f) => /Dockerfile/.test(f))).toBe(true);
    const named = manifests.filter((f) => /duckdb/i.test(readFileSync(join(REPO, f), "utf8")));
    expect(named).toEqual([]);
  });

  it("recognises a library import and not the local writer module", () => {
    for (const text of ['import { x } from "@duckdb/node-api";', 'const d = require("duckdb");', "import duckdb", 'await import("duckdb-wasm")']) expect(library.test(text), text).toBe(true);
    for (const text of ['import { DuckdbWriter } from "./duckdb.ts";', 'import x from "../src/shards/duckdb.ts"']) expect(library.test(text), text).toBe(false);
  });

  it("is imported by no source file (the writer runs a command-line tool)", () => {
    const sources = tracked.filter((f) => /\.(ts|js|mjs|py)$/.test(f) && !f.endsWith("no-duckdb-dependency.test.ts"));
    const importing = sources.filter((f) => library.test(readFileSync(join(REPO, f), "utf8")));
    expect(importing).toEqual([]);
  });
});
