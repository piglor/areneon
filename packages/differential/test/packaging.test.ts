// The packaging rule (ADR-030 rule 1, ticket #1461): the core is engine-free, holds no float, sits in no image and on no
// evaluation path; only the composition root (packages/team-dataset) joins it with the bridge, which does not import it.
import { execFileSync } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const PACKAGES = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const ROOT = join(PACKAGES, "..");
const SRC = join(PACKAGES, "differential", "src");
const sources = readdirSync(SRC).filter((f) => f.endsWith(".ts"));
const pkg = (name: string): { dependencies?: Record<string, string>; devDependencies?: Record<string, string> } => JSON.parse(readFileSync(join(PACKAGES, name, "package.json"), "utf8"));
const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 })
  .split("\0")
  .filter((p) => p !== "");
const NAME = "@areneon/differential";
/** The composition root's files that may join the core with the bridge (and their tests and scripts). */
const JOINING = /^packages\/team-dataset\/(src|test|scripts)\/differential[a-z.-]*\.ts$/;

describe("the differential core package", () => {
  it("depends on the schema package only", () => {
    expect(Object.keys(pkg("differential").dependencies ?? {})).toEqual(["@areneon/schema"]);
    expect(pkg("differential").devDependencies).toBeUndefined();
  });

  it("imports no engine, bridge, network module or Node built-in but node:crypto", () => {
    for (const file of sources) {
      const code = readFileSync(join(SRC, file), "utf8");
      for (const m of code.matchAll(/^(?:import|export)\s[^;]*?from\s+"([^"]+)";/gms)) {
        const spec = m[1] as string;
        const allowed = spec.startsWith("./") || spec === "node:crypto" || spec === "@areneon/schema/ajv" || spec.startsWith("@areneon/schema/contracts/v1/") || spec.startsWith("@areneon/schema/types/v1/");
        expect(allowed, `${file} imports ${spec}`).toBe(true);
      }
      expect(code, file).not.toMatch(/\brequire\(|\bimport\(|\bfetch\(|process\.|Math\.random/);
    }
  });

  it("holds no floating-point arithmetic", () => {
    for (const file of sources) expect(readFileSync(join(SRC, file), "utf8"), file).not.toMatch(/parseFloat|toFixed|Number\.parseFloat|Math\.(floor|round|ceil|pow|sqrt|log|exp|trunc|fround)|\/ 100\b|\d\.\d/);
  });

  it("is depended on by the composition root alone, never by the bridge", () => {
    const dependents = readdirSync(PACKAGES).filter((name) => {
      try {
        const p = pkg(name);
        return NAME in { ...p.dependencies, ...p.devDependencies };
      } catch {
        return false;
      }
    });
    expect(dependents).toEqual(["team-dataset"]);
  });

  it("is imported only by the joining files of the composition root (no evaluation path, no bridge)", () => {
    const imports = new RegExp(`(?:from|import\\(|require\\()\\s*["']${NAME}["']`);
    const importers = tracked.filter((path) => /\.(ts|tsx|js|mjs|cjs)$/.test(path) && !path.startsWith("packages/differential/") && imports.test(readFileSync(join(ROOT, path), "utf8")));
    for (const path of importers) expect(path, path).toMatch(JOINING);
    expect(importers.length).toBeGreaterThan(0);
  });

  it("is named by no image or deployment file", () => {
    const builds = tracked.filter((path) => /(^|\/)(Dockerfile[^/]*|(docker-)?compose[^/]*\.ya?ml|\.?coolify[^/]*|[^/]*\.dockerignore)$/i.test(path));
    expect(builds.length).toBeGreaterThan(0);
    for (const path of builds) {
      const text = readFileSync(join(ROOT, path), "utf8");
      expect(text, path).not.toContain("packages/differential");
      expect(text, path).not.toContain(NAME);
      expect(text, path).not.toContain("team-dataset");
    }
  });
});
