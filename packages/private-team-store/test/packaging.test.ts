import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { PRIVATE_TEAM_STORE_PACKAGE } from "../src/index.ts";

// The packaging rule of ADR-014 (section 10, item 5): the private team store is
// an unshipped, exact-pinned package that no image or other package can reach.

const PACKAGE_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = join(PACKAGE_DIR, "..", "..");
const SKIPPED = new Set(["node_modules", "vendor", ".git", "coverage", "dist", "private-data"]);
const NAME = PRIVATE_TEAM_STORE_PACKAGE;
const DEP_FIELDS = ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"];
const EXACT = /^\d+\.\d+\.\d+$/;
const REQUIRED_PINS = [
  "age-encryption",
  "@noble/ciphers",
  "@noble/curves",
  "@noble/hashes",
  "@noble/post-quantum",
  "@scure/base",
];

function walk(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (SKIPPED.has(entry.name)) continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walk(path));
    else if (entry.isFile()) files.push(path);
  }
  return files;
}

const FILES = walk(ROOT);
const rel = (path: string): string => relative(ROOT, path);
const text = (path: string): string => readFileSync(path, "utf8");
// Dockerfile, Dockerfile.e2e, ... but not Dockerfile.dockerignore (a file list).
const isDockerfile = (path: string): boolean => {
  const base = path.slice(path.lastIndexOf("/") + 1);
  return base.startsWith("Dockerfile") && !base.endsWith(".dockerignore");
};
const isBuildFile = (path: string): boolean =>
  isDockerfile(path) ||
  /(^|\/)(docker-)?compose[^/]*\.ya?ml$/.test(path) ||
  /(^|\/)\.?coolify[^/]*$/i.test(path) ||
  /\.dockerignore$/.test(path);
const noComments = (source: string): string =>
  source
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("#"))
    .join("\n");

const BUILD_FILES = FILES.filter((path) => isBuildFile(path));
const DOCKERFILES = FILES.filter((path) => isDockerfile(path));
const PACKAGE_JSONS = FILES.filter((path) => path.endsWith("/package.json"));
const PYPROJECTS = FILES.filter((path) => path.endsWith("/pyproject.toml"));

describe("the private team store package", () => {
  const manifest = JSON.parse(text(join(PACKAGE_DIR, "package.json"))) as Record<string, unknown>;

  it("is private and named as the placeholder says", () => {
    expect(manifest["private"]).toBe(true);
    expect(manifest["name"]).toBe(NAME);
  });

  it("pins every dependency to an exact version, with no range", () => {
    for (const field of DEP_FIELDS) {
      for (const [dep, version] of Object.entries((manifest[field] ?? {}) as Record<string, string>)) {
        expect(version, `${field}.${dep}`).toMatch(EXACT);
      }
    }
    const deps = (manifest["dependencies"] ?? {}) as Record<string, string>;
    expect(Object.keys(deps)).toEqual(expect.arrayContaining(REQUIRED_PINS));
    expect(deps["age-encryption"]).toBe("0.3.1");
  });

  it("has no build step", () => {
    expect(Object.keys((manifest["scripts"] ?? {}) as object)).toEqual(["typecheck"]);
  });
});

describe("the packaging rule", () => {
  it("finds the build files and manifests it scans", () => {
    expect(DOCKERFILES.map(rel)).toContain("packages/rules-importer/Dockerfile");
    expect(BUILD_FILES.map(rel)).toContain("db/compose.yaml");
    expect(PACKAGE_JSONS.map(rel)).toContain("packages/private-team-store/package.json");
    expect(PYPROJECTS.map(rel)).toContain("pyproject.toml");
  });

  it("is referenced by no Dockerfile, compose file or Coolify build file", () => {
    for (const path of BUILD_FILES) {
      expect(text(path), rel(path)).not.toMatch(/private-team-store/);
    }
  });

  it("is a dependency of no other workspace package", () => {
    for (const path of PACKAGE_JSONS) {
      if (path === join(PACKAGE_DIR, "package.json")) continue;
      const other = JSON.parse(text(path)) as Record<string, Record<string, string> | undefined>;
      for (const field of DEP_FIELDS) {
        expect(Object.keys(other[field] ?? {}), `${rel(path)} ${field}`).not.toContain(NAME);
      }
    }
  });

  it("is reached by no Dockerfile that copies a whole directory of packages", () => {
    for (const path of DOCKERFILES) {
      const code = noComments(text(path));
      expect(code, rel(path)).not.toMatch(/^\s*(COPY|ADD)\s+(--\S+\s+)*(\.|packages\/?\*?|\*)\s/m);
      for (const line of code.split("\n").filter((l) => /pnpm\s+(-r\s+)?(install|i)\b/.test(l))) {
        expect(line, `${rel(path)} installs without a workspace filter`).toMatch(/--filter\b/);
      }
    }
  });

  it("keeps the DuckDB CLI, the age CLI and OpenSSL out of every manifest and Dockerfile", () => {
    const tools = [/duckdb/i, /openssl/i, /(?<![\w./@-])age(?![\w-])/i];
    for (const path of [...PACKAGE_JSONS, ...PYPROJECTS, ...DOCKERFILES]) {
      const code = path.endsWith(".toml") || isDockerfile(path) ? noComments(text(path)) : text(path);
      for (const tool of tools) expect(code, `${rel(path)} ${String(tool)}`).not.toMatch(tool);
    }
  });

  it("is named by no importer's dependencies in pnpm-lock.yaml", () => {
    const lock = text(join(ROOT, "pnpm-lock.yaml"));
    expect(lock).toContain("  packages/private-team-store:\n");
    expect(lock).not.toContain(`'${NAME}':`);
  });
});

describe("the Dependabot configuration", () => {
  it("ignores age-encryption, @noble/* and @scure/* for version updates in the npm ecosystem", () => {
    const config = text(join(ROOT, ".github", "dependabot.yml"));
    const blocks = config.split(/^\s*- package-ecosystem:/m).slice(1);
    const npm = blocks.find((block) => /^\s*npm\b/.test(block));
    expect(npm).toBeDefined();
    for (const name of ["age-encryption", "@noble/*", "@scure/*"]) {
      expect(npm).toMatch(new RegExp(`dependency-name:\\s*["']${name.replace(/[*/]/g, "\\$&")}["']`));
    }
  });
});
