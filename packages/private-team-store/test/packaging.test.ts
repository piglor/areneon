import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { PRIVATE_TEAM_STORE_PACKAGE } from "../src/index.ts";
import {
  PACKAGE_NAME,
  PACKAGE_PATH,
  dependabotNpmIgnored,
  dockerInstructions,
  dockerfileProblems,
  forbiddenManifestDeps,
  forbiddenPyprojectDeps,
  lockedVersion,
  overrideProblems,
  referencesPackage,
  toolsIn,
  unsafeCopySource,
} from "./packaging-rules.ts";

// The packaging rule of ADR-014 (section 10, item 5): the private team store is
// an unshipped, exact-pinned package that no image or other package can reach.
// The checks live in packaging-rules.ts; the "checks catch bad input" blocks
// below prove each one against a table of bad and good cases, so a check that
// stops matching fails here, not silently.

const PACKAGE_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = join(PACKAGE_DIR, "..", "..");
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
const IGNORED_FOR_VERSION_UPDATES = ["age-encryption", "@noble/*", "@scure/*"];

// Tracked files only: what a checkout (and so any image build context) holds.
function trackedFiles(): string[] {
  const out = execFileSync("git", ["ls-files", "-z"], { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 });
  return out
    .split("\0")
    .filter((path) => path !== "")
    .map((path) => join(ROOT, path));
}

const FILES = trackedFiles();
const rel = (path: string): string => relative(ROOT, path);
const text = (path: string): string => readFileSync(path, "utf8");
const json = (path: string): Record<string, unknown> => JSON.parse(text(path)) as Record<string, unknown>;
const base = (path: string): string => path.slice(path.lastIndexOf("/") + 1);
// Dockerfile, Dockerfile.e2e, ... but not Dockerfile.dockerignore (a file list).
const isDockerfile = (path: string): boolean => base(path).startsWith("Dockerfile") && !base(path).endsWith(".dockerignore");
const isBuildFile = (path: string): boolean =>
  isDockerfile(path) ||
  /(^|\/)(docker-)?compose[^/]*\.ya?ml$/.test(path) ||
  /(^|\/)\.?coolify[^/]*$/i.test(path) ||
  /\.dockerignore$/.test(path);

const BUILD_FILES = FILES.filter(isBuildFile);
const DOCKERFILES = FILES.filter(isDockerfile);
const PACKAGE_JSONS = FILES.filter((path) => path.endsWith("/package.json"));
const PYPROJECTS = FILES.filter((path) => path.endsWith("/pyproject.toml"));
const OWN_PACKAGE_JSON = join(PACKAGE_DIR, "package.json");
const OWN_FILES = FILES.filter((path) => rel(path).startsWith(`${PACKAGE_PATH}/`));
// Code and config outside the package that could import or build it.
const OUTSIDE_SOURCES = FILES.filter(
  (path) =>
    /\.(ts|tsx|js|mjs|cjs|py|json|toml|sh|ya?ml)$/.test(path) &&
    !rel(path).startsWith(`${PACKAGE_PATH}/`) &&
    base(path) !== "pnpm-lock.yaml",
);

describe("the private team store package", () => {
  const manifest = json(OWN_PACKAGE_JSON);

  it("is private and named as the placeholder says", () => {
    expect(manifest["private"]).toBe(true);
    expect(manifest["name"]).toBe(PRIVATE_TEAM_STORE_PACKAGE);
    expect(PACKAGE_NAME).toBe(PRIVATE_TEAM_STORE_PACKAGE);
  });

  it("pins every dependency to an exact version, with no range", () => {
    for (const field of DEP_FIELDS) {
      for (const [dep, version] of Object.entries((manifest[field] ?? {}) as Record<string, string>)) {
        // A workspace link to this repository's own package is not a registry version.
        if (dep.startsWith("@areneon/")) expect(version, `${field}.${dep}`).toBe("workspace:*");
        else expect(version, `${field}.${dep}`).toMatch(EXACT);
      }
    }
    const deps = (manifest["dependencies"] ?? {}) as Record<string, string>;
    expect(Object.keys(deps)).toEqual(expect.arrayContaining(REQUIRED_PINS));
    expect(deps["age-encryption"]).toBe("0.3.1");
  });

  it("has the lockfile resolve each pin to exactly the version in package.json", () => {
    const lock = text(join(ROOT, "pnpm-lock.yaml"));
    const deps = manifest["dependencies"] as Record<string, string>;
    for (const dep of REQUIRED_PINS) {
      expect(lockedVersion(lock, PACKAGE_PATH, dep), dep).toEqual({ specifier: deps[dep], version: deps[dep] });
    }
  });

  it("has no build step", () => {
    expect(Object.keys((manifest["scripts"] ?? {}) as object)).toEqual(["typecheck"]);
  });

  it("has nothing that can rewrite its pinned versions", () => {
    const rootManifest = json(join(ROOT, "package.json"));
    const workspace = text(join(ROOT, "pnpm-workspace.yaml"));
    expect(overrideProblems(rootManifest, workspace)).toEqual([]);
    expect(overrideProblems(manifest, "")).toEqual([]);
  });
});

describe("the packaging rule on this repository", () => {
  it("finds the build files and manifests it scans", () => {
    expect(DOCKERFILES.map(rel)).toContain("packages/rules-importer/Dockerfile");
    expect(BUILD_FILES.map(rel)).toContain("db/compose.yaml");
    expect(PACKAGE_JSONS.map(rel)).toContain("packages/private-team-store/package.json");
    expect(PYPROJECTS.map(rel)).toContain("pyproject.toml");
    expect(OUTSIDE_SOURCES.length).toBeGreaterThan(50);
    expect(OWN_FILES.map(rel)).toContain("packages/private-team-store/test/packaging.test.ts");
  });

  it("is referenced by no Dockerfile, compose file or Coolify build file", () => {
    for (const path of BUILD_FILES) expect(text(path), rel(path)).not.toMatch(/private-team-store/);
  });

  it("is named by no code or config outside the package", () => {
    for (const path of OUTSIDE_SOURCES) expect(text(path), rel(path)).not.toMatch(/private-team-store/);
  });

  it("is a dependency of no other workspace package", () => {
    for (const path of PACKAGE_JSONS) {
      if (path === OWN_PACKAGE_JSON) continue;
      expect(referencesPackage(json(path)), rel(path)).toBe(false);
    }
  });

  it("is named by no importer's dependencies in pnpm-lock.yaml", () => {
    const lock = text(join(ROOT, "pnpm-lock.yaml"));
    expect(lock).toContain(`  ${PACKAGE_PATH}:\n`);
    expect(lock).not.toContain(`'${PACKAGE_NAME}':`);
    expect(lock).not.toContain("private-team-store@");
  });

  it("is reached by no Dockerfile through a whole-context copy or an unfiltered install", () => {
    for (const path of DOCKERFILES) expect(dockerfileProblems(text(path)), rel(path)).toEqual([]);
  });

  it("keeps the DuckDB CLI, the age CLI and OpenSSL out of every manifest and Dockerfile", () => {
    for (const path of [...PACKAGE_JSONS, ...PYPROJECTS, ...DOCKERFILES]) {
      const source = path.endsWith("package.json")
        ? Object.entries(json(path).scripts ?? {}).map(([, v]) => String(v)).join("\n")
        : dockerInstructions(path.endsWith(".toml") ? "" : text(path)).map((i) => `${i.cmd} ${i.args}`).join("\n");
      expect(toolsIn(source), rel(path)).toEqual([]);
    }
    for (const path of PACKAGE_JSONS) expect(forbiddenManifestDeps(json(path)), rel(path)).toEqual([]);
    for (const path of PYPROJECTS) {
      expect(forbiddenPyprojectDeps(text(path)), rel(path)).toEqual([]);
      expect(toolsIn(text(path).split("\n").filter((l) => !l.trimStart().startsWith("#")).join("\n")), rel(path)).toEqual([]);
    }
  });

  it("has dependabot.yml ignore age-encryption, @noble/* and @scure/* for version updates", () => {
    expect(dependabotNpmIgnored(text(join(ROOT, ".github", "dependabot.yml")))).toEqual(
      expect.arrayContaining(IGNORED_FOR_VERSION_UPDATES),
    );
  });

  it("has CODEOWNERS cover the files that can undo these guarantees", () => {
    const owners = text(join(ROOT, ".github", "CODEOWNERS"));
    for (const path of ["/pnpm-lock.yaml", "/package.json", "/pnpm-workspace.yaml", `/${PACKAGE_PATH}/`]) {
      const line = owners.split("\n").find((l) => l.split(/\s+/)[0] === path);
      expect(line, path).toMatch(/\s@\S+$/);
    }
  });

  it("tells the operator what a repository test cannot see", () => {
    expect(text(join(PACKAGE_DIR, "README.md"))).toMatch(/Coolify application configured in\s+the UI/);
  });
});

describe("the checks catch bad input", () => {
  const bad: [string, string][] = [
    ["whole context", "COPY . ."],
    ["whole context with ./", "COPY ./ ./"],
    ["packages directory", "COPY ./packages ./packages"],
    ["packages glob", "COPY packages/*/package.json ./"],
    ["glob in a directory", "COPY packages/pri*/src ./"],
    ["JSON form", 'COPY [".", "/app"]'],
    ["parent context", "COPY ../ ./"],
    ["the package itself", "COPY packages/private-team-store/package.json packages/private-team-store/"],
    ["ADD of the package", "ADD packages/private-team-store/ /app/"],
    ["flags before a bad source", "COPY --chown=1000:1000 . /app"],
    ["continuation line", "COPY package.json \\\n  . ./"],
    ["variable source", "COPY $CTX /app"],
    ["unfiltered install", "RUN pnpm install --frozen-lockfile"],
    ["flag before verb", "RUN pnpm --prod install"],
    ["dir flag", "RUN pnpm -C . install"],
    ["short alias", "RUN pnpm i"],
    ["deploy", "RUN pnpm deploy /out"],
    ["fetch", "RUN pnpm fetch"],
    ["chained", "RUN corepack enable && pnpm install --prod"],
    ["after semicolon", "RUN set -eu; pnpm add left-pad"],
    ["exec form", 'RUN ["pnpm", "install"]'],
    ["npm install", "RUN npm install"],
    ["yarn install", "RUN yarn install --frozen-lockfile"],
  ];
  const good: [string, string][] = [
    ["explicit files", "COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./"],
    ["explicit directories", "COPY packages/rules-importer/src packages/rules-importer/src"],
    ["from another stage", "COPY --from=build /app /app"],
    ["filtered install", 'RUN corepack enable && pnpm install --frozen-lockfile --prod --filter "@areneon/rules-importer..."'],
    ["filter with equals", "RUN pnpm install --filter=@areneon/x..."],
    ["short filter", "RUN pnpm -F @areneon/x install"],
    ["comment", "# COPY . .\nRUN echo ok"],
    ["no pnpm", "RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl"],
  ];

  it.each(bad)("flags a Dockerfile with %s", (_name, source) => {
    expect(dockerfileProblems(source).length).toBeGreaterThan(0);
  });

  it.each(good)("passes a Dockerfile with %s", (_name, source) => {
    expect(dockerfileProblems(source)).toEqual([]);
  });

  it("rejects each unsafe COPY source and accepts explicit paths", () => {
    for (const src of [".", "./", "..", "../x", "packages", "./packages/", "a/*", "a/b?", "$X", "https://x/y", PACKAGE_PATH]) {
      expect(unsafeCopySource(src), src).not.toBeNull();
    }
    for (const src of ["package.json", "packages/schema/src", "./fixtures/rules-releases", "packages/private-team-stores-not"]) {
      expect(unsafeCopySource(src), src).toBeNull();
    }
  });

  it("joins continuation lines and drops comments", () => {
    expect(dockerInstructions("RUN a \\\n  && b\n# x\nCOPY c d")).toEqual([
      { cmd: "RUN", args: "a    && b" },
      { cmd: "COPY", args: "c d" },
    ]);
    expect(dockerInstructions("\n  \n")).toEqual([]);
  });

  const toolsBad: [string, string][] = [
    ["duckdb CLI", "RUN duckdb -safe"],
    ["duckdb package", "RUN pip install duckdb"],
    ["openssl", "RUN apt-get install -y openssl"],
    ["age via go", "RUN go install filippo.io/age/cmd/age@latest"],
    ["age tarball", "RUN curl -fsSL https://github.com/FiloSottile/age/releases/download/v1.3.2/age-v1.3.2-linux-amd64.tar.gz | tar xz"],
    ["age tarball name only", "RUN tar xzf age-v1.3.2-linux-amd64.tar.gz"],
    ["age repository", "RUN git clone https://github.com/FiloSottile/age"],
    ["pyrage", "RUN pip install pyrage"],
    ["rage", "RUN cargo install rage"],
    ["apt age", "RUN apt-get install -y age"],
    ["apk age", "RUN apk add --no-cache age"],
    ["brew age", "RUN brew install age"],
  ];
  const toolsGood: [string, string][] = [
    ["age-encryption package", "RUN pnpm install --filter x age-encryption"],
    ["plain apt packages", "RUN apt-get install -y --no-install-recommends ca-certificates curl"],
    ["the word usage", "RUN echo usage and image and package"],
    ["a manager", "RUN pip install pydantic"],
  ];

  it.each(toolsBad)("finds a tool in %s", (_name, source) => {
    expect(toolsIn(source).length).toBeGreaterThan(0);
  });

  it.each(toolsGood)("finds no tool in %s", (_name, source) => {
    expect(toolsIn(source)).toEqual([]);
  });

  it("finds forbidden dependency names in a package.json and a pyproject", () => {
    expect(forbiddenManifestDeps({ dependencies: { "@duckdb/node-api": "1.0.0", age: "1.0.0" }, devDependencies: { rage: "1.0.0" } })).toEqual([
      "@duckdb/node-api",
      "age",
      "rage",
    ]);
    expect(forbiddenManifestDeps({ dependencies: { "age-encryption": "0.3.1" } })).toEqual([]);
    expect(forbiddenPyprojectDeps('dependencies = ["pyrage>=1", "duckdb", "age ; python_version>\'3\'", "ageing>=1"]')).toEqual([
      "pyrage",
      "duckdb",
      "age",
    ]);
    expect(forbiddenPyprojectDeps('# "duckdb"\ndependencies = ["pydantic>=2"]')).toEqual([]);
  });

  it("finds every way a manifest can name the package", () => {
    const ways = [
      { dependencies: { [PACKAGE_NAME]: "workspace:*" } },
      { devDependencies: { alias: "workspace:@areneon/private-team-store@*" } },
      { dependencies: { x: "link:../private-team-store" } },
      { dependencies: { x: "file:../../packages/private-team-store" } },
      { optionalDependencies: { x: "npm:@areneon/private-team-store@0.0.0" } },
      { peerDependencies: { "private-team-store": "*" } },
      { overrides: { x: "link:../private-team-store" } },
      { pnpm: { overrides: { x: "workspace:../private-team-store" } } },
      { bundledDependencies: ["@areneon/private-team-store"] },
    ];
    for (const manifest of ways) expect(referencesPackage(manifest), JSON.stringify(manifest)).toBe(true);
    expect(referencesPackage({ dependencies: { "@areneon/schema": "workspace:*" }, name: "private-team-store" })).toBe(false);
    expect(referencesPackage({})).toBe(false);
  });

  it("finds overrides, resolutions and patches", () => {
    expect(overrideProblems({ overrides: {}, resolutions: {}, pnpm: { overrides: {}, packageExtensions: {}, patchedDependencies: {} } }, "overrides:\n  a: 1\npackageExtensions:\n")).toEqual([
      "package.json overrides",
      "package.json resolutions",
      "package.json pnpm.overrides",
      "package.json pnpm.packageExtensions",
      "package.json pnpm.patchedDependencies",
      "pnpm-workspace.yaml overrides",
      "pnpm-workspace.yaml packageExtensions",
    ]);
    expect(overrideProblems({ name: "x" }, "packages:\n  - a\n")).toEqual([]);
  });

  const head = "version: 2\nupdates:\n  - package-ecosystem: npm\n    directory: /\n";
  const entries = IGNORED_FOR_VERSION_UPDATES.map((n) => `      - dependency-name: "${n}"\n`).join("");

  it("reads the ignore list of the npm block only", () => {
    expect(dependabotNpmIgnored(`${head}    ignore:\n${entries}  - package-ecosystem: uv\n    directory: /\n`)).toEqual(IGNORED_FOR_VERSION_UPDATES);
    expect(dependabotNpmIgnored(`${head}  - package-ecosystem: uv\n    directory: /\n    ignore:\n${entries}`)).toEqual([]);
    expect(dependabotNpmIgnored(`${head}    # ignore:\n${entries.replace(/ {6}-/g, "      #-")}`)).toEqual([]);
    expect(dependabotNpmIgnored(`${head}    schedule: { interval: weekly }\n`)).toEqual([]);
    expect(dependabotNpmIgnored("version: 2\n")).toEqual([]);
    expect(dependabotNpmIgnored(`${head}    ignore:\n      - dependency-name: "a"\n\n      - dependency-name: "b"\n    schedule: x\n      - dependency-name: "c"\n`)).toEqual(["a", "b"]);
  });

  it("reads a locked version and misses an absent or differing one", () => {
    const lock = [
      "importers:",
      "",
      "  .:",
      "    devDependencies:",
      "      x:",
      "        specifier: 1.0.0",
      "        version: 1.0.0",
      "",
      "  packages/p:",
      "    dependencies:",
      "      '@s/a':",
      "        specifier: 1.2.3",
      "        version: 1.2.3",
      "      b:",
      "        specifier: 2.0.0",
      "        version: 2.0.1",
      "",
      "  packages/q:",
      "    dependencies:",
      "      c:",
      "        specifier: 3.0.0",
      "        version: 3.0.0",
      "",
      "packages:",
      "",
    ].join("\n");
    expect(lockedVersion(lock, "packages/p", "@s/a")).toEqual({ specifier: "1.2.3", version: "1.2.3" });
    expect(lockedVersion(lock, "packages/p", "b")).toEqual({ specifier: "2.0.0", version: "2.0.1" });
    expect(lockedVersion(lock, "packages/p", "c")).toBeNull();
    expect(lockedVersion(lock, "packages/q", "c")).toEqual({ specifier: "3.0.0", version: "3.0.0" });
    expect(lockedVersion(lock, "packages/none", "c")).toBeNull();
  });
});
