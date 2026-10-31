// The checks behind the packaging rule (ADR-014 section 10, item 5), kept as
// pure functions over text so packaging.test.ts can prove each one against bad
// input (table-driven positive and negative cases) as well as run it on the repo.

export const PACKAGE_PATH = "packages/private-team-store";
export const PACKAGE_NAME = "@areneon/private-team-store";
const DEP_FIELDS = ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"];

/** Dockerfile text as instructions: comments dropped, `\` continuations joined. */
export function dockerInstructions(source: string): { cmd: string; args: string }[] {
  const lines = source
    .replace(/\r/g, "")
    .replace(/\\\n/g, " ")
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("#"));
  const out: { cmd: string; args: string }[] = [];
  for (const line of lines) {
    const m = /^\s*([A-Za-z]+)\s+(.*)$/.exec(line);
    if (m?.[1] !== undefined && m[2] !== undefined) out.push({ cmd: m[1].toUpperCase(), args: m[2].trim() });
  }
  return out;
}

/** The sources of a COPY/ADD, or null when they come from another stage or image. */
function copySources(args: string): string[] | null {
  let rest = args;
  let fromStage = false;
  for (;;) {
    const flag = /^--([\w-]+)(?:=(\S*))?\s+/.exec(rest);
    if (flag === null) break;
    if (flag[1] === "from") fromStage = true;
    rest = rest.slice(flag[0].length);
  }
  if (fromStage) return null;
  let words: string[];
  if (rest.startsWith("[")) {
    try {
      words = (JSON.parse(rest) as unknown[]).map(String);
    } catch {
      return [rest];
    }
  } else {
    words = rest.split(/\s+/).filter((w) => w !== "");
  }
  return words.slice(0, -1);
}

/** Why a COPY/ADD source could pull the private package into an image, or null. */
export function unsafeCopySource(source: string): string | null {
  const path = source.replace(/^(\.\/)+/, "").replace(/\/+$/, "");
  if (/[*?[\]{}$]/.test(path)) return `glob or variable source ${source}`;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(path)) return `remote source ${source}`;
  if (path === "" || path === "." || path === ".." || path.startsWith("../")) return `whole-context source ${source}`;
  if (path === "packages") return `the whole packages directory (${source})`;
  if (path === PACKAGE_PATH || path.startsWith(`${PACKAGE_PATH}/`)) return `the private package (${source})`;
  return null;
}

const PNPM_VERBS = /^(install|i|add|fetch|deploy|update|up|upgrade|import|rebuild|rb|dlx|exec|run)$/;
const OTHER_INSTALLERS = /^(npm|npx|yarn|bun|bunx|corepack)$/;

/** Problems in one Dockerfile that could let the private package into an image. */
export function dockerfileProblems(source: string): string[] {
  const problems: string[] = [];
  for (const { cmd, args } of dockerInstructions(source)) {
    if (cmd === "COPY" || cmd === "ADD") {
      for (const src of copySources(args) ?? []) {
        const why = unsafeCopySource(src);
        if (why !== null) problems.push(`${cmd}: ${why}`);
      }
    }
    if (cmd === "RUN" || cmd === "CMD" || cmd === "ENTRYPOINT") {
      for (const segment of args.split(/&&|\|\||;|\||\n/)) {
        const words = segment.replace(/[[\]",]/g, " ").trim().split(/\s+/).filter((w) => w !== "");
        const exe = words.findIndex((w) => /(^|\/)pnpm$/.test(w));
        if (exe >= 0) {
          const verb = words.slice(exe + 1).find((w) => PNPM_VERBS.test(w));
          const filtered = words.some((w) => w === "-F" || w === "--filter" || w.startsWith("--filter="));
          if (verb !== undefined && !filtered) problems.push(`${cmd}: pnpm ${verb} without a workspace filter`);
        }
        const other = words.find((w) => OTHER_INSTALLERS.test(w.replace(/^.*\//, "")));
        if (other !== undefined && words.some((w) => /^(install|i|ci|add|deploy)$/.test(w))) {
          problems.push(`${cmd}: ${other} install bypasses the pnpm workspace filter`);
        }
      }
    }
  }
  return problems;
}

// The age CLI (and its ports), the DuckDB CLI and OpenSSL are user-installed
// tools. These are explicit patterns for how one would get them into a build.
const INSTALL_VERB = String.raw`(?:install|add|get|i)`;
const TOOL_PATTERNS: { name: string; pattern: RegExp }[] = [
  { name: "duckdb", pattern: /duckdb/i },
  { name: "openssl", pattern: /openssl/i },
  { name: "age release tarball", pattern: /\bage-v\d/i },
  { name: "age go module", pattern: /filippo\.io\/age/i },
  { name: "age repository", pattern: /FiloSottile\/age\b/i },
  { name: "age python port", pattern: /\b(pyrage|rage)\b/i },
  {
    name: "age package",
    pattern: new RegExp(
      String.raw`\b(?:apt-get|apt|apk|brew|dnf|yum|pacman|zypper|nix-env|nix|conda|mamba|snap|cargo|go|pip3?|uv|pipx)\b[^\n;&|]*\b${INSTALL_VERB}\b[^\n;&|]*(?<![\w@./-])age(?![\w./@-])`,
      "i",
    ),
  },
];

/** Names of the tools found in Dockerfile, manifest or pyproject text. */
export function toolsIn(text: string): string[] {
  return TOOL_PATTERNS.filter(({ pattern }) => pattern.test(text)).map(({ name }) => name);
}

/** Dependency names that are the age, rage or DuckDB packages themselves. */
const FORBIDDEN_DEP = /^(age|rage|pyrage|duckdb(-.*)?|@duckdb\/.*)$/i;

export function forbiddenManifestDeps(manifest: Record<string, unknown>): string[] {
  const names: string[] = [];
  for (const field of DEP_FIELDS) {
    for (const dep of Object.keys((manifest[field] ?? {}) as object)) {
      if (FORBIDDEN_DEP.test(dep)) names.push(dep);
    }
  }
  return names;
}

/** Python dependency lines naming age, rage, pyrage or duckdb. */
export function forbiddenPyprojectDeps(text: string): string[] {
  const code = text
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("#"))
    .join("\n");
  return [...code.matchAll(/["']\s*(age|rage|pyrage|duckdb[\w.-]*)\s*(?=[<>=!~[;"',\s])/gi)].map((m) => m[1] ?? "");
}

/** True when a package.json refers to the private package in any dependency position. */
export function referencesPackage(manifest: Record<string, unknown>): boolean {
  const fields = [...DEP_FIELDS, "overrides", "resolutions", "bundledDependencies"];
  const hit = (value: unknown): boolean => {
    if (typeof value === "string") return value.includes("private-team-store");
    if (Array.isArray(value)) return value.some(hit);
    if (value !== null && typeof value === "object") {
      return Object.entries(value).some(([k, v]) => k.includes("private-team-store") || hit(v));
    }
    return false;
  };
  const pnpm = (manifest["pnpm"] ?? {}) as Record<string, unknown>;
  return [...fields.map((f) => manifest[f]), pnpm["overrides"]].some(hit);
}

/**
 * The only root `pnpm.overrides` entries allowed, each reviewed (a code-owner edit):
 * minimatch forced to ^11.0.1 clears GHSA-g3hv-75wg-m249 in dev-only lint tooling
 * (eslint, @eslint/config-array, @typescript-eslint/typescript-estree).
 */
export const ALLOWED_OVERRIDES: Readonly<Record<string, string>> = { minimatch: "^11.0.1" };

function stripQuotes(name: string): string {
  return name.replace(/^'|'$/g, "");
}

/** Names in the production dependency closure of a workspace importer, read from the lockfile. */
export function productionClosure(lock: string, importer: string): Set<string> {
  const names = new Set<string>();
  const snapshotsAt = lock.indexOf("\nsnapshots:\n");
  const importersAt = lock.indexOf("\nimporters:\n");
  const importerText = lock.slice(importersAt, snapshotsAt < 0 ? undefined : snapshotsAt);
  const snapshotText = snapshotsAt < 0 ? "" : lock.slice(snapshotsAt);
  const deps = (text: string): { name: string; version: string }[] => {
    const out: { name: string; version: string }[] = [];
    let inProd = false;
    let pending: string | null = null;
    for (const line of text.split("\n")) {
      const sec = /^ {4}(\w+):\s*$/.exec(line);
      if (sec !== null) {
        inProd = sec[1] === "dependencies" || sec[1] === "optionalDependencies";
        pending = null;
        continue;
      }
      if (!inProd) continue;
      const head = /^ {6}(\S+):\s*$/.exec(line);
      if (head?.[1] !== undefined) pending = stripQuotes(head[1]);
      const ver = /^ {8}version: (\S+)\s*$/.exec(line);
      if (ver?.[1] !== undefined && pending !== null) out.push({ name: pending, version: ver[1] });
    }
    return out;
  };
  const importerBody = (path: string): string => {
    const at = importerText.indexOf(`\n  ${path}:`);
    if (at < 0) return "";
    const tail = importerText.slice(importerText.indexOf("\n", at + 1) + 1);
    const stop = tail.search(/\n {2}\S/);
    return stop < 0 ? tail : tail.slice(0, stop);
  };
  const snapshotDeps = (key: string): { name: string; version: string }[] => {
    const out: { name: string; version: string }[] = [];
    const lines = snapshotText.split("\n");
    const at = lines.findIndex((l) => l === `  '${key}':` || l === `  ${key}:`);
    if (at < 0) return out;
    let inProd = false;
    for (const line of lines.slice(at + 1)) {
      if (/^ {2}\S/.test(line)) break;
      const sec = /^ {4}(\w+):\s*$/.exec(line);
      if (sec !== null) inProd = sec[1] === "dependencies" || sec[1] === "optionalDependencies";
      const dep = /^ {6}(\S+): (\S+)\s*$/.exec(line);
      if (inProd && dep?.[1] !== undefined && dep[2] !== undefined) out.push({ name: stripQuotes(dep[1]), version: dep[2] });
    }
    return out;
  };
  const seenImporters = new Set<string>();
  const seenSnapshots = new Set<string>();
  const queue: { name: string; version: string }[] = [];
  const visitImporter = (path: string): void => {
    if (seenImporters.has(path)) return;
    seenImporters.add(path);
    queue.push(...deps(importerBody(path)));
  };
  visitImporter(importer);
  for (let item = queue.pop(); item !== undefined; item = queue.pop()) {
    names.add(item.name);
    if (item.version.startsWith("link:")) {
      const target = item.version.slice(5).replace(/^\.\.\//, "packages/");
      visitImporter(target);
      continue;
    }
    const key = `${item.name}@${item.version}`;
    if (seenSnapshots.has(key)) continue;
    seenSnapshots.add(key);
    queue.push(...snapshotDeps(key));
  }
  return names;
}

/**
 * Settings that can rewrite resolved versions behind the exact pins. The one exception is a
 * root `pnpm.overrides` entry that is in ALLOWED_OVERRIDES exactly and names no package of the
 * private package's production closure (`closure`, from productionClosure).
 */
export function overrideProblems(
  manifest: Record<string, unknown>,
  workspaceYaml: string,
  closure: ReadonlySet<string> = new Set(),
): string[] {
  const out: string[] = [];
  for (const key of ["overrides", "resolutions"]) if (key in manifest) out.push(`package.json ${key}`);
  const pnpm = (manifest["pnpm"] ?? {}) as Record<string, unknown>;
  for (const key of ["overrides", "packageExtensions", "patchedDependencies"]) {
    if (key in pnpm) out.push(`package.json pnpm.${key}`);
  }
  const overrides = pnpm["overrides"];
  if (overrides !== null && typeof overrides === "object" && !Array.isArray(overrides)) {
    const entries = Object.entries(overrides as Record<string, unknown>);
    const allowed = entries.length > 0 && entries.every(([k, v]) => Object.hasOwn(ALLOWED_OVERRIDES, k) && ALLOWED_OVERRIDES[k] === v);
    const reached = entries.some(([k]) => closure.has(k));
    if (allowed && !reached) out.splice(out.indexOf("package.json pnpm.overrides"), 1);
  }
  for (const m of workspaceYaml.matchAll(/^(overrides|packageExtensions|patchedDependencies):/gm)) {
    out.push(`pnpm-workspace.yaml ${m[1]}`);
  }
  return out;
}

/** The dependency-name entries under `ignore:` in the npm block of dependabot.yml. */
export function dependabotNpmIgnored(config: string): string[] {
  const block = config.split(/^\s*- package-ecosystem:/m).slice(1).find((b) => /^\s*npm\b/.test(b));
  if (block === undefined) return [];
  const lines = block.split("\n").filter((line) => !line.trimStart().startsWith("#"));
  const start = lines.findIndex((line) => /^\s*ignore:\s*$/.test(line));
  if (start < 0) return [];
  const indentOf = (line: string): number => line.length - line.trimStart().length;
  const indent = indentOf(lines[start] ?? "");
  const names: string[] = [];
  for (const line of lines.slice(start + 1)) {
    if (line.trim() === "") continue;
    if (indentOf(line) < indent) break;
    const m = /^\s*-\s*dependency-name:\s*["']([^"']+)["']\s*$/.exec(line);
    if (m === null) {
      if (/^\s*[\w-]+:/.test(line) && indentOf(line) <= indent) break;
      continue;
    }
    names.push(m[1] ?? "");
  }
  return names;
}

/** The (specifier, version) the lockfile records for one importer dependency. */
export function lockedVersion(lock: string, importer: string, dep: string): { specifier: string; version: string } | null {
  const start = lock.indexOf(`\n  ${importer}:\n`);
  if (start < 0) return null;
  const rest = lock.slice(start + 1);
  const end = rest.search(/\n {2}\S[^\n]*:\n|\n\S/);
  const section = end < 0 ? rest : rest.slice(0, end);
  const escaped = dep.replace(/[\\^$.*+?()[\]{}|/]/g, "\\$&");
  const m = new RegExp(`^\\s+'?${escaped}'?:\\n\\s+specifier: (\\S+)\\n\\s+version: (\\S+)`, "m").exec(section);
  return m?.[1] === undefined || m[2] === undefined ? null : { specifier: m[1], version: m[2] };
}
