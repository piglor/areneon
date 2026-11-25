// Shared by ninjask-tree.test.ts (the supply-chain invariants of the vendored tree, run before anything is built) and
// ninjask-pin.test.ts (the pin, the workflows, the built engine and the isolation rule), #1531. Pure functions, so that
// each check has negative controls; the constants say where the vendored engine lives and whether a run requires it.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { it } from "vitest";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..");
export const NINJASK = join(ROOT, "vendor", "ninjask");
export const OUT = join(ROOT, ".tmp", "ninjask");
export const PRESENT = existsSync(join(NINJASK, "Cargo.toml"));
/** ARENEON_REQUIRE_NINJASK=1: an absent submodule or an absent build is a failure, not a skip (CI's `ninjask` job sets it). */
export const REQUIRED = process.env.ARENEON_REQUIRE_NINJASK === "1";
export const NOT_INITIALISED = " [skipped: vendor/ninjask is not initialised; run `git submodule update --init vendor/ninjask`]";
export const git = (...args: string[]): string => execFileSync("git", args, { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 });
export const read = (path: string): string => readFileSync(join(NINJASK, path), "utf8");
export const readRepo = (path: string): string => readFileSync(join(ROOT, path), "utf8");

/** A test that reads the submodule's files: skipped, with the reason in its name, when it is not initialised. */
export const withSubmodule = it.skipIf(!PRESENT);

/** The first of `candidates` that is CPython 3.12 or later (the built module is abi3 for 3.12 and later), else the reason. */
export function pickPython(candidates: string[], versionOf: (python: string) => string | null): { python: string } | { error: string } {
  const seen: string[] = [];
  for (const python of candidates) {
    const version = versionOf(python);
    seen.push(`${python}: ${version ?? "not found"}`);
    const m = /^(\d+)\.(\d+)/.exec(version ?? "");
    if (m && (Number(m[1]) > 3 || (Number(m[1]) === 3 && Number(m[2]) >= 12))) return { python };
  }
  return { error: `the built module is abi3 for CPython 3.12 and later, and no interpreter found is one (${seen.join("; ")}); set NINJASK_PYTHON to a 3.12 or later python` };
}

const CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index";

/** Cargo.lock packages that do not come from crates.io with a checksum (the one `root` package is the crate itself). */
export function lockProblems(lock: string, root: string): string[] {
  const out: string[] = [];
  for (const block of lock.split(/\n\[\[package\]\]\n/).slice(1)) {
    const field = (key: string): string | undefined => new RegExp(`^${key} = "([^"]*)"$`, "m").exec(block)?.[1];
    const name = field("name") ?? "?";
    const source = field("source");
    if (source === undefined) {
      if (name !== root) out.push(`${name}: a package with no source (a path dependency)`);
    } else if (source !== CRATES_IO) {
      out.push(`${name}: source ${source}`);
    } else if (!/^[0-9a-f]{64}$/.test(field("checksum") ?? "")) {
      out.push(`${name}: no checksum`);
    }
  }
  return out;
}

/** A Cargo.toml that rewrites dependencies, declares a build script, or names a git, path or registry source. */
export function manifestProblems(path: string, text: string): string[] {
  const code = text.replace(/^\s*#.*$/gm, "");
  const out: string[] = [];
  if (/^\s*\[(patch|replace)\b/m.test(code)) out.push(`${path}: a [patch] or [replace] section`);
  if (/^\s*build\s*=/m.test(code)) out.push(`${path}: a build script is declared`);
  if (/\b(git|path|registry)\s*=\s*"/.test(code)) out.push(`${path}: a git, path or registry dependency`);
  return out;
}

/** Tracked files that run code at build time or change how cargo resolves: a build script anywhere, a `.cargo/` directory. */
export function treeProblems(files: string[]): string[] {
  return files.filter((f) => /(^|\/)build\.rs$/.test(f) || /(^|\/)\.cargo\//.test(f)).map((f) => `${f}: a build script or a cargo configuration`);
}

/** The text of the job `name` of a GitHub workflow (its two-space-indented block). */
export function jobBlock(workflow: string, name: string): string {
  const lines = workflow.split("\n");
  const start = lines.findIndex((l) => l === `  ${name}:`);
  if (start < 0) return "";
  const end = lines.findIndex((l, i) => i > start && /^ {2}\S/.test(l));
  return lines.slice(start, end < 0 ? undefined : end).join("\n");
}

/** The recipe of the Makefile target `name` (the lines after `name:` that start with a tab), tabs removed. */
export function makeRecipe(makefile: string, name: string): string[] {
  const lines = makefile.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`${name}:`));
  if (start < 0) return [];
  const end = lines.findIndex((l, i) => i > start && l !== "" && !l.startsWith("\t"));
  return lines.slice(start + 1, end < 0 ? undefined : end).filter((l) => l.startsWith("\t")).map((l) => l.slice(1));
}

/** The paths a CODEOWNERS file assigns to an owner (the first field of each rule line). */
export function ownedPaths(codeowners: string): string[] {
  return codeowners
    .split("\n")
    .filter((l) => l.trim() !== "" && !l.trim().startsWith("#") && /\s@\S+/.test(l))
    .map((l) => l.trim().split(/\s+/)[0] as string);
}

/**
 * The workflows that would fetch vendor/ninjask where they must not: `submodules: true` or `recursive` outside ci.yml (whose
 * jobs that build or test the engine need it), and, in any workflow, a `git submodule update` or `foreach` that does not name
 * vendor/pokemon-showdown alone, or a git command with recursive submodules (not another tool's `--recursive`). A line continued with a backslash is one line.
 */
export function fetchEverySubmodule(workflows: Record<string, string>): string[] {
  return Object.keys(workflows)
    .filter((name) => {
      const code = (workflows[name] as string).replace(/^\s*#.*$/gm, "").replace(/\\\n\s*/g, " ");
      const flag = name !== "ci.yml" && /^\s*submodules:\s*(true|recursive)\b/m.test(code);
      const update = code.split("\n").some((l) => /git\s+submodule\s+(update|foreach)\b/.test(l) && (!/vendor\/pokemon-showdown\b/.test(l) || /vendor\/ninjask\b/.test(l)));
      const clone = /--recurse-submodules\b/.test(code) || /git\s+clone\b[^\n]*--recursive\b/.test(code);
      return flag || update || clone;
    })
    .sort();
}

/** Tracked files that may never name the native engine: everything that ships, evaluates, tests or is configured. */
export const SCANNED =
  /^(packages\/[^/]+\/(src|scripts|testing|web)\/|packages\/schema\/python\/|ml\/|scripts\/|tests\/|config\/|services\/|crates\/|\.github\/|Makefile$)/;
export const MENTION = /vendor\/ninjask|\.tmp\/ninjask|^\s*(import|from)\s+ninjask\b|["']ninjask["']/m;
/** Files that are about the pin itself. A backend adapter's directory joins this list by a reviewed edit. */
export const ALLOWED = new Set(["packages/showdown-bridge/src/ninjask-pin.ts"]);
/** The guards themselves name what they forbid. */
export const GUARDS = new Set(["packages/showdown-bridge/test/ninjask-pin.test.ts", "packages/showdown-bridge/test/ninjask-tree.test.ts", "packages/showdown-bridge/test/support/ninjask-guards.ts"]);
/** The only places outside it that build the engine: the Makefile target `ninjask` and the job `ninjask` of the CI workflow. */
export const BUILD_FILES = new Set(["Makefile", ".github/workflows/ci.yml"]);
/** The vendored tree carries real public player teams (pools/, third-party, attributed in its NOTICE): nothing here reads them. */
export const POOLS = /ninjask\/pools|vgcpastes/i;
export const TEXT = /\.(ts|js|mjs|cjs|py|sh|json|jsonl|ya?ml|toml|sql|html|css)$|^Makefile$/;

/** What `text` says outside comments and the build blocks that a build file is allowed (the Makefile's `ninjask` target and `.PHONY`, ci.yml's `ninjask` job). */
export function outsideBuildBlocks(path: string, text: string): string {
  if (path === "Makefile") {
    const lines = text.split("\n");
    const start = lines.findIndex((l) => /^ninjask:/.test(l));
    const recipe = start < 0 ? 0 : lines.slice(start + 1).findIndex((l) => !/^\t/.test(l) && l !== "");
    const end = start < 0 ? start : recipe < 0 ? lines.length : start + 1 + recipe;
    // The variables of the target (`NINJASK_OUT ?= ...`) and the .PHONY list are part of it.
    const own = (l: string): boolean => /^\.PHONY:/.test(l) || /^NINJASK_[A-Z_]+\s*[?:]?=/.test(l);
    return lines.filter((l, i) => !(i >= start && i < end) && !own(l)).join("\n").replace(/^\s*#.*$/gm, "");
  }
  if (path === ".github/workflows/ci.yml") return text.replace(jobBlock(text, "ninjask"), "").replace(/^\s*#.*$/gm, "");
  // A comment in a workflow or other YAML explains; it runs nothing.
  return /\.ya?ml$/.test(path) ? text.replace(/^\s*#.*$/gm, "") : text;
}

/** The tracked files that name the engine and are not allowed to. */
export function offenders(files: Record<string, string>): string[] {
  return Object.keys(files)
    .filter((path) => SCANNED.test(path) && !ALLOWED.has(path) && MENTION.test(outsideBuildBlocks(path, files[path] as string)))
    .sort();
}

/** The tracked files that name the fork's pools of real player teams. */
export function poolReaders(files: Record<string, string>): string[] {
  return Object.keys(files)
    .filter((path) => !/^(vendor|docs|fixtures)\//.test(path) && !path.endsWith(".md") && !ALLOWED.has(path) && !GUARDS.has(path) && POOLS.test(files[path] as string))
    .sort();
}
