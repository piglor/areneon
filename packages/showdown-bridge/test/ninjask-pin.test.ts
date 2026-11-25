// The vendored Areneon fork of Ninjask (vendor/ninjask, #1531): the pin tests, the supply-chain invariants of the vendored
// tree, and the isolation rule that no evaluation, progress, search or admission path can use the native engine (ADR-003:
// Showdown is the referee, the native engine plays training games only).
//
// The checks on the submodule's files need `git submodule update --init vendor/ninjask` and skip, naming that, without it,
// unless ARENEON_REQUIRE_NINJASK=1, which makes an absent submodule (or an absent build, `make ninjask`) a failure: CI sets
// it in the `ninjask` job, which builds the engine and runs this file.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { PINNED_NINJASK_COMMIT } from "../src/ninjask-pin.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const NINJASK = join(ROOT, "vendor", "ninjask");
const OUT = join(ROOT, ".tmp", "ninjask");
const BUILT = join(OUT, "python", "ninjask", "_core.abi3.so");
const DATA = join(OUT, "data");
const PRESENT = existsSync(join(NINJASK, "Cargo.toml"));
const REQUIRED = process.env.ARENEON_REQUIRE_NINJASK === "1";
const NOT_INITIALISED = " [skipped: vendor/ninjask is not initialised; run `git submodule update --init vendor/ninjask`]";
const NOT_BUILT = " [skipped: not built; run `make ninjask`]";
const git = (...args: string[]): string => execFileSync("git", args, { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 });
const read = (path: string): string => readFileSync(join(NINJASK, path), "utf8");

/** A test that reads the submodule's files: skipped, with the reason in its name, when it is not initialised. */
const withSubmodule = it.skipIf(!PRESENT);
/** A test that needs `make ninjask` to have run: skipped unless this run requires it (and, like every test of the
 * submodule's files, without the submodule: the one that fails then says why). */
const withBuild = it.skipIf(!PRESENT || !(existsSync(BUILT) || REQUIRED));

// ---------------------------------------------------------------------------------------------------------------------
// The interpreter that loads the built module, and the invariants of the vendored tree: pure functions, so that each has
// negative controls.

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

describe("the vendored Ninjask pin", () => {
  it("is a gitlink at the pinned commit", () => {
    expect(git("ls-files", "--stage", "vendor/ninjask").trim()).toBe(`160000 ${PINNED_NINJASK_COMMIT} 0\tvendor/ninjask`);
    expect(git("config", "--file", ".gitmodules", "--get", "submodule.vendor/ninjask.url").trim()).toBe("https://github.com/piglor/ninjask.git");
  });

  it("is initialised in a run that requires it (ARENEON_REQUIRE_NINJASK=1)", () => {
    if (REQUIRED) expect(PRESENT, "ARENEON_REQUIRE_NINJASK=1, but vendor/ninjask is empty: git submodule update --init vendor/ninjask").toBe(true);
  });

  withSubmodule(`is at the commit its gitlink names${PRESENT ? "" : NOT_INITIALISED}`, () => {
    expect(execFileSync("git", ["rev-parse", "HEAD"], { cwd: NINJASK, encoding: "utf8" }).trim()).toBe(PINNED_NINJASK_COMMIT);
  });

  withSubmodule(`is on the same Showdown as the referee${PRESENT ? "" : NOT_INITIALISED}`, () => {
    const gitlink = git("ls-files", "--stage", "vendor/pokemon-showdown");
    expect(gitlink).toContain(PINNED_SHOWDOWN_COMMIT);
    expect(read("refsim/SHOWDOWN_COMMIT").trim()).toBe(PINNED_SHOWDOWN_COMMIT);
    const provenance = JSON.parse(read("refsim/PROVENANCE.json")) as { showdown: { commit: string } };
    expect(provenance.showdown.commit).toBe(PINNED_SHOWDOWN_COMMIT);
    expect(read("Cargo.toml")).toContain(`showdown_commit = "${PINNED_SHOWDOWN_COMMIT}"`);
  });

  withSubmodule(`is the Areneon fork of the release it is based on, MIT, with the Showdown notice${PRESENT ? "" : NOT_INITIALISED}`, () => {
    expect(read("Cargo.toml")).toMatch(/^version = "\d+\.\d+\.\d+\+areneon\.\d+"$/m);
    expect(read("LICENSE")).toContain("MIT License");
    expect(read("NOTICE")).toContain(`commit ${PINNED_SHOWDOWN_COMMIT}`);
  });
});

describe("the vendored tree runs no code of its own at build time, and takes dependencies from crates.io only", () => {
  const files = PRESENT ? execFileSync("git", ["ls-files", "-z"], { cwd: NINJASK, encoding: "utf8", maxBuffer: 1 << 26 }).split("\0").filter(Boolean) : [];

  withSubmodule(`has no build script and no .cargo directory${PRESENT ? "" : NOT_INITIALISED}`, () => {
    expect(files.length).toBeGreaterThan(100);
    expect(treeProblems(files)).toEqual([]);
  });

  withSubmodule(`has manifests without [patch], build scripts or git, path and registry dependencies${PRESENT ? "" : NOT_INITIALISED}`, () => {
    const manifests = files.filter((f) => /(^|\/)Cargo\.toml$/.test(f));
    expect(manifests).toContain("Cargo.toml");
    expect(manifests.flatMap((f) => manifestProblems(f, read(f)))).toEqual([]);
  });

  withSubmodule(`locks every package to crates.io with a checksum${PRESENT ? "" : NOT_INITIALISED}`, () => {
    const lock = read("Cargo.lock");
    expect((lock.match(/^\[\[package\]\]$/gm) ?? []).length).toBeGreaterThan(10);
    expect(lockProblems(lock, "ninjask")).toEqual([]);
  });

  withSubmodule(`builds with the toolchain CI installs${PRESENT ? "" : NOT_INITIALISED}`, () => {
    const channel = /^channel = "([^"]+)"$/m.exec(read("rust-toolchain.toml"))?.[1];
    const job = jobBlock(readFileSync(join(ROOT, ".github", "workflows", "ci.yml"), "utf8"), "ninjask");
    expect(job).toContain(`toolchain: ${channel}`);
  });

  it("has checks with teeth (negative controls)", () => {
    const lock = (body: string): string => `# generated\nversion = 4\n\n[[package]]\n${body}\n`;
    const ok = 'name = "serde"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "' + "a".repeat(64) + '"';
    expect(lockProblems(lock(ok), "ninjask")).toEqual([]);
    expect(lockProblems(lock('name = "ninjask"\nversion = "0.4.1"'), "ninjask")).toEqual([]);
    expect(lockProblems(lock('name = "other"\nversion = "0.1.0"'), "ninjask")).toEqual(["other: a package with no source (a path dependency)"]);
    expect(lockProblems(lock('name = "g"\nversion = "1"\nsource = "git+https://example.com/g#abc"'), "ninjask")).toEqual(["g: source git+https://example.com/g#abc"]);
    expect(lockProblems(lock('name = "r"\nversion = "1"\nsource = "registry+https://example.com/index"'), "ninjask")).toEqual(["r: source registry+https://example.com/index"]);
    expect(lockProblems(lock(ok.replace(/checksum = .*/, "")), "ninjask")).toEqual(["serde: no checksum"]);
    expect(lockProblems(lock(ok.replace("a".repeat(64), "short")), "ninjask")).toEqual(["serde: no checksum"]);

    expect(manifestProblems("Cargo.toml", '[dependencies]\nserde = "1"\n# git = "x" is a comment\n')).toEqual([]);
    expect(manifestProblems("Cargo.toml", '[patch.crates-io]\nserde = { path = "../serde" }\n')).toHaveLength(2);
    expect(manifestProblems("Cargo.toml", '[replace]\n"serde:1.0.0" = {}\n')).toEqual(["Cargo.toml: a [patch] or [replace] section"]);
    expect(manifestProblems("Cargo.toml", '[package]\nbuild = "build.rs"\n')).toEqual(["Cargo.toml: a build script is declared"]);
    expect(manifestProblems("Cargo.toml", '[dependencies]\nx = { git = "https://example.com/x" }\n')).toEqual(["Cargo.toml: a git, path or registry dependency"]);
    expect(manifestProblems("Cargo.toml", '[dependencies]\nx = { version = "1", registry = "other" }\n')).toEqual(["Cargo.toml: a git, path or registry dependency"]);

    expect(treeProblems(["src/lib.rs", "Cargo.toml"])).toEqual([]);
    expect(treeProblems(["build.rs"])).toHaveLength(1);
    expect(treeProblems(["crates/x/build.rs"])).toHaveLength(1);
    expect(treeProblems([".cargo/config.toml"])).toHaveLength(1);
    expect(treeProblems(["src/.cargo/config"])).toHaveLength(1);
    expect(treeProblems(["src/rebuild.rs"])).toEqual([]);

    const workflow = "jobs:\n  python:\n    steps: a\n  ninjask:\n    steps: b\n      toolchain: 1.97.1\n  team-guard:\n    steps: c\n";
    expect(jobBlock(workflow, "ninjask")).toContain("1.97.1");
    expect(jobBlock(workflow, "ninjask")).not.toContain("team-guard");
    expect(jobBlock(workflow, "missing")).toBe("");
  });
});

/** The workflows that fetch every submodule (`submodules: true`), vendor/ninjask included, and that are not the CI workflow. */
export function fetchEverySubmodule(workflows: Record<string, string>): string[] {
  return Object.keys(workflows)
    .filter((name) => name !== "ci.yml" && /^\s*submodules:\s*(true|recursive)\b/m.test(workflows[name] as string))
    .sort();
}

describe("the workflows that publish, and the CI job that builds the engine", () => {
  const dir = join(ROOT, ".github", "workflows");
  const workflows: Record<string, string> = {};
  for (const name of readdirSync(dir).filter((f) => /\.ya?ml$/.test(f))) workflows[name] = readFileSync(join(dir, name), "utf8");

  it("fetch only the pinned Showdown: the vendored engine is neither built nor read where they publish", () => {
    expect(Object.keys(workflows)).toEqual(expect.arrayContaining(["ci.yml", "rules-publish.yml", "teams-publish.yml"]));
    expect(fetchEverySubmodule(workflows)).toEqual([]);
    for (const name of ["rules-publish.yml", "teams-publish.yml"]) expect(workflows[name], name).toContain("git submodule update --init --force --depth=1 vendor/pokemon-showdown");
  });

  it("keeps no credentials in the ninjask job's checkout, and requires the submodule and the build in its test run", () => {
    const job = jobBlock(workflows["ci.yml"] as string, "ninjask");
    expect(job).toMatch(/persist-credentials: false/);
    expect(job).toMatch(/submodules: true/);
    expect(job).toMatch(/ARENEON_REQUIRE_NINJASK: "1"[\s\S]*vitest run packages\/showdown-bridge\/test\/ninjask-pin\.test\.ts/);
    expect(job).toMatch(/cargo-deny[\s\S]*--locked[\s\S]*check/);
    expect(job).toMatch(/sha256sum -c/);
  });

  it("has a check with teeth (negative control)", () => {
    expect(fetchEverySubmodule({ "ci.yml": "submodules: true", "a.yml": "      - uses: x\n        with:\n          submodules: true\n" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "submodules: recursive" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "# submodules: true is not used here\nrun: git submodule update --init vendor/pokemon-showdown" })).toEqual([]);
  });
});

describe("the interpreter that loads the built module", () => {
  const versions: Record<string, string | null> = { old: "3.10", new: "3.13", odd: "garbage", gone: null, tip: "3.12" };
  const versionOf = (p: string): string | null => versions[p] ?? null;

  it("is CPython 3.12 or later, and says so when none is", () => {
    expect(pickPython(["old", "new"], versionOf)).toEqual({ python: "new" });
    expect(pickPython(["tip"], versionOf)).toEqual({ python: "tip" });
    const none = pickPython(["old", "gone", "odd"], versionOf);
    expect("error" in none && none.error).toMatch(/CPython 3\.12 and later.*old: 3\.10.*gone: not found.*odd: garbage.*NINJASK_PYTHON/);
    expect(pickPython([], versionOf)).toHaveProperty("error");
  });
});

/** The python that runs the built package: NINJASK_PYTHON alone if set (an explicit choice is never overridden), else the repository's. */
function interpreter(): string {
  const explicit = process.env.NINJASK_PYTHON;
  const venv = join(ROOT, ".venv", "bin", "python");
  const candidates = explicit ? [explicit] : [...(existsSync(venv) ? [venv] : []), "python3.13", "python3.12", "python3"];
  const versionOf = (python: string): string | null => {
    try {
      return execFileSync(python, ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
    } catch {
      return null;
    }
  };
  const picked = pickPython(candidates, versionOf);
  if ("error" in picked) throw new Error(picked.error);
  return picked.python;
}

/** Run python code against the package as `make ninjask` assembles it, and nothing else of this checkout's Python. */
function runBuilt(code: string, extraPath: string[] = [], env: Record<string, string> = {}): string {
  return execFileSync(interpreter(), ["-c", code], {
    cwd: tmpdir(),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: { PATH: process.env.PATH ?? "", PYTHONPATH: [join(OUT, "python"), ...extraPath].join(":"), ...env },
  }).trim();
}

describe("the built engine (make ninjask)", () => {
  withBuild(`is a module that names the pinned Showdown${existsSync(BUILT) ? "" : NOT_BUILT}`, () => {
    expect(existsSync(BUILT), "run `make ninjask`").toBe(true);
    expect(runBuilt("import ninjask; print(ninjask.SHOWDOWN_COMMIT)")).toBe(PINNED_SHOWDOWN_COMMIT);
  });

  withBuild(`loads the formats exported from this repository's Showdown, which are the recorded ones${existsSync(BUILT) ? "" : NOT_BUILT}`, () => {
    // The package is assembled with no refsim/ beside it, as an installed wheel is: `NINJASK_DATA` is how it finds its data.
    expect(existsSync(join(DATA, "format")), "run `make ninjask`: it exports the formats into .tmp/ninjask/data").toBe(true);
    const code = "import ninjask, ninjask.paths as p; assert not p.IN_CHECKOUT; print(ninjask.dex('gen9championsvgc2026regmc').format_id)";
    expect(runBuilt(code, [], { NINJASK_DATA: DATA })).toBe("gen9championsvgc2026regmc");
    // Without the data it refuses, and says what to give it (it cannot export: there is no Showdown beside it).
    expect(() => runBuilt("import ninjask; ninjask.dex('gen9championsvgc2026regmc')", [], { NINJASK_DATA: join(OUT, "no-such-data") })).toThrow(/NINJASK_DATA/);
    // The same files are the ones refsim/PROVENANCE.json records (sizes and hashes, no Showdown needed).
    const checked = execFileSync(interpreter(), [join(NINJASK, "scripts", "repin.py"), "provenance", "check-data", DATA], { encoding: "utf8" });
    expect(checked).toContain("provenance data: ok");
  });

  withBuild(`does not honour the ninjask.extension entry-point group${existsSync(BUILT) ? "" : NOT_BUILT}`, () => {
    // Any installed distribution can declare the group; it must not replace the engine's classes (the fork makes it opt-in).
    const dir = mkdtempSync(join(tmpdir(), "areneon-ext-"));
    mkdirSync(join(dir, "hostile_ext-1.0.dist-info"));
    writeFileSync(join(dir, "hostile_ext.py"), "raise SystemExit('the declared extension module was imported')\n");
    writeFileSync(join(dir, "hostile_ext-1.0.dist-info", "METADATA"), "Metadata-Version: 2.1\nName: hostile-ext\nVersion: 1.0\n");
    writeFileSync(join(dir, "hostile_ext-1.0.dist-info", "entry_points.txt"), "[ninjask.extension]\nhostile = hostile_ext\n");
    const code = "import warnings; warnings.simplefilter('ignore'); import ninjask, ninjask._core as c; print(ninjask.Battle is c.Battle, ninjask.Dex is c.Dex)";
    expect(runBuilt(code, [dir])).toBe("True True");
    // Naming it in the process is the opt-in; the assembled package is not made to do that here.
    expect(() => runBuilt(code, [dir], { NINJASK_EXTENSION: "hostile" })).toThrow(/declared extension module was imported/);
  });
});

// ---------------------------------------------------------------------------------------------------------------------

/** Tracked files that may never name the native engine: everything that ships, evaluates, tests or is configured. */
const SCANNED = /^(packages\/[^/]+\/(src|scripts)\/|ml\/areneon_ml\/|scripts\/|tests\/|config\/|services\/|crates\/|\.github\/|Makefile$)/;
const MENTION = /vendor\/ninjask|\.tmp\/ninjask|^\s*(import|from)\s+ninjask\b|["']ninjask["']/m;
/** Files that are about the pin itself. A backend adapter's directory joins this list by a reviewed edit. */
const ALLOWED = new Set(["packages/showdown-bridge/src/ninjask-pin.ts"]);
const SELF = "packages/showdown-bridge/test/ninjask-pin.test.ts";
/** The only places outside it that build the engine: the Makefile target `ninjask` and the job `ninjask` of the CI workflow. */
const BUILD_FILES = new Set(["Makefile", ".github/workflows/ci.yml"]);
/** The vendored tree carries real public player teams (pools/, third-party, attributed in its NOTICE): nothing here reads them. */
const POOLS = /ninjask\/pools|vgcpastes/i;
const TEXT = /\.(ts|js|mjs|cjs|py|sh|json|jsonl|ya?ml|toml|sql)$|^Makefile$/;

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
    .filter((path) => !/^(vendor|docs|fixtures)\//.test(path) && !path.endsWith(".md") && !ALLOWED.has(path) && path !== SELF && POOLS.test(files[path] as string))
    .sort();
}

describe("the native engine is training play only", () => {
  const tracked = git("ls-files", "-z")
    .split("\0")
    .filter((p) => p !== "" && !p.startsWith("vendor/") && existsSync(join(ROOT, p)));
  const texts = (accept: (path: string) => boolean): Record<string, string> => {
    const files: Record<string, string> = {};
    for (const path of tracked) if (accept(path) && TEXT.test(path)) files[path] = readFileSync(join(ROOT, path), "utf8");
    return files;
  };

  it("is named by no shipped, evaluating, testing or configuring source", () => {
    const files = texts((path) => SCANNED.test(path));
    expect(Object.keys(files).length).toBeGreaterThan(100);
    expect(Object.keys(files).filter((p) => BUILD_FILES.has(p)).sort()).toEqual([...BUILD_FILES].sort());
    expect(offenders(files)).toEqual([]);
  });

  it("is built by the Makefile target and the CI job and by nothing else", () => {
    const makefile = readFileSync(join(ROOT, "Makefile"), "utf8");
    expect(MENTION.test(makefile)).toBe(true);
    expect(MENTION.test(outsideBuildBlocks("Makefile", makefile))).toBe(false);
    const ci = readFileSync(join(ROOT, ".github", "workflows", "ci.yml"), "utf8");
    expect(MENTION.test(jobBlock(ci, "ninjask"))).toBe(true);
    expect(MENTION.test(outsideBuildBlocks(".github/workflows/ci.yml", ci))).toBe(false);
  });

  it("does not read the vendored tree's pools of real player teams", () => {
    const files = texts((path) => !/^(fixtures|private-data|node_modules)\//.test(path) && !/(^|\/)(pnpm-lock\.yaml|uv\.lock)$/.test(path));
    expect(Object.keys(files).length).toBeGreaterThan(500);
    expect(poolReaders(files)).toEqual([]);
  });

  it("is named by no Dockerfile or compose file", () => {
    const images = tracked.filter((p) => /(^|\/)(Dockerfile[^/]*|[^/]*compose[^/]*\.ya?ml)$/.test(p));
    expect(images.length).toBeGreaterThan(0);
    for (const path of images) expect(readFileSync(join(ROOT, path), "utf8"), path).not.toMatch(/ninjask/i);
  });

  it("has a scanner with teeth (negative controls)", () => {
    expect(offenders({ "packages/x/src/a.ts": 'const p = "vendor/ninjask/python";' })).toEqual(["packages/x/src/a.ts"]);
    expect(offenders({ "ml/areneon_ml/a.py": "import ninjask\n" })).toEqual(["ml/areneon_ml/a.py"]);
    expect(offenders({ "scripts/a.sh": "PYTHONPATH=.tmp/ninjask/python" })).toEqual(["scripts/a.sh"]);
    expect(offenders({ "tests/scripts/test_x.py": "import ninjask\n" })).toEqual(["tests/scripts/test_x.py"]);
    expect(offenders({ "config/x.json": '{"engine": ".tmp/ninjask/python"}' })).toEqual(["config/x.json"]);
    expect(offenders({ "services/x/main.py": "from ninjask import Battle\n" })).toEqual(["services/x/main.py"]);
    expect(offenders({ ".github/workflows/other.yml": "run: PYTHONPATH=.tmp/ninjask/python python x.py" })).toEqual([".github/workflows/other.yml"]);
    expect(offenders({ ".github/workflows/other.yml": "# the vendored vendor/ninjask engine is not fetched here\nrun: echo" })).toEqual([]);
    expect(offenders({ "packages/x/src/a.ts": "const p = 1;" })).toEqual([]);
    expect(offenders({ "docs/a.md": "vendor/ninjask" })).toEqual([]);
    expect(offenders({ "packages/showdown-bridge/src/ninjask-pin.ts": "vendor/ninjask" })).toEqual([]);
    // The build files may build the engine in their own block and nowhere else.
    const make = ".PHONY: a ninjask\nNINJASK_OUT ?= .tmp/ninjask\n# vendor/ninjask is built here\nninjask:\n\tcd vendor/ninjask && cargo build\n\nts:\n\tpnpm test\n";
    expect(offenders({ Makefile: make })).toEqual([]);
    expect(offenders({ Makefile: make + "check:\n\tPYTHONPATH=.tmp/ninjask/python pytest\n" })).toEqual(["Makefile"]);
    expect(offenders({ Makefile: make.replace("ts:\n\tpnpm test", "ts:\n\tcd vendor/ninjask && make") })).toEqual(["Makefile"]);
    const ci = "jobs:\n  python:\n    steps: [a]\n  ninjask:\n    steps: [make ninjask, vendor/ninjask]\n  ci:\n    needs: [python, ninjask]\n";
    expect(offenders({ ".github/workflows/ci.yml": ci })).toEqual([]);
    expect(offenders({ ".github/workflows/ci.yml": ci.replace("steps: [a]", "steps: [PYTHONPATH=.tmp/ninjask/python pytest]") })).toEqual([".github/workflows/ci.yml"]);

    expect(poolReaders({ "packages/x/src/a.ts": 'read("vendor/ninjask/pools/vgcpastes-regmc.txt")' })).toEqual(["packages/x/src/a.ts"]);
    expect(poolReaders({ "config/a.json": '{"pool": "vgcpastes-regmc.txt"}' })).toEqual(["config/a.json"]);
    expect(poolReaders({ "docs/architecture/ninjask.md": "vendor/ninjask/pools" })).toEqual([]);
    expect(poolReaders({ "packages/x/src/a.ts": "const p = 1;" })).toEqual([]);
  });
});
