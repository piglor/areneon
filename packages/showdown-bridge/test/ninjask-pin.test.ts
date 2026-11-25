// The vendored Areneon fork of Ninjask (vendor/ninjask, #1531): the pin tests, and the isolation rule that no evaluation,
// progress, search or admission path can use the native engine (ADR-003: Showdown is the referee, the native engine plays
// training games only). The checks on the submodule's files need `git submodule update --init vendor/ninjask`, which CI
// does (`submodules: true`); the build check needs `make ninjask` (the `ninjask` job; ARENEON_REQUIRE_NINJASK=1 makes a
// missing build a failure there).
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { PINNED_NINJASK_COMMIT } from "../src/ninjask-pin.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const NINJASK = join(ROOT, "vendor", "ninjask");
const BUILT = join(ROOT, ".tmp", "ninjask", "python", "ninjask", "_core.abi3.so");
const git = (...args: string[]): string => execFileSync("git", args, { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 });
const read = (path: string): string => readFileSync(join(NINJASK, path), "utf8");

describe("the vendored Ninjask pin", () => {
  it("is a gitlink at the pinned commit", () => {
    expect(git("ls-files", "--stage", "vendor/ninjask").trim()).toBe(`160000 ${PINNED_NINJASK_COMMIT} 0\tvendor/ninjask`);
    expect(git("config", "--file", ".gitmodules", "--get", "submodule.vendor/ninjask.url").trim()).toBe("https://github.com/piglor/ninjask.git");
  });

  it("is on the same Showdown as the referee", () => {
    const gitlink = git("ls-files", "--stage", "vendor/pokemon-showdown");
    expect(gitlink).toContain(PINNED_SHOWDOWN_COMMIT);
    expect(read("refsim/SHOWDOWN_COMMIT").trim()).toBe(PINNED_SHOWDOWN_COMMIT);
    const provenance = JSON.parse(read("refsim/PROVENANCE.json")) as { showdown: { commit: string } };
    expect(provenance.showdown.commit).toBe(PINNED_SHOWDOWN_COMMIT);
    expect(read("Cargo.toml")).toContain(`showdown_commit = "${PINNED_SHOWDOWN_COMMIT}"`);
  });

  it("is the Areneon fork of the release it is based on, MIT, with the Showdown notice", () => {
    expect(read("Cargo.toml")).toMatch(/^version = "\d+\.\d+\.\d+\+areneon\.\d+"$/m);
    expect(read("LICENSE")).toContain("MIT License");
    expect(read("NOTICE")).toContain(`commit ${PINNED_SHOWDOWN_COMMIT}`);
  });

  it("builds to a module that names the pin (make ninjask)", () => {
    if (!existsSync(BUILT)) {
      expect(process.env.ARENEON_REQUIRE_NINJASK, "run `make ninjask`").toBeUndefined();
      return;
    }
    // The package as `make ninjask` assembles it, on PYTHONPATH and nothing else of this checkout's Python.
    // The module is abi3 for CPython 3.12 and later: the repository's interpreter, else NINJASK_PYTHON, else python3.
    const venv = join(ROOT, ".venv", "bin", "python");
    const python = process.env.NINJASK_PYTHON ?? (existsSync(venv) ? venv : "python3");
    const out = execFileSync(python, ["-c", "import ninjask; print(ninjask.SHOWDOWN_COMMIT)"], {
      cwd: tmpdir(),
      encoding: "utf8",
      env: { PATH: process.env.PATH ?? "", PYTHONPATH: join(ROOT, ".tmp", "ninjask", "python") },
    });
    expect(out.trim()).toBe(PINNED_SHOWDOWN_COMMIT);
  });
});

/** Tracked files that may never name the native engine: everything that ships or evaluates. */
const SCANNED = /^(packages\/[^/]+\/(src|scripts)\/|ml\/areneon_ml\/|scripts\/)/;
const MENTION = /vendor\/ninjask|\.tmp\/ninjask|^\s*(import|from)\s+ninjask\b|["']ninjask["']/m;
/** Files that are about the pin itself. A backend adapter's directory joins this list by a reviewed edit. */
const ALLOWED = new Set(["packages/showdown-bridge/src/ninjask-pin.ts"]);

/** The tracked files that name the engine and are not allowed to. */
export function offenders(files: Record<string, string>): string[] {
  return Object.keys(files)
    .filter((path) => SCANNED.test(path) && !ALLOWED.has(path) && MENTION.test(files[path] as string))
    .sort();
}

describe("the native engine is training play only", () => {
  const tracked = git("ls-files", "-z")
    .split("\0")
    .filter((p) => p !== "" && !p.startsWith("vendor/") && existsSync(join(ROOT, p)));

  it("is named by no shipped or evaluating source", () => {
    const files: Record<string, string> = {};
    for (const path of tracked) if (SCANNED.test(path) && /\.(ts|js|py|json|sh)$/.test(path)) files[path] = readFileSync(join(ROOT, path), "utf8");
    expect(Object.keys(files).length).toBeGreaterThan(100);
    expect(offenders(files)).toEqual([]);
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
    expect(offenders({ "packages/x/src/a.ts": "const p = 1;" })).toEqual([]);
    expect(offenders({ "docs/a.md": "vendor/ninjask" })).toEqual([]);
  });
});
