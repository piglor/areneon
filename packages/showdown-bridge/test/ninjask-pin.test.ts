// The vendored Areneon fork of Ninjask (vendor/ninjask, #1531): the pin tests, the build and CI rules around it, and the
// isolation rule that no evaluation, progress, search or admission path can use the native engine (ADR-003: Showdown is the
// referee, the native engine plays training games only). The supply-chain invariants of the vendored tree are in
// ninjask-tree.test.ts, which CI runs before anything is built.
//
// The checks on the submodule's files need `git submodule update --init vendor/ninjask` and skip, naming that, without it,
// unless ARENEON_REQUIRE_NINJASK=1, which makes an absent submodule (or an absent build, `make ninjask`) a failure: CI sets
// it in the `ninjask` job, which builds the engine and runs this file.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { PINNED_NINJASK_COMMIT } from "../src/ninjask-pin.ts";
import {
  BUILD_FILES,
  fetchEverySubmodule,
  git,
  jobBlock,
  makeRecipe,
  MENTION,
  NINJASK,
  NOT_INITIALISED,
  OUT,
  offenders,
  outsideBuildBlocks,
  ownedPaths,
  pickPython,
  poolReaders,
  PRESENT,
  read,
  readRepo,
  REQUIRED,
  ROOT,
  SCANNED,
  TEXT,
  withSubmodule,
} from "./support/ninjask-guards.ts";

const BUILT = join(OUT, "python", "ninjask", "_core.abi3.so");
const DATA = join(OUT, "data");
const NOT_BUILT = " [skipped: not built; run `make ninjask`]";
/** A test that needs `make ninjask` to have run: skipped unless this run requires it (and, like every test of the
 * submodule's files, without the submodule: the one that fails then says why). */
const withBuild = it.skipIf(!PRESENT || !(existsSync(BUILT) || REQUIRED));

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

  it("is guarded by code owners: the gitlink, the pin and the tests that hold it", () => {
    const owned = ownedPaths(readRepo(".github/CODEOWNERS"));
    for (const path of [
      "/.gitmodules",
      "/vendor/",
      "/packages/showdown-bridge/src/ninjask-pin.ts",
      "/packages/showdown-bridge/test/ninjask-pin.test.ts",
      "/packages/showdown-bridge/test/ninjask-tree.test.ts",
      "/packages/showdown-bridge/test/support/ninjask-guards.ts",
    ])
      expect(owned, path).toContain(path);
    // A check with teeth.
    expect(ownedPaths("# a comment\n/vendor/   @a\n\n/x/y.ts\n/z @b @c\n")).toEqual(["/vendor/", "/z"]);
  });
});

describe("make ninjask and the CI job that runs it", () => {
  const makefile = readRepo("Makefile");
  const recipe = makeRecipe(makefile, "ninjask");
  const at = (needle: string): number => recipe.findIndex((l) => l.includes(needle));

  it("checks the vendored tree's supply chain before anything of it is built", () => {
    expect(at("ninjask-tree.test.ts")).toBeGreaterThanOrEqual(0);
    expect(at("ARENEON_REQUIRE_NINJASK=1")).toBeGreaterThanOrEqual(0);
    expect(at("ninjask-tree.test.ts")).toBeLessThan(at("cargo build"));
    expect(at("git submodule update --init vendor/ninjask")).toBeLessThan(at("ninjask-tree.test.ts"));
  });

  it("assembles the package from nothing, so a file of an older commit is not importable", () => {
    expect(at("rm -rf")).toBeGreaterThanOrEqual(0);
    expect(recipe[at("rm -rf")]).toContain("python/ninjask");
    expect(recipe[at("rm -rf")]).toMatch(/NINJASK_OUT/);
    expect(at("rm -rf")).toBeLessThan(at("cp -r"));
    expect(at("test -n")).toBeLessThan(at("rm -rf")); // an empty output directory never reaches the removal
  });

  it("keeps vendor/ninjask/pools, real public player teams, out of the working tree", () => {
    const line = recipe[at("sparse-checkout")];
    expect(line).toBeDefined();
    expect(line).toContain("vendor/ninjask");
    expect(line).toContain("'!/pools/'");
    expect(at("sparse-checkout")).toBeGreaterThan(at("git submodule update --init vendor/ninjask"));
  });

  it("has helpers with teeth (negative controls)", () => {
    const make = "a:\n\techo a\nninjask:\n\techo 1\n\techo 2\n\nb:\n\techo b\n";
    expect(makeRecipe(make, "ninjask")).toEqual(["echo 1", "echo 2"]);
    expect(makeRecipe(make, "missing")).toEqual([]);
  });

  const workflows: Record<string, string> = {};
  const dir = join(ROOT, ".github", "workflows");
  for (const name of readdirSync(dir).filter((f) => /\.ya?ml$/.test(f))) workflows[name] = readFileSync(join(dir, name), "utf8");

  it("fetch only the pinned Showdown where workflows publish: the vendored engine is neither built nor read there", () => {
    expect(Object.keys(workflows)).toEqual(expect.arrayContaining(["ci.yml", "rules-publish.yml", "teams-publish.yml"]));
    expect(fetchEverySubmodule(workflows)).toEqual([]);
    for (const name of ["rules-publish.yml", "teams-publish.yml"]) expect(workflows[name], name).toContain("git submodule update --init --force --depth=1 vendor/pokemon-showdown");
  });

  it("is a job with a time limit, no credentials, the supply chain checked before any build, and the test run requiring the build", () => {
    const job = jobBlock(workflows["ci.yml"] as string, "ninjask");
    const at = (needle: string): number => job.indexOf(needle);
    expect(job).toMatch(/timeout-minutes: \d+/);
    expect(job).toMatch(/persist-credentials: false/);
    expect(job).toMatch(/submodules: true/);
    expect(job).toMatch(/cargo-deny[\s\S]*--locked[\s\S]*check/);
    expect(job).toMatch(/sha256sum -c/);
    expect(job).toMatch(/ARENEON_REQUIRE_NINJASK: "1"[\s\S]*vitest run packages\/showdown-bridge\/test\/ninjask-pin\.test\.ts/);
    // The invariants of the vendored tree, then cargo deny, before Showdown or the engine is built.
    const tree = at("vitest run packages/showdown-bridge/test/ninjask-tree.test.ts");
    expect(tree).toBeGreaterThan(-1);
    expect(job.slice(tree - 200, tree)).toContain('ARENEON_REQUIRE_NINJASK: "1"');
    for (const build of ["make showdown", "make ninjask"]) {
      expect(tree, build).toBeLessThan(at(build));
      expect(at("cargo-deny --locked"), build).toBeLessThan(at(build));
    }
    // cargo-deny is pinned on purpose, and the comment says when to bump it.
    expect(job).toMatch(/pinned on purpose[\s\S]*bump/i);
  });

  it("has a check with teeth (negative controls)", () => {
    expect(fetchEverySubmodule({ "ci.yml": "submodules: true", "a.yml": "      - uses: x\n        with:\n          submodules: true\n" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "submodules: recursive" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "# submodules: true is not used here\nrun: git submodule update --init vendor/pokemon-showdown" })).toEqual([]);
    // A bare update fetches every submodule, vendor/ninjask included, in any workflow.
    expect(fetchEverySubmodule({ "a.yml": "      - run: git submodule update --init" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "ci.yml": "      - run: git submodule update --init --recursive" })).toEqual(["ci.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: |\n          git submodule update --init \\\n            --force --depth=1" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git submodule update --init vendor/pokemon-showdown vendor/ninjask" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git submodule foreach 'git fetch'" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git clone --recurse-submodules https://example.com/x.git" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git clone --recursive https://example.com/x.git" })).toEqual(["a.yml"]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git pull --recurse-submodules" })).toEqual(["a.yml"]);
    // Another tool's own --recursive flag (osv-scanner) is not a submodule fetch.
    expect(fetchEverySubmodule({ "ci.yml": "          scan-args: |-\n            --recursive\n            ./" })).toEqual([]);
    expect(fetchEverySubmodule({ "a.yml": "      - run: git submodule update --init --force --depth=1 vendor/pokemon-showdown" })).toEqual([]);
    expect(fetchEverySubmodule({ "a.yml": "      # git submodule update --init is not run here\n      - run: make showdown" })).toEqual([]);
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
    const makefile = readRepo("Makefile");
    expect(MENTION.test(makefile)).toBe(true);
    expect(MENTION.test(outsideBuildBlocks("Makefile", makefile))).toBe(false);
    const ci = readRepo(".github/workflows/ci.yml");
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
    for (const path of images) expect(readRepo(path), path).not.toMatch(/ninjask/i);
  });

  it("has a scanner with teeth (negative controls)", () => {
    expect(offenders({ "packages/x/src/a.ts": 'const p = "vendor/ninjask/python";' })).toEqual(["packages/x/src/a.ts"]);
    expect(offenders({ "packages/x/scripts/a.ts": 'const p = "vendor/ninjask/python";' })).toEqual(["packages/x/scripts/a.ts"]);
    // Every place a package keeps code that runs or ships: testing helpers, the replay viewer's web files, the generated Python
    // contracts, and all of ml/ (the GPU environment included).
    expect(offenders({ "packages/x/testing/a.ts": 'const p = ".tmp/ninjask";' })).toEqual(["packages/x/testing/a.ts"]);
    expect(offenders({ "packages/replay-view/web/a.cjs": 'const p = "vendor/ninjask";' })).toEqual(["packages/replay-view/web/a.cjs"]);
    expect(offenders({ "packages/schema/python/areneon_contracts/a.py": "import ninjask\n" })).toEqual(["packages/schema/python/areneon_contracts/a.py"]);
    expect(offenders({ "ml/gpu-env/a.py": "from ninjask import Battle\n" })).toEqual(["ml/gpu-env/a.py"]);
    expect(offenders({ "ml/gpu-env/pyproject.toml": 'ninjask = { path = "../../vendor/ninjask" }' })).toEqual(["ml/gpu-env/pyproject.toml"]);
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
    // The scanned places are text the scanner reads: web files and stylesheets too.
    expect(TEXT.test("packages/replay-view/web/view.cjs") && TEXT.test("packages/x/web/a.html") && TEXT.test("packages/x/web/a.css")).toBe(true);
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
