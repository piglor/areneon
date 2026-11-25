// The supply-chain invariants of the vendored Areneon fork of Ninjask (vendor/ninjask, #1531): it runs no code of its own at
// build time and takes its dependencies from crates.io, checksummed, only. This file reads the submodule's files and builds
// nothing, so CI's `ninjask` job (and `make ninjask`) run it BEFORE anything of the fork is built; the pin, the workflows,
// the built engine and the isolation rule are in ninjask-pin.test.ts.
//
// It skips, naming the reason, when the submodule is not initialised (`git submodule update --init vendor/ninjask`) unless
// ARENEON_REQUIRE_NINJASK=1, which makes an absent submodule a failure (CI and `make ninjask` set it).
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { jobBlock, lockProblems, manifestProblems, NINJASK, NOT_INITIALISED, PRESENT, read, readRepo, REQUIRED, treeProblems, withSubmodule } from "./support/ninjask-guards.ts";

describe("the vendored tree runs no code of its own at build time, and takes dependencies from crates.io only", () => {
  it("is initialised in a run that requires it (ARENEON_REQUIRE_NINJASK=1)", () => {
    if (REQUIRED) expect(PRESENT, "ARENEON_REQUIRE_NINJASK=1, but vendor/ninjask is empty: git submodule update --init vendor/ninjask").toBe(true);
  });

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
    expect(channel).toBeDefined();
    expect(jobBlock(readRepo(".github/workflows/ci.yml"), "ninjask")).toContain(`toolchain: ${channel}`);
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
