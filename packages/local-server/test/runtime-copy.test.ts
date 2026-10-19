import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  renameSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR, EngineError, PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import {
  createRuntimeCopy,
  DEFAULT_RUNTIME_ROOT,
  RUNTIME_ENTRIES,
  RUNTIME_MANIFEST,
  RuntimeCopyError,
  treeDigest,
  verifyRuntimeCopy,
} from "../src/index.ts";

const temps: string[] = [];
afterAll(() => {
  for (const dir of temps) rmSync(dir, { recursive: true, force: true });
});

function tempDir(label: string): string {
  const dir = mkdtempSync(join(tmpdir(), `areneon-local-server-${label}-`));
  temps.push(dir);
  return dir;
}

function write(root: string, path: string, content: string): void {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), content);
}

/** A small committed fake Showdown checkout with every runtime entry. */
function fakeVendor(edit: (dir: string) => void = () => {}): { dir: string; commit: string } {
  const dir = tempDir("vendor");
  write(dir, "dist/sim/index.js", "module.exports = {};\n");
  write(dir, "dist/server/index.js", "// server\n");
  write(dir, "server/static/index.html", "<!doctype html>\n");
  write(dir, "config/config-example.js", "exports.port = 8000;\n");
  write(dir, "node_modules/dep/bin/dep.js", "#!/usr/bin/env node\n");
  mkdirSync(join(dir, "node_modules/.bin"));
  symlinkSync("../dep/bin/dep.js", join(dir, "node_modules/.bin/dep"));
  write(dir, "package.json", "{}\n");
  write(dir, "pokemon-showdown", "#!/usr/bin/env node\n");
  edit(dir);
  const git = (...args: string[]): string =>
    execFileSync("git", ["-C", dir, ...args], { encoding: "utf8" }).trim();
  git("init", "-q");
  git("add", "-A");
  git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "fake");
  return { dir, commit: git("rev-parse", "HEAD") };
}

/** Path, type, size and mtime of everything under `root`, to prove nothing changed. */
function listing(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isSymbolicLink()) out.push(`${relative(root, path)} -> ${readlinkSync(path)}`);
      else {
        const stat = statSync(path);
        out.push(`${relative(root, path)} ${stat.size} ${stat.mtimeMs}`);
        if (entry.isDirectory()) walk(path);
      }
    }
  };
  walk(root);
  return out.sort();
}

describe("createRuntimeCopy refuses", () => {
  it("a checkout at another commit than the pin", () => {
    const vendor = fakeVendor();
    const root = tempDir("root");
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: "0".repeat(40) }),
    ).toThrow(EngineError);
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: "0".repeat(40) }),
    ).toThrow(/expected pinned 0{40}/);
    expect(readdirSync(root)).toEqual([]);
  });

  it("a checkout without a build, before touching the default runtime root", () => {
    const vendor = fakeVendor((dir) => rmSync(join(dir, "dist"), { recursive: true }));
    expect(() => createRuntimeCopy({ vendorDir: vendor.dir })).toThrow(/Showdown build not found/);
  });

  it("a checkout missing a runtime entry", () => {
    const vendor = fakeVendor((dir) => rmSync(join(dir, "server"), { recursive: true }));
    const root = tempDir("root");
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit }),
    ).toThrow(new RuntimeCopyError("vendor checkout has no server/static; run `make showdown`"));
    expect(readdirSync(root)).toEqual([]);
  });

  it("a runtime root inside the vendor checkout, without creating it", () => {
    const vendor = fakeVendor();
    const before = listing(vendor.dir);
    expect(() =>
      createRuntimeCopy({
        vendorDir: vendor.dir,
        runtimeRoot: join(vendor.dir, "runtime", "nested"),
        expectedCommit: vendor.commit,
      }),
    ).toThrow(/is inside the vendor checkout/);
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: vendor.dir, expectedCommit: vendor.commit }),
    ).toThrow(RuntimeCopyError);
    expect(listing(vendor.dir)).toEqual(before);
  });

  it.each([
    ["an absolute symlink", (dir: string) => symlinkSync(join(dir, "package.json"), join(dir, "node_modules/abs"))],
    ["a symlink out of the copy", (dir: string) => symlinkSync("../../package.json", join(dir, "node_modules/out"))],
  ])("%s, leaving no partial copy", (_label, edit) => {
    const vendor = fakeVendor(edit);
    const root = tempDir("root");
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit }),
    ).toThrow(/leaves the runtime copy/);
    expect(readdirSync(root)).toEqual([]);
  });

  it("a dist/ that holds anything but directories and regular files", () => {
    const vendor = fakeVendor((dir) => symlinkSync("index.js", join(dir, "dist/sim/alias.js")));
    const root = tempDir("root");
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit }),
    ).toThrow(/alias\.js is not a regular file/);
    expect(readdirSync(root)).toEqual([]);
  });
});

describe("a runtime copy of a fake checkout", () => {
  const vendor = fakeVendor();
  const root = tempDir("root");
  const options = { vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit };

  function freshCopy(): string {
    return createRuntimeCopy(options).dir;
  }

  it("holds exactly the runtime entries and a manifest of commit and dist/ digest", () => {
    const before = listing(vendor.dir);
    const copy = createRuntimeCopy(options);
    expect(listing(vendor.dir)).toEqual(before);
    expect(copy.dir).toBe(join(root, vendor.commit));
    expect(readdirSync(root)).toEqual([vendor.commit]);
    expect(readdirSync(copy.dir).sort()).toEqual(
      [RUNTIME_MANIFEST, "config", "dist", "node_modules", "package.json", "pokemon-showdown", "server"].sort(),
    );
    for (const entry of RUNTIME_ENTRIES) expect(existsSync(join(copy.dir, entry))).toBe(true);
    expect(readlinkSync(join(copy.dir, "node_modules/.bin/dep"))).toBe("../dep/bin/dep.js");
    expect(readFileSync(join(copy.dir, "node_modules/.bin/dep"), "utf8")).toBe("#!/usr/bin/env node\n");
    expect(copy.distSha256).toBe(treeDigest(join(vendor.dir, "dist")));
    expect(JSON.parse(readFileSync(join(copy.dir, RUNTIME_MANIFEST), "utf8"))).toEqual({
      commit: vendor.commit,
      dist_sha256: copy.distSha256,
    });
    expect(verifyRuntimeCopy(copy.dir, vendor.commit)).toEqual(copy);
  });

  it("replaces an earlier copy for the same commit", () => {
    const dir = freshCopy();
    write(dir, "logs/stale.log", "x");
    expect(freshCopy()).toBe(dir);
    expect(existsSync(join(dir, "logs"))).toBe(false);
    expect(readdirSync(root)).toEqual([vendor.commit]);
  });

  it.each([
    ["a changed byte", (dir: string) => appendFileSync(join(dir, "dist/sim/index.js"), " ")],
    ["an added file", (dir: string) => write(dir, "dist/sim/extra.js", "")],
    ["an added empty file in a new directory", (dir: string) => write(dir, "dist/new/x.js", "")],
    ["a removed file", (dir: string) => rmSync(join(dir, "dist/server/index.js"))],
    [
      "a renamed file",
      (dir: string) => renameSync(join(dir, "dist/server/index.js"), join(dir, "dist/server/main.js")),
    ],
  ])("refuses tampering with dist/: %s", (_label, tamper) => {
    const dir = freshCopy();
    tamper(dir);
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(/does not match the recorded/);
  });

  it("refuses a symlink planted in dist/", () => {
    const dir = freshCopy();
    symlinkSync(join(vendor.dir, "dist/sim/index.js"), join(dir, "dist/sim/planted.js"));
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(/is not a regular file/);
  });

  it("refuses a copy missing a runtime entry", () => {
    const dir = freshCopy();
    rmSync(join(dir, "pokemon-showdown"));
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(/has no pokemon-showdown/);
  });

  it("refuses a copy made for another commit, by default the pin", () => {
    const dir = freshCopy();
    expect(() => verifyRuntimeCopy(dir)).toThrow(`expected pinned ${PINNED_SHOWDOWN_COMMIT}`);
  });

  it.each([
    ["missing", null],
    ["not JSON", "{"],
    ["null", "null"],
    ["an extra field", JSON.stringify({ commit: "c", dist_sha256: "d", extra: "e" })],
    ["a missing field", JSON.stringify({ commit: "c" })],
    ["a non-string field", JSON.stringify({ commit: "c", dist_sha256: 1 })],
  ])("refuses a manifest that is %s", (_label, content) => {
    const dir = freshCopy();
    if (content === null) rmSync(join(dir, RUNTIME_MANIFEST));
    else writeFileSync(join(dir, RUNTIME_MANIFEST), content);
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(RuntimeCopyError);
  });
});

describe("treeDigest", () => {
  it("depends on paths and bytes only, not on where the tree is", () => {
    const a = tempDir("tree");
    const b = tempDir("tree");
    for (const dir of [a, b]) {
      write(dir, "x/y.js", "1");
      write(dir, "z.js", "22");
    }
    expect(treeDigest(a)).toBe(treeDigest(b));
    expect(treeDigest(a)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("does not let content and path boundaries shift", () => {
    const a = tempDir("tree");
    const b = tempDir("tree");
    write(a, "a", "b");
    write(b, "ab", "");
    expect(treeDigest(a)).not.toBe(treeDigest(b));
  });
});

describe("a runtime copy of the real pinned checkout", () => {
  it("defaults to the gitignored runtime root of the repository", () => {
    expect(relative(join(DEFAULT_SHOWDOWN_DIR, "..", ".."), DEFAULT_RUNTIME_ROOT)).toBe(
      join(".local-server", "runtime"),
    );
  });

  it(
    "copies the pin without writing under vendor/, and verifies",
    () => {
      const gitStatus = (): string =>
        execFileSync("git", ["-C", DEFAULT_SHOWDOWN_DIR, "status", "--porcelain", "--ignored"], {
          encoding: "utf8",
        });
      const before = listing(DEFAULT_SHOWDOWN_DIR);
      const statusBefore = gitStatus();
      const root = tempDir("real");
      const copy = createRuntimeCopy({ runtimeRoot: root });
      expect(listing(DEFAULT_SHOWDOWN_DIR)).toEqual(before);
      expect(gitStatus()).toBe(statusBefore);
      expect(copy.commit).toBe(PINNED_SHOWDOWN_COMMIT);
      expect(copy.dir).toBe(join(root, PINNED_SHOWDOWN_COMMIT));
      expect(copy.distSha256).toBe(treeDigest(join(DEFAULT_SHOWDOWN_DIR, "dist")));
      expect(verifyRuntimeCopy(copy.dir)).toEqual(copy);
      expect(existsSync(join(copy.dir, "server/static/index.html"))).toBe(true);
      expect(existsSync(join(copy.dir, "config/config.js"))).toBe(false);
    },
    300_000,
  );
});
