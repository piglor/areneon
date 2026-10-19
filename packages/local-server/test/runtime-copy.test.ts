import { execFile, execFileSync } from "node:child_process";
import {
  appendFileSync,
  chmodSync,
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
  utimesSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR, EngineError, PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import {
  createRuntimeCopy,
  DEFAULT_RUNTIME_ROOT,
  DIGESTED_ENTRIES,
  entryDigest,
  RUNTIME_ENTRIES,
  RUNTIME_MANIFEST,
  RUNTIME_MANIFEST_VERSION,
  RuntimeCopyError,
  STALE_STAGING_MS,
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
  write(dir, "dist/config/config-example.js", "exports.port = 8000;\n");
  write(dir, "node_modules/dep/bin/dep.js", "#!/usr/bin/env node\n");
  mkdirSync(join(dir, "node_modules/.bin"));
  symlinkSync("../dep/bin/dep.js", join(dir, "node_modules/.bin/dep"));
  write(dir, "package.json", "{}\n");
  write(dir, "pokemon-showdown", "#!/usr/bin/env node\n");
  chmodSync(join(dir, "pokemon-showdown"), 0o755);
  edit(dir);
  const git = (...args: string[]): string =>
    execFileSync("git", ["-C", dir, ...args], { encoding: "utf8" }).trim();
  git("init", "-q");
  git("add", "-A");
  git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "fake");
  return { dir, commit: git("rev-parse", "HEAD") };
}

function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
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
    [
      "an absolute symlink",
      (dir: string) => symlinkSync(join(dir, "package.json"), join(dir, "node_modules/abs")),
      /is absolute/,
    ],
    [
      "a dangling symlink",
      (dir: string) => symlinkSync("../../package.json", join(dir, "node_modules/out")),
      /is dangling/,
    ],
    [
      // Lexically `x -> l/..` stays inside, but `l` is the copy's root, so
      // `x` really is the directory holding the copy.
      "a chained symlink out of the copy",
      (dir: string) => {
        mkdirSync(join(dir, "node_modules/p/q"), { recursive: true });
        symlinkSync("../../..", join(dir, "node_modules/p/q/l"));
        symlinkSync("l/..", join(dir, "node_modules/p/q/x"));
      },
      /node_modules\/p\/q\/x -> l\/\.\. leaves the runtime copy/,
    ],
  ])("%s, leaving no partial copy", (_label, edit, message) => {
    const vendor = fakeVendor(edit);
    const root = tempDir("root");
    expect(() =>
      createRuntimeCopy({ vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit }),
    ).toThrow(message);
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

  it("holds exactly the runtime entries and a manifest of commit and entry digests", () => {
    const before = listing(vendor.dir);
    const copy = createRuntimeCopy(options);
    expect(listing(vendor.dir)).toEqual(before);
    expect(copy.dir).toBe(join(root, vendor.commit));
    expect(readdirSync(root)).toEqual([vendor.commit]);
    expect(readdirSync(copy.dir).sort()).toEqual(
      [RUNTIME_MANIFEST, "config", "dist", "node_modules", "package.json", "pokemon-showdown", "server"].sort(),
    );
    for (const entry of RUNTIME_ENTRIES) expect(existsSync(join(copy.dir, entry))).toBe(true);
    expect(statSync(join(copy.dir, "pokemon-showdown")).mode & 0o111).toBe(0o111);
    expect(readlinkSync(join(copy.dir, "node_modules/.bin/dep"))).toBe("../dep/bin/dep.js");
    expect(readFileSync(join(copy.dir, "node_modules/.bin/dep"), "utf8")).toBe("#!/usr/bin/env node\n");
    expect(copy.sha256).toEqual({
      dist: treeDigest(join(vendor.dir, "dist")),
      "server/static": treeDigest(join(vendor.dir, "server/static")),
      "config/config-example.js": sha256File(join(vendor.dir, "config/config-example.js")),
      "package.json": sha256File(join(vendor.dir, "package.json")),
      "pokemon-showdown": sha256File(join(vendor.dir, "pokemon-showdown")),
    });
    expect(JSON.parse(readFileSync(join(copy.dir, RUNTIME_MANIFEST), "utf8"))).toEqual({
      version: RUNTIME_MANIFEST_VERSION,
      commit: vendor.commit,
      sha256: copy.sha256,
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

  it("sweeps this commit's stale staging directories but leaves young and foreign ones", () => {
    const stale = join(root, `.${vendor.commit}-stale`);
    const young = join(root, `.${vendor.commit}-young`);
    const foreign = join(root, `.${"f".repeat(40)}-stale`);
    for (const dir of [stale, young, foreign]) mkdirSync(dir);
    const old = (Date.now() - STALE_STAGING_MS - 60_000) / 1000;
    utimesSync(stale, old, old);
    utimesSync(foreign, old, old);
    freshCopy();
    expect(readdirSync(root).sort()).toEqual(
      [`.${"f".repeat(40)}-stale`, `.${vendor.commit}-young`, vendor.commit].sort(),
    );
    for (const dir of [young, foreign]) rmSync(dir, { recursive: true });
  });

  it("lets concurrent creators of one commit each succeed or fail with RuntimeCopyError", async () => {
    const module = new URL("../src/index.ts", import.meta.url).href;
    const script = `
      import { createRuntimeCopy } from ${JSON.stringify(module)};
      try {
        createRuntimeCopy(${JSON.stringify(options)});
        console.log("ok");
      } catch (error) {
        console.log(error.name);
      }`;
    const run = (): Promise<string> =>
      new Promise((done, fail) => {
        execFile(process.execPath, ["--input-type=module", "-e", script], (error, stdout) =>
          error ? fail(error) : done(stdout.trim()),
        );
      });
    const outcomes = await Promise.all(Array.from({ length: 6 }, run));
    for (const outcome of outcomes) expect(["ok", "RuntimeCopyError"]).toContain(outcome);
    expect(outcomes).toContain("ok");
    expect(readdirSync(root)).toEqual([vendor.commit]);
    expect(verifyRuntimeCopy(join(root, vendor.commit), vendor.commit).commit).toBe(vendor.commit);
  }, 60_000);

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

  it.each([
    ["the entry script", "pokemon-showdown", "pokemon-showdown"],
    ["config-example.js", "config/config-example.js", "config/config-example.js"],
    ["the config defaults the server loads", "dist/config/config-example.js", "dist"],
    ["package.json", "package.json", "package.json"],
    ["a static file", "server/static/index.html", "server/static"],
  ])("refuses a changed byte in %s", (_label, path, entry) => {
    const dir = freshCopy();
    appendFileSync(join(dir, path), " ");
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(
      new RegExp(`${entry} digest [0-9a-f]{64} does not match the recorded [0-9a-f]{64}; recreate the copy`),
    );
  });

  it.each(["pokemon-showdown", "config/config-example.js"])(
    "refuses %s replaced by a symlink to the same bytes",
    (path) => {
      const dir = freshCopy();
      renameSync(join(dir, path), join(dir, `${path}.real`));
      symlinkSync(basename(`${path}.real`), join(dir, path));
      expect(readFileSync(join(dir, path), "utf8")).toBe(readFileSync(join(vendor.dir, path), "utf8"));
      expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(`${join(dir, path)} is not a regular file`);
    },
  );

  it("refuses a copy with a manifest of the older version, which digested only dist/", () => {
    const dir = freshCopy();
    writeFileSync(
      join(dir, RUNTIME_MANIFEST),
      JSON.stringify({ commit: vendor.commit, dist_sha256: treeDigest(join(dir, "dist")) }),
    );
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(
      new RuntimeCopyError(
        `runtime manifest ${join(dir, RUNTIME_MANIFEST)} is version 1, not 2; recreate the copy with createRuntimeCopy`,
      ),
    );
    expect(freshCopy()).toBe(dir);
    expect(verifyRuntimeCopy(dir, vendor.commit).commit).toBe(vendor.commit);
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
    ["missing", null, /cannot read runtime manifest/],
    ["not JSON", "{", /cannot read runtime manifest/],
    ["null", "null", /is not an object/],
    ["of another version", { version: 3 }, /is version 3, not 2; recreate the copy/],
    ["with an extra field", { extra: "e" }, /is not \{version, commit, sha256/],
    ["without a commit", { commit: undefined }, /is not \{version, commit, sha256/],
    ["with a non-string commit", { commit: 1 }, /is not \{version, commit, sha256/],
    ["with null digests", { sha256: null }, /is not \{version, commit, sha256/],
    ["with digests that are not an object", { sha256: "d" }, /is not \{version, commit, sha256/],
    ["without an entry's digest", { sha256: { dist: "d" } }, /is not \{version, commit, sha256/],
    [
      "with a non-string digest",
      { sha256: Object.fromEntries(DIGESTED_ENTRIES.map((entry) => [entry, 1])) },
      /is not \{version, commit, sha256/,
    ],
  ])("refuses a manifest that is %s", (_label, change, message) => {
    const dir = freshCopy();
    const path = join(dir, RUNTIME_MANIFEST);
    if (change === null) rmSync(path);
    else if (typeof change === "string") writeFileSync(path, change);
    else writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, "utf8")), ...change }));
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(RuntimeCopyError);
    expect(() => verifyRuntimeCopy(dir, vendor.commit)).toThrow(message);
  });
});

describe("the digested entries", () => {
  it("are the runtime entries but node_modules", () => {
    expect([...DIGESTED_ENTRIES]).toEqual(RUNTIME_ENTRIES.filter((entry) => entry !== "node_modules"));
  });
});

describe("entryDigest", () => {
  it("is treeDigest of a directory and the plain SHA-256 of a file", () => {
    const dir = tempDir("entry");
    write(dir, "d/x.js", "1");
    write(dir, "f.js", "22");
    expect(entryDigest(join(dir, "d"))).toBe(treeDigest(join(dir, "d")));
    expect(entryDigest(join(dir, "f.js"))).toBe(createHash("sha256").update("22").digest("hex"));
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
      expect(copy.sha256).toEqual(
        Object.fromEntries(DIGESTED_ENTRIES.map((entry) => [entry, entryDigest(join(DEFAULT_SHOWDOWN_DIR, entry))])),
      );
      expect(copy.sha256.dist).toBe(treeDigest(join(DEFAULT_SHOWDOWN_DIR, "dist")));
      expect(verifyRuntimeCopy(copy.dir)).toEqual(copy);
      expect(existsSync(join(copy.dir, "server/static/index.html"))).toBe(true);
      expect(statSync(join(copy.dir, "pokemon-showdown")).mode & 0o111).not.toBe(0);
      expect(existsSync(join(copy.dir, "config/config.js"))).toBe(false);
    },
    300_000,
  );
});
