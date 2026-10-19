import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";

// Fail chosen renames in the module under test, so the paths a concurrent
// creator or an unwritable root would take are exercised deterministically.
type RenameFault = (from: string, to: string) => string | null;
const failRename = vi.hoisted(() => ({ when: (() => null) as RenameFault }));
vi.mock("node:fs", async (original) => {
  const fs = await original<typeof import("node:fs")>();
  return {
    ...fs,
    renameSync: (from: string, to: string): void => {
      const code = failRename.when(from, to);
      if (code !== null) throw Object.assign(new Error(`${code}: rename ${from}`), { code });
      fs.renameSync(from, to);
    },
  };
});

const { createRuntimeCopy, RuntimeCopyError, verifyRuntimeCopy } = await import("../src/index.ts");

const temps: string[] = [];
afterAll(() => {
  for (const dir of temps) rmSync(dir, { recursive: true, force: true });
});

function fakeVendor(): { dir: string; commit: string } {
  const dir = mkdtempSync(join(tmpdir(), "areneon-local-server-rename-"));
  temps.push(dir);
  for (const path of [
    "dist/sim/index.js",
    "server/static/index.html",
    "config/config-example.js",
    "node_modules/dep/index.js",
    "package.json",
    "pokemon-showdown",
  ]) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), path);
  }
  const git = (...args: string[]): string =>
    execFileSync("git", ["-C", dir, ...args], { encoding: "utf8" }).trim();
  git("init", "-q");
  git("add", "-A");
  git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "fake");
  return { dir, commit: git("rev-parse", "HEAD") };
}

describe("putting a runtime copy in place", () => {
  const vendor = fakeVendor();
  const root = mkdtempSync(join(tmpdir(), "areneon-local-server-rename-root-"));
  temps.push(root);
  const options = { vendorDir: vendor.dir, runtimeRoot: root, expectedCommit: vendor.commit };

  it("wraps a failure to set the old copy aside, and keeps the old copy", () => {
    const first = createRuntimeCopy(options);
    failRename.when = (from) => (from === first.dir ? "EACCES" : null);
    expect(() => createRuntimeCopy(options)).toThrow(RuntimeCopyError);
    expect(() => createRuntimeCopy(options)).toThrow(/cannot put the runtime copy in place.*EACCES/);
    failRename.when = () => null;
    expect(readdirSync(root)).toEqual([vendor.commit]);
    expect(verifyRuntimeCopy(first.dir, vendor.commit)).toEqual(first);
  });

  it("wraps a failure to rename the staging copy into place, leaving no staging or set-aside directory", () => {
    failRename.when = (_from, to) => (basename(to) === vendor.commit ? "ENOTEMPTY" : null);
    expect(() => createRuntimeCopy(options)).toThrow(/cannot put the runtime copy in place.*ENOTEMPTY/);
    failRename.when = () => null;
    expect(readdirSync(root)).toEqual([]);
    expect(createRuntimeCopy(options).commit).toBe(vendor.commit);
  });
});
