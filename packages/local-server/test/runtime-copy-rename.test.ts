import { execFileSync } from "node:child_process";
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";

// Inject faults into the module under test's fs calls, so the paths a
// concurrent creator or an unwritable root would take run deterministically.
type RenameFault = (from: string, to: string) => string | null;
type PathFault = (path: string) => "missing" | string | null;
const failRename = vi.hoisted(() => ({ when: (() => null) as RenameFault }));
const failLstat = vi.hoisted(() => ({ when: (() => null) as PathFault }));
const failReaddir = vi.hoisted(() => ({ when: (() => null) as PathFault }));
const failRm = vi.hoisted(() => ({ when: (() => null) as PathFault }));
const fault = (code: string, path: string): Error =>
  Object.assign(new Error(`${code}: ${path}`), { code });
vi.mock("node:fs", async (original) => {
  const fs = await original<typeof import("node:fs")>();
  return {
    ...fs,
    renameSync: (from: string, to: string): void => {
      const code = failRename.when(from, to);
      if (code !== null) throw fault(code, from);
      fs.renameSync(from, to);
    },
    lstatSync: ((path: string, options?: object) => {
      const code = failLstat.when(path);
      if (code === "missing") return undefined;
      if (code !== null) throw fault(code, path);
      return fs.lstatSync(path, options);
    }) as typeof fs.lstatSync,
    rmSync: ((path: string, options?: object) => {
      const code = failRm.when(path);
      if (code !== null) throw fault(code, path);
      fs.rmSync(path, options);
    }) as typeof fs.rmSync,
    readdirSync: ((path: string, options?: object) => {
      const code = failReaddir.when(path);
      if (code !== null) throw fault(code, path);
      return fs.readdirSync(path, options as undefined);
    }) as typeof fs.readdirSync,
  };
});

const { createRuntimeCopy, RuntimeCopyError, STALE_STAGING_MS, verifyRuntimeCopy } = await import(
  "../src/index.ts"
);

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

  it("touches the old copy before setting it aside, so a concurrent sweep leaves it alone", () => {
    const first = createRuntimeCopy(options);
    const old = (Date.now() - STALE_STAGING_MS - 60_000) / 1000;
    utimesSync(first.dir, old, old);
    let asideAge: number | null = null;
    failRename.when = (from, to) => {
      if (basename(to) === vendor.commit) asideAge = Date.now() - lstatSync(`${from}.old`).mtimeMs;
      return null;
    };
    createRuntimeCopy(options);
    failRename.when = () => null;
    expect(asideAge).not.toBeNull();
    expect(asideAge).toBeLessThan(STALE_STAGING_MS);
  });

  it("sweeps on a best-effort basis: a vanished, unreadable or unlistable entry never fails the copy", () => {
    const stale = (Date.now() - STALE_STAGING_MS - 60_000) / 1000;
    const names = ["vanished", "unreadable", "removable"].map((name) => `.${vendor.commit}-${name}`);
    for (const name of names) {
      mkdirSync(join(root, name));
      utimesSync(join(root, name), stale, stale);
    }
    // Vanished and unreadable in one sweep: the copy still succeeds.
    failLstat.when = (path) =>
      path.endsWith("-vanished") ? "missing" : path.endsWith("-unreadable") ? "EACCES" : null;
    expect(createRuntimeCopy(options).commit).toBe(vendor.commit);
    failLstat.when = () => null;
    expect(readdirSync(root).sort()).toEqual([names[0], names[1], vendor.commit].sort());

    // Unlistable root: the copy still succeeds.
    failReaddir.when = (path) => (path === root ? "EACCES" : null);
    expect(createRuntimeCopy(options).commit).toBe(vendor.commit);
    failReaddir.when = () => null;
    expect(readdirSync(root).sort()).toEqual([names[0], names[1], vendor.commit].sort());
  });

  it("keeps a copy that is in place when the old copy cannot be removed", () => {
    createRuntimeCopy(options);
    failRm.when = (path) => (path.endsWith(".old") ? "EBUSY" : null);
    const copy = createRuntimeCopy(options);
    failRm.when = () => null;
    expect(verifyRuntimeCopy(copy.dir, vendor.commit)).toEqual(copy);
    const leftovers = readdirSync(root).filter((name) => name.endsWith(".old"));
    expect(leftovers).toHaveLength(1);
    for (const name of leftovers) rmSync(join(root, name), { recursive: true });
  });

  it("reports the failed rename, not a failed clean-up", () => {
    createRuntimeCopy(options);
    failRename.when = (_from, to) => (basename(to) === vendor.commit ? "ENOTEMPTY" : null);
    failRm.when = () => "EBUSY";
    expect(() => createRuntimeCopy(options)).toThrow(/cannot put the runtime copy in place.*ENOTEMPTY/);
    failRename.when = () => null;
    failRm.when = () => null;
    for (const name of readdirSync(root).filter((entry) => entry.startsWith("."))) {
      rmSync(join(root, name), { recursive: true });
    }
    expect(createRuntimeCopy(options).commit).toBe(vendor.commit);
  });
});
