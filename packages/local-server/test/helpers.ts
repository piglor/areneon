import { execFileSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterAll } from "vitest";
import { createRuntimeCopy, type RuntimeCopy } from "../src/index.ts";

const temps: string[] = [];
afterAll(() => {
  for (const dir of temps) rmSync(dir, { recursive: true, force: true });
});

export function tempDir(label: string): string {
  const dir = mkdtempSync(join(tmpdir(), `areneon-local-server-${label}-`));
  temps.push(dir);
  return dir;
}

export function write(root: string, path: string, content: string): void {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), content);
}

/**
 * A runtime copy of a small committed fake Showdown checkout whose
 * `pokemon-showdown` entry script is `entry` (a Node script).
 */
export function fakeCopy(entry: string): RuntimeCopy & { vendorDir: string; runtimeRoot: string } {
  const dir = tempDir("vendor");
  write(dir, "dist/sim/index.js", "module.exports = {};\n");
  write(dir, "dist/server/index.js", "// server\n");
  write(dir, "server/static/index.html", "<!doctype html>\n");
  write(dir, "config/config-example.js", "exports.port = 8000;\nexports.bindaddress = '0.0.0.0';\n");
  write(dir, "node_modules/dep/bin/dep.js", "#!/usr/bin/env node\n");
  mkdirSync(join(dir, "node_modules/.bin"));
  symlinkSync("../dep/bin/dep.js", join(dir, "node_modules/.bin/dep"));
  write(dir, "package.json", "{}\n");
  write(dir, "pokemon-showdown", entry);
  chmodSync(join(dir, "pokemon-showdown"), 0o755);
  const git = (...args: string[]): string =>
    execFileSync("git", ["-C", dir, ...args], { encoding: "utf8" }).trim();
  git("init", "-q");
  git("add", "-A");
  git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "fake");
  const runtimeRoot = tempDir("runtime");
  const copy = createRuntimeCopy({ vendorDir: dir, runtimeRoot, expectedCommit: git("rev-parse", "HEAD") });
  return { ...copy, vendorDir: dir, runtimeRoot };
}

/** A port nothing listens on right now, chosen by the OS on 127.0.0.1. */
export function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.once("error", fail);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      probe.close(() => done(port));
    });
  });
}
