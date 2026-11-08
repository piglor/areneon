/* eslint-disable @typescript-eslint/no-explicit-any */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { isSafeRelativePath, readRunInputManifest, RunInputManifestError, runInputManifestBytes, runInputManifestHash } from "../src/index.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const dir = (kind: string): string => join(SAMPLES_DIR, `v1/run-input-manifest/${kind}`);
const load = (kind: string, name: string): Record<string, any> => JSON.parse(readFileSync(join(dir(kind), `${name}.json`), "utf8")) as Record<string, any>;
const names = (kind: string): string[] => readdirSync(dir(kind)).map((f) => f.slice(0, -5)).sort();

const PATHS = "paths must be safe relative paths";
// The first refusal each reader-invalid sample draws.
const MESSAGES: Record<string, string> = {
  "files-unsorted": "files must be sorted by path, each once",
  "duplicate-path": "files must be sorted by path, each once",
  "path-parent-segment": PATHS,
  "path-absolute": PATHS,
  "path-empty-segment": PATHS,
  "path-dot-segment": PATHS,
  "path-backslash": PATHS,
  "path-space": PATHS,
  "path-trailing-slash": PATHS,
  "config-file-unsafe": PATHS,
  "dataset-dir-unsafe": PATHS,
  "config-file-unlisted": "the config file must be one of the files",
  "config-inside-dataset": "the config file cannot be inside the dataset directory",
  "dataset-dir-empty": "the dataset directory must hold at least one file",
  "file-is-directory": "a path cannot be both a file and a directory",
  "dataset-dir-is-file": "a path cannot be both a file and a directory",
};

describe("the RunInputManifest reader (ADR-024 section 3.5)", () => {
  it("reads every valid sample and hands back a copy", () => {
    for (const name of names("valid")) {
      const raw = load("valid", name);
      const manifest = readRunInputManifest(raw);
      expect(manifest).toEqual(raw);
      expect(manifest).not.toBe(raw);
    }
  });

  it("refuses every invalid sample at the schema", () => {
    for (const name of names("invalid")) expect(() => readRunInputManifest(load("invalid", name)), name).toThrow("invalid RunInputManifest");
  });

  it("holds a message for every reader-invalid sample and refuses it with those words", () => {
    expect(names("reader-invalid")).toEqual(Object.keys(MESSAGES).sort());
    for (const name of names("reader-invalid")) {
      expect(() => readRunInputManifest(load("reader-invalid", name)), name).toThrow(RunInputManifestError);
      expect(() => readRunInputManifest(load("reader-invalid", name)), name).toThrow(MESSAGES[name] as string);
    }
  });

  it("never quotes a path in a reader refusal", () => {
    const raw = load("reader-invalid", "path-parent-segment");
    try {
      readRunInputManifest(raw);
    } catch (error) {
      expect((error as Error).message).not.toContain("manifest.json");
    }
  });

  it("accepts only plain relative paths", () => {
    for (const path of ["a", "a/b", "dataset/records.dev.jsonl", "A_b-c.d/e", ".hidden"]) expect(isSafeRelativePath(path), path).toBe(true);
    for (const path of ["", "/a", "a/", "a//b", ".", "..", "a/./b", "a/../b", "a\\b", "a b", "a:b", "é", "a/\u0000", "x".repeat(513)]) expect(isSafeRelativePath(path), path).toBe(false);
  });

  it("stores the manifest as its JCS bytes and hashes those bytes", async () => {
    const manifest = readRunInputManifest(load("valid", "tiny-run"));
    const bytes = runInputManifestBytes(manifest);
    expect(new TextDecoder().decode(bytes)).toBe(JSON.stringify(JSON.parse(new TextDecoder().decode(bytes))));
    expect(new TextDecoder().decode(bytes).startsWith('{"commit":')).toBe(true);
    const { createHash } = await import("node:crypto");
    expect(await runInputManifestHash(manifest)).toBe(createHash("sha256").update(bytes).digest("hex"));
  });
});
