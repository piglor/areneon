import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { writeLocalCopy } from "../src/local-copy.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { workDir } from "./snapshot-support.ts";

describe("writeLocalCopy", () => {
  it("removes its partial directory and says only local_copy_failed when a file cannot be written", () => {
    const dir = join(dirname(workDir()), "snapshots");
    const hash = "ab".repeat(32);
    let caught: unknown;
    try {
      writeLocalCopy(dir, hash, new Uint8Array([1]), [null as unknown as Uint8Array]);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(SnapshotIoError);
    expect((caught as SnapshotIoError).code).toBe("local_copy_failed");
    expect(readdirSync(dir)).toEqual([]);
    expect(existsSync(join(dir, hash))).toBe(false);
  });

  it("clears a partial directory a killed run left behind", () => {
    const dir = join(dirname(workDir()), "snapshots");
    const hash = "cd".repeat(32);
    mkdirSync(join(dir, `.${hash}.partial`), { recursive: true, mode: 0o700 });
    writeFileSync(join(dir, `.${hash}.partial`, "manifest.json"), "left behind");
    writeLocalCopy(dir, hash, new Uint8Array([1]), []);
    expect(readdirSync(dir)).toEqual([hash]);
    expect(readdirSync(join(dir, hash)).sort()).toEqual(["manifest.json", "shards"]);
  });
});
