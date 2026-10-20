import { createHash, randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { jcs } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { DedupeError, FileStage, KnownHashes, MemoryStage, prefixOf, referencePoolHash, sortedUnique } from "../src/dedupe.ts";

const dir = mkdtempSync(join(tmpdir(), "dedupe-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const sha = (text: string): string => createHash("sha256").update(text).digest("hex");
const hashes = (n: number): string[] => Array.from({ length: n }, (_, k) => sha(`team ${k}`));

describe("the two-stage dedupe index (ADR-013 section 2)", () => {
  it("answers membership exactly, from memory or from a sorted file", () => {
    const known = sortedUnique(hashes(500));
    for (const stage of [new MemoryStage(known), FileStage.write(join(dir, "pool.bin"), known)]) {
      const index = new KnownHashes(known, stage);
      expect(index.size).toBe(500);
      for (const h of known.filter((_, k) => k % 7 === 0)) expect(index.has(h)).toBe(true);
      for (const h of hashes(600).slice(500)) expect(index.has(h)).toBe(false);
      index.close();
    }
  });

  it("decides most absent hashes from the prefix array alone, and asks the exact stage only on a prefix hit", () => {
    const known = sortedUnique(hashes(2000));
    const index = new KnownHashes(known);
    for (const h of known.slice(0, 50)) index.has(h);
    expect(index.exactChecks).toBe(50);
    for (let k = 0; k < 500; k++) index.has(sha(`other ${k}`));
    expect(index.prefixMisses).toBe(500);
    expect(index.exactChecks).toBe(50);
  });

  it("is not fooled by a 64-bit prefix collision: the exact stage compares the whole hash", () => {
    const a = `${"ab".repeat(8)}${"00".repeat(24)}`;
    const b = `${"ab".repeat(8)}${"11".repeat(24)}`;
    expect(prefixOf(a)).toBe(prefixOf(b));
    const index = new KnownHashes(sortedUnique([a]));
    expect(index.has(a)).toBe(true);
    expect(index.has(b)).toBe(false);
    expect(index.exactChecks).toBe(2);
  });

  it("holds 8 bytes of prefix per known hash", () => {
    expect(new KnownHashes(sortedUnique(hashes(1000))).prefixBytes).toBe(8000);
    expect(new KnownHashes([]).has(sha("x"))).toBe(false);
  });

  it("finds hashes at the ends of the sorted file", () => {
    const known = sortedUnique(hashes(64));
    const stage = FileStage.write(join(dir, "ends.bin"), known);
    expect(stage.has(known[0] as string)).toBe(true);
    expect(stage.has(known[63] as string)).toBe(true);
    expect(stage.has("f".repeat(64))).toBe(false);
    stage.close();
  });

  it("refuses anything that is not a lowercase 64-digit hash", () => {
    expect(() => sortedUnique(["xyz"])).toThrow(DedupeError);
    expect(() => new KnownHashes([]).has("AB".repeat(32))).toThrow(/is not a 64-digit lowercase hex hash/);
  });
});

describe("the reference pool hash", () => {
  it("is the SHA-256 of the JCS bytes of the sorted unique list, and sorts and merges repeats", () => {
    const list = sortedUnique([...hashes(20), ...hashes(5)]);
    expect(list).toHaveLength(20);
    expect(referencePoolHash(list)).toBe(sha(jcs(list as never)));
    expect(referencePoolHash([])).toBe(sha("[]"));
    expect(sortedUnique(hashes(20).reverse())).toEqual(list);
  });
});

describe("the memory of the prefix index (measured, ADR-013 'Not verified')", () => {
  it("is 8 bytes per hash in its typed array, plus the cost of building it", () => {
    const n = 200_000;
    const raw = Array.from({ length: n }, () => randomBytes(32).toString("hex"));
    const index = new KnownHashes(sortedUnique(raw));
    expect(index.prefixBytes).toBe(8 * n);
    // a random hash is a hit only if it is known, and a prefix hit that is not known has probability n / 2^64
    expect(index.has(raw[n >> 1] as string)).toBe(true);
    expect(index.has(randomBytes(32).toString("hex"))).toBe(false);
    expect(index.exactChecks).toBe(1);
  });
});
