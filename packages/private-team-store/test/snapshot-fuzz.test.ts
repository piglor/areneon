import { createHash } from "node:crypto";
import { privateLogicalSnapshotHash, privateSnapshotHash, recordHash } from "@areneon/showdown-bridge";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot, type WrittenSnapshot } from "../src/snapshot-writer.ts";
import { at, engine, input, labelSet, realDuckdb, records } from "./snapshot-support.ts";

// Deterministic corruption fuzz (seeded): damage a shard file, reseal the
// manifest's size and digest so the damage reaches the Parquet decoder, and
// require that every case either decodes to the very same records or is a
// SnapshotIoError, and that the decoder leaves no rejected promise unhandled
// (an unhandled rejection kills the process with library text).

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function corrupt(file: Uint8Array, random: () => number): Uint8Array {
  const out = file.slice();
  const at = (n: number): number => Math.floor(random() * n);
  switch (at(5)) {
    case 0:
      for (let k = 0, n = 1 + at(3); k < n; k++) out[at(out.length)] = at(256);
      return out;
    case 1:
      return out.slice(0, 1 + at(out.length - 1));
    case 2: {
      const from = at(out.length);
      out.fill(0, from, Math.min(out.length, from + 1 + at(64)));
      return out;
    }
    case 3: {
      const from = at(out.length - 8);
      out.copyWithin(from, at(out.length - 8), at(out.length - 8) + 8);
      return out;
    }
    default: {
      // Damage near the end, where the footer lives.
      for (let k = 0, n = 1 + at(4); k < n; k++) out[out.length - 1 - at(Math.min(out.length, 200))] = at(256);
      return out;
    }
  }
}

const unhandled: unknown[] = [];
const onUnhandled = (reason: unknown): void => {
  unhandled.push(reason);
};
beforeAll(() => process.on("unhandledRejection", onUnhandled));
beforeEach(() => {
  unhandled.length = 0;
});
afterAll(() => process.off("unhandledRejection", onUnhandled));

async function fuzz(w: WrittenSnapshot, cases: number, seed: number): Promise<{ same: number; refused: number }> {
  const original = await readPrivateSnapshot({ engine, manifest: w.manifest, expectedSnapshotHash: w.snapshotHash, shard: (i) => at(w.shards, i), labelSet, decodeTimeoutMs: 20_000 });
  const wanted = original.records.map((r) => recordHash(r));
  const random = mulberry32(seed);
  let same = 0;
  let refused = 0;
  for (let k = 0; k < cases; k++) {
    const shard = Math.floor(random() * w.shards.length);
    const file = corrupt(at(w.shards, shard), random);
    const shards = w.manifest.shards.map((s, i) => (i === shard ? { ...s, bytes: file.byteLength, file_sha256: createHash("sha256").update(file).digest("hex") } : s));
    const manifest = { ...w.manifest, shards } as Record<string, unknown>;
    manifest.logical_snapshot_hash = privateLogicalSnapshotHash(manifest as never);
    try {
      const read = await readPrivateSnapshot({
        engine,
        manifest,
        expectedSnapshotHash: privateSnapshotHash(manifest as never),
        shard: (i) => (i === shard ? file : at(w.shards, i)),
        labelSet,
        decodeTimeoutMs: 2000,
      });
      expect(read.records.map((r) => recordHash(r)), `case ${k}`).toEqual(wanted);
      same += 1;
    } catch (error) {
      expect(error, `case ${k}`).toBeInstanceOf(SnapshotIoError);
      refused += 1;
    }
    await new Promise((resolve) => setImmediate(resolve));
    expect(unhandled.length, `unhandled rejection after case ${k}`).toBe(0);
  }
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(unhandled).toEqual([]);
  return { same, refused };
}

describe("corrupted shard files with a resealed digest", () => {
  it("never leave a rejection unhandled: each case reads back identically or is a SnapshotIoError", async () => {
    const w = await writePrivateSnapshot(input({ records: records(12) }));
    const { same, refused } = await fuzz(w, 150, 20261021);
    expect(same + refused).toBe(150);
    expect(refused).toBeGreaterThan(30);
  }, 120_000);

  const real = realDuckdb();
  it.skipIf(real === undefined && process.env.ARENEON_REQUIRE_DUCKDB !== "1")("the same for files the pinned DuckDB tool wrote (zstd)", async () => {
    if (real === undefined) throw new Error("no DuckDB");
    const w = await writePrivateSnapshot(input({ records: records(12) }, real as never));
    const { same, refused } = await fuzz(w, 60, 7);
    expect(same + refused).toBe(60);
  }, 120_000);
});
