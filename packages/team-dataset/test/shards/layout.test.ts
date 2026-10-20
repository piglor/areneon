import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { ShardError } from "../../src/shards/columns.ts";
import { byStratumThenHash, cellRanges, layoutSplit, MAX_SHARDS, SHARD_ROWS, shardAssigner, shardCount, sortShard } from "../../src/shards/layout.ts";
import { SHARD_KEY } from "./support.ts";

const hash = (n: number): string => createHash("sha256").update(`team ${n}`).digest("hex");

describe("shardCount", () => {
  it("is ceil(R / 2^18)", () => {
    expect(shardCount(0)).toBe(0);
    expect(shardCount(1)).toBe(1);
    expect(shardCount(SHARD_ROWS)).toBe(1);
    expect(shardCount(SHARD_ROWS + 1)).toBe(2);
    expect(shardCount(10 ** 8)).toBe(382);
    expect(shardCount(10, 4)).toBe(3);
  });
  it("refuses a negative or fractional row count, and more shards than a manifest holds", () => {
    expect(() => shardCount(-1)).toThrow(ShardError);
    expect(() => shardCount(1.5)).toThrow(/cannot have/);
    expect(() => shardCount(MAX_SHARDS * SHARD_ROWS + 1)).toThrow(/more than 4096/);
  });
});

describe("shardAssigner", () => {
  it("is the first 8 bytes of HMAC-SHA256(key, canonical hash) mod S, and depends on the key and the hash alone", () => {
    const assign = shardAssigner(SHARD_KEY, 7);
    expect(assign(hash(1))).toBe(assign(hash(1)));
    expect(shardAssigner(SHARD_KEY, 7)(hash(1))).toBe(assign(hash(1)));
    // Fixed vectors, computed independently (Python hmac): the 32-byte key over the 64-character hash,
    // first 8 bytes big endian (2490258485879265742 for this key and hash), mod S.
    expect(shardAssigner(SHARD_KEY, 1)(hash(1))).toBe(0);
    expect(assign(hash(1))).toBe(4);
    expect(shardAssigner("cd".repeat(32), 7)(hash(1))).toBe(2);
  });
  it("spreads rows over the shards about evenly", () => {
    const assign = shardAssigner(SHARD_KEY, 4);
    const counts = [0, 0, 0, 0];
    for (let n = 0; n < 4000; n++) counts[assign(hash(n))] = (counts[assign(hash(n))] as number) + 1;
    for (const count of counts) expect(count).toBeGreaterThan(850);
  });
  it("refuses a bad key, a bad shard count and a hash that is not a digest", () => {
    expect(() => shardAssigner("AB".repeat(32), 2)).toThrow(/64 lowercase hex/);
    expect(() => shardAssigner("ab", 2)).toThrow(ShardError);
    expect(() => shardAssigner(SHARD_KEY, 0)).toThrow(/at least one shard/);
    expect(() => shardAssigner(SHARD_KEY, 1.5)).toThrow(ShardError);
    expect(() => shardAssigner(SHARD_KEY, 2)("nope")).toThrow(/not a canonical team hash/);
  });
});

describe("sorting and cell ranges", () => {
  const row = (stratum: number, n: number) => ({ stratum, canonical_team_hash: hash(n) });
  it("sorts by stratum, then canonical hash", () => {
    const rows = [row(3, 1), row(1, 2), row(3, 3), row(1, 4), row(2, 5)];
    sortShard(rows);
    expect(rows.map((r) => r.stratum)).toEqual([1, 1, 2, 3, 3]);
    for (let k = 1; k < rows.length; k++) expect(byStratumThenHash(rows[k - 1] as (typeof rows)[number], rows[k] as (typeof rows)[number])).toBeLessThan(0);
  });
  it("refuses a team twice in a shard", () => {
    expect(() => sortShard([row(1, 1), row(1, 1)])).toThrow(/in the shard twice/);
  });
  it("derives each stratum's contiguous range, and refuses unsorted rows", () => {
    expect(cellRanges([1, 1, 4, 9, 9, 9])).toEqual([
      { stratum: 1, first_row: 0, rows: 2 },
      { stratum: 4, first_row: 2, rows: 1 },
      { stratum: 9, first_row: 3, rows: 3 },
    ]);
    expect(cellRanges([])).toEqual([]);
    expect(() => cellRanges([2, 1])).toThrow(/not in stratum order/);
  });
});

describe("layoutSplit", () => {
  const rows = Array.from({ length: 100 }, (_, n) => ({ stratum: n % 5, canonical_team_hash: hash(n) }));
  it("assigns every row to one shard by the HMAC rule and sorts each shard", () => {
    const shards = layoutSplit(rows, SHARD_KEY, 30);
    expect(shards).toHaveLength(4);
    expect(shards.flat()).toHaveLength(100);
    const assign = shardAssigner(SHARD_KEY, 4);
    shards.forEach((shard, k) => {
      for (const r of shard) expect(assign(r.canonical_team_hash)).toBe(k);
      expect(shard.map((r) => r.stratum)).toEqual([...shard.map((r) => r.stratum)].sort((a, b) => a - b));
    });
  });
  it("does not depend on the order the rows arrive in", () => {
    const a = layoutSplit(rows, SHARD_KEY, 30);
    const b = layoutSplit([...rows].reverse(), SHARD_KEY, 30);
    expect(b).toEqual(a);
  });
  it("makes no shard for an empty split and refuses an empty shard", () => {
    expect(layoutSplit([], SHARD_KEY)).toEqual([]);
    expect(() => layoutSplit(rows.slice(0, 3), SHARD_KEY, 1)).toThrow(/received no row/);
  });
});
