// The shard layout of ADR-013 section 4. Within a split of `R` rows the shard
// count is `S = ceil(R / 2^18)`. A row goes to shard
//   first 8 bytes (big endian) of HMAC-SHA256(shard_key, canonical_team_hash) mod S
// (the key is the manifest's public `shard_key`, 32 bytes as hex; the message
// is the canonical hash as its 64 hex characters, the same convention as the
// split assignment). Each shard so carries close to the split's cell mix. Rows
// are then sorted within the shard by stratum, then canonical hash, and
// `cell_ranges` give each stratum's contiguous rows. Pure functions only.

import { createHmac } from "node:crypto";
import type { TeamSnapshotManifest } from "@areneon/showdown-bridge";
import { refuse } from "./columns.ts";

/** Rows per shard (2^18): the target, and the divisor of the shard count. */
export const SHARD_ROWS = 2 ** 18;
/** The most shards a manifest may list (its schema's `maxItems`). */
export const MAX_SHARDS = 4096;

export type CellRange = TeamSnapshotManifest["shards"][number]["cell_ranges"][number];

const HEX_KEY = /^[0-9a-f]{64}$/;
const HEX_HASH = HEX_KEY;

/** The shard count of a split of `rows` rows; a split with no rows has no shard. */
export function shardCount(rows: number, shardRows: number = SHARD_ROWS): number {
  if (!Number.isSafeInteger(rows) || rows < 0) return refuse(`a split cannot have ${rows} rows`);
  const count = Math.ceil(rows / shardRows);
  if (count > MAX_SHARDS) return refuse(`${rows} rows need ${count} shards, more than ${MAX_SHARDS}`);
  return count;
}

/** The shard of a team: a function of the key and the canonical hash alone. */
export function shardAssigner(shardKey: string, shards: number): (canonicalTeamHash: string) => number {
  if (!HEX_KEY.test(shardKey)) return refuse("the shard key must be 64 lowercase hex digits");
  if (!Number.isInteger(shards) || shards < 1) return refuse("a split needs at least one shard");
  const key = Buffer.from(shardKey, "hex");
  const count = BigInt(shards);
  return (canonicalTeamHash) => {
    if (!HEX_HASH.test(canonicalTeamHash)) return refuse(`${canonicalTeamHash} is not a canonical team hash`);
    return Number(createHmac("sha256", key).update(canonicalTeamHash, "utf8").digest().readBigUInt64BE(0) % count);
  };
}

/** What sorting a shard needs of a row. */
export interface Keyed {
  stratum: number;
  canonical_team_hash: string;
}

/** The order of rows in a shard: by stratum, then canonical hash (code units). */
export const byStratumThenHash = (a: Keyed, b: Keyed): number =>
  a.stratum - b.stratum || (a.canonical_team_hash < b.canonical_team_hash ? -1 : a.canonical_team_hash > b.canonical_team_hash ? 1 : 0);

/** Sort a shard's rows in place; a repeated canonical hash refuses (no team twice). */
export function sortShard<T extends Keyed>(rows: T[]): T[] {
  rows.sort(byStratumThenHash);
  for (let k = 1; k < rows.length; k++) {
    if ((rows[k - 1] as T).canonical_team_hash === (rows[k] as T).canonical_team_hash) return refuse(`team ${(rows[k] as T).canonical_team_hash} is in the shard twice`);
  }
  return rows;
}

/** The contiguous row range of each stratum of rows already in shard order. */
export function cellRanges(strata: Iterable<number>): CellRange[] {
  const ranges: CellRange[] = [];
  let row = 0;
  for (const stratum of strata) {
    const last = ranges[ranges.length - 1];
    if (last !== undefined && last.stratum === stratum) last.rows += 1;
    else if (last !== undefined && last.stratum > stratum) return refuse(`rows are not in stratum order at row ${row}`);
    else ranges.push({ stratum, first_row: row, rows: 1 });
    row += 1;
  }
  return ranges;
}

/**
 * Lay out one split: the rows (with their stratum) go to `shardCount(rows.length)`
 * shards by the key, each sorted. Returns the shards in index order; an index
 * that receives no row is an empty shard, which the manifest cannot hold, so
 * it refuses (with 2^18-row shards and a real key this does not happen).
 */
export function layoutSplit<T extends Keyed>(rows: readonly T[], shardKey: string, shardRows: number = SHARD_ROWS): T[][] {
  const count = shardCount(rows.length, shardRows);
  const assign = shardAssigner(shardKey, Math.max(count, 1));
  const shards: T[][] = Array.from({ length: count }, () => []);
  for (const row of rows) (shards[assign(row.canonical_team_hash)] as T[]).push(row);
  shards.forEach((shard, k) => {
    if (shard.length === 0) refuse(`shard ${k} of ${count} received no row; use another shard key`);
    sortShard(shard);
  });
  return shards;
}
