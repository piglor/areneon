// The two-platform check (ADR-013 section 2, section 6): the same pipeline run
// on two platforms (x86-64 and arm64 runners) must give the same logical
// snapshot. Parquet FILE bytes depend on the writer and are not compared;
// the logical hashes (writer-independent) are. Up to a size bound the whole
// pipeline is compared; above it the check compares 16 ranges of the draw
// stream chosen by an HMAC of a public key (so neither platform can know in
// advance which ranges are checked) plus the re-derived logical hashes.
//
// This module holds the pure parts: choosing the checks, the report each
// platform writes, and the comparison. Computing a range digest is
// `rangeSummary` (range.ts).

import { createHmac } from "node:crypto";
import { jcs } from "@areneon/showdown-bridge";
import { PublishError, readSnapshotDir } from "./plan.ts";

/** Draws per generation job (ADR-013 section 2). */
export const RANGE_DRAWS = 2 ** 20;
/** Ranges compared above the size bound. */
export const CHECKED_RANGES = 16;

export type CheckPlan = { mode: "full" } | { mode: "ranges"; ranges: { first: number; count: number }[] };

const HEX64 = /^[0-9a-f]{64}$/;

/**
 * Which checks a run of `totalDraws` draws gets. At or below `sizeBound` the
 * full pipeline is compared. Above it, `CHECKED_RANGES` distinct counter
 * ranges of `RANGE_DRAWS` (the last may be short), picked by
 * HMAC-SHA256(key, JCS({attempt, purpose: "two-platform-range"})) modulo the
 * number of ranges, skipping repeats; with no more ranges than that, all.
 */
export function checkPlan(totalDraws: number, sizeBound: number, key: string, rangeDraws: number = RANGE_DRAWS): CheckPlan {
  if (!Number.isSafeInteger(totalDraws) || totalDraws < 1) throw new PublishError("totalDraws must be a positive integer");
  if (!Number.isSafeInteger(sizeBound) || sizeBound < 1) throw new PublishError("sizeBound must be a positive integer");
  if (!Number.isSafeInteger(rangeDraws) || rangeDraws < 1) throw new PublishError("rangeDraws must be a positive integer");
  if (!HEX64.test(key)) throw new PublishError("the range key must be 64 lowercase hex digits");
  if (totalDraws <= sizeBound) return { mode: "full" };
  const n = Math.ceil(totalDraws / rangeDraws);
  const chosen = new Set<number>();
  if (n <= CHECKED_RANGES) for (let i = 0; i < n; i++) chosen.add(i);
  for (let attempt = 0; chosen.size < CHECKED_RANGES; attempt++) {
    const mac = createHmac("sha256", Buffer.from(key, "hex")).update(jcs({ attempt, purpose: "two-platform-range" })).digest();
    chosen.add(Number(mac.readBigUInt64BE(0) % BigInt(n)));
  }
  const ranges = [...chosen].sort((a, b) => a - b).map((i) => ({ first: i * rangeDraws, count: Math.min(rangeDraws, totalDraws - i * rangeDraws) }));
  return { mode: "ranges", ranges };
}

export interface RangeDigest {
  first: number;
  count: number;
  digest: string;
}

/** What one platform writes: its name, the logical hashes of its snapshot and the digests of the checked ranges. */
export interface PlatformReport {
  schema_version: 1;
  platform: string;
  logical_snapshot_hash: string;
  /** Sorted logical hashes of the shards. */
  shard_logical_hashes: string[];
  ranges: RangeDigest[];
}

const PLATFORM = /^[a-z0-9][a-z0-9._-]{0,63}$/;

export function platformReport(dir: string, platform: string, ranges: readonly RangeDigest[] = []): PlatformReport {
  if (!PLATFORM.test(platform)) throw new PublishError("the platform name must be 1 to 64 of a-z, 0-9, '.', '_' and '-'");
  const manifest = readSnapshotDir(dir);
  return {
    schema_version: 1,
    platform,
    logical_snapshot_hash: manifest.logical_snapshot_hash,
    shard_logical_hashes: manifest.shards.map((s) => s.logical_hash).sort(),
    ranges: [...ranges].sort((a, b) => a.first - b.first),
  };
}

const isObject = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);

export function parseReport(text: string): PlatformReport {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    throw new PublishError("a platform report is not JSON");
  }
  const ok =
    isObject(v) &&
    Object.keys(v).sort().join() === "logical_snapshot_hash,platform,ranges,schema_version,shard_logical_hashes" &&
    v.schema_version === 1 &&
    typeof v.platform === "string" &&
    PLATFORM.test(v.platform) &&
    typeof v.logical_snapshot_hash === "string" &&
    HEX64.test(v.logical_snapshot_hash) &&
    Array.isArray(v.shard_logical_hashes) &&
    v.shard_logical_hashes.every((h) => typeof h === "string" && HEX64.test(h)) &&
    Array.isArray(v.ranges) &&
    v.ranges.every(
      (r) =>
        isObject(r) &&
        Object.keys(r).sort().join() === "count,digest,first" &&
        Number.isSafeInteger(r.first) &&
        Number.isSafeInteger(r.count) &&
        typeof r.digest === "string" &&
        HEX64.test(r.digest),
    );
  if (!ok) throw new PublishError("not a platform report");
  return v as unknown as PlatformReport;
}

/** The differences between two platforms' reports; empty means they agree. */
export function comparePlatforms(a: PlatformReport, b: PlatformReport): string[] {
  const problems: string[] = [];
  if (a.platform === b.platform) problems.push("both reports name the same platform");
  if (a.logical_snapshot_hash !== b.logical_snapshot_hash) problems.push("the logical snapshot hashes differ");
  if (a.shard_logical_hashes.join() !== b.shard_logical_hashes.join()) problems.push("the shards' logical hashes differ");
  if (a.ranges.length !== b.ranges.length) problems.push("the platforms checked different numbers of ranges");
  else {
    a.ranges.forEach((r, i) => {
      const o = b.ranges[i] as RangeDigest;
      if (r.first !== o.first || r.count !== o.count) problems.push(`range ${i} is not the same range on both platforms`);
      else if (r.digest !== o.digest) problems.push(`range ${r.first}+${r.count} has different digests`);
    });
  }
  return problems;
}

/** Throws a `PublishError` listing the differences, if any. */
export function assertPlatformsAgree(a: PlatformReport, b: PlatformReport): void {
  const problems = comparePlatforms(a, b);
  if (problems.length > 0) throw new PublishError(`the two-platform check failed:\n- ${problems.join("\n- ")}`);
}
