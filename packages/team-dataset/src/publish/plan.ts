// The publish plan of one teams snapshot (ADR-013 section 6, in the shape of
// rules-publish, ADR-007 section 6). A snapshot directory is what the shard
// writer leaves: `snapshot.json` and `pub/v1/shards/<sha256>`. The plan names
// what is released (the manifest's JCS bytes, as the Release asset
// `teams-<snapshot_hash>.json`) and what is uploaded (each shard under its
// key), each with its SHA-256, so the job that attests and uploads can check
// that it holds exactly the bytes the read-only plan job built.
//
// Everything here is pure over a directory; no network, no credential.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { jcsBytes, readTeamSnapshotManifest, snapshotHash, type TeamSnapshotManifest } from "@areneon/showdown-bridge";

export class PublishError extends Error {
  override name = "PublishError";
}

export const PLAN_FILE = "plan.json";
export const RELEASE_DIR = "release";
export const SNAPSHOT_FILE = "snapshot.json";

const HEX64 = /^[0-9a-f]{64}$/;
const SHARD_KEY = /^pub\/v1\/shards\/[0-9a-f]{64}$/;

export const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
export const teamsTag = (hash: string): string => `teams-${hash}`;

export interface PlannedShard {
  object_key: string;
  sha256: string;
  bytes: number;
}

export interface PublishPlan {
  schema_version: 1;
  /** SHA-256 of the manifest's JCS bytes: names the Release and the manifest object. */
  snapshot_hash: string;
  logical_snapshot_hash: string;
  /** The Release tag, `teams-<snapshot_hash>`. */
  tag: string;
  /** SHA-256 of the Release asset's bytes (equal to `snapshot_hash`). */
  manifest_sha256: string;
  shards: PlannedShard[];
}

/** The manifest of a snapshot directory, through the bridge's reader. */
export function readSnapshotDir(dir: string): TeamSnapshotManifest {
  let text: string;
  try {
    text = readFileSync(join(dir, SNAPSHOT_FILE), "utf8");
  } catch {
    throw new PublishError(`${dir} has no ${SNAPSHOT_FILE}`);
  }
  try {
    return readTeamSnapshotManifest(JSON.parse(text));
  } catch (error) {
    throw new PublishError(`${dir}/${SNAPSHOT_FILE} is not a valid snapshot manifest: ${(error as Error).message}`);
  }
}

/** The Release asset: the manifest's canonical bytes, whose SHA-256 is the snapshot hash. */
export const manifestBytes = (manifest: TeamSnapshotManifest): Buffer => jcsBytes(manifest);

/** Read a shard file of the directory by its manifest key. */
export function shardFile(dir: string, objectKey: string): Buffer {
  if (!SHARD_KEY.test(objectKey)) throw new PublishError("a shard key is not pub/v1/shards/<sha256>");
  const path = join(dir, objectKey);
  if (!existsSync(path)) throw new PublishError(`${objectKey} is missing from ${dir}`);
  return readFileSync(path);
}

/**
 * Plan the publication of a snapshot directory. Every shard file must exist
 * and match the manifest's SHA-256, size and key (a file's key IS its hash);
 * a directory that does not is refused, never planned.
 */
export function buildPlan(dir: string): PublishPlan {
  const manifest = readSnapshotDir(dir);
  const hash = snapshotHash(manifest);
  const shards = manifest.shards.map((entry): PlannedShard => {
    const file = shardFile(dir, entry.object_key);
    const digest = sha256(file);
    if (digest !== entry.file_sha256 || entry.object_key !== `pub/v1/shards/${digest}`) {
      throw new PublishError(`${entry.object_key} does not match its SHA-256 in the manifest`);
    }
    if (file.byteLength !== entry.bytes) throw new PublishError(`${entry.object_key} does not have the size the manifest says`);
    return { object_key: entry.object_key, sha256: digest, bytes: file.byteLength };
  });
  const manifestSha = sha256(manifestBytes(manifest));
  if (manifestSha !== hash) throw new PublishError("the snapshot hash is not the SHA-256 of the manifest's canonical bytes");
  return {
    schema_version: 1,
    snapshot_hash: hash,
    logical_snapshot_hash: manifest.logical_snapshot_hash,
    tag: teamsTag(hash),
    manifest_sha256: manifestSha,
    shards,
  };
}

const isObject = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
const hasKeys = (value: Record<string, unknown>, keys: string[]): boolean => Object.keys(value).sort().join() === [...keys].sort().join();

/** Parse a plan, failing closed on any shape `buildPlan` does not write. */
export function parsePlan(text: string): PublishPlan {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new PublishError(`${PLAN_FILE} is not JSON`);
  }
  if (
    !isObject(value) ||
    !hasKeys(value, ["schema_version", "snapshot_hash", "logical_snapshot_hash", "tag", "manifest_sha256", "shards"]) ||
    value.schema_version !== 1 ||
    typeof value.snapshot_hash !== "string" ||
    !HEX64.test(value.snapshot_hash) ||
    typeof value.logical_snapshot_hash !== "string" ||
    !HEX64.test(value.logical_snapshot_hash) ||
    value.tag !== teamsTag(value.snapshot_hash) ||
    value.manifest_sha256 !== value.snapshot_hash ||
    !Array.isArray(value.shards)
  ) {
    throw new PublishError(`${PLAN_FILE} is not a publish plan`);
  }
  const seen = new Set<string>();
  const shards = value.shards.map((entry: unknown, i): PlannedShard => {
    if (
      !isObject(entry) ||
      !hasKeys(entry, ["object_key", "sha256", "bytes"]) ||
      typeof entry.object_key !== "string" ||
      !SHARD_KEY.test(entry.object_key) ||
      typeof entry.sha256 !== "string" ||
      entry.object_key !== `pub/v1/shards/${entry.sha256}` ||
      !Number.isSafeInteger(entry.bytes) ||
      (entry.bytes as number) < 1 ||
      seen.has(entry.object_key)
    ) {
      throw new PublishError(`${PLAN_FILE}: shards[${i}] is not a planned shard`);
    }
    seen.add(entry.object_key);
    return { object_key: entry.object_key, sha256: entry.sha256, bytes: entry.bytes as number };
  });
  return {
    schema_version: 1,
    snapshot_hash: value.snapshot_hash,
    logical_snapshot_hash: value.logical_snapshot_hash,
    tag: value.tag as string,
    manifest_sha256: value.manifest_sha256 as string,
    shards,
  };
}

export const readPlan = (dir: string): PublishPlan => {
  let text: string;
  try {
    text = readFileSync(join(dir, PLAN_FILE), "utf8");
  } catch {
    throw new PublishError(`${dir} has no ${PLAN_FILE}: run plan first`);
  }
  return parsePlan(text);
};

/**
 * The upload job's check before it attests or uploads anything: the plan in
 * the directory must equal the plan made again from the files now. Returns it.
 */
export function verifyDir(dir: string): PublishPlan {
  const recorded = readPlan(dir);
  const fresh = buildPlan(dir);
  if (JSON.stringify(recorded) !== JSON.stringify(fresh)) {
    throw new PublishError("the plan in the directory differs from the plan made from its files now");
  }
  return fresh;
}

/** Write the plan and the Release asset (`release/<tag>.json`, the manifest's canonical bytes). */
export function writePlan(dir: string): PublishPlan {
  const plan = buildPlan(dir);
  mkdirSync(join(dir, RELEASE_DIR), { recursive: true });
  writeFileSync(join(dir, RELEASE_DIR, `${plan.tag}.json`), manifestBytes(readSnapshotDir(dir)));
  writeFileSync(join(dir, PLAN_FILE), `${JSON.stringify(plan, null, 2)}\n`);
  return plan;
}

export const releaseFile = (dir: string, plan: Pick<PublishPlan, "tag">): string => join(dir, RELEASE_DIR, `${plan.tag}.json`);
