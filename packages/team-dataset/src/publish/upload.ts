// Upload a verified snapshot to the public bucket (ADR-013 section 6): every
// shard under its content-addressed key, then the manifest, each with
// `If-None-Match: *`. A key that already exists is fine only if it holds the
// very same bytes (a retry after a half-finished upload); other bytes under a
// key are a refusal, never an overwrite. Manifest last, so a reader that finds
// a manifest finds all its shards.

import { ObjectStoreError, type ObjectStore } from "@areneon/object-store";
import { snapshotKey } from "./layout.ts";
import { PublishError, type PublishPlan, readSnapshotDir, manifestBytes, sha256, shardFile } from "./plan.ts";

export interface UploadResult {
  uploaded: string[];
  /** Keys that were already there with identical bytes. */
  present: string[];
}

async function putSame(store: ObjectStore, key: string, body: Uint8Array, result: UploadResult): Promise<void> {
  try {
    await store.putIfAbsent(key, body);
    result.uploaded.push(key);
    return;
  } catch (error) {
    if (!(error instanceof ObjectStoreError) || error.code !== "exists") throw error;
  }
  const stored = await store.get(key);
  if (stored === null || sha256(stored) !== sha256(body)) throw new PublishError(`${key} already holds other bytes; nothing was overwritten`);
  result.present.push(key);
}

/** `plan` must already be verified against `dir` (`verifyDir`). */
export async function uploadSnapshot(store: ObjectStore, dir: string, plan: PublishPlan): Promise<UploadResult> {
  const result: UploadResult = { uploaded: [], present: [] };
  for (const shard of plan.shards) await putSame(store, shard.object_key, shardFile(dir, shard.object_key), result);
  await putSame(store, snapshotKey(plan.snapshot_hash), manifestBytes(readSnapshotDir(dir)), result);
  return result;
}
