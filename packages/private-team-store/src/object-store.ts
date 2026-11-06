// The object store layout of the private team store (ADR-015 section 1). The
// port, the errors, the R2 client and the in-memory store live in the shared
// `@areneon/object-store` package; this module declares what THIS store admits
// and hands it to them as an injected layout:
//   - `putIfAbsent` is the only way to write under `snapshots/`;
//   - `overwriteCurrent` is the only overwrite, and it can only reach
//     `current.json` (the key is not an argument);
//   - keys are checked against the layout (`snapshotKeys`): anything else,
//     `snapshots/_lock-check/` included, can be neither written nor fetched
//     through the port. A lister may still see such keys in a prefix listing
//     (the owner puts a lock-check object there); `listSnapshotHashes` ignores
//     them, and so does every other consumer.

import { checkKey, checkPrefix, type ObjectLayout, ObjectStoreError } from "@areneon/object-store";

export {
  MAX_OBJECT_BYTES,
  OBJECT_STORE_REFUSALS,
  type ObjectHead,
  type ObjectStore,
  ObjectStoreError,
  type ObjectStoreRefusal,
} from "@areneon/object-store";

export const CURRENT_KEY = "current.json";
export const SNAPSHOTS_PREFIX = "snapshots/";
/** The prefix the runbook's lock check writes under; never a snapshot. */
export const LOCK_CHECK_PREFIX = "snapshots/_lock-check/";

const HASH = /^[0-9a-f]{64}$/;
const SNAPSHOT_OBJECT = /^snapshots\/([0-9a-f]{64})\/(manifest\.json|shards\/(?:0|[1-9][0-9]{0,8})\.parquet)$/;

function keyOf(hash: string, rest: string): string {
  if (!HASH.test(hash)) throw new ObjectStoreError("key_invalid");
  return checkObjectKey(`${SNAPSHOTS_PREFIX}${hash}/${rest}`);
}

export const snapshotKeys = {
  manifest: (hash: string): string => keyOf(hash, "manifest.json"),
  shard: (hash: string, index: number): string => keyOf(hash, `shards/${index}.parquet`),
};

/** A key the port accepts: `current.json` or an object of the snapshot layout. */
export function checkObjectKey(key: string): string {
  return checkKey(PRIVATE_LAYOUT, key);
}

/** A listing prefix the port accepts: `snapshots/` or `snapshots/<hash>/`. */
export function checkListPrefix(prefix: string): string {
  return checkPrefix(PRIVATE_LAYOUT, prefix);
}

/** The snapshot hash a manifest key names, or null for any other key (unknown keys, `_lock-check/`, shards). */
export function manifestKeyHash(key: string): string | null {
  const match = SNAPSHOT_OBJECT.exec(key);
  return match !== null && match[2] === "manifest.json" ? (match[1] as string) : null;
}

/** What the private store admits, injected into the shared client and the in-memory store. */
export const PRIVATE_LAYOUT: ObjectLayout = {
  allowsKey: (key) => key === CURRENT_KEY || SNAPSHOT_OBJECT.test(key),
  allowsListPrefix: (prefix) => prefix === SNAPSHOTS_PREFIX || /^snapshots\/[0-9a-f]{64}\/$/.test(prefix),
  overwriteKey: CURRENT_KEY,
};
