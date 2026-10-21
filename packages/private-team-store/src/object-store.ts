// The object store port of the private team store (ADR-015 section 1). The
// import and the readers talk to the bucket only through this interface; the
// R2 client (r2-client.ts) is the one implementation that touches the network
// and the in-memory store (memory-store.ts) is the one tests use.
//
// The port is deliberately narrow:
//   - `putIfAbsent` is the only way to write under `snapshots/`: it is
//     `If-None-Match: *`, and an existing object is a refusal ("exists"),
//     never an overwrite;
//   - `overwriteCurrent` is the only overwrite, and it can only reach
//     `current.json` (the key is not an argument);
//   - there is no delete;
//   - keys are checked against the layout (`snapshotKeys`): anything else,
//     `snapshots/_lock-check/` included, can be neither written nor fetched
//     through the port. A lister may still see such keys in a prefix listing
//     (the owner puts a lock-check object there); `listSnapshotHashes` ignores
//     them, and so does every other consumer.
//
// An `ObjectStoreError` carries a code and a fixed message. It never holds the
// response body, the URL, a header, the bucket, a credential or a key's
// contents, and no cause: whatever a lower layer said is dropped.

export const CURRENT_KEY = "current.json";
export const SNAPSHOTS_PREFIX = "snapshots/";
/** The prefix the runbook's lock check writes under; never a snapshot. */
export const LOCK_CHECK_PREFIX = "snapshots/_lock-check/";

const HASH = /^[0-9a-f]{64}$/;
const SNAPSHOT_OBJECT = /^snapshots\/([0-9a-f]{64})\/(manifest\.json|shards\/(?:0|[1-9][0-9]{0,8})\.parquet)$/;

export const OBJECT_STORE_REFUSALS = [
  "key_invalid",
  "exists",
  "too_large",
  "denied",
  "unavailable",
  "failed",
  "response_invalid",
  "config_invalid",
] as const;
export type ObjectStoreRefusal = (typeof OBJECT_STORE_REFUSALS)[number];

const MESSAGES: Readonly<Record<ObjectStoreRefusal, string>> = {
  key_invalid: "the key is not part of the private store's layout",
  exists: "an object is already stored under this key; nothing was overwritten",
  too_large: "the object is larger than the store accepts",
  denied: "the store refused the credentials or their permission",
  unavailable: "the store could not be reached or is busy",
  failed: "the store answered with an error",
  response_invalid: "the store's answer was not in the expected form",
  config_invalid: "the store's settings are not valid",
};

export class ObjectStoreError extends Error {
  override name = "ObjectStoreError";
  readonly code: ObjectStoreRefusal;

  constructor(code: ObjectStoreRefusal) {
    super(`object store refused: ${MESSAGES[code]}`);
    this.code = code;
  }
}

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
  if (key !== CURRENT_KEY && !SNAPSHOT_OBJECT.test(key)) throw new ObjectStoreError("key_invalid");
  return key;
}

/** A listing prefix the port accepts: `snapshots/` or `snapshots/<hash>/`. */
export function checkListPrefix(prefix: string): string {
  if (prefix !== SNAPSHOTS_PREFIX && !/^snapshots\/[0-9a-f]{64}\/$/.test(prefix)) throw new ObjectStoreError("key_invalid");
  return prefix;
}

/** The snapshot hash a manifest key names, or null for any other key (unknown keys, `_lock-check/`, shards). */
export function manifestKeyHash(key: string): string | null {
  const match = SNAPSHOT_OBJECT.exec(key);
  return match !== null && match[2] === "manifest.json" ? (match[1] as string) : null;
}

export interface ObjectHead {
  readonly bytes: number;
}

export interface ObjectStore {
  /** Write `body` at `key` only if nothing is there (`If-None-Match: *`); an existing object is `exists`. */
  putIfAbsent(key: string, body: Uint8Array): Promise<void>;
  /** The object's bytes, or null if there is none. */
  get(key: string): Promise<Uint8Array | null>;
  /** The object's size, or null if there is none. */
  head(key: string): Promise<ObjectHead | null>;
  /** Every key under `prefix` (`snapshots/` or `snapshots/<hash>/`), sorted by code unit, unknown keys included. */
  list(prefix: string): Promise<string[]>;
  /** Replace `current.json`. The only overwrite the port has. */
  overwriteCurrent(body: Uint8Array): Promise<void>;
}

/** Upper bound of one object, for the client and the in-memory store alike. */
export const MAX_OBJECT_BYTES = 256 * 1024 * 1024;
