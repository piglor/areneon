// The object store port and the key-layout policy (ADR-015 section 1, ADR-024
// section 3.6). The port is narrow, and the client behind it admits no key its
// user did not declare:
//   - the user injects an `ObjectLayout`: a predicate over object keys, a
//     predicate over listing prefixes and at most one key that may be
//     overwritten; every other key is refused (`key_invalid`) BEFORE a request
//     is signed or sent;
//   - `putIfAbsent` is `If-None-Match: *`: an existing object is a refusal
//     ("exists"), never an overwrite, and it cannot reach the overwritable key;
//   - `overwriteCurrent` is the only overwrite, and the key is not an argument;
//   - there is no delete.
//
// An `ObjectStoreError` carries a code and a fixed message. It never holds the
// response body, the URL, a header, the bucket, a credential or a key's
// contents, and no cause: whatever a lower layer said is dropped.

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
  key_invalid: "the key is not part of this store's layout",
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

/** What a user of the client declares: the keys, the listing prefixes and the one overwritable key it admits. */
export interface ObjectLayout {
  /** True for a key the user may read or write. Anything else is refused before a request. */
  readonly allowsKey: (key: string) => boolean;
  /** True for a prefix the user may list. Anything else is refused before a request. */
  readonly allowsListPrefix: (prefix: string) => boolean;
  /** The one key `overwriteCurrent` replaces (it must be an allowed key), or null for a store with no overwrite. */
  readonly overwriteKey: string | null;
}

/** A layout must be usable: its overwritable key, if any, is one of its own keys. */
export function assertLayout(layout: ObjectLayout): ObjectLayout {
  if (layout.overwriteKey !== null && !layout.allowsKey(layout.overwriteKey)) throw new ObjectStoreError("config_invalid");
  return layout;
}

/** The key, if the layout admits it; else `key_invalid`. */
export function checkKey(layout: ObjectLayout, key: string): string {
  if (!layout.allowsKey(key)) throw new ObjectStoreError("key_invalid");
  return key;
}

/** The prefix, if the layout admits it for a listing; else `key_invalid`. */
export function checkPrefix(layout: ObjectLayout, prefix: string): string {
  if (!layout.allowsListPrefix(prefix)) throw new ObjectStoreError("key_invalid");
  return prefix;
}

/** A key a conditional write may create: allowed by the layout and not the overwritable one. */
export function checkCreateKey(layout: ObjectLayout, key: string): string {
  checkKey(layout, key);
  if (key === layout.overwriteKey) throw new ObjectStoreError("key_invalid");
  return key;
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
  /** Every key under `prefix` (a prefix the layout allows), sorted by code unit, unknown keys included. */
  list(prefix: string): Promise<string[]>;
  /** Replace the layout's overwritable key. The only overwrite the port has. */
  overwriteCurrent(body: Uint8Array): Promise<void>;
}

/** Upper bound of one object, for the client and the in-memory store alike. */
export const MAX_OBJECT_BYTES = 256 * 1024 * 1024;
