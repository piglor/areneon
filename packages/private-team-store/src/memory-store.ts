// The in-memory object store tests use behind the port. It honours the same
// rules as the R2 client: `If-None-Match: *` semantics (an existing key is a
// refusal and the stored bytes stay), keys checked against the layout, one
// overwrite only (`current.json`), listing by prefix. `seed` puts an object
// where the port would refuse to (a key outside the layout, as the owner's
// lock-check object is), to test that readers ignore it.

import {
  checkListPrefix,
  checkObjectKey,
  CURRENT_KEY,
  MAX_OBJECT_BYTES,
  type ObjectHead,
  type ObjectStore,
  ObjectStoreError,
} from "./object-store.ts";

export class InMemoryObjectStore implements ObjectStore {
  readonly #objects = new Map<string, Uint8Array>();
  /** The keys written through `putIfAbsent` and `overwriteCurrent`, in order (a test reads it). */
  readonly writes: string[] = [];

  putIfAbsent(key: string, body: Uint8Array): Promise<void> {
    try {
      checkObjectKey(key);
      if (key === CURRENT_KEY) throw new ObjectStoreError("key_invalid");
      if (body.byteLength > MAX_OBJECT_BYTES) throw new ObjectStoreError("too_large");
      if (this.#objects.has(key)) throw new ObjectStoreError("exists");
      this.#objects.set(key, body.slice());
      this.writes.push(key);
      return Promise.resolve();
    } catch (error) {
      return Promise.reject(error);
    }
  }

  get(key: string): Promise<Uint8Array | null> {
    try {
      checkObjectKey(key);
      return Promise.resolve(this.#objects.get(key)?.slice() ?? null);
    } catch (error) {
      return Promise.reject(error);
    }
  }

  head(key: string): Promise<ObjectHead | null> {
    try {
      checkObjectKey(key);
      const found = this.#objects.get(key);
      return Promise.resolve(found === undefined ? null : { bytes: found.byteLength });
    } catch (error) {
      return Promise.reject(error);
    }
  }

  list(prefix: string): Promise<string[]> {
    try {
      checkListPrefix(prefix);
      return Promise.resolve([...this.#objects.keys()].filter((k) => k.startsWith(prefix)).sort());
    } catch (error) {
      return Promise.reject(error);
    }
  }

  overwriteCurrent(body: Uint8Array): Promise<void> {
    this.#objects.set(CURRENT_KEY, body.slice());
    this.writes.push(CURRENT_KEY);
    return Promise.resolve();
  }

  /** Test hook: store an object under any key, as another client (the owner's) would. */
  seed(key: string, body: Uint8Array): void {
    this.#objects.set(key, body.slice());
  }

  /** Test hook: every stored key. */
  keys(): string[] {
    return [...this.#objects.keys()].sort();
  }
}
