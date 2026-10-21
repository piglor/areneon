import { describe, expect, it } from "vitest";
import { InMemoryObjectStore } from "../src/memory-store.ts";
import { checkListPrefix, checkObjectKey, CURRENT_KEY, MAX_OBJECT_BYTES, manifestKeyHash, type ObjectStore, ObjectStoreError, snapshotKeys } from "../src/object-store.ts";
import { R2ObjectStore } from "../src/r2-client.ts";
import { FakeR2, HASH_A, HASH_B, manifestKey, settings, shardKey } from "./r2-support.ts";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);
const text = (data: Uint8Array | null): string | null => (data === null ? null : new TextDecoder().decode(data));
const code = async (run: () => Promise<unknown>): Promise<string> => {
  try {
    await run();
  } catch (error) {
    expect(error).toBeInstanceOf(ObjectStoreError);
    return (error as ObjectStoreError).code;
  }
  return "no error";
};

// The same contract, run against the in-memory store and against the R2 client
// talking to the fake bucket: a test written for one holds for the other.
const stores: [string, () => ObjectStore][] = [
  ["the in-memory store", () => new InMemoryObjectStore()],
  ["the R2 client on a fake bucket", () => new R2ObjectStore(settings(), { fetch: new FakeR2().fetch })],
];

describe.each(stores)("the object store port: %s", (_name, make) => {
  it("writes an object once and refuses a second write without changing it", async () => {
    const store = make();
    await store.putIfAbsent(manifestKey(HASH_A), bytes("first"));
    expect(await code(() => store.putIfAbsent(manifestKey(HASH_A), bytes("second")))).toBe("exists");
    expect(text(await store.get(manifestKey(HASH_A)))).toBe("first");
  });

  it("answers get and head for a stored object and null for an absent one", async () => {
    const store = make();
    await store.putIfAbsent(shardKey(HASH_A, 0), bytes("12345"));
    expect(text(await store.get(shardKey(HASH_A, 0)))).toBe("12345");
    expect(await store.head(shardKey(HASH_A, 0))).toEqual({ bytes: 5 });
    expect(await store.get(shardKey(HASH_A, 1))).toBeNull();
    expect(await store.head(shardKey(HASH_A, 1))).toBeNull();
    expect(await store.get(CURRENT_KEY)).toBeNull();
  });

  it("lists the keys under a prefix, sorted", async () => {
    const store = make();
    await store.putIfAbsent(manifestKey(HASH_B), bytes("b"));
    await store.putIfAbsent(shardKey(HASH_A, 1), bytes("1"));
    await store.putIfAbsent(manifestKey(HASH_A), bytes("a"));
    expect(await store.list("snapshots/")).toEqual([manifestKey(HASH_A), shardKey(HASH_A, 1), manifestKey(HASH_B)]);
    expect(await store.list(`snapshots/${HASH_A}/`)).toEqual([manifestKey(HASH_A), shardKey(HASH_A, 1)]);
    expect(await store.list(`snapshots/${"c3".repeat(32)}/`)).toEqual([]);
  });

  it("overwrites current.json and nothing else", async () => {
    const store = make();
    await store.overwriteCurrent(bytes("one"));
    await store.overwriteCurrent(bytes("two"));
    expect(text(await store.get(CURRENT_KEY))).toBe("two");
    expect(await code(() => store.putIfAbsent(CURRENT_KEY, bytes("x")))).toBe("key_invalid");
  });

  it("refuses keys and prefixes outside the layout, the lock-check prefix included", async () => {
    const store = make();
    for (const key of ["", "snapshots/_lock-check/2026-10-21.txt", "snapshots/x/manifest.json", `snapshots/${HASH_A}/other.bin`, `snapshots/${HASH_A}/shards/01.parquet`, `snapshots/${HASH_A}/../manifest.json`, `/${CURRENT_KEY}`, "unknown.txt"]) {
      expect(await code(() => store.putIfAbsent(key, bytes("x"))), key).toBe("key_invalid");
      expect(await code(() => store.get(key)), key).toBe("key_invalid");
      expect(await code(() => store.head(key)), key).toBe("key_invalid");
    }
    for (const prefix of ["", "snapshots", "snapshots/_lock-check/", `snapshots/${HASH_A}`, "snapshots//"]) {
      expect(await code(() => store.list(prefix)), prefix).toBe("key_invalid");
    }
  });
});

describe("the layout helpers", () => {
  it("builds the manifest and shard keys from a snapshot hash only", () => {
    expect(snapshotKeys.manifest(HASH_A)).toBe(manifestKey(HASH_A));
    expect(snapshotKeys.shard(HASH_A, 12)).toBe(shardKey(HASH_A, 12));
    expect(() => snapshotKeys.manifest("../x")).toThrow(ObjectStoreError);
    expect(() => snapshotKeys.shard(HASH_A, -1)).toThrow(ObjectStoreError);
    expect(() => snapshotKeys.shard(HASH_A, 1.5)).toThrow(ObjectStoreError);
  });

  it("names a snapshot only by its manifest key", () => {
    expect(manifestKeyHash(manifestKey(HASH_A))).toBe(HASH_A);
    for (const key of [shardKey(HASH_A, 0), "snapshots/_lock-check/manifest.json", `snapshots/${HASH_A.toUpperCase()}/manifest.json`, `snapshots/${HASH_A}/manifest.json.bak`, CURRENT_KEY, ""]) {
      expect(manifestKeyHash(key), key).toBeNull();
    }
    expect(checkObjectKey(CURRENT_KEY)).toBe(CURRENT_KEY);
    expect(checkListPrefix("snapshots/")).toBe("snapshots/");
  });
});

describe("the in-memory store's own rules", () => {
  it("refuses an object over the size limit and a seeded key is invisible to get but listed", async () => {
    const store = new InMemoryObjectStore();
    expect(await code(() => store.putIfAbsent(manifestKey(HASH_A), { byteLength: MAX_OBJECT_BYTES + 1 } as Uint8Array))).toBe("too_large");
    store.seed("snapshots/_lock-check/2026-10-21.txt", bytes("lock check"));
    expect(await store.list("snapshots/")).toEqual(["snapshots/_lock-check/2026-10-21.txt"]);
    expect(await code(() => store.get("snapshots/_lock-check/2026-10-21.txt"))).toBe("key_invalid");
    expect(store.keys()).toEqual(["snapshots/_lock-check/2026-10-21.txt"]);
    await store.putIfAbsent(manifestKey(HASH_A), bytes("a"));
    expect(store.writes).toEqual([manifestKey(HASH_A)]);
  });
});
