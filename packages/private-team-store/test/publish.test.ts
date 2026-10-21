import { chmodSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { canonicalJson, privateSnapshotHash } from "@areneon/showdown-bridge";
import { describe, expect, it, vi } from "vitest";
import { readCurrentPointer } from "../src/current-pointer.ts";
import { InMemoryObjectStore } from "../src/memory-store.ts";
import { CURRENT_KEY, type ObjectStore, ObjectStoreError, snapshotKeys } from "../src/object-store.ts";
import { publishSnapshot } from "../src/publish.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { INSPECT_ONLY } from "../src/snapshot-reader.ts";
import { fetchSnapshot, listSnapshotHashes, readCurrent } from "../src/snapshot-store.ts";
import { writePrivateSnapshot, type WrittenSnapshot } from "../src/snapshot-writer.ts";
import { engine, format, input, labelSet, records, workDir } from "./snapshot-support.ts";

vi.setConfig({ testTimeout: 60_000 });

const expected = { manifest: format.manifest, vocabularyHash: "ee".repeat(32) };
const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

const made = (over: Parameters<typeof input>[0] extends infer T ? Partial<T> : never = {}): Promise<WrittenSnapshot> =>
  writePrivateSnapshot(input({ records: records(8), shardRows: 4, ...over }));
const snapshotsDir = (): string => join(dirname(workDir()), "snapshots");

const publish = (store: ObjectStore, written: WrittenSnapshot, dir: string, expectedCurrent: string | null = null, over: Record<string, unknown> = {}) =>
  publishSnapshot({ store, engine, labelSet, written, expected, expectedCurrent, snapshotsDir: dir, shardRows: 4, decodeTimeoutMs: 20_000, ...over });

const failure = async (run: () => Promise<unknown>): Promise<SnapshotIoError | ObjectStoreError> => {
  try {
    await run();
  } catch (error) {
    expect(error instanceof SnapshotIoError || error instanceof ObjectStoreError, String(error)).toBe(true);
    return error as SnapshotIoError | ObjectStoreError;
  }
  throw new Error("expected a refusal");
};

/** A store that lets a test change what a read returns, or fail a write. */
class Tampered implements ObjectStore {
  getHook: ((key: string, real: Uint8Array | null, calls: number) => Uint8Array | null) | undefined;
  failPutAt: number | undefined;
  puts = 0;
  readonly gets = new Map<string, number>();
  readonly inner: InMemoryObjectStore;
  constructor(inner: InMemoryObjectStore) {
    this.inner = inner;
  }
  putIfAbsent(key: string, body: Uint8Array): Promise<void> {
    this.puts += 1;
    if (this.failPutAt === this.puts) return Promise.reject(new ObjectStoreError("unavailable"));
    return this.inner.putIfAbsent(key, body);
  }
  async get(key: string): Promise<Uint8Array | null> {
    const calls = (this.gets.get(key) ?? 0) + 1;
    this.gets.set(key, calls);
    const real = await this.inner.get(key);
    return this.getHook === undefined ? real : this.getHook(key, real, calls);
  }
  head(key: string) {
    return this.inner.head(key);
  }
  list(prefix: string) {
    return this.inner.list(prefix);
  }
  overwriteCurrent(body: Uint8Array): Promise<void> {
    return this.inner.overwriteCurrent(body);
  }
}

describe("publishSnapshot", () => {
  it("uploads shards then the manifest, verifies the download, copies locally, and moves current.json last", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    const dir = snapshotsDir();
    const done = await publish(store, written, dir);

    const hash = written.snapshotHash;
    expect(done).toEqual({ snapshotHash: hash, localDir: join(dir, hash), previous: null });
    expect(store.writes).toEqual([...written.shards.keys()].map((k) => snapshotKeys.shard(hash, k)).concat(snapshotKeys.manifest(hash), CURRENT_KEY));
    expect(store.writes.at(-1)).toBe(CURRENT_KEY);
    expect(readCurrentPointer((await store.get(CURRENT_KEY)) as Uint8Array)).toEqual({ schema_version: 1, snapshot_hash: hash, parent_snapshot_hash: null });
    expect(await readCurrent(store)).toEqual({ schema_version: 1, snapshot_hash: hash, parent_snapshot_hash: null });
    expect(await listSnapshotHashes(store)).toEqual([hash]);

    // The local copy: the stored bytes, directories 0700, files 0600.
    expect(readFileSync(join(done.localDir, "manifest.json"), "utf8")).toBe(canonicalJson(written.manifest as never));
    written.shards.forEach((shard, k) => expect(Buffer.compare(readFileSync(join(done.localDir, "shards", `${k}.parquet`)), shard)).toBe(0));
    for (const d of [dir, done.localDir, join(done.localDir, "shards")]) expect(statSync(d).mode & 0o777).toBe(0o700);
    expect(statSync(join(done.localDir, "manifest.json")).mode & 0o777).toBe(0o600);
    expect(readdirSync(dir)).toEqual([hash]);

    const fetched = await fetchSnapshot({ store, engine, labelSet, snapshotHash: hash, expected, shardRows: 4, decodeTimeoutMs: 20_000 });
    expect(fetched.records).toHaveLength(8);
    expect(privateSnapshotHash(fetched.manifest)).toBe(hash);
  });

  it("publishes a snapshot laid out with the default shard size when no size is given", async () => {
    const store = new InMemoryObjectStore();
    const written = await writePrivateSnapshot(input({ records: records(5) }));
    const done = await publishSnapshot({ store, engine, labelSet, written, expected, expectedCurrent: null, snapshotsDir: snapshotsDir() });
    expect(written.shards).toHaveLength(1);
    expect(done.snapshotHash).toBe(written.snapshotHash);
  });

  it("chains a second snapshot on the first and names the parent in current.json", async () => {
    const store = new InMemoryObjectStore();
    const dir = snapshotsDir();
    const first = await made();
    await publish(store, first, dir);
    const second = await made({ parentSnapshotHash: first.snapshotHash, snapshotId: "owner-test-2" });
    const done = await publish(store, second, dir, first.snapshotHash);
    expect(done.previous).toBe(first.snapshotHash);
    expect(await readCurrent(store)).toEqual({ schema_version: 1, snapshot_hash: second.snapshotHash, parent_snapshot_hash: first.snapshotHash });
    expect(await listSnapshotHashes(store)).toEqual([first.snapshotHash, second.snapshotHash].sort());
    expect(readdirSync(dir).sort()).toEqual([first.snapshotHash, second.snapshotHash].sort());
  });

  it("refuses a parent that is not the expected current snapshot, before anything is uploaded", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    expect((await failure(() => publish(store, written, snapshotsDir(), "cc".repeat(32)))).message).toMatch(/parent/);
    const child = await made({ parentSnapshotHash: "dd".repeat(32), snapshotId: "owner-test-3" });
    expect(((await failure(() => publish(store, child, snapshotsDir(), null))) as SnapshotIoError).code).toBe("parent_mismatch");
    expect(store.writes).toEqual([]);
  });

  it("refuses when current.json is not what the import was planned against", async () => {
    const written = await made();
    const other = new InMemoryObjectStore();
    other.seed(CURRENT_KEY, bytes(canonicalJson({ parent_snapshot_hash: null, schema_version: 1, snapshot_hash: "ee".repeat(32) })));
    expect(((await failure(() => publish(other, written, snapshotsDir()))) as SnapshotIoError).code).toBe("current_changed");
    expect(other.writes).toEqual([]);
    const bad = new InMemoryObjectStore();
    bad.seed(CURRENT_KEY, bytes("{}"));
    expect(((await failure(() => publish(bad, written, snapshotsDir()))) as SnapshotIoError).code).toBe("current_invalid");
    const child = await made({ parentSnapshotHash: "ee".repeat(32), snapshotId: "owner-test-4" });
    expect(((await failure(() => publish(new InMemoryObjectStore(), child, snapshotsDir(), "ee".repeat(32)))) as SnapshotIoError).code).toBe("current_changed");
  });

  it("never overwrites: an object that exists is a refusal, current.json stays and nothing else is written", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    const key = snapshotKeys.shard(written.snapshotHash, 1);
    store.seed(key, bytes("someone else's bytes"));
    const error = await failure(() => publish(store, written, snapshotsDir()));
    expect((error as ObjectStoreError).code).toBe("exists");
    expect(Buffer.from((await store.get(key)) as Uint8Array).toString()).toBe("someone else's bytes");
    expect(store.writes).toEqual([snapshotKeys.shard(written.snapshotHash, 0)]);
    expect(await store.get(CURRENT_KEY)).toBeNull();
  });

  it("leaves current.json and the local copy alone when an upload fails midway", async () => {
    const tampered = new Tampered(new InMemoryObjectStore());
    tampered.failPutAt = 2;
    const written = await made();
    const dir = snapshotsDir();
    expect(((await failure(() => publish(tampered, written, dir))) as ObjectStoreError).code).toBe("unavailable");
    expect(await tampered.inner.get(CURRENT_KEY)).toBeNull();
    expect(existsSync(join(dir, written.snapshotHash))).toBe(false);
  });

  it("verifies the download: a shard that differs from the manifest's digest is refused and nothing is copied or moved", async () => {
    const tampered = new Tampered(new InMemoryObjectStore());
    tampered.getHook = (key, real) => (key.endsWith("/shards/0.parquet") && real !== null ? real.map((b, k) => (k === 10 ? b ^ 1 : b)) : real);
    const written = await made();
    const dir = snapshotsDir();
    expect(((await failure(() => publish(tampered, written, dir))) as SnapshotIoError).code).toBe("shard_digest_mismatch");
    expect(await tampered.inner.get(CURRENT_KEY)).toBeNull();
    expect(existsSync(join(dir, written.snapshotHash))).toBe(false);
  });

  it("compares the manifest's bytes with what it sent", async () => {
    const tampered = new Tampered(new InMemoryObjectStore());
    tampered.getHook = (key, real) => (key.endsWith("/manifest.json") && real !== null ? bytes(`${new TextDecoder().decode(real)}\n`) : real);
    const written = await made();
    expect(((await failure(() => publish(tampered, written, snapshotsDir()))) as SnapshotIoError).code).toBe("download_mismatch");
    expect(await tampered.inner.get(CURRENT_KEY)).toBeNull();
  });

  it("checks current.json again just before moving it", async () => {
    const tampered = new Tampered(new InMemoryObjectStore());
    const written = await made();
    tampered.getHook = (key, real, calls) => (key === CURRENT_KEY && calls === 2 ? bytes(canonicalJson({ parent_snapshot_hash: null, schema_version: 1, snapshot_hash: "ff".repeat(32) })) : real);
    expect(((await failure(() => publish(tampered, written, snapshotsDir()))) as SnapshotIoError).code).toBe("current_changed");
    expect(tampered.inner.writes.includes(CURRENT_KEY)).toBe(false);
  });

  it("refuses a local copy that exists, or a directory that is not private, before uploading", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    const dir = snapshotsDir();
    mkdirSync(join(dir, written.snapshotHash), { recursive: true, mode: 0o700 });
    expect(((await failure(() => publish(store, written, dir))) as SnapshotIoError).code).toBe("local_copy_exists");
    const open = snapshotsDir();
    mkdirSync(open, { mode: 0o755 });
    chmodSync(open, 0o755);
    expect(((await failure(() => publish(store, written, open))) as SnapshotIoError).code).toBe("work_dir_unsafe");
    expect(((await failure(() => publish(store, written, join(dirname(workDir()), "missing", "snapshots")))) as SnapshotIoError).code).toBe("work_dir_unsafe");
    expect(store.writes).toEqual([]);
  });

  it("does not take INSPECT_ONLY: a publish compares against the pin", async () => {
    const written = await made();
    const error = await failure(() => publish(new InMemoryObjectStore(), written, snapshotsDir(), null, { expected: INSPECT_ONLY }));
    expect((error as SnapshotIoError).code).toBe("version_manifest_mismatch");
    const wrong = await failure(() => publish(new InMemoryObjectStore(), written, snapshotsDir(), null, { expected: { ...expected, vocabularyHash: "00".repeat(32) } }));
    expect((wrong as SnapshotIoError).code).toBe("version_manifest_mismatch");
  });
});

describe("readers and listers", () => {
  it("ignore snapshots/_lock-check/ and every unknown key, and list only directories with a manifest", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    await publish(store, written, snapshotsDir());
    store.seed("snapshots/_lock-check/2026-10-21.txt", bytes("lock check"));
    store.seed("snapshots/_lock-check/manifest.json", bytes("{}"));
    store.seed("snapshots/stray.txt", bytes("x"));
    store.seed("snapshots/manifest.json", bytes("{}"));
    store.seed(`snapshots/${"ab".repeat(32)}/shards/0.parquet`, bytes("a half-uploaded directory"));
    store.seed(`snapshots/${"AB".repeat(32)}/manifest.json`, bytes("{}"));
    store.seed(`snapshots/${written.snapshotHash}/manifest.json.bak`, bytes("{}"));
    expect(await listSnapshotHashes(store)).toEqual([written.snapshotHash]);
  });

  it("do not treat a manifest under a made-up hash as a snapshot: the fetch verifies it", async () => {
    const store = new InMemoryObjectStore();
    const written = await made();
    await publish(store, written, snapshotsDir());
    const fake = "ab".repeat(32);
    store.seed(snapshotKeys.manifest(fake), (await store.get(snapshotKeys.manifest(written.snapshotHash))) as Uint8Array);
    expect(await listSnapshotHashes(store)).toEqual([fake, written.snapshotHash].sort());
    const code = async (hash: string): Promise<string> => ((await failure(() => fetchSnapshot({ store, engine, labelSet, snapshotHash: hash, expected, shardRows: 4 }))) as SnapshotIoError).code;
    expect(await code(fake)).toBe("snapshot_hash_mismatch");
    expect(await code("cd".repeat(32))).toBe("manifest_missing");
    store.seed(snapshotKeys.manifest("ef".repeat(32)), bytes("not json {"));
    expect(await code("ef".repeat(32))).toBe("manifest_invalid");
    store.seed(snapshotKeys.manifest("12".repeat(32)), bytes("{}"));
    expect(await code("12".repeat(32))).toBe("manifest_invalid");
    expect(((await failure(() => fetchSnapshot({ store, engine, labelSet, snapshotHash: "../_lock-check", expected }))) as ObjectStoreError).code).toBe("key_invalid");
  });

  it("refuse a snapshot with a shard missing from the bucket", async () => {
    const store = new InMemoryObjectStore();
    const tampered = new Tampered(store);
    const written = await made();
    await publish(store, written, snapshotsDir());
    tampered.getHook = (key, real) => (key.endsWith("/shards/1.parquet") ? null : real);
    const error = (await failure(() => fetchSnapshot({ store: tampered, engine, labelSet, snapshotHash: written.snapshotHash, expected, shardRows: 4 }))) as SnapshotIoError;
    expect(error.code).toBe("shard_missing");
    expect(error.position.shard).toBe(1);
  });
});
