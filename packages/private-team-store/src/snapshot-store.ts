// Reading a stored snapshot through the object store port (ADR-015 sections 1
// and 4). Three rules, shared by the import and every reader (#888, #890):
//   - a snapshot is a directory `snapshots/<64 hex>/` with a `manifest.json`
//     that hashes to that name and whose shards verify (`fetchSnapshot` runs
//     `readPrivateSnapshot` on the downloaded bytes); a key that is not part
//     of that layout is nothing: the runbook's `snapshots/_lock-check/`
//     object, a stray file or a half-uploaded directory is ignored, never
//     read as a snapshot (`listSnapshotHashes` returns hashes that have a
//     manifest object and no more; the verification is `fetchSnapshot`);
//   - `current.json` is read only as a `CurrentPointer`;
//   - nothing here writes.

import type { Engine, TeamLabelSet } from "@areneon/showdown-bridge";
import { readCurrentPointer, type CurrentPointer } from "./current-pointer.ts";
import { CURRENT_KEY, manifestKeyHash, type ObjectStore, snapshotKeys, SNAPSHOTS_PREFIX } from "./object-store.ts";
import { type ExpectedVersion, INSPECT_ONLY, readPrivateSnapshot, type SnapshotRead } from "./snapshot-reader.ts";
import { SnapshotIoError } from "./snapshot-io-error.ts";

/** The pointer in `current.json`, or null when the bucket has none (before the first import). */
export async function readCurrent(store: ObjectStore): Promise<CurrentPointer | null> {
  const bytes = await store.get(CURRENT_KEY);
  return bytes === null ? null : readCurrentPointer(bytes);
}

/** The snapshot hashes that have a manifest object under `snapshots/`, sorted. Everything else under the prefix is ignored. */
export async function listSnapshotHashes(store: ObjectStore): Promise<string[]> {
  const hashes = new Set<string>();
  for (const key of await store.list(SNAPSHOTS_PREFIX)) {
    const hash = manifestKeyHash(key);
    if (hash !== null) hashes.add(hash);
  }
  return [...hashes].sort();
}

export interface FetchSnapshotInput {
  readonly store: ObjectStore;
  readonly engine: Engine;
  readonly labelSet: TeamLabelSet;
  /** The snapshot hash to fetch (a bucket directory name, or the hash a run pinned). */
  readonly snapshotHash: string;
  readonly expected: ExpectedVersion | typeof INSPECT_ONLY;
  readonly shardRows?: number;
  readonly decodeTimeoutMs?: number;
}

export interface FetchedSnapshot extends SnapshotRead {
  /** The bytes as stored, for the local copy. */
  readonly manifestBytes: Uint8Array;
  readonly shardBytes: readonly Uint8Array[];
}

/** Download a snapshot and verify it completely; throws `SnapshotIoError` or `ObjectStoreError` and nothing else. */
export async function fetchSnapshot(input: FetchSnapshotInput): Promise<FetchedSnapshot> {
  const manifestBytes = await input.store.get(snapshotKeys.manifest(input.snapshotHash));
  if (manifestBytes === null) throw new SnapshotIoError("manifest_missing");
  let manifest: unknown;
  try {
    manifest = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(manifestBytes));
  } catch {
    throw new SnapshotIoError("manifest_invalid");
  }
  const fetched = new Map<number, Uint8Array>();
  const read = await readPrivateSnapshot({
    engine: input.engine,
    manifest,
    expectedSnapshotHash: input.snapshotHash,
    expected: input.expected,
    labelSet: input.labelSet,
    ...(input.shardRows === undefined ? {} : { shardRows: input.shardRows }),
    ...(input.decodeTimeoutMs === undefined ? {} : { decodeTimeoutMs: input.decodeTimeoutMs }),
    shard: async (index) => {
      const bytes = await input.store.get(snapshotKeys.shard(input.snapshotHash, index));
      if (bytes === null) throw new SnapshotIoError("shard_missing", { shard: index });
      fetched.set(index, bytes);
      return bytes;
    },
  });
  const shardBytes = read.manifest.shards.map((entry) => fetched.get(entry.index) as Uint8Array);
  return { ...read, manifestBytes, shardBytes };
}
