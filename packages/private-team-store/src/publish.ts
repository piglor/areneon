// The publish sequence of an import (ADR-015 section 3, steps after "write
// the shards and build the manifest"), against the object store port:
//
//   1. the parent: the snapshot's parent must be the snapshot `current.json`
//      is expected to name (none for the first), and the bucket must say the
//      same now; the local copy's place must be free and safe;
//   2. upload every shard, then the manifest, each with `If-None-Match: *`: an
//      object that exists is a refusal and is never overwritten;
//   3. download every object again and verify it as a reader would
//      (`fetchSnapshot` = `readPrivateSnapshot` with the required anchors),
//      and compare the manifest's bytes with the ones uploaded (the reader
//      has already checked every shard against the manifest's digests);
//   4. write the verified local copy under `private-data/snapshots/<hash>/`;
//   5. only then, check `current.json` once more and overwrite it with the
//      new pointer. If anything before this fails, `current.json` is still the
//      old one and the snapshot objects are unreferenced, inert objects (they
//      cannot be removed under the lock; they do no harm).
//
// A retry of a failed publish with the very same bytes meets `exists` on the
// first object it uploaded before: the snapshot_id of the next attempt must
// differ (it is part of the manifest, so it makes another hash).
// The compare-and-set of step 5 is a read followed by a write, not one atomic
// request: a second writer between the two is not excluded. The import is one
// owner on one workstation holding the only write token (ADR-015 section 7).

import type { Engine, TeamLabelSet } from "@areneon/showdown-bridge";
import { canonicalJson } from "@areneon/showdown-bridge";
import { encodeCurrent } from "./current-pointer.ts";
import { checkLocalCopyTarget, writeLocalCopy } from "./local-copy.ts";
import { type ObjectStore, snapshotKeys } from "./object-store.ts";
import { type ExpectedVersion, INSPECT_ONLY } from "./snapshot-reader.ts";
import { SnapshotIoError } from "./snapshot-io-error.ts";
import { fetchSnapshot, readCurrent } from "./snapshot-store.ts";
import type { WrittenSnapshot } from "./snapshot-writer.ts";

export interface PublishInput {
  readonly store: ObjectStore;
  readonly engine: Engine;
  readonly labelSet: TeamLabelSet;
  readonly written: WrittenSnapshot;
  /** The version manifest and vocabulary hash the snapshot was made under; required (no inspect-only publish). */
  readonly expected: ExpectedVersion;
  /** The snapshot hash `current.json` must name now, or null when the bucket must have none (the first import). */
  readonly expectedCurrent: string | null;
  /** `private-data/snapshots`: the directory that holds the local copies (see `assertWorkDir`). */
  readonly snapshotsDir: string;
  readonly shardRows?: number;
  readonly decodeTimeoutMs?: number;
}

export interface Published {
  readonly snapshotHash: string;
  /** The local copy's directory. */
  readonly localDir: string;
  /** The snapshot `current.json` named before (null for the first import). */
  readonly previous: string | null;
}

async function checkCurrent(store: ObjectStore, expectedCurrent: string | null): Promise<void> {
  const pointer = await readCurrent(store);
  if ((pointer?.snapshot_hash ?? null) !== expectedCurrent) throw new SnapshotIoError("current_changed");
}

/** Upload, verify, copy locally, then move `current.json`. Throws `SnapshotIoError` or `ObjectStoreError`. */
export async function publishSnapshot(input: PublishInput): Promise<Published> {
  if ((input.expected as ExpectedVersion | typeof INSPECT_ONLY) === INSPECT_ONLY) throw new SnapshotIoError("version_manifest_mismatch");
  const { store, written } = input;
  const hash = written.snapshotHash;
  const parent = written.manifest.parent_snapshot_hash ?? null;
  if (parent !== input.expectedCurrent) throw new SnapshotIoError("parent_mismatch");
  await checkCurrent(store, input.expectedCurrent);
  checkLocalCopyTarget(input.snapshotsDir, hash);

  const manifestBytes = new TextEncoder().encode(canonicalJson(written.manifest as unknown as Parameters<typeof canonicalJson>[0]));
  for (const [index, bytes] of written.shards.entries()) await store.putIfAbsent(snapshotKeys.shard(hash, index), bytes);
  await store.putIfAbsent(snapshotKeys.manifest(hash), manifestBytes);

  const fetched = await fetchSnapshot({
    store,
    engine: input.engine,
    labelSet: input.labelSet,
    snapshotHash: hash,
    expected: input.expected,
    ...(input.shardRows === undefined ? {} : { shardRows: input.shardRows }),
    ...(input.decodeTimeoutMs === undefined ? {} : { decodeTimeoutMs: input.decodeTimeoutMs }),
  });
  // The shards were checked against the manifest's digests by the reader; the manifest itself must be the bytes that were sent.
  if (Buffer.compare(fetched.manifestBytes, manifestBytes) !== 0) throw new SnapshotIoError("download_mismatch");

  const localDir = writeLocalCopy(input.snapshotsDir, hash, fetched.manifestBytes, fetched.shardBytes);

  await checkCurrent(store, input.expectedCurrent);
  await store.overwriteCurrent(encodeCurrent({ schema_version: 1, snapshot_hash: hash, parent_snapshot_hash: parent }));
  return { snapshotHash: hash, localDir, previous: input.expectedCurrent };
}
