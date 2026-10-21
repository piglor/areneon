// The verified local copy of a snapshot (ADR-015 sections 1 and 3):
// `private-data/snapshots/<snapshot_hash>/manifest.json` and
// `shards/<index>.parquet`. It is the backup and a reader source, so it is
// written only from bytes that have already passed `readPrivateSnapshot`, and
// only into a directory `assertWorkDir` accepts (a `private-data` path or
// tmpfs, owned by this user, mode 0700, no loosely-writable parent).
// Directories are 0700 and files 0600. The copy is built in a hidden partial
// directory, flushed, and renamed into place, so a reader never sees half a
// copy; an existing copy is never replaced. A failure removes the partial
// directory and says only `local_copy_failed`.

import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, renameSync, writeSync } from "node:fs";
import { join } from "node:path";
import { removeTree } from "@areneon/team-dataset";
import { SnapshotIoError } from "./snapshot-io-error.ts";
import { assertWorkDir } from "./work-dir.ts";

function writeFile(path: string, bytes: Uint8Array): void {
  const fd = openSync(path, "wx", 0o600);
  try {
    let at = 0;
    while (at < bytes.byteLength) at += writeSync(fd, bytes, at);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function syncDirectory(path: string): void {
  const fd = openSync(path, "r");
  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

/** Create `dir` (mode 0700) if it is not there; the parent must already exist. */
export function ensurePrivateDirectory(dir: string): void {
  try {
    if (!existsSync(dir)) mkdirSync(dir, { mode: 0o700 });
  } catch {
    throw new SnapshotIoError("work_dir_unsafe");
  }
}

/** Refuse early (before anything is uploaded) if the local copy cannot be made: unsafe directory or a copy that exists. */
export function checkLocalCopyTarget(snapshotsDir: string, snapshotHash: string): string {
  ensurePrivateDirectory(snapshotsDir);
  const root = assertWorkDir(snapshotsDir);
  if (existsSync(join(root, snapshotHash))) throw new SnapshotIoError("local_copy_exists");
  return root;
}

/** Write the verified copy; returns its directory. */
export function writeLocalCopy(snapshotsDir: string, snapshotHash: string, manifest: Uint8Array, shards: readonly Uint8Array[]): string {
  const root = checkLocalCopyTarget(snapshotsDir, snapshotHash);
  const target = join(root, snapshotHash);
  const partial = join(root, `.${snapshotHash}.partial`);
  try {
    removeTree(partial);
    mkdirSync(partial, { mode: 0o700 });
    mkdirSync(join(partial, "shards"), { mode: 0o700 });
    writeFile(join(partial, "manifest.json"), manifest);
    shards.forEach((bytes, index) => writeFile(join(partial, "shards", `${index}.parquet`), bytes));
    syncDirectory(join(partial, "shards"));
    syncDirectory(partial);
    renameSync(partial, target);
    syncDirectory(root);
  } catch {
    removeTree(partial);
    throw new SnapshotIoError("local_copy_failed");
  }
  return target;
}
