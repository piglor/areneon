// The directory a snapshot write may keep intermediates in (ADR-015 section 3,
// "No leaks"): row files and shard files hold pastes in the clear, so they live
// only under a `private-data/` directory or on tmpfs, owned by the invoking
// user and closed to everyone else (mode 0700). The caller gives the
// directory; nothing is created here.

import { lstatSync, realpathSync, statfsSync } from "node:fs";
import { isAbsolute, resolve, sep } from "node:path";
import { SnapshotIoError } from "./snapshot-io-error.ts";

/** `f_type` of tmpfs and ramfs (linux/magic.h). */
const MEMORY_FILESYSTEMS: readonly number[] = [0x01021994, 0x858458f6];

export interface WorkDirProbe {
  /** The file system type of a path; the default asks the operating system. */
  readonly fsType?: (path: string) => number;
  /** The invoking user; the default is the process's. */
  readonly uid?: number;
}

/** Refuse unless `dir` is an acceptable place for plaintext intermediates; returns its real path. */
export function assertWorkDir(dir: string, probe: WorkDirProbe = {}): string {
  const fail = (): never => {
    throw new SnapshotIoError("work_dir_unsafe");
  };
  if (!isAbsolute(dir)) return fail();
  try {
    // lstat: a symbolic link to a directory is not the directory.
    const stat = lstatSync(dir);
    if (!stat.isDirectory()) return fail();
    const real = realpathSync(dir);
    if (real !== resolve(dir)) return fail();
    if (stat.uid !== (probe.uid ?? process.getuid?.())) return fail();
    if ((stat.mode & 0o077) !== 0) return fail();
    const fsType = probe.fsType ?? ((p: string) => Number(statfsSync(p).type));
    if (real.split(sep).includes("private-data") || MEMORY_FILESYSTEMS.includes(fsType(real))) return real;
    return fail();
  } catch (error) {
    if (error instanceof SnapshotIoError) throw error;
    return fail();
  }
}
