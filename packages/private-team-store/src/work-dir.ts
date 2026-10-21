// The directory a snapshot write may keep intermediates in (ADR-015 section 3,
// "No leaks"): row files and shard files hold pastes in the clear, so they live
// only under a `private-data/` directory or on tmpfs, owned by the invoking
// user and closed to everyone else (mode 0700). The caller gives the
// directory; nothing is created here. Following the sealed store's rules
// (`packages/team-dataset/src/sealed-store.ts`), no parent may be writable by
// group or other (a sticky bit, as on /tmp, is fine) and every parent must
// belong to the invoking user or to root, or the directory could be swapped
// under the writer.
//
// A 0700 mode proves little on a Windows drive mounted into WSL (`/mnt/c`:
// DrvFs/9p keeps no real owner or mode), so such a path is accepted only if it
// is on tmpfs, which it cannot be.
//
// `assertWorkDir`'s second argument is a seam for tests (the file-system type
// and the user id); production code never passes it, and a test reads the
// writer's source to say so.

import { lstatSync, realpathSync, statfsSync } from "node:fs";
import { dirname, isAbsolute, resolve, sep } from "node:path";
import { SnapshotIoError } from "./snapshot-io-error.ts";

/** `f_type` of tmpfs and ramfs (linux/magic.h). */
const MEMORY_FILESYSTEMS: readonly number[] = [0x01021994, 0x858458f6];
const GROUP_OTHER = 0o077;
const GROUP_OTHER_WRITE = 0o022;
const STICKY = 0o1000;

export interface WorkDirProbe {
  /** The file system type of a path; the default asks the operating system. */
  readonly fsType?: (path: string) => number;
  /** The invoking user; the default is the process's. */
  readonly uid?: number;
}

/** A directory above the work directory may belong to the invoking user or to root, and to nobody else. */
export const parentOwnerOk = (owner: number, me: number): boolean => owner === me || owner === 0;

/** A Windows drive as WSL mounts it (`/mnt/c/...`): no real owner or mode there. */
export const onWindowsMount = (path: string): boolean => /^\/mnt\/[A-Za-z](\/|$)/.test(path);

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
    const me = probe.uid ?? (process.getuid as () => number)();
    if (stat.uid !== me) return fail();
    if ((stat.mode & GROUP_OTHER) !== 0) return fail();
    for (let here = real, up = dirname(here); up !== here; here = up, up = dirname(here)) {
      const parent = lstatSync(up);
      if (!parentOwnerOk(parent.uid, me)) return fail();
      if ((parent.mode & GROUP_OTHER_WRITE) !== 0 && (parent.mode & STICKY) === 0) return fail();
    }
    const fsType = probe.fsType ?? ((p: string) => Number(statfsSync(p).type));
    const inMemory = MEMORY_FILESYSTEMS.includes(fsType(real));
    // A Windows drive counts only if it is in memory; elsewhere a private-data path counts too.
    return inMemory || (!onWindowsMount(real) && real.split(sep).includes("private-data")) ? real : fail();
  } catch (error) {
    if (error instanceof SnapshotIoError) throw error;
    return fail();
  }
}
