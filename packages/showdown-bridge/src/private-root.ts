// The private output root (ADR-015 erratum E1, items 7-9): the one rule for
// where a command that reads private teams may write. A directory is accepted
// only if it is absolute, a real directory (no symbolic link in the path),
// owned by the invoking user, mode 0700, below no group- or world-writable
// parent (a sticky bit, as on /tmp, is fine), every parent owned by the user
// or root, not on a Windows drive mounted into WSL (DrvFs/9p keeps no real
// owner or mode), and has a `private-data` path segment. These are the
// `assertWorkDir` rules of `packages/private-team-store/src/work-dir.ts`
// (which also admits tmpfs for snapshot intermediates; an output root does
// not) with a fixed code each; the Python twin is
// `ml/areneon_ml/private/private_root.py`. A refusal carries a code and the
// fixed words of `fixtures/private-guard/refusals.json`, never a path.
// `probe.uid` is a seam for tests only. Node built-ins only.

import { closeSync, lstatSync, openSync, realpathSync, writeSync } from "node:fs";
import { dirname, isAbsolute, join, resolve, sep } from "node:path";

export const PRIVATE_ROOT_REFUSALS = {
  not_absolute: "refusing: the private output root must be an absolute path",
  unreadable: "refusing: the private output root cannot be read",
  not_a_directory: "refusing: the private output root is not a real directory",
  symbolic_link: "refusing: the private output root is, or lies under, a symbolic link",
  wrong_owner: "refusing: the private output root is not owned by the invoking user",
  group_or_other_access: "refusing: the private output root must have mode 0700",
  unsafe_parent: "refusing: a parent of the private output root is writable by others or owned by another user",
  windows_mount: "refusing: the private output root is on a Windows drive mount",
  no_private_data_segment: "refusing: the private output root has no private-data path segment",
  bad_file_name: "refusing: not a plain file name for the private output root",
  exists: "refusing: the private output file already exists",
} as const;

export type PrivateRootCode = keyof typeof PRIVATE_ROOT_REFUSALS;

export class PrivateRootError extends Error {
  readonly code: PrivateRootCode;
  constructor(code: PrivateRootCode) {
    super(PRIVATE_ROOT_REFUSALS[code]);
    this.name = "PrivateRootError";
    this.code = code;
  }
}

export interface PrivateRootProbe {
  /** The invoking user; the default is the process's. */
  readonly uid?: number;
  /** The owner of a parent directory; the default asks the operating system. */
  readonly parentUid?: (parent: string) => number;
}

const GROUP_OTHER = 0o077;
const GROUP_OTHER_WRITE = 0o022;
const STICKY = 0o1000;

/** A Windows drive as WSL mounts it (`/mnt/c/...`). */
export const onWindowsMount = (path: string): boolean => /^\/mnt\/[A-Za-z](\/|$)/.test(path);

/** Refuse unless `dir` is an acceptable private output root; returns its real path. */
export function assertPrivateRoot(dir: string, probe: PrivateRootProbe = {}): string {
  const fail = (code: PrivateRootCode): never => {
    throw new PrivateRootError(code);
  };
  if (!isAbsolute(dir)) return fail("not_absolute");
  const me = probe.uid ?? (process.getuid as () => number)();
  let real: string;
  try {
    // lstat: a symbolic link to a directory is not the directory.
    const stat = lstatSync(dir);
    if (stat.isSymbolicLink()) return fail("symbolic_link");
    if (!stat.isDirectory()) return fail("not_a_directory");
    real = realpathSync(dir);
    if (real !== resolve(dir)) return fail("symbolic_link");
    if (stat.uid !== me) return fail("wrong_owner");
    if ((stat.mode & GROUP_OTHER) !== 0) return fail("group_or_other_access");
    for (let here = real, up = dirname(here); up !== here; here = up, up = dirname(here)) {
      const parent = lstatSync(up);
      const owner = probe.parentUid?.(up) ?? parent.uid;
      if (owner !== me && owner !== 0) return fail("unsafe_parent");
      if ((parent.mode & GROUP_OTHER_WRITE) !== 0 && (parent.mode & STICKY) === 0) return fail("unsafe_parent");
    }
  } catch (error) {
    return error instanceof PrivateRootError ? fail(error.code) : fail("unreadable");
  }
  const place = placeRefusal(real);
  return place === null ? real : fail(place);
}

/** The refusal for where a real path lies (a Windows drive, or no `private-data` segment), or null. */
export function placeRefusal(real: string): PrivateRootCode | null {
  if (onWindowsMount(real)) return "windows_mount";
  return real.split(sep).includes("private-data") ? null : "no_private_data_segment";
}

/** `name` must be one plain file name: no separator, no `.` or `..`, no NUL, not empty. */
const plainName = (name: string): boolean => /^[^/\\\0]+$/.test(name) && name !== "." && name !== "..";

/**
 * The path of a file in a checked private root (re-checked on every call).
 * Only a plain file name is accepted, so nothing resolves outside the root.
 */
export function privateRootFile(root: string, name: string, probe: PrivateRootProbe = {}): string {
  const real = assertPrivateRoot(root, probe);
  if (!plainName(name)) throw new PrivateRootError("bad_file_name");
  return join(real, name);
}

/** Write a new file (mode 0600) in the private root; an existing file is never overwritten. */
export function writePrivateFile(root: string, name: string, data: string | Uint8Array, probe: PrivateRootProbe = {}): string {
  const path = privateRootFile(root, name, probe);
  let fd: number;
  try {
    fd = openSync(path, "wx", 0o600);
  } catch (error) {
    throw new PrivateRootError((error as NodeJS.ErrnoException).code === "EEXIST" ? "exists" : "unreadable");
  }
  try {
    writeSync(fd, typeof data === "string" ? Buffer.from(data, "utf8") : data);
  } finally {
    closeSync(fd);
  }
  return path;
}
