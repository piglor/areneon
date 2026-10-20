// The files of the isolated evaluation store, as the sealed-set tooling sees
// them (evaluation protocol I1; docs/runbooks/evaluation-store.md). Nothing
// secret is written anywhere but below a store directory the caller names, and
// only if that directory is the caller's own, private and outside any git
// working tree: the key, the salt and the sealed teams must not become
// readable to another user (training runs as one) by a mistake of the caller.
//
// Directories are created 0700 and files 0600 and never overwritten; a file
// that is read back must still be the invoking user's and private, so a copy
// somebody loosened is refused instead of used.

import { type Stats, chmodSync, closeSync, fstatSync, fsyncSync, lstatSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

export class SealedStoreError extends Error {
  override name = "SealedStoreError";
}

const refuse = (message: string): never => {
  throw new SealedStoreError(message);
};

const GROUP_OTHER = 0o077;
const GROUP_OTHER_WRITE = 0o022;
const STICKY = 0o1000;

function uid(): number {
  if (typeof process.getuid !== "function") return refuse("the sealed store needs POSIX file ownership (process.getuid)");
  return process.getuid();
}

/**
 * The store directory, if it may hold secrets: an absolute path to a real
 * directory (not a link) that the invoking user owns, with no group or other
 * permission bit; whose parents are not writable by group or other (a sticky
 * bit, as on /tmp, is fine); and which is not inside a git working tree.
 * Returns the path.
 */
export function assertPrivateStore(dir: string): string {
  if (!isAbsolute(dir)) refuse("the store directory must be an absolute path");
  const path = resolve(dir);
  const stat = lstatOrRefuse(path);
  if (stat.isSymbolicLink() || !stat.isDirectory()) refuse(`${path} is not a directory (a link is refused)`);
  if (stat.uid !== uid()) refuse(`${path} is not owned by the invoking user`);
  if ((stat.mode & GROUP_OTHER) !== 0) refuse(`${path} has group or other permission bits (mode ${(stat.mode & 0o7777).toString(8)}); it must be 0700`);
  for (let here = path; ; here = dirname(here)) {
    if (exists(join(here, ".git"))) refuse(`${path} is inside a git working tree (${here}); the store must be outside every checkout`);
    const parent = dirname(here);
    if (parent === here) break;
    const up = lstatOrRefuse(parent);
    if ((up.mode & GROUP_OTHER_WRITE) !== 0 && (up.mode & STICKY) === 0) refuse(`${parent} is writable by group or other, so ${path} could be swapped`);
  }
  return path;
}

function lstatOrRefuse(path: string): Stats {
  try {
    return lstatSync(path) as Stats;
  } catch (error) {
    return refuse(`${path}: ${(error as NodeJS.ErrnoException).code ?? "cannot be read"}`);
  }
}

function exists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

/** Create one new directory, mode 0700 whatever the umask. An existing one is refused: a set is never made twice. */
export function createPrivateDir(path: string): void {
  try {
    mkdirSync(path, { mode: 0o700 });
  } catch (error) {
    refuse(`${path} cannot be created: ${(error as NodeJS.ErrnoException).code ?? "error"}`);
  }
  chmodSync(path, 0o700);
}

/** Create `path` if it is missing, as a private directory; an existing one must itself be private. */
export function ensurePrivateDir(path: string): void {
  if (!exists(path)) return createPrivateDir(path);
  const stat = lstatOrRefuse(path);
  if (stat.isSymbolicLink() || !stat.isDirectory() || stat.uid !== uid() || (stat.mode & GROUP_OTHER) !== 0) refuse(`${path} is not a private directory of the invoking user`);
}

/** Write a new file, mode 0600, fsynced; an existing file is never overwritten. */
export function writeSecretFile(path: string, data: string | Buffer): void {
  let fd: number;
  try {
    fd = openSync(path, "wx", 0o600);
  } catch (error) {
    return refuse(`${path} cannot be created: ${(error as NodeJS.ErrnoException).code ?? "error"}`);
  }
  try {
    writeSync(fd, typeof data === "string" ? Buffer.from(data, "utf8") : data);
    fsyncSync(fd);
    if ((fstatSync(fd).mode & GROUP_OTHER) !== 0) refuse(`${path} was created with group or other permission bits`);
  } finally {
    closeSync(fd);
  }
}

/** An empty private file, for a log that is written as the run goes. */
export const createSecretFile = (path: string): void => writeSecretFile(path, "");

/** Read a file of the store: it must be a regular file of the invoking user with no group or other bit. */
export function readSecretFile(path: string): string {
  const stat = lstatOrRefuse(path);
  if (!stat.isFile()) refuse(`${path} is not a regular file`);
  if (stat.uid !== uid()) refuse(`${path} is not owned by the invoking user`);
  if ((stat.mode & GROUP_OTHER) !== 0) refuse(`${path} has group or other permission bits (mode ${(stat.mode & 0o7777).toString(8)}); it must be 0600`);
  return readFileSync(path, "utf8");
}

/** JSON.parse that never echoes the text: a parse error message can quote the input, and the input may be a secret. */
export function parseSecretJson(text: string, what: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return refuse(`${what} is not valid JSON`);
  }
}
