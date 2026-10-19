import { createHash } from "node:crypto";
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  renameSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_SHOWDOWN_DIR,
  PINNED_SHOWDOWN_COMMIT,
  verifyShowdownCheckout,
} from "@areneon/showdown-bridge";
import { acquireCopyLock, blockedMessage } from "./copy-lock.ts";

/**
 * Where runtime copies live by default: `<repo>/.local-server/runtime/<pin>/`
 * (gitignored). The server writes `logs/` and its `config/config.js` into its
 * install root, so it must run from a copy, never from `vendor/` (ADR-005 §4).
 */
export const DEFAULT_RUNTIME_ROOT = fileURLToPath(
  new URL("../../../.local-server/runtime", import.meta.url),
);

/** What the server needs at runtime, relative to the Showdown install root (ADR-005 §4). */
export const RUNTIME_ENTRIES = [
  "dist",
  "server/static",
  "config/config-example.js",
  "node_modules",
  "package.json",
  "pokemon-showdown",
] as const;

/**
 * The runtime entries the manifest digests: every file the server executes
 * or loads from the copy outside `node_modules/`. At the pin, `launch` runs
 * the `pokemon-showdown` entry script, which loads `dist/server/index.js`;
 * the server then requires only from `dist/` (its code, `dist/data`,
 * `dist/translations`, and the config defaults `dist/config/config-example.js`
 * that its config loader merges under `config/config.js`), serves
 * `server/static/`, and reads `package.json` (Node reads it to load the entry
 * script as CommonJS; the server parses it as its package data). The copy's
 * `config/config-example.js` is digested too: the server never loads it
 * under `--skip-build`, but it is the source the build copies to
 * `dist/config/config-example.js`, and the build also copies it to
 * `config/config.js` when that file is missing.
 *
 * Not digested, by design:
 * - `node_modules/`: third-party packages that npm installs from the
 *   checkout's lockfile, not files of the pinned commit, and not the whole of
 *   what the server can require: Node resolves a package missing there from
 *   every ancestor `node_modules/`, outside the copy, so a digest of the
 *   copy's tree would claim more than it proves. `createRuntimeCopy` checks
 *   its symlinks stay inside the copy.
 * - `config/config.js`: written by `launch` before every start, checked from
 *   disk, and recorded in the session manifest (`config_sha256`).
 * - Everything else the server reads under `config/` or writes under `logs/`
 *   (`RUNTIME_STATE_DIRS`, `config/usergroups.csv`, ...): server state, not
 *   part of the copy.
 * - `databases/` (the SQL schemas and migrations): not copied. At the pin
 *   only the SQLite stores read them (friends, offline PMs, modlog, chat
 *   plugins), behind `usesqlite`, which neither the upstream defaults nor
 *   Areneon's `config.js` set (the friends child processes start, but open
 *   no store without it), and the Postgres stores (room logs behind
 *   `replaysdb`, teams behind `usepostgres`), both off in Areneon's
 *   `config.js`. So nothing reads them under Areneon's config.
 * - `.git`: never copied, and `verifyRuntimeCopy` refuses a copy that holds
 *   one. `Monitor.version()` reads `.git/index` at every start; were one
 *   present, it would run git from `dist/server/` against the repository
 *   enclosing the copy.
 */
export const DIGESTED_ENTRIES = [
  "dist",
  "server/static",
  "config/config-example.js",
  "package.json",
  "pokemon-showdown",
] as const;

export type DigestedEntry = (typeof DIGESTED_ENTRIES)[number];

/**
 * The runtime manifest's format. A copy whose manifest has another version
 * (version 1 had no `version` field and digested only `dist/`) is refused,
 * never migrated: the caller recreates it with `createRuntimeCopy`.
 */
export const RUNTIME_MANIFEST_VERSION = 2;

/**
 * A staging or set-aside directory (`.<commit>-*`, `.<commit>-*.old`) whose
 * mtime is older than this is left over from a crashed run and is removed; a
 * younger one may belong to a concurrent run and is left alone. A staging
 * directory is new when created, and an old copy is touched just before it
 * is set aside, so both start young.
 */
export const STALE_STAGING_MS = 10 * 60 * 1000;

/** The record of a runtime copy's commit and entry digests, inside the copy. */
export const RUNTIME_MANIFEST = "areneon-runtime.json";

export interface RuntimeCopy {
  /** The copy's root: the server's install root and working directory. */
  dir: string;
  commit: string;
  /** `entryDigest` of each digested runtime entry, by entry. */
  sha256: Record<DigestedEntry, string>;
}

export interface RuntimeCopyOptions {
  vendorDir?: string;
  runtimeRoot?: string;
  expectedCommit?: string;
}

export class RuntimeCopyError extends Error {
  override name = "RuntimeCopyError";
}

/**
 * SHA-256 over a directory tree: every regular file in byte order of its
 * `/`-separated relative path, each as `path NUL size NUL bytes`. The size
 * prefix keeps the encoding unambiguous. Anything but a directory or a
 * regular file (a symlink, a socket) is refused, so the digest always covers
 * the bytes the server will load.
 */
export function treeDigest(dir: string): string {
  const files: string[] = [];
  const walk = (rel: string): void => {
    for (const entry of readdirSync(join(dir, rel), { withFileTypes: true })) {
      const path = rel === "" ? entry.name : `${rel}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) files.push(path);
      else throw new RuntimeCopyError(`${join(dir, path)} is not a regular file`);
    }
  };
  walk("");
  files.sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
  const hash = createHash("sha256");
  for (const path of files) {
    const bytes = readFileSync(join(dir, path));
    hash.update(`${path}\0${bytes.length}\0`);
    hash.update(bytes);
  }
  return hash.digest("hex");
}

/**
 * SHA-256 of one runtime entry: `treeDigest` of a directory, the plain
 * SHA-256 of a regular file's bytes. Anything else (a symlink, even to a
 * file inside the copy) is refused.
 */
export function entryDigest(path: string): string {
  const stat = lstatSync(path);
  if (stat.isDirectory()) return treeDigest(path);
  if (!stat.isFile()) throw new RuntimeCopyError(`${path} is not a regular file`);
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function digestEntries(dir: string): Record<DigestedEntry, string> {
  return Object.fromEntries(
    DIGESTED_ENTRIES.map((entry) => [entry, entryDigest(join(dir, entry))]),
  ) as Record<DigestedEntry, string>;
}

/**
 * Copy the pinned, built Showdown checkout's runtime files into
 * `<runtimeRoot>/<commit>/` and record the digests of its `DIGESTED_ENTRIES` there
 * (ADR-005 §4). The checkout is verified exactly as `loadEngine()` does
 * (HEAD = pin, build present). Nothing is written under `vendorDir`: the
 * copy is staged next to its target and renamed into place.
 *
 * An earlier copy for the same commit is renamed aside, replaced, then
 * removed, under the copy's lock (`lockPath`). If that lock is held (by
 * `launch` while a server may run from the copy, or by another creator) or
 * cannot be read, this refuses with a `RuntimeCopyError`: a server running
 * from the copy would lose its install root.
 * Concurrent creators of the same commit each either put a complete copy in
 * place or fail with a `RuntimeCopyError`. Other I/O errors (a full disk,
 * EACCES while creating or copying) propagate as they are. Either way the
 * staging directory is removed; anything a failed removal leaves behind is
 * swept by a later run once stale.
 */
export function createRuntimeCopy(options: RuntimeCopyOptions = {}): RuntimeCopy {
  const vendorDir = resolve(options.vendorDir ?? DEFAULT_SHOWDOWN_DIR);
  const runtimeRoot = resolve(options.runtimeRoot ?? DEFAULT_RUNTIME_ROOT);
  const { commit } = verifyShowdownCheckout(
    vendorDir,
    options.expectedCommit ?? PINNED_SHOWDOWN_COMMIT,
  );
  if (isWithin(realTarget(runtimeRoot), realpathSync(vendorDir))) {
    throw new RuntimeCopyError(`runtime root ${runtimeRoot} is inside the vendor checkout`);
  }
  for (const entry of RUNTIME_ENTRIES) {
    if (!existsSync(join(vendorDir, entry))) {
      throw new RuntimeCopyError(`vendor checkout has no ${entry}; run \`make showdown\``);
    }
  }
  mkdirSync(runtimeRoot, { recursive: true });
  sweepStale(runtimeRoot, commit);
  const staging = mkdtempSync(join(runtimeRoot, `.${commit}-`));
  try {
    for (const entry of RUNTIME_ENTRIES) {
      // Relative symlinks (node_modules/.bin) stay relative, so they resolve
      // inside the copy; checkContained refuses any that would not.
      cpSync(join(vendorDir, entry), join(staging, entry), {
        recursive: true,
        verbatimSymlinks: true,
        errorOnExist: true,
        force: false,
      });
    }
    checkContained(staging);
    const sha256 = digestEntries(staging);
    writeFileSync(
      join(staging, RUNTIME_MANIFEST),
      `${JSON.stringify({ version: RUNTIME_MANIFEST_VERSION, commit, sha256 }, null, 2)}\n`,
    );
    const dir = join(runtimeRoot, commit);
    const lock = acquireCopyLock(dir);
    if (!("release" in lock)) throw new RuntimeCopyError(blockedMessage(dir, lock));
    const aside = `${staging}.old`;
    try {
      try {
        const now = new Date();
        utimesSync(dir, now, now);
        renameSync(dir, aside);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      renameSync(staging, dir);
    } catch (error) {
      removeBestEffort(aside);
      throw new RuntimeCopyError(`cannot put the runtime copy in place at ${dir}: ${String(error)}`);
    } finally {
      try {
        lock.release();
      } catch {
        // Left in place, the lock names this process: stale once it exits.
      }
    }
    // The new copy is in place; a leftover old copy is swept later.
    removeBestEffort(aside);
    return { dir, commit, sha256 };
  } catch (error) {
    removeBestEffort(staging);
    throw error;
  }
}

/** Remove a directory without letting a failure mask the caller's outcome. */
function removeBestEffort(path: string): void {
  try {
    rmSync(path, { recursive: true, force: true });
  } catch {
    // Left for a later sweep of stale `.<commit>-*` directories.
  }
}

/**
 * Re-verify a runtime copy before it is used: its manifest is of version
 * `RUNTIME_MANIFEST_VERSION` and names `expectedCommit`, every runtime entry
 * is present, it holds no `.git` (of any kind, symlinks included), and every
 * digested entry still has its recorded digest. Any
 * difference, including a manifest of an older version, is refused as a
 * `RuntimeCopyError`; the caller recreates the copy with `createRuntimeCopy`.
 */
export function verifyRuntimeCopy(
  dir: string,
  expectedCommit: string = PINNED_SHOWDOWN_COMMIT,
): RuntimeCopy {
  const manifest = readManifest(join(dir, RUNTIME_MANIFEST));
  if (manifest.commit !== expectedCommit) {
    throw new RuntimeCopyError(
      `runtime copy ${dir} is for ${manifest.commit}, expected pinned ${expectedCommit}`,
    );
  }
  for (const entry of RUNTIME_ENTRIES) {
    if (!existsSync(join(dir, entry))) {
      throw new RuntimeCopyError(`runtime copy ${dir} has no ${entry}`);
    }
  }
  if (lstatSync(join(dir, ".git"), { throwIfNoEntry: false }) !== undefined) {
    throw new RuntimeCopyError(
      `runtime copy ${dir} has a .git entry; the runtime copy must not contain .git: ` +
        "Monitor.version would run git against the enclosing repository; recreate the copy",
    );
  }
  const sha256 = digestEntries(dir);
  for (const entry of DIGESTED_ENTRIES) {
    if (sha256[entry] !== manifest.sha256[entry]) {
      throw new RuntimeCopyError(
        `runtime copy ${dir} ${entry} digest ${sha256[entry]} does not match the recorded ` +
          `${manifest.sha256[entry]}; recreate the copy with createRuntimeCopy`,
      );
    }
  }
  return { dir, commit: manifest.commit, sha256 };
}

interface RuntimeManifest {
  version: number;
  commit: string;
  sha256: Record<DigestedEntry, string>;
}

function readManifest(path: string): RuntimeManifest {
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new RuntimeCopyError(`cannot read runtime manifest ${path}: ${String(error)}`);
  }
  if (typeof value !== "object" || value === null) {
    throw new RuntimeCopyError(`runtime manifest ${path} is not an object`);
  }
  const version = (value as Record<string, unknown>)["version"];
  if (version !== RUNTIME_MANIFEST_VERSION) {
    throw new RuntimeCopyError(
      `runtime manifest ${path} is version ${JSON.stringify(version ?? 1)}, not ` +
        `${RUNTIME_MANIFEST_VERSION}; recreate the copy with createRuntimeCopy`,
    );
  }
  const { commit, sha256 } = value as Partial<RuntimeManifest>;
  if (
    Object.keys(value).sort().join() !== "commit,sha256,version" ||
    typeof commit !== "string" ||
    typeof sha256 !== "object" ||
    sha256 === null ||
    Object.keys(sha256).sort().join() !== [...DIGESTED_ENTRIES].sort().join() ||
    !Object.values(sha256).every((digest) => typeof digest === "string")
  ) {
    throw new RuntimeCopyError(
      `runtime manifest ${path} is not {version, commit, sha256: {${DIGESTED_ENTRIES.join(", ")}}}`,
    );
  }
  return value as RuntimeManifest;
}

/**
 * Remove this commit's stale staging and set-aside directories left by a
 * crashed run. Best effort: a concurrent run may remove its own directories
 * while this runs, and a failed sweep never fails the copy.
 */
function sweepStale(runtimeRoot: string, commit: string): void {
  const now = Date.now();
  let names: string[];
  try {
    names = readdirSync(runtimeRoot);
  } catch {
    return;
  }
  for (const name of names.filter((entry) => entry.startsWith(`.${commit}-`))) {
    const path = join(runtimeRoot, name);
    try {
      const stat = lstatSync(path, { throwIfNoEntry: false });
      if (stat !== undefined && now - stat.mtimeMs > STALE_STAGING_MS) {
        rmSync(path, { recursive: true, force: true });
      }
    } catch {
      // Gone or not ours to remove: leave it to its owner or a later sweep.
    }
  }
}

/**
 * Refuse a symlink in the copy that is absolute, dangling, or whose real path
 * (following every link in the chain) is outside the copy's real path. Fails
 * closed: the pinned checkout's only links are relative `node_modules/.bin`
 * entries whose targets exist in the copy.
 */
function checkContained(root: string): void {
  // `realpathSync.native` asks the OS, which follows each link before the
  // next `..`; the JS `realpathSync` collapses `l/..` lexically first.
  const realRoot = realpathSync.native(root);
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.isSymbolicLink()) {
        const link = readlinkSync(path);
        if (isAbsolute(link)) {
          throw new RuntimeCopyError(`symlink ${path} -> ${link} is absolute`);
        }
        if (!existsSync(path)) {
          throw new RuntimeCopyError(`symlink ${path} -> ${link} is dangling`);
        }
        if (!isWithin(realpathSync.native(path), realRoot)) {
          throw new RuntimeCopyError(`symlink ${path} -> ${link} leaves the runtime copy`);
        }
      }
    }
  };
  walk(root);
}

/** The real path `path` would have, resolving its nearest existing ancestor. */
function realTarget(path: string): string {
  if (existsSync(path)) return realpathSync(path);
  return join(realTarget(dirname(path)), basename(path));
}

function isWithin(path: string, root: string): boolean {
  const rel = relative(root, path);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}
