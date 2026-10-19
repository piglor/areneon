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
  writeFileSync,
} from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_SHOWDOWN_DIR,
  PINNED_SHOWDOWN_COMMIT,
  verifyShowdownCheckout,
} from "@areneon/showdown-bridge";

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
 * A staging or set-aside directory (`.<commit>-*`) older than this is left
 * over from a crashed run and is removed; a younger one may belong to a
 * concurrent run and is left alone.
 */
export const STALE_STAGING_MS = 10 * 60 * 1000;

/** The record of a runtime copy's commit and `dist/` digest, inside the copy. */
export const RUNTIME_MANIFEST = "areneon-runtime.json";

export interface RuntimeCopy {
  /** The copy's root: the server's install root and working directory. */
  dir: string;
  commit: string;
  /** `treeDigest` of the copy's `dist/`. */
  distSha256: string;
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
 * Copy the pinned, built Showdown checkout's runtime files into
 * `<runtimeRoot>/<commit>/` and record the copied `dist/` digest there
 * (ADR-005 §4). The checkout is verified exactly as `loadEngine()` does
 * (HEAD = pin, build present). Nothing is written under `vendorDir`: the
 * copy is staged next to its target and renamed into place.
 *
 * An earlier copy for the same commit is renamed aside, replaced, then
 * removed. A server already running from it keeps its open files, but the
 * launcher must not recreate the copy while one of its servers runs from it.
 * Concurrent creators of the same commit each either put a complete copy in
 * place or fail with a `RuntimeCopyError`; none leaves a staging directory.
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
    const distSha256 = treeDigest(join(staging, "dist"));
    writeFileSync(
      join(staging, RUNTIME_MANIFEST),
      `${JSON.stringify({ commit, dist_sha256: distSha256 }, null, 2)}\n`,
    );
    const dir = join(runtimeRoot, commit);
    const aside = `${staging}.old`;
    try {
      try {
        renameSync(dir, aside);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      renameSync(staging, dir);
    } catch (error) {
      rmSync(aside, { recursive: true, force: true });
      throw new RuntimeCopyError(`cannot put the runtime copy in place at ${dir}: ${String(error)}`);
    }
    rmSync(aside, { recursive: true, force: true });
    return { dir, commit, distSha256 };
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }
}

/**
 * Re-verify a runtime copy before it is used: its manifest names
 * `expectedCommit`, every runtime entry is present, and its `dist/` still has
 * the recorded digest. Any difference is refused as a `RuntimeCopyError`; the
 * caller rebuilds the copy with `createRuntimeCopy`.
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
  const distSha256 = treeDigest(join(dir, "dist"));
  if (distSha256 !== manifest.dist_sha256) {
    throw new RuntimeCopyError(
      `runtime copy ${dir} dist/ digest ${distSha256} does not match the recorded ${manifest.dist_sha256}`,
    );
  }
  return { dir, commit: manifest.commit, distSha256 };
}

function readManifest(path: string): { commit: string; dist_sha256: string } {
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new RuntimeCopyError(`cannot read runtime manifest ${path}: ${String(error)}`);
  }
  if (
    typeof value !== "object" ||
    value === null ||
    Object.keys(value).sort().join() !== "commit,dist_sha256" ||
    !Object.values(value).every((field) => typeof field === "string")
  ) {
    throw new RuntimeCopyError(`runtime manifest ${path} is not {commit, dist_sha256}`);
  }
  return value as { commit: string; dist_sha256: string };
}

/** Remove this commit's staging and set-aside directories left by a crashed run. */
function sweepStale(runtimeRoot: string, commit: string): void {
  const now = Date.now();
  for (const name of readdirSync(runtimeRoot)) {
    const path = join(runtimeRoot, name);
    if (name.startsWith(`.${commit}-`) && now - lstatSync(path).mtimeMs > STALE_STAGING_MS) {
      rmSync(path, { recursive: true, force: true });
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
