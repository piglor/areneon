// The RunInputManifest (ADR-024 section 3.5, ADR-008): the public inputs of one
// training run as the owner put them in the inputs bucket, each file with its
// path, size and SHA-256, the frozen commit, the run config hash and the
// dataset hash. The schema checks the shape; the rules the contract profile
// cannot say are checked here: every path is a safe relative path (no absolute
// path, no empty, dot or dot-dot segment, no backslash), the paths are sorted
// and unique and none is also a directory, the config file is one of the files
// and outside the dataset directory, and the dataset directory holds a file.
// Reader messages are fixed words that quote no path or value.
//
// The manifest is stored as JCS bytes (`runInputManifestBytes`); its hash is
// the SHA-256 of those bytes, and the fetch command can pin it.

import { ajvFor } from "@areneon/schema/ajv";
import runInputManifestSchema from "@areneon/schema/contracts/v1/run-input-manifest.schema.json" with { type: "json" };
import type { RunInputManifest } from "@areneon/schema/types/v1/run-input-manifest.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { nodeSha256 } from "./player-channel.ts";

export type { RunInputManifest };
export type RunInputFile = RunInputManifest["files"][number];

export class RunInputManifestError extends Error {
  override name = "RunInputManifestError";
}

const validManifest = ajvFor("contract").compile(runInputManifestSchema);

const fail = (message: string): never => {
  throw new RunInputManifestError(message);
};

const SEGMENT = /^[A-Za-z0-9._-]+$/;

/** A relative path with `/` separators whose every segment is plain: not empty, not `.` or `..`, only letters, digits, `.`, `_` and `-`. */
export function isSafeRelativePath(path: string): boolean {
  if (path.length === 0 || path.length > 512) return false;
  return path.split("/").every((segment) => segment !== "." && segment !== ".." && SEGMENT.test(segment));
}

/** The directories above a path, nearest last (`a/b/c` gives `a`, `a/b`). */
function ancestors(path: string): string[] {
  const parts = path.split("/");
  return parts.slice(0, -1).map((_, index) => parts.slice(0, index + 1).join("/"));
}

/** Read a RunInputManifest: the schema, then the rules above. Returns a copy. */
export function readRunInputManifest(value: unknown): RunInputManifest {
  if (!validManifest(value)) throw new RunInputManifestError(`invalid RunInputManifest: ${JSON.stringify(validManifest.errors)}`);
  const manifest = structuredClone(value) as unknown as RunInputManifest;
  const paths = manifest.files.map((file) => file.path);
  for (const path of [manifest.config_file, manifest.dataset_dir, ...paths]) {
    if (!isSafeRelativePath(path)) fail("paths must be safe relative paths: no absolute path, no empty, dot or dot-dot segment and no backslash");
  }
  if (paths.some((path, index) => index > 0 && path <= (paths[index - 1] as string))) fail("files must be sorted by path, each once");
  const listed = new Set(paths);
  const directories = new Set(paths.flatMap(ancestors));
  if (paths.some((path) => directories.has(path))) fail("a path cannot be both a file and a directory");
  if (!listed.has(manifest.config_file)) fail("the config file must be one of the files");
  const prefix = `${manifest.dataset_dir}/`;
  if (manifest.config_file.startsWith(prefix)) fail("the config file cannot be inside the dataset directory");
  if (!paths.some((path) => path.startsWith(prefix))) fail("the dataset directory must hold at least one file");
  return manifest;
}

/** The bytes the manifest is stored as: its JCS form. */
export const runInputManifestBytes = (manifest: RunInputManifest): Uint8Array => new TextEncoder().encode(jcs(manifest as unknown as Json));

/** The manifest's hash: SHA-256 of its JCS bytes. */
export const runInputManifestHash = (manifest: RunInputManifest): Promise<string> => nodeSha256(runInputManifestBytes(manifest));
