// The team pool, loaded once and shared with the further game processes of a generation (#1192, #980).
//
// Every game process used to load and normalise the pool itself (the dataset is read, validated and each
// team normalised by the pinned validator: about 34 s at full size, repeated S times in parallel). Now the
// process that starts the others loads it once and leaves it in the scratch directory they share (a
// private 0700 directory this process made); the others read it back instead. Operational only: the pool
// is the same value (same teams, same order, same key order), so no seed, plan or hash moves.
//
// A pool read back is checked, not trusted: the starter vouches for the file by sending its SHA-256 to each
// child over the pipe it holds (`parent-link.ts`), and the child refuses bytes that do not hash to it, or a
// file that is not owned by the invoking user with mode 0600. Beyond that its request key must be the one
// derived from this process's own run config (so a file of another format, dataset or excluded-ability list
// is refused), its hash list must be sorted and unique, and every team must hash to the hash listed for it.
// The `known_synthetic` set is rebuilt from the verified hash list, which the starter vouched for.

import { createHash } from "node:crypto";
import { chmodSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { type Json, jcs } from "./canonical-json.ts";
import { GenerationError, type PoolRequest, type TrainingPool } from "./league-generation.ts";
import type { PokemonSet } from "./engine.ts";
import { timed, timedAsync } from "./profile.ts";
import { syntheticTeamHashes, teamHash } from "./team-class.ts";

/** The version of the pool file's layout. */
export const POOL_FILE_VERSION = 1;

/** The key a pool file is bound to: SHA-256 of the JCS of what the pool is built from (format, dataset, the excluded abilities as a sorted set). */
export function poolRequestKey(request: PoolRequest): string {
  const named = { format_id: request.format_id, team_dataset: request.team_dataset ?? null, excluded_abilities: [...request.generation.excluded_abilities].sort() };
  return createHash("sha256").update(jcs(named as unknown as Json)).digest("hex");
}

/**
 * Write the pool for the other game processes (mode 0600, to a temporary name first, then renamed, so a reader never sees half a file) and
 * return the SHA-256 of the bytes written, which the caller sends to each of them. Plain JSON, so the sets keep their key order.
 */
export function writePoolFile(path: string, pool: TrainingPool, request: PoolRequest): string {
  const temporary = `${path}.partial`;
  const bytes = JSON.stringify({ schema_version: POOL_FILE_VERSION, request: poolRequestKey(request), hashes: pool.hashes, teams: pool.teams });
  writeFileSync(temporary, bytes, { mode: 0o600 });
  chmodSync(temporary, 0o600); // whatever the umask said
  renameSync(temporary, path);
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * Read a pool file back and check it as the module comment says; a refusal is a `GenerationError`.
 * The whole file is read as ONE string: a pool is a few megabytes (some thousand teams), far below Node's string limit.
 * The bytes are hashed and parsed from the same read, so the file cannot change between the check and the use.
 */
export function readPoolFile(path: string, request: PoolRequest, digest: string, uid: number | undefined = process.getuid?.()): TrainingPool {
  let bytes: Buffer;
  try {
    const status = statSync(path);
    if ((uid !== undefined && status.uid !== uid) || (status.mode & 0o777) !== 0o600) throw new GenerationError("the shared team pool is not owned by this user with mode 0600");
    bytes = readFileSync(path);
  } catch (error) {
    if (error instanceof GenerationError) throw error;
    throw new GenerationError(`the shared team pool cannot be read: ${(error as Error).name}`);
  }
  if (createHash("sha256").update(bytes).digest("hex") !== digest) throw new GenerationError("the shared team pool is not the file its starter vouched for");
  let value: { schema_version?: unknown; request?: unknown; hashes?: unknown; teams?: unknown };
  try {
    value = JSON.parse(bytes.toString("utf8")) as typeof value;
  } catch (error) {
    throw new GenerationError(`the shared team pool cannot be read: ${(error as Error).name}`);
  }
  if (value.schema_version !== POOL_FILE_VERSION) throw new GenerationError("the shared team pool has another layout version");
  if (value.request !== poolRequestKey(request)) throw new GenerationError("the shared team pool was built for another format, dataset or excluded-ability list");
  const { hashes, teams } = value;
  if (!Array.isArray(hashes) || !Array.isArray(teams) || hashes.length !== teams.length || hashes.length === 0) throw new GenerationError("the shared team pool is empty or its hashes and teams differ in number");
  const listed = hashes as unknown[];
  if (listed.some((hash, index) => typeof hash !== "string" || (index > 0 && !((listed[index - 1] as string) < hash)))) throw new GenerationError("the shared team pool's hashes are not sorted and unique");
  if ((teams as unknown[]).some((team, index) => !Array.isArray(team) || teamHash(team) !== listed[index])) throw new GenerationError("a team of the shared team pool does not hash to its listed hash");
  return { teams: teams as PokemonSet[][], hashes: listed as string[], known_synthetic: syntheticTeamHashes(listed as string[]) };
}

/**
 * The pool of one game process, and the digest of the file that carries it. With `read` (a further game process:
 * the file, and the digest its starter sends) it reads what its starter left once the digest arrives; otherwise it
 * loads the pool with `load` (the load runs before this returns, so a caller that starts its workers after calling
 * this starts them after the load, as before) and, with `write`, leaves it in that file and returns the digest to
 * send (`vouch` is called with it at once). Both ways are timed as the `pool_load` phase. There is no polling: a reader waits for the digest, and ends
 * with its starter if that dies (`parent-link.ts`).
 */
export function obtainPool(
  config: PoolRequest,
  load: () => TrainingPool,
  { read, write, vouch }: { read?: { file: string; digest: Promise<string> } | undefined; write?: string | undefined; vouch?: ((digest: string) => void) | undefined },
): Promise<{ pool: TrainingPool; digest: string | undefined }> {
  if (read !== undefined) return timedAsync("pool_load", async () => ({ pool: readPoolFile(read.file, config, await read.digest), digest: undefined }));
  const pool = timed("pool_load", load);
  const digest = write === undefined ? undefined : timed("pool_load", () => writePoolFile(write, pool, config));
  if (digest !== undefined) vouch?.(digest); // right away, before the caller starts its workers
  return Promise.resolve({ pool, digest });
}
