// The team pool, loaded once and shared with the further game processes of a generation (#1192, #980).
//
// Every game process used to load and normalise the pool itself (the dataset is read, validated and each
// team normalised by the pinned validator: about 34 s at full size, repeated S times in parallel). Now the
// process that starts the others loads it once and leaves it in the scratch directory they share (a
// private 0700 directory this process made); the others read it back instead. Operational only: the pool
// is the same value (same teams, same order, same key order), so no seed, plan or hash moves.
//
// A pool read back is not taken on trust beyond that: its request key must be the one derived from this
// process's own run config (so a file of another format, dataset or excluded-ability list is refused),
// its hash list must be sorted and unique, and every team must hash to the hash listed for it. The
// `known_synthetic` set is rebuilt from the verified hash list and nothing else.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { type Json, jcs } from "./canonical-json.ts";
import { GenerationError, type PoolRequest, type TrainingPool } from "./league-generation.ts";
import type { PokemonSet } from "./engine.ts";
import { syntheticTeamHashes, teamHash } from "./team-class.ts";

/** The version of the pool file's layout. */
export const POOL_FILE_VERSION = 1;

/** The key a pool file is bound to: SHA-256 of the JCS of what the pool is built from (format, dataset, excluded abilities). */
export function poolRequestKey(request: PoolRequest): string {
  const named = { format_id: request.format_id, team_dataset: request.team_dataset ?? null, excluded_abilities: [...request.generation.excluded_abilities] };
  return createHash("sha256").update(jcs(named as unknown as Json)).digest("hex");
}

/** Write the pool for the other game processes: to a temporary name first, then renamed, so a reader never sees half a file. Plain JSON, so the sets keep their key order. */
export function writePoolFile(path: string, pool: TrainingPool, request: PoolRequest): void {
  const temporary = `${path}.partial`;
  writeFileSync(temporary, JSON.stringify({ schema_version: POOL_FILE_VERSION, request: poolRequestKey(request), hashes: pool.hashes, teams: pool.teams }), { mode: 0o600 });
  renameSync(temporary, path);
}

/** Read a pool file back and check it as the module comment says; a refusal is a `GenerationError`. */
export function readPoolFile(path: string, request: PoolRequest): TrainingPool {
  let value: { schema_version?: unknown; request?: unknown; hashes?: unknown; teams?: unknown };
  try {
    value = JSON.parse(readFileSync(path, "utf8")) as typeof value;
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

/** Wait for the pool file to appear (the starting process writes it when its own load is done), then read it. */
export async function waitForPoolFile(path: string, request: PoolRequest, { poll_ms = 100, timeout_ms = 30 * 60 * 1000 }: { poll_ms?: number; timeout_ms?: number } = {}): Promise<TrainingPool> {
  const deadline = Date.now() + timeout_ms;
  while (!existsSync(path)) {
    if (Date.now() > deadline) throw new GenerationError("the shared team pool did not appear in time");
    await new Promise((resolve) => setTimeout(resolve, poll_ms));
  }
  return readPoolFile(path, request);
}
