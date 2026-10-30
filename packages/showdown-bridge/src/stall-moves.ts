// Which moves the engine treats as stalling moves (the Protect family, Endure, Quick Guard, Wide Guard: the moves
// that add the `stall` volatile), for the Python reducer's Protect streak (#1013). The protocol never names the
// volatile, so a reader that counts consecutive successful uses must know the moves. They are read from the pinned
// engine's own move source by the one reader the world facts use (`stallMoves`), restricted to the format's moves,
// and written to a reviewed, hashed data file (package data of the reducer) that a test re-exports byte for byte.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { jcs, type Json } from "./canonical-json.ts";
import { DEFAULT_SHOWDOWN_DIR, type Engine } from "./engine.ts";
import { type FactsTable, stallMoves } from "./world-facts-data.ts";

export class StallMovesError extends Error {
  override name = "StallMovesError";
}

export { VOLATILE_COPY_DIR as STALL_MOVES_DIR } from "./volatile-copy.ts";
export const stallMovesFile = (formatId: string): string => `stall-moves.${formatId}.json`;

export interface StallMoves {
  schema_version: 1;
  showdown_commit: string;
  format_id: string;
  /** The engine source the list was read from, and its SHA-256. */
  source_path: string;
  source_sha256: string;
  /** Ids of the format's moves that add the `stall` volatile, sorted. */
  stall_moves: string[];
  /** SHA-256 of the JCS bytes of the other fields. */
  stall_moves_hash: string;
}

const SOURCE_PATH = "data/moves.ts";
const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

export function stallMovesHash(body: Omit<StallMoves, "stall_moves_hash">): string {
  return sha256(jcs(body as unknown as Json));
}

/** Read the stall moves of `formatId` from the pinned engine; `table` is the format's engine-data table. */
export function exportStallMoves(engine: Engine, formatId: string, table: FactsTable, dir: string = DEFAULT_SHOWDOWN_DIR): StallMoves {
  const body = {
    schema_version: 1 as const,
    showdown_commit: engine.commit,
    format_id: formatId,
    source_path: SOURCE_PATH,
    source_sha256: sha256(readFileSync(join(dir, SOURCE_PATH), "utf8")),
    stall_moves: stallMoves(dir, table),
  };
  if (body.stall_moves.length === 0) throw new StallMovesError("the engine's move source names no stalling move");
  return { ...body, stall_moves_hash: stallMovesHash(body) };
}

const KEYS = ["format_id", "schema_version", "showdown_commit", "source_path", "source_sha256", "stall_moves", "stall_moves_hash"];
const bad = (message: string): never => {
  throw new StallMovesError(message);
};

/** Read a data file: the exact key set, sorted unique ids, and a hash that matches the rest. */
export function readStallMoves(value: unknown): StallMoves {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad("not an object");
  const raw = value as Record<string, unknown>;
  if (Object.keys(raw).sort().join() !== KEYS.join()) bad("unexpected key set");
  if (raw.schema_version !== 1) bad("unsupported schema_version");
  for (const key of ["format_id", "showdown_commit", "source_path", "source_sha256", "stall_moves_hash"] as const) {
    if (typeof raw[key] !== "string" || raw[key] === "") bad(`${key}: not a string`);
  }
  const list = raw.stall_moves;
  if (!Array.isArray(list) || list.some((v) => typeof v !== "string" || !/^[a-z0-9]+$/.test(v))) bad("stall_moves: not a list of ids");
  const ids = list as string[];
  if (ids.some((v, i) => i > 0 && (ids[i - 1] as string) >= v)) bad("stall_moves: not sorted and unique");
  const table: StallMoves = {
    schema_version: 1,
    showdown_commit: raw.showdown_commit as string,
    format_id: raw.format_id as string,
    source_path: raw.source_path as string,
    source_sha256: raw.source_sha256 as string,
    stall_moves: ids,
    stall_moves_hash: raw.stall_moves_hash as string,
  };
  const { stall_moves_hash: hash, ...body } = table;
  if (stallMovesHash(body) !== hash) bad("hash does not match the content");
  return table;
}
