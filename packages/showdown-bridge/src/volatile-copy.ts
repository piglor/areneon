// Which volatile statuses a Baton Pass or Shed Tail switch hands to the incoming Pokemon (#1102).
// The engine's `Pokemon.copyVolatileFrom` copies every volatile of the outgoing Pokemon except the
// ones whose condition data says `noCopy`, and a Shed Tail switch copies Substitute alone. The
// protocol prints no line for the copy, so the readers must know the rule: this module reads it
// from the pinned engine (the mod's conditions, moves, abilities and items) and writes it to a
// reviewed, hashed data file that the Python reducer and the shared interpreter both load. No
// list of volatiles is written by hand; the Shed Tail rule is a literal in the engine's source,
// so the export refuses a checkout whose `copyVolatileFrom` text no longer says it.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { jcs, type Json } from "./canonical-json.ts";
import { DEFAULT_SHOWDOWN_DIR, type Engine, EngineError } from "./engine.ts";

export class VolatileCopyError extends Error {
  override name = "VolatileCopyError";
}

/** The data file's directory: package data of the Python reducer, loaded by the interpreter's wrapper too. */
export const VOLATILE_COPY_DIR = fileURLToPath(new URL("../../../ml/areneon_ml/reducer/data", import.meta.url));
export const volatileCopyFile = (formatId: string): string => `volatile-copy.${formatId}.json`;

export interface VolatileCopy {
  schema_version: 1;
  showdown_commit: string;
  format_id: string;
  /** The engine source the rule was read from, and its SHA-256. */
  source_path: string;
  source_sha256: string;
  /** Ids of the conditions with `noCopy`: a Baton Pass leaves them behind. Every other volatile is copied. */
  not_copied: string[];
  /** What a Shed Tail switch copies and nothing else. */
  shed_tail_copies: string[];
  /** SHA-256 of the JCS bytes of the other fields. */
  volatile_copy_hash: string;
}

interface RawDex {
  conditions: { get(id: string): unknown };
  data: {
    Conditions: Record<string, unknown>;
    Moves: Record<string, { condition?: unknown }>;
    Abilities: Record<string, { condition?: unknown }>;
    Items: Record<string, { condition?: unknown }>;
  };
}

const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** The ids of every condition the mod defines: its condition table and the `condition` of its moves, abilities and items. Sorted. */
export function conditionIds(dex: Pick<RawDex, "data">): string[] {
  const ids = new Set<string>([
    ...Object.keys(dex.data.Conditions),
    ...Object.entries(dex.data.Moves).filter(([, e]) => e.condition !== undefined).map(([id]) => id),
    ...Object.entries(dex.data.Abilities).filter(([, e]) => e.condition !== undefined).map(([id]) => id),
    ...Object.entries(dex.data.Items).filter(([, e]) => e.condition !== undefined).map(([id]) => id),
  ]);
  return [...ids].sort(cmp);
}
interface RawSim {
  Dex: { formats: { get(id: string): { exists: boolean; mod: string } }; mod(name: string): RawDex };
}

const SOURCE_PATH = "sim/pokemon.ts";
const SHED_TAIL_TEXT = "if (switchCause === 'shedtail' && i !== 'substitute') continue;";
const NO_COPY_TEXT = "if (this.battle.dex.conditions.getByID(i as ID).noCopy) continue;";

const sha256 = (text: string): string => createHash("sha256").update(text).digest("hex");

export function volatileCopyHash(copy: Omit<VolatileCopy, "volatile_copy_hash">): string {
  return sha256(jcs(copy as unknown as Json));
}

/** Read the copy rule of `formatId`'s mod from the pinned engine (`dir` defaults to the pinned checkout). */
export function exportVolatileCopy(engine: Engine, formatId: string, dir: string = DEFAULT_SHOWDOWN_DIR): VolatileCopy {
  const sim = engine.sim as unknown as RawSim;
  const format = sim.Dex.formats.get(formatId);
  if (!format.exists) throw new EngineError(`unknown format ${formatId}`);
  const source = readFileSync(join(dir, SOURCE_PATH), "utf8");
  if (!source.includes(SHED_TAIL_TEXT) || !source.includes(NO_COPY_TEXT)) {
    throw new VolatileCopyError("the engine's copyVolatileFrom no longer reads as the rule this export records");
  }
  const dex = sim.Dex.mod(format.mod);
  const notCopied = conditionIds(dex)
    .filter((id) => (dex.conditions.get(id) as { noCopy?: unknown }).noCopy === true)
    .sort(cmp);
  const body = {
    schema_version: 1 as const,
    showdown_commit: engine.commit,
    format_id: formatId,
    source_path: SOURCE_PATH,
    source_sha256: sha256(source),
    not_copied: notCopied,
    shed_tail_copies: ["substitute"],
  };
  return { ...body, volatile_copy_hash: volatileCopyHash(body) };
}

const KEYS = [
  "format_id",
  "not_copied",
  "schema_version",
  "shed_tail_copies",
  "showdown_commit",
  "source_path",
  "source_sha256",
  "volatile_copy_hash",
];
const bad = (message: string): never => {
  throw new VolatileCopyError(message);
};
function ids(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || value.some((v) => typeof v !== "string" || !/^[a-z0-9]+$/.test(v))) bad(`${what}: not a list of ids`);
  const list = value as string[];
  if (list.some((v, i) => i > 0 && cmp(list[i - 1] as string, v) >= 0)) bad(`${what}: not sorted and unique`);
  return list;
}

/** Read a data file: the exact key set, sorted unique ids, and a hash that matches the rest. */
export function readVolatileCopy(value: unknown): VolatileCopy {
  if (typeof value !== "object" || value === null || Array.isArray(value)) bad("not an object");
  const raw = value as Record<string, unknown>;
  if (Object.keys(raw).sort(cmp).join() !== KEYS.join()) bad("unexpected key set");
  if (raw.schema_version !== 1) bad("unsupported schema_version");
  for (const key of ["format_id", "showdown_commit", "source_path", "source_sha256", "volatile_copy_hash"] as const) {
    if (typeof raw[key] !== "string" || raw[key] === "") bad(`${key}: not a string`);
  }
  const copy: VolatileCopy = {
    schema_version: 1,
    showdown_commit: raw.showdown_commit as string,
    format_id: raw.format_id as string,
    source_path: raw.source_path as string,
    source_sha256: raw.source_sha256 as string,
    not_copied: ids(raw.not_copied, "not_copied"),
    shed_tail_copies: ids(raw.shed_tail_copies, "shed_tail_copies"),
    volatile_copy_hash: raw.volatile_copy_hash as string,
  };
  const { volatile_copy_hash: hash, ...body } = copy;
  if (volatileCopyHash(body) !== hash) bad("hash does not match the content");
  return copy;
}
