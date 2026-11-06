// The verified input of the situational move-use diagnostic (#1283): a committed progress generation read as the Python
// move-use diagnostic reads it. The generation's `manifest.json` must be that generation's and list exactly the SHA-256 of
// every other file under it; `progress.json` is read by the progress record reader of its schema version; every stored game
// goes through `readProgressGameRecord` (schema, the choice among its candidates, team hashes, the omniscient log hash) and
// must belong to the record's run, generation and a listed match. Anything else is refused, never skipped.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type ProgressEvaluationRecord, type ProgressEvaluationRecordV2, readProgressEvaluationRecord, readProgressEvaluationRecordV2, readProgressGameRecord } from "./progress-records.ts";
import type { StoredGame } from "./situational-move-use.ts";

export class ProgressInputError extends Error {
  override name = "ProgressInputError";
}

const MANIFEST = "manifest.json";
const sha256File = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");

/** SHA-256 of every file under `root` but the manifest, by relative path (the loop layout's `hash_tree`). */
function hashTree(root: string, base = ""): Record<string, string> {
  const out: Record<string, string> = {};
  for (const entry of readdirSync(join(root, base), { withFileTypes: true })) {
    const relative = base === "" ? entry.name : `${base}/${entry.name}`;
    if (entry.isDirectory()) Object.assign(out, hashTree(root, relative));
    else if (entry.isFile() && entry.name !== MANIFEST) out[relative] = sha256File(join(root, relative));
  }
  return out;
}

const json = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));

export interface VerifiedGeneration {
  record: ProgressEvaluationRecord | ProgressEvaluationRecordV2;
  /** The stored games, in match then game order. */
  games: (StoredGame & { run_id: string; evaluation_key?: string })[];
}

/** Read and verify the generation directory `<progress root>/g<NNNNNN>`; `matches` keeps only those match ids. */
export function readVerifiedGeneration(dir: string, generation: number, matches?: readonly string[]): VerifiedGeneration {
  const manifest = json(join(dir, MANIFEST)) as { generation?: unknown; files?: unknown };
  if (manifest.generation !== generation) throw new ProgressInputError("the manifest is not this generation's");
  const found = hashTree(dir);
  const listed = (manifest.files ?? {}) as Record<string, string>;
  const names = Object.keys(found).sort();
  if (names.join("\n") !== Object.keys(listed).sort().join("\n") || names.some((name) => found[name] !== listed[name])) throw new ProgressInputError("the files differ from the manifest");
  const raw = json(join(dir, "progress.json")) as { schema_version?: unknown };
  const record = raw.schema_version === 2 ? readProgressEvaluationRecordV2(raw) : readProgressEvaluationRecord(raw);
  if (record.generation !== generation) throw new ProgressInputError("progress.json is not this generation's");
  const known = new Set(record.matches.map((m) => m.match_id));
  const games = readdirSync(join(dir, "games"))
    .filter((name) => name.endsWith(".json"))
    .map((name) => readProgressGameRecord(json(join(dir, "games", name))))
    .filter((game) => matches === undefined || matches.includes(game.match_id));
  for (const game of games) {
    if (game.run_id !== record.run_id || game.generation !== generation) throw new ProgressInputError("a stored game is not of this run and generation");
    if (!known.has(game.match_id)) throw new ProgressInputError("a stored game names a match the record lacks");
  }
  games.sort((a, b) => (a.match_id < b.match_id ? -1 : a.match_id > b.match_id ? 1 : a.game - b.game));
  return { record, games: games as unknown as VerifiedGeneration["games"] };
}
