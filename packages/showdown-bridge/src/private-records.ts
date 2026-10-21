// The reader of the PrivateTeamRecord contract (ADR-015 sections 2 and 5). The
// schema (`packages/schema/contracts/v1/private-team-record.schema.json`) holds
// the shape; this module holds what the contract profile cannot say: sorted
// and unique lists, hashes recomputed from the paste, revision rules and the
// shape of the hold-out flag.
//
// Nothing here holds a key. The hold-out flag is derived from a snapshot's
// keyed HMAC of `team_id` (ADR-015 section 2); the snapshot reader, which has
// the key, recomputes it. This reader checks only that it is a boolean. The
// paste is team content, so a failure message never repeats it.
//
// record_hash is SHA-256 of the JCS bytes of the whole PrivateTeamRecord
// (`recordHash`).

import { ajvFor } from "@areneon/schema/ajv";
import privateTeamRecordSchema from "@areneon/schema/contracts/v1/private-team-record.schema.json" with { type: "json" };
import type { PrivateTeamRecord } from "@areneon/schema/types/v1/private-team-record.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { teamHash } from "./team-class.ts";
import { canonicalTeamHash } from "./team-records.ts";
import { exportPaste, importPaste, TeamImportError } from "./teams.ts";

export type { PrivateTeamRecord };

export class PrivateRecordError extends Error {
  override name = "PrivateRecordError";
}

const bad = (message: string): never => {
  throw new PrivateRecordError(message);
};

const validators = {
  record: ajvFor("contract").compile(privateTeamRecordSchema),
};

function parsed<T>(validate: ((value: unknown) => boolean) & { errors?: unknown }, value: unknown, what: string): T {
  // Ajv's errors name schema paths and keywords, never the offending value, unless verbose is on (it is not).
  if (!validate(value)) return bad(`invalid ${what}: ${JSON.stringify(validate.errors)}`);
  return value as T;
}

/** `values` must be strictly ascending: sorted by code unit, no repeat. */
function ordered(values: readonly string[], what: string): void {
  values.forEach((v, k) => {
    if (k > 0 && !((values[k - 1] as string) < v)) bad(`${what}: a value is repeated or out of order`);
  });
}

/**
 * Parse a PrivateTeamRecord: the schema, then the engine's round trip of the
 * paste (it must be exactly the engine's own export), both team hashes
 * recomputed from the imported sets, revision 1 having no predecessor and
 * every later revision having one, tags sorted and unique, and unique label
 * ids. Legality is not judged here: the engine's validator decides that when
 * a record is committed.
 */
export function readPrivateTeamRecord(engine: Engine, value: unknown): PrivateTeamRecord {
  const record = parsed<PrivateTeamRecord>(validators.record, value, "private team record");
  let sets: PokemonSet[];
  try {
    sets = importPaste(engine, record.paste);
  } catch (error) {
    if (!(error instanceof TeamImportError)) throw error;
    return bad("the paste is refused by the pinned engine");
  }
  if (exportPaste(engine, sets) !== record.paste) bad("the paste is not the engine's own export of its sets");
  if (teamHash(sets) !== record.team_hash) bad("team_hash is not the hash of the paste's sets in paste order");
  if (canonicalTeamHash(sets) !== record.canonical_team_hash) bad("canonical_team_hash is not the hash of the paste's sets in canonical order");
  if ((record.revision === 1) !== (record.supersedes_record_hash === null)) {
    bad("supersedes_record_hash is null exactly for revision 1");
  }
  ordered(record.tags, "tags");
  if (new Set(record.labels.map((l) => l.label_id)).size !== record.labels.length) bad("label ids: a value is repeated");
  return record;
}

/**
 * The rules over the records of one snapshot (ADR-015 section 3): `team_id` is
 * unique, and `canonical_team_hash` is unique among the active records. A
 * retired record may share its team's hash.
 */
export function checkPrivateRecordSet(records: readonly PrivateTeamRecord[]): void {
  const ids = new Set<string>();
  const active = new Set<string>();
  for (const record of records) {
    if (ids.has(record.team_id)) bad(`team_id ${record.team_id} is in the snapshot twice`);
    ids.add(record.team_id);
    if (record.status !== "active") continue;
    if (active.has(record.canonical_team_hash)) bad(`canonical_team_hash ${record.canonical_team_hash} is held by two active records`);
    active.add(record.canonical_team_hash);
  }
}
