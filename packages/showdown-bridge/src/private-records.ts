// The readers of the private team store's contracts (ADR-014 sections 3 and
// 4): PrivateTeamRecord and the `private-head` SignedStatement. The schemas
// (`packages/schema/contracts/v1/{private-team-record,signed-statement}.schema.json`)
// hold the shape; this module holds what the contract profile cannot say:
// sorted and unique lists, hashes recomputed from the paste, the key_id rule
// and the signature acceptance rule.
//
// Nothing here holds a key, an identity or a ciphertext. It does not verify a
// signature: the crypto core does that, then hands `acceptHeadSignatures` one
// flag per signature. So an auditor can use this module without the private
// store's dependencies. The paste is team content, so a failure message never
// repeats it.
//
// Identities, all SHA-256 of RFC 8785 (JCS) bytes:
//   record_hash   the whole PrivateTeamRecord (`recordHash`)
//   head hash     the whole SignedStatement, signatures included
//   signed bytes  {"domain", "schema_version", "body"} of a head

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import privateTeamRecordSchema from "@areneon/schema/contracts/v1/private-team-record.schema.json" with { type: "json" };
import signedStatementSchema from "@areneon/schema/contracts/v1/signed-statement.schema.json" with { type: "json" };
import type { PrivateTeamRecord } from "@areneon/schema/types/v1/private-team-record.ts";
import type { SignedStatement } from "@areneon/schema/types/v1/signed-statement.ts";
import { jcs, type Json } from "./canonical-json.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { teamHash } from "./team-class.ts";
import { canonicalTeamHash, recordHash } from "./team-records.ts";
import { exportPaste, importPaste, TeamImportError } from "./teams.ts";

export type { PrivateTeamRecord, SignedStatement };

export class PrivateRecordError extends Error {
  override name = "PrivateRecordError";
}

const bad = (message: string): never => {
  throw new PrivateRecordError(message);
};

/** The domain string signed with every private head (ADR-014 section 3). */
export const PRIVATE_HEAD_DOMAIN = "areneon/private-head/v1";

const validators = {
  record: ajvFor("contract").compile(privateTeamRecordSchema),
  statement: ajvFor("contract").compile(signedStatementSchema),
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
 * The rules over the records of one snapshot (ADR-014 section 4): `team_id` is
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

/** The key id of a public key: SHA-256 of its raw bytes (not of the hex text), as hex. */
export const keyIdOf = (publicKey: string): string => createHash("sha256").update(Buffer.from(publicKey, "hex")).digest("hex");

/** The bytes each signature of a private head covers: the JCS of its domain, schema version and body. */
export const privateHeadSigningBytes = (statement: SignedStatement): Uint8Array =>
  Buffer.from(jcs({ domain: PRIVATE_HEAD_DOMAIN, schema_version: statement.schema_version, body: statement.body } as unknown as Json), "utf8");

/** A head's hash, as the next head's `parent_head_hash` names it: SHA-256 of the whole statement's JCS bytes. */
export const privateHeadHash = (statement: SignedStatement): string => recordHash(statement);

/**
 * Parse a SignedStatement (a private head): the schema, then signatures sorted
 * by key_id with no repeat, each key_id the SHA-256 of its public key,
 * object_digests sorted and unique, a parent exactly when seq is above 1, and
 * the signature count that next_public_key calls for: one, or two when it is
 * set, one of them by that key. Which key may sign is the pin's decision
 * (`acceptHeadSignatures`); no signature is verified here.
 */
export function readSignedStatement(value: unknown): SignedStatement {
  const statement = parsed<SignedStatement>(validators.statement, value, "signed statement");
  const { body, signatures } = statement;
  ordered(signatures.map((s) => s.key_id), "signatures");
  for (const signature of signatures) {
    if (keyIdOf(signature.public_key) !== signature.key_id) bad(`key_id ${signature.key_id} is not the SHA-256 of its public_key`);
  }
  ordered(body.object_digests, "object_digests");
  if ((body.seq === 1) !== (body.parent_head_hash === null)) bad("parent_head_hash is null exactly for seq 1");
  if (body.next_public_key === null) {
    if (signatures.length !== 1) bad("a head without next_public_key carries exactly one signature");
  } else if (signatures.length !== 2 || !signatures.some((s) => s.public_key === body.next_public_key)) {
    bad("a head with next_public_key carries two signatures, one by that key");
  }
  return statement;
}

/**
 * The signature acceptance rule (ADR-014 section 3), over signatures a verifier
 * already checked. `verified[k]` says that `statement.signatures[k]` is a valid
 * signature over `privateHeadSigningBytes` by its `public_key`. A statement
 * must come from `readSignedStatement`. Refuses unless every signature
 * verified and either one signature is by `pinnedPublicKey` and the head
 * rotates nothing, or two are and `next_public_key` is the second signer's.
 * Returns the key the pin holds afterwards.
 */
export function acceptHeadSignatures(statement: SignedStatement, pinnedPublicKey: string, verified: readonly boolean[]): string {
  const { signatures, body } = statement;
  if (verified.length !== signatures.length) bad("one verification result per signature is needed");
  if (!verified.every(Boolean)) bad("a signature does not verify");
  const signers = signatures.map((s) => s.public_key);
  if (body.next_public_key === null) {
    if (signers.length !== 1 || signers[0] !== pinnedPublicKey) bad("the head is not signed by exactly the pinned key");
    return pinnedPublicKey;
  }
  const rotating = signers.length === 2 && signers.includes(pinnedPublicKey) && signers.includes(body.next_public_key);
  if (!rotating || pinnedPublicKey === body.next_public_key) {
    bad("a rotation head needs exactly the pinned key and next_public_key, both signing");
  }
  return body.next_public_key;
}
