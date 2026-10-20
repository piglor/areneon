// The private team store's share of the sample corpus (ADR-008 §2, ADR-014
// sections 3 and 4). Synthetic only: the pastes are the committed synthetic
// team corpus (`fixtures/teams/`), the keys are two fixed test seeds (a seed
// is 32 repeated bytes and appears nowhere in the output; only the public
// keys and the signatures do).
//
// A PrivateTeamRecord sample says `data_class: private`, so each such file is
// listed in `PRIVATE_SAMPLE_MARKERS` (scripts/check_data_class.py), which also
// requires its paste to be a corpus paste and its shape to be exactly a
// record. The guard therefore cannot hold an invalid sample that carries the
// private label with a broken field, and an `invalid` sample here says
// `data_class: synthetic` instead (itself a violation of the contract's
// const). A test flips the label back and requires the sample to stay
// invalid, so each one is refused for its own fault.

import { createPrivateKey, createPublicKey, sign } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Engine } from "./engine.ts";
import { TEAMS_DIR } from "./golden.ts";
import { keyIdOf, privateHeadHash, privateHeadSigningBytes, type SignedStatement } from "./private-records.ts";
import { canonicalTeamHash, recordHash } from "./team-records.ts";
import { teamHash } from "./team-class.ts";
import { importPaste } from "./teams.ts";

type Cases = Record<string, unknown>;

/** The corpus pastes the record samples carry. */
export const PRIVATE_SAMPLE_PASTES = ["legal-synthetic.txt", "legal-opponent.txt"] as const;

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

const hexOf = (c: string): string => c.repeat(64);
const PKCS8_ED25519_PREFIX = Buffer.from("302e020100300506032b657004220420", "hex");

/** An Ed25519 test key from a 32-byte seed of one repeated byte. */
function testKey(byte: number): { publicKey: string; sign: (message: Uint8Array) => string } {
  const key = createPrivateKey({ key: Buffer.concat([PKCS8_ED25519_PREFIX, Buffer.alloc(32, byte)]), format: "der", type: "pkcs8" });
  const publicKey = createPublicKey(key).export({ format: "der", type: "spki" }).subarray(-32).toString("hex");
  return { publicKey, sign: (message) => sign(null, message, key).toString("hex") };
}

export const SAMPLE_KEYS = { a: testKey(1), b: testKey(2) } as const;

type Body = SignedStatement["body"];

/** A private head signed by the given keys, in key_id order. */
function head(body: Body, signers: readonly (typeof SAMPLE_KEYS)[keyof typeof SAMPLE_KEYS][]): SignedStatement {
  const statement = { schema_version: 1, kind: "private-head", body, signatures: [] } as unknown as SignedStatement;
  const bytes = privateHeadSigningBytes(statement);
  statement.signatures = signers
    .map((k) => ({ key_id: keyIdOf(k.publicKey), public_key: k.publicKey, signature: k.sign(bytes) }))
    .sort((x, y) => Number(x.key_id > y.key_id) - Number(x.key_id < y.key_id)) as SignedStatement["signatures"];
  return statement;
}

/** The record samples' `valid` and `reader-invalid` cases carry the private label; `invalid` ones the synthetic one. */
export function privateSamples(engine: Engine): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const pastes = PRIVATE_SAMPLE_PASTES.map((file) => readFileSync(join(TEAMS_DIR, file), "utf8")) as [string, string];
  const hashesOf = (paste: string): { team_hash: string; canonical_team_hash: string } => {
    const sets = importPaste(engine, paste);
    return { team_hash: teamHash(sets), canonical_team_hash: canonicalTeamHash(sets) };
  };
  const first = {
    schema_version: 1,
    data_class: "private",
    team_id: "00112233445566778899aabbccddeeff",
    revision: 1,
    supersedes_record_hash: null,
    ...hashesOf(pastes[0]),
    paste: pastes[0],
    status: "active",
    tags: [],
    labels: [],
  };
  const edited = {
    ...first,
    revision: 2,
    supersedes_record_hash: recordHash(first),
    ...hashesOf(pastes[1]),
    paste: pastes[1],
    tags: ["synthetic", "vgc"],
    labels: [{ label_id: "archetype", value: "offense" }],
  };
  const retired = { ...edited, revision: 3, supersedes_record_hash: recordHash(edited), status: "retired" };
  const withRecord = (change: (r: Record<string, unknown>) => void): Record<string, unknown> => mutated(edited, change);
  const synthetic = (change: (r: Record<string, unknown>) => void): Record<string, unknown> =>
    mutated(edited, (r) => {
      r.data_class = "synthetic";
      change(r);
    });

  const body1: Body = {
    seq: 1,
    parent_head_hash: null,
    issued_at: 1_760_000_000,
    snapshot_hash: hexOf("1"),
    logical_snapshot_hash: hexOf("2"),
    manifest_ciphertext_sha256: hexOf("3"),
    object_digests: [hexOf("4"), hexOf("5")],
    recipient_type: "mlkem768x25519",
    next_public_key: null,
  };
  const head1 = head(body1, [SAMPLE_KEYS.a]);
  const body2: Body = { ...body1, seq: 2, parent_head_hash: privateHeadHash(head1), issued_at: 1_760_000_060, snapshot_hash: hexOf("6"), object_digests: [hexOf("4"), hexOf("5"), hexOf("7")] };
  const head2 = head(body2, [SAMPLE_KEYS.a]);
  const body3: Body = { ...body2, seq: 3, parent_head_hash: privateHeadHash(head2), issued_at: 1_760_000_120, recipient_type: "x25519", next_public_key: SAMPLE_KEYS.b.publicKey };
  const rotation = head(body3, [SAMPLE_KEYS.a, SAMPLE_KEYS.b]);
  const headWith = (change: (s: Record<string, unknown>) => void): Record<string, unknown> => mutated(head2, change);
  const bodyOf = (s: Record<string, unknown>): Record<string, unknown> => s.body as Record<string, unknown>;
  const sigsOf = (s: Record<string, unknown>): Record<string, string>[] => s.signatures as Record<string, string>[];
  const rotationWith = (change: (s: Record<string, unknown>) => void): Record<string, unknown> => mutated(rotation, change);
  const [signatureA] = head2.signatures;

  return {
    valid: {
      "private-team-record/first": first,
      "private-team-record/edited": edited,
      "private-team-record/retired": retired,
      "signed-statement/head-1": head1,
      "signed-statement/head-2": head2,
      "signed-statement/rotation": rotation,
    },
    invalid: {
      "private-team-record/synthetic-class": synthetic(() => undefined),
      "private-team-record/extra-field": synthetic((r) => (r.rating = 1500)),
      "private-team-record/next-version": synthetic((r) => (r.schema_version = 2)),
      "private-team-record/no-team-id": synthetic((r) => delete r.team_id),
      "private-team-record/short-team-id": synthetic((r) => (r.team_id = "0".repeat(31))),
      "private-team-record/uppercase-team-id": synthetic((r) => (r.team_id = "A".repeat(32))),
      "private-team-record/revision-zero": synthetic((r) => (r.revision = 0)),
      "private-team-record/revision-over-max": synthetic((r) => (r.revision = 2 ** 31)),
      "private-team-record/fractional-revision": synthetic((r) => (r.revision = 1.5)),
      "private-team-record/no-supersedes": synthetic((r) => delete r.supersedes_record_hash),
      "private-team-record/short-supersedes": synthetic((r) => (r.supersedes_record_hash = "abc")),
      "private-team-record/uppercase-team-hash": synthetic((r) => (r.team_hash = hexOf("A"))),
      "private-team-record/empty-paste": synthetic((r) => (r.paste = "")),
      "private-team-record/paste-too-long": synthetic((r) => (r.paste = "x".repeat(8193))),
      "private-team-record/unknown-status": synthetic((r) => (r.status = "archived")),
      "private-team-record/too-many-tags": synthetic((r) => (r.tags = Array.from({ length: 17 }, (_, i) => `tag-${String.fromCharCode(97 + i)}`))),
      "private-team-record/uppercase-tag": synthetic((r) => (r.tags = ["Vgc"])),
      "private-team-record/long-tag": synthetic((r) => (r.tags = ["a".repeat(33)])),
      "private-team-record/too-many-labels": synthetic((r) => (r.labels = Array.from({ length: 65 }, (_, i) => ({ label_id: `l${i}`, value: "yes" })))),
      "private-team-record/label-extra-field": synthetic((r) => (r.labels = [{ label_id: "archetype", value: "offense", weight: 1 }])),
      "signed-statement/extra-field": headWith((s) => (s.notes = "x")),
      "signed-statement/next-version": headWith((s) => (s.schema_version = 2)),
      "signed-statement/run-kind": headWith((s) => (s.kind = "run-result")),
      "signed-statement/no-signatures": headWith((s) => (s.signatures = [])),
      "signed-statement/three-signatures": headWith((s) => (s.signatures = [signatureA, signatureA, signatureA])),
      "signed-statement/short-signature": headWith((s) => (sigsOf(s)[0] as Record<string, string>).signature = "ab".repeat(63)),
      "signed-statement/short-key-id": headWith((s) => (sigsOf(s)[0] as Record<string, string>).key_id = "abc"),
      "signed-statement/uppercase-public-key": headWith((s) => (sigsOf(s)[0] as Record<string, string>).public_key = hexOf("A")),
      "signed-statement/signature-extra-field": headWith((s) => (sigsOf(s)[0] as Record<string, unknown>).algorithm = "ed25519"),
      "signed-statement/no-body": headWith((s) => delete s.body),
      "signed-statement/seq-zero": headWith((s) => (bodyOf(s).seq = 0)),
      "signed-statement/fractional-seq": headWith((s) => (bodyOf(s).seq = 1.5)),
      "signed-statement/negative-issued-at": headWith((s) => (bodyOf(s).issued_at = -1)),
      "signed-statement/short-parent": headWith((s) => (bodyOf(s).parent_head_hash = "abc")),
      "signed-statement/unknown-recipient-type": headWith((s) => (bodyOf(s).recipient_type = "scrypt")),
      "signed-statement/short-next-public-key": headWith((s) => (bodyOf(s).next_public_key = "abc")),
      "signed-statement/body-extra-field": headWith((s) => (bodyOf(s).domain = "areneon/private-head/v1")),
    },
    readerInvalid: {
      "private-team-record/team-hash-mismatch": withRecord((r) => (r.team_hash = hexOf("a"))),
      "private-team-record/canonical-hash-mismatch": withRecord((r) => (r.canonical_team_hash = hexOf("a"))),
      "private-team-record/first-revision-supersedes": mutated(first, (r) => (r.supersedes_record_hash = hexOf("a"))),
      "private-team-record/later-revision-no-predecessor": withRecord((r) => (r.supersedes_record_hash = null)),
      "private-team-record/unsorted-tags": withRecord((r) => (r.tags = ["vgc", "synthetic"])),
      "private-team-record/duplicate-tags": withRecord((r) => (r.tags = ["vgc", "vgc"])),
      "private-team-record/duplicate-label-ids": withRecord((r) => (r.labels = [{ label_id: "archetype", value: "offense" }, { label_id: "archetype", value: "defense" }])),
      "signed-statement/unsorted-signatures": rotationWith((s) => sigsOf(s).reverse()),
      "signed-statement/duplicate-signatures": rotationWith((s) => (s.signatures = [sigsOf(s)[0], sigsOf(s)[0]])),
      "signed-statement/key-id-mismatch": headWith((s) => (sigsOf(s)[0] as Record<string, string>).key_id = hexOf("a")),
      "signed-statement/unsorted-object-digests": headWith((s) => (bodyOf(s).object_digests = [hexOf("7"), hexOf("4")])),
      "signed-statement/duplicate-object-digests": headWith((s) => (bodyOf(s).object_digests = [hexOf("4"), hexOf("4")])),
      "signed-statement/first-head-has-parent": mutated(head1, (s) => (bodyOf(s).parent_head_hash = hexOf("a"))),
      "signed-statement/later-head-no-parent": headWith((s) => (bodyOf(s).parent_head_hash = null)),
      "signed-statement/rotation-one-signature": rotationWith((s) => (s.signatures = [sigsOf(s)[0]])),
      "signed-statement/two-signatures-no-rotation": rotationWith((s) => (bodyOf(s).next_public_key = null)),
      "signed-statement/rotation-target-not-a-signer": rotationWith((s) => (bodyOf(s).next_public_key = hexOf("c"))),
    },
  };
}
