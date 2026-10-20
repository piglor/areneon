import { createPublicKey, verify } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import privateTeamRecordSchema from "@areneon/schema/contracts/v1/private-team-record.schema.json" with { type: "json" };
import signedStatementSchema from "@areneon/schema/contracts/v1/signed-statement.schema.json" with { type: "json" };
import teamRecordSchema from "@areneon/schema/contracts/v1/team-record.schema.json" with { type: "json" };
import { describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { syntheticFixtureHashes, TEAMS_DIR } from "../src/golden.ts";
import {
  acceptHeadSignatures,
  checkPrivateRecordSet,
  keyIdOf,
  PRIVATE_HEAD_DOMAIN,
  PrivateRecordError,
  privateHeadHash,
  privateHeadSigningBytes,
  type PrivateTeamRecord,
  readPrivateTeamRecord,
  readSignedStatement,
  type SignedStatement,
} from "../src/private-records.ts";
import { PRIVATE_SAMPLE_PASTES, SAMPLE_KEYS } from "../src/private-samples.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { teamHash } from "../src/team-class.ts";
import { importPaste } from "../src/teams.ts";

const REPO = join(SAMPLES_DIR, "..", "..");
const engine = loadEngine();
const sample = <T>(path: string): T => JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", path), "utf8")) as T;
const names = (contract: string, kind: string): string[] => readdirSync(join(SAMPLES_DIR, "v1", contract, kind)).sort();
const edit = <T>(value: T, change: (copy: T) => void): T => {
  const copy = structuredClone(value);
  change(copy);
  return copy;
};
const hex = (c: string): string => c.repeat(64);
const first = sample<PrivateTeamRecord>("private-team-record/valid/first.json");
const edited = sample<PrivateTeamRecord>("private-team-record/valid/edited.json");
const head1 = sample<SignedStatement>("signed-statement/valid/head-1.json");
const head2 = sample<SignedStatement>("signed-statement/valid/head-2.json");
const rotation = sample<SignedStatement>("signed-statement/valid/rotation.json");
const keyA = SAMPLE_KEYS.a.publicKey;
const keyB = SAMPLE_KEYS.b.publicKey;

describe("the private-sample markers (ADR-014 section 10)", () => {
  const source = readFileSync(join(REPO, "scripts/check_data_class.py"), "utf8");
  const markers = (/PRIVATE_SAMPLE_MARKERS = frozenset\(\s*\{([^}]*)\}/.exec(source)?.[1] ?? "")
    .split("\n")
    .map((line) => line.trim().replace(/^"|",?$/g, ""))
    .filter(Boolean);

  it("lists exactly the committed record samples that carry the private label", () => {
    const labelled = ["valid", "invalid", "reader-invalid"].flatMap((kind) =>
      names("private-team-record", kind)
        .filter((n) => sample<{ data_class: string }>(`private-team-record/${kind}/${n}`).data_class === "private")
        .map((n) => `fixtures/contracts/v1/private-team-record/${kind}/${n}`),
    );
    expect(markers.length).toBeGreaterThan(0);
    expect(markers.sort()).toEqual(labelled.sort());
  });

  it("round-trips every marked paste through importPaste, and its team_hash is a synthetic fixture hash", () => {
    const known = syntheticFixtureHashes(engine);
    const corpus = PRIVATE_SAMPLE_PASTES.map((file) => readFileSync(join(TEAMS_DIR, file), "utf8"));
    for (const marker of markers) {
      const record = JSON.parse(readFileSync(join(REPO, marker), "utf8")) as PrivateTeamRecord;
      expect(corpus, marker).toContain(record.paste);
      expect(known.has(teamHash(importPaste(engine, record.paste))), marker).toBe(true);
    }
  });

  it("keeps the guard's record keys equal to the schema's property set", () => {
    const keys = /PRIVATE_RECORD_KEYS = frozenset\(\s*\{([^}]*)\}/.exec(source)?.[1] ?? "";
    const guard = [...keys.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]).sort();
    expect(guard).toEqual(Object.keys(privateTeamRecordSchema.properties).sort());
    expect(guard).toEqual([...privateTeamRecordSchema.required].sort());
  });
});

describe("the PrivateTeamRecord contract", () => {
  const valid = ajvFor("contract").compile(privateTeamRecordSchema);

  it("takes its labels exactly as ADR-013's TeamRecord does", () => {
    expect(privateTeamRecordSchema.properties.labels).toEqual({ ...teamRecordSchema.properties.labels, description: expect.any(String) });
  });

  it("refuses each invalid sample for its own fault: with the private label put back it stays invalid", () => {
    for (const name of names("private-team-record", "invalid")) {
      const value = sample<Record<string, unknown>>(`private-team-record/invalid/${name}`);
      expect(valid(value), name).toBe(false);
      if (name !== "synthetic-class.json") expect(valid({ ...value, data_class: "private" }), name).toBe(false);
    }
    // The negative control: that one sample is invalid for its label alone.
    const only = sample<Record<string, unknown>>("private-team-record/invalid/synthetic-class.json");
    expect(valid({ ...only, data_class: "private" })).toBe(true);
  });

  it("bounds revision, tags and labels at the edges ADR-014 section 4 names", () => {
    const at = (change: Partial<PrivateTeamRecord>): boolean => valid({ ...first, ...change }) as boolean;
    expect(at({ revision: 2 ** 31 - 1 })).toBe(true);
    expect(at({ revision: 2 ** 31 })).toBe(false);
    expect(at({ tags: Array.from({ length: 16 }, (_, i) => `t${i}`) })).toBe(true);
    expect(at({ tags: ["a".repeat(32)] })).toBe(true);
    expect(at({ tags: ["-a"] })).toBe(false);
    expect(at({ labels: Array.from({ length: 64 }, (_, i) => ({ label_id: `l${i}`, value: "yes" })) })).toBe(true);
    expect(at({ paste: "x".repeat(8192) })).toBe(true);
  });
});

describe("readPrivateTeamRecord", () => {
  it("accepts every valid sample", () => {
    for (const name of names("private-team-record", "valid")) {
      expect(readPrivateTeamRecord(engine, sample(`private-team-record/valid/${name}`)).data_class).toBe("private");
    }
  });

  it.each([
    ["team-hash-mismatch", /team_hash is not the hash/],
    ["canonical-hash-mismatch", /canonical_team_hash is not the hash/],
    ["first-revision-supersedes", /null exactly for revision 1/],
    ["later-revision-no-predecessor", /null exactly for revision 1/],
    ["unsorted-tags", /tags: a value is repeated or out of order/],
    ["duplicate-tags", /tags: a value is repeated or out of order/],
    ["duplicate-label-ids", /label ids: a value is repeated/],
  ])("refuses private-team-record/%s", (name, message) => {
    const value = sample(`private-team-record/reader-invalid/${name}.json`);
    expect(() => readPrivateTeamRecord(engine, value)).toThrow(PrivateRecordError);
    expect(() => readPrivateTeamRecord(engine, value)).toThrow(message);
  });

  it("refuses a schema violation, a paste that is not the engine's export and one it rejects, without repeating the paste", () => {
    expect(() => readPrivateTeamRecord(engine, { ...first, status: "archived" })).toThrow(/invalid private team record/);
    expect(() => readPrivateTeamRecord(engine, edit(first, (r) => (r.paste = `${r.paste}\n\n`)))).toThrow(/not the engine's own export/);
    const unreproduced = edit(first, (r) => (r.paste = `${r.paste}\nSPs: 1 HP`));
    expect(() => readPrivateTeamRecord(engine, unreproduced)).toThrow("the paste is refused by the pinned engine");
    try {
      readPrivateTeamRecord(engine, unreproduced);
    } catch (error) {
      expect((error as Error).message).not.toContain(first.paste.split("\n")[0]);
    }
  });

  it("lets only a foreign error through the import step", () => {
    const broken = { ...engine, Teams: undefined } as unknown as typeof engine;
    expect(() => readPrivateTeamRecord(broken, first)).not.toThrow(PrivateRecordError);
  });
});

describe("checkPrivateRecordSet", () => {
  it("accepts distinct teams, and a retired record may repeat an active team's hash", () => {
    const other = edit(edited, (r) => (r.team_id = "e".repeat(32)));
    expect(() => checkPrivateRecordSet([first, other])).not.toThrow();
    const retired = edit(first, (r) => {
      r.team_id = "f".repeat(32);
      r.status = "retired";
    });
    expect(() => checkPrivateRecordSet([first, retired])).not.toThrow();
  });

  it("refuses a repeated team_id and a canonical_team_hash held by two active records", () => {
    expect(() => checkPrivateRecordSet([first, first])).toThrow(/is in the snapshot twice/);
    const twin = edit(first, (r) => (r.team_id = "f".repeat(32)));
    expect(() => checkPrivateRecordSet([first, twin])).toThrow(/held by two active records/);
  });
});

describe("the SignedStatement contract", () => {
  const valid = ajvFor("contract").compile(signedStatementSchema);

  it("accepts every valid sample and refuses every invalid one", () => {
    for (const name of names("signed-statement", "valid")) expect(valid(sample(`signed-statement/valid/${name}`)), name).toBe(true);
    for (const name of names("signed-statement", "invalid")) expect(valid(sample(`signed-statement/invalid/${name}`)), name).toBe(false);
  });

  it("holds the bounds ADR-014 section 3 names: 4096 digests, 2^53-1 integers, 1 to 2 signatures", () => {
    const body = (change: Partial<SignedStatement["body"]>): SignedStatement => ({ ...head2, body: { ...head2.body, ...change } });
    const digests = (n: number): string[] => Array.from({ length: n }, (_, i) => i.toString(16).padStart(64, "0"));
    expect(valid(body({ object_digests: digests(4096) }))).toBe(true);
    expect(valid(body({ object_digests: digests(4097) }))).toBe(false);
    expect(valid(body({ seq: Number.MAX_SAFE_INTEGER, issued_at: Number.MAX_SAFE_INTEGER }))).toBe(true);
    expect(valid(body({ seq: Number.MAX_SAFE_INTEGER + 1 }))).toBe(false);
    expect(valid(body({ issued_at: Number.MAX_SAFE_INTEGER + 1 }))).toBe(false);
    expect(valid(body({ issued_at: 0 }))).toBe(true);
  });

  it("signs with real Ed25519 over the JCS of the domain, schema version and body", () => {
    const text = Buffer.from(privateHeadSigningBytes(head2)).toString("utf8");
    expect(text.startsWith('{"body":{"issued_at":')).toBe(true);
    expect(text.endsWith(`,"domain":"${PRIVATE_HEAD_DOMAIN}","schema_version":1}`)).toBe(true);
    expect(PRIVATE_HEAD_DOMAIN).toBe("areneon/private-head/v1");
    for (const statement of [head1, head2, rotation]) {
      for (const s of statement.signatures) {
        const key = createPublicKey({ key: Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), Buffer.from(s.public_key, "hex")]), format: "der", type: "spki" });
        expect(verify(null, privateHeadSigningBytes(statement), key, Buffer.from(s.signature, "hex"))).toBe(true);
      }
    }
  });
});

describe("readSignedStatement", () => {
  it("accepts every valid sample, chained by head hash", () => {
    for (const name of names("signed-statement", "valid")) expect(readSignedStatement(sample(`signed-statement/valid/${name}`)).kind).toBe("private-head");
    expect(head2.body.parent_head_hash).toBe(privateHeadHash(head1));
    expect(rotation.body.parent_head_hash).toBe(privateHeadHash(head2));
    expect(rotation.signatures.map((s) => s.public_key).sort()).toEqual([keyA, keyB].sort());
  });

  it("derives a key id from the raw key bytes, not from its hex text", () => {
    expect(head1.signatures[0]?.key_id).toBe(keyIdOf(keyA));
    expect(keyIdOf(keyA)).not.toBe(keyIdOf(keyB));
  });

  it.each([
    ["unsorted-signatures", /signatures: a value is repeated or out of order/],
    ["duplicate-signatures", /signatures: a value is repeated or out of order/],
    ["key-id-mismatch", /is not the SHA-256 of its public_key/],
    ["unsorted-object-digests", /object_digests: a value is repeated or out of order/],
    ["duplicate-object-digests", /object_digests: a value is repeated or out of order/],
    ["first-head-has-parent", /parent_head_hash is null exactly for seq 1/],
    ["later-head-no-parent", /parent_head_hash is null exactly for seq 1/],
    ["rotation-one-signature", /with next_public_key carries two signatures/],
    ["two-signatures-no-rotation", /without next_public_key carries exactly one signature/],
    ["rotation-target-not-a-signer", /with next_public_key carries two signatures, one by that key/],
  ])("refuses signed-statement/%s", (name, message) => {
    const value = sample(`signed-statement/reader-invalid/${name}.json`);
    expect(() => readSignedStatement(value)).toThrow(PrivateRecordError);
    expect(() => readSignedStatement(value)).toThrow(message);
  });

  it("refuses a schema violation", () => {
    expect(() => readSignedStatement({ ...head1, kind: "run-result" })).toThrow(/invalid signed statement/);
  });
});

describe("acceptHeadSignatures (ADR-014 section 3)", () => {
  const rotates = (to: string): SignedStatement => edit(rotation, (s) => (s.body.next_public_key = to));

  it("accepts one signature by the pinned key and keeps the pin", () => {
    expect(acceptHeadSignatures(head2, keyA, [true])).toBe(keyA);
  });

  it("accepts a rotation signed by the pinned and the next key, and moves the pin to the next", () => {
    expect(acceptHeadSignatures(rotation, keyA, [true, true])).toBe(keyB);
  });

  it("refuses a signature that did not verify, or a missing or surplus flag", () => {
    expect(() => acceptHeadSignatures(head2, keyA, [false])).toThrow(/does not verify/);
    expect(() => acceptHeadSignatures(rotation, keyA, [true, false])).toThrow(/does not verify/);
    expect(() => acceptHeadSignatures(rotation, keyA, [true])).toThrow(/one verification result per signature/);
    expect(() => acceptHeadSignatures(head2, keyA, [true, true])).toThrow(/one verification result per signature/);
  });

  it("refuses a head not signed by exactly the pinned key", () => {
    expect(() => acceptHeadSignatures(head2, keyB, [true])).toThrow(/not signed by exactly the pinned key/);
  });

  it("refuses a rotation whose signers are not the pinned key and the next key", () => {
    expect(() => acceptHeadSignatures(rotation, hex("9"), [true, true])).toThrow(/needs exactly the pinned key and next_public_key/);
    // The pinned key is already the next key: not a rotation.
    expect(() => acceptHeadSignatures(rotates(keyA), keyA, [true, true])).toThrow(/needs exactly the pinned key and next_public_key/);
    // Two signers that are the pinned key and a third key, with next_public_key naming neither.
    const third = edit(rotation, (s) => (s.body.next_public_key = hex("c")));
    expect(() => acceptHeadSignatures(third, keyA, [true, true])).toThrow(/needs exactly the pinned key and next_public_key/);
  });

  it("refuses a statement that has not been through the reader's signer count", () => {
    const lone = edit(rotation, (s) => (s.signatures = [s.signatures[0] as SignedStatement["signatures"][number]]));
    expect(() => acceptHeadSignatures(lone, keyA, [true])).toThrow(/needs exactly the pinned key and next_public_key/);
  });
});
