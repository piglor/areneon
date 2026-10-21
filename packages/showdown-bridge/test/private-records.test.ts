import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import privateTeamRecordSchema from "@areneon/schema/contracts/v1/private-team-record.schema.json" with { type: "json" };
import teamRecordSchema from "@areneon/schema/contracts/v1/team-record.schema.json" with { type: "json" };
import { describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { syntheticFixtureHashes, TEAMS_DIR } from "../src/golden.ts";
import { checkPrivateRecordSet, PrivateRecordError, type PrivateTeamRecord, readPrivateTeamRecord } from "../src/private-records.ts";
import { PRIVATE_SAMPLE_PASTES } from "../src/private-samples.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { teamHash } from "../src/team-class.ts";
import { recordHash } from "../src/team-records.ts";
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
const first = sample<PrivateTeamRecord>("private-team-record/valid/first.json");
const edited = sample<PrivateTeamRecord>("private-team-record/valid/edited.json");
const KNOWN_FIRST_RECORD_HASH = "5a41fcb61f1cb78ab3db8f982d1ad9fe8cbaf01b002097554315f4380d6c1260";
const holdout = sample<PrivateTeamRecord>("private-team-record/valid/holdout.json");

describe("the private-sample markers (ADR-015 section 6)", () => {
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
    for (const marker of markers) expect(existsSync(join(REPO, marker)), marker).toBe(true);
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

  it("takes holdout as a boolean and nothing else", () => {
    for (const value of [true, false]) expect(valid({ ...first, holdout: value })).toBe(true);
    for (const value of ["true", 1, null]) expect(valid({ ...first, holdout: value })).toBe(false);
    const { holdout: _omitted, ...without } = first;
    expect(valid(without)).toBe(false);
  });

  it("bounds revision, tags and labels at the edges ADR-015 section 2 names", () => {
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

  it("keeps hold-out a per-team_id flag: the sample chain agrees on it and a hold-out sample is true", () => {
    expect(first.holdout).toBe(edited.holdout);
    expect(readPrivateTeamRecord(engine, holdout).holdout).toBe(true);
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

  it("refuses a schema violation, a paste that is not the engine's export and one it rejects", () => {
    expect(() => readPrivateTeamRecord(engine, { ...first, status: "archived" })).toThrow(/invalid private team record/);
    expect(() => readPrivateTeamRecord(engine, edit(first, (r) => (r.paste = `${r.paste}\n\n`)))).toThrow(/not the engine's own export/);
    const unreproduced = edit(first, (r) => (r.paste = `${r.paste}\nSPs: 1 HP`));
    expect(() => readPrivateTeamRecord(engine, unreproduced)).toThrow("the paste is refused by the pinned engine");
  });

  it("never echoes the paste: not on a schema failure, a too-long paste, an engine refusal or a hash mismatch", () => {
    const lines = first.paste.split("\n").filter((l) => l.length > 5);
    expect(lines.length).toBeGreaterThan(3);
    const marker = "CANARYPASTELINE";
    const cases: unknown[] = [
      { ...first, status: "archived" },
      { ...first, paste: `${marker}${"x".repeat(8200)}` },
      { ...first, paste: marker },
      { ...first, paste: `${first.paste}\nSPs: 1 HP ${marker}` },
      { ...first, team_hash: "a".repeat(64) },
      { ...first, revision: 0 },
      { ...first, holdout: first.paste },
      { ...first, tags: [first.paste], extra: first.paste },
    ];
    for (const value of cases) {
      let message = "";
      try {
        readPrivateTeamRecord(engine, value);
      } catch (error) {
        expect(error).toBeInstanceOf(PrivateRecordError);
        message = (error as Error).message;
      }
      expect(message, JSON.stringify(value).slice(0, 60)).not.toBe("");
      expect(message).not.toContain(marker);
      for (const line of lines) expect(message).not.toContain(line);
    }
  });

  it("refuses a holdout that is not a boolean", () => {
    expect(() => readPrivateTeamRecord(engine, { ...first, holdout: "yes" })).toThrow(/invalid private team record/);
  });

  it("hashes a record to a known answer computed outside the module (Python hashlib over sorted-key compact JSON)", () => {
    expect(recordHash(first)).toBe(KNOWN_FIRST_RECORD_HASH);
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
