import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  canonicalTeamHash,
  checkPrivateRecordSet,
  type Engine,
  importPaste,
  loadEngine,
  type PokemonSet,
  type PrivateTeamRecord,
  readPrivateTeamRecord,
  recordHash,
  resolveFormat,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { ImportPlanError, type ImportRefusal } from "../src/import-error.ts";
import { assertNoActiveRecordLost, type ImportPlanInput, planImport } from "../src/import-plan.ts";
import type { StagedFile } from "../src/staged-file.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const TEAMS = join(ROOT, "fixtures", "teams", "gen9championsvgc2026regmc");
const engine: Engine = loadEngine();
const format = resolveFormat(engine, "gen9championsvgc2026regmc");

// Committed synthetic teams only: the dev split of the fixture dataset (distinct teams) and the paste fixtures.
const pool = readFileSync(join(TEAMS, "datasets", "fixtures-v1", "teams-dev.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => (JSON.parse(line) as { paste: string }).paste);
const fixture = (name: string): string => readFileSync(join(TEAMS, name), "utf8");
const [A, B, C, D] = [pool[0] as string, pool[1] as string, pool[2] as string, pool[3] as string];

const hex = (n: number): string => n.toString(16).padStart(32, "0");
const idSource = (start = 1): (() => string) => {
  let n = start;
  return () => hex(n++);
};
const file = (name: string, text: string): StagedFile => ({ name, text });
const header = (id: string, tags?: string): string => `#! team_id=${id}${tags === undefined ? "" : ` tags=${tags}`}\n`;

const base = (over: Partial<ImportPlanInput> = {}): ImportPlanInput => ({
  engine,
  format,
  staged: [],
  parent: null,
  retire: [],
  labels: () => [{ label_id: "demo", value: "x" }],
  holdout: (id) => id.endsWith("7"),
  newTeamId: idSource(),
  ...over,
});
const plan = (over: Partial<ImportPlanInput>) => planImport(base(over));
const refusal = (over: Partial<ImportPlanInput>): ImportPlanError => {
  try {
    plan(over);
  } catch (error) {
    if (error instanceof ImportPlanError) return error;
    throw error;
  }
  throw new Error("expected a refusal");
};
const code = (over: Partial<ImportPlanInput>): ImportRefusal => refusal(over).code;

// A parent snapshot of A (id 1, tags none), B (id 2, tags ["keep"]) and C (id 3), made by the planner itself.
const first = plan({ staged: [file("a", A), file("b", `#! tags=keep\n${B}`), file("c", C)] });
const parent = first.records;
const idOf = (paste: string): string => (parent.find((r) => r.canonical_team_hash === canonicalTeamHash(importPaste(engine, paste))) as PrivateTeamRecord).team_id;
const [idA, idB, idC] = [idOf(A), idOf(B), idOf(C)];
const record = (id: string): PrivateTeamRecord => parent.find((r) => r.team_id === id) as PrivateTeamRecord;

describe("a first import", () => {
  it("makes revision-1 active records with fresh ids, sorted by team_id, and counts them", () => {
    expect(first.summary).toEqual({ new: 3, revised: 0, unchanged: 0, retired: 0 });
    expect(first.records.map((r) => r.team_id)).toEqual([hex(1), hex(2), hex(3)]);
    for (const r of first.records) {
      expect(readPrivateTeamRecord(engine, r)).toEqual(r);
      expect(r).toMatchObject({ revision: 1, supersedes_record_hash: null, status: "active", data_class: "private" });
    }
    expect(() => checkPrivateRecordSet(first.records)).not.toThrow();
    expect(record(idB).tags).toEqual(["keep"]);
    expect(record(idA).tags).toEqual([]);
  });

  it("fills holdout and labels through the ports, per team_id", () => {
    expect(first.records.map((r) => r.holdout)).toEqual([false, false, false]);
    const out = plan({ staged: [file("a", A)], newTeamId: () => hex(7), holdout: (id) => id === hex(7), labels: () => [] });
    expect(out.records[0]).toMatchObject({ holdout: true, labels: [] });
  });

  it("issues ids in file-name order, not list order", () => {
    const one = plan({ staged: [file("b", B), file("a", A)] }).records.find((r) => r.team_id === hex(1)) as PrivateTeamRecord;
    expect(one.canonical_team_hash).toBe(canonicalTeamHash(importPaste(engine, A)));
  });

  it("is a pure function of its inputs", () => {
    const again = plan({ staged: [file("a", A), file("b", `#! tags=keep\n${B}`), file("c", C)] });
    expect(again).toEqual(first);
  });

  it("stores the engine's export, not the staged text", () => {
    const loose = A.replace(/ +\n/g, "\n");
    expect(importPaste(engine, A)).toBeTruthy();
    const out = plan({ staged: [file("a", `${loose}`)] });
    expect(out.records[0]?.paste).toBe(first.records.find((r) => r.team_id === hex(1))?.paste);
  });

  it("refuses an import that leaves no record", () => {
    expect(code({})).toBe("no_teams");
  });
});

describe("a later import", () => {
  const next = (over: Partial<ImportPlanInput>) => plan({ parent, newTeamId: idSource(100), ...over });

  it("carries absent teams unchanged: the same record objects, summary unchanged", () => {
    const out = next({ staged: [file("d", D)] });
    expect(out.summary).toEqual({ new: 1, revised: 0, unchanged: 3, retired: 0 });
    expect(out.records).toHaveLength(4);
    for (const r of parent) expect(out.records).toContainEqual(r);
  });

  it("makes no change for a revision file that matches (paste and tags), whatever its header", () => {
    const out = next({ staged: [file("a", header(idA) + A), file("b", header(idB, "keep") + B)] });
    expect(out.records).toEqual(parent);
    expect(out.summary).toEqual({ new: 0, revised: 0, unchanged: 3, retired: 0 });
  });

  it("raises the revision by 1 and links the record it replaces", () => {
    const out = next({ staged: [file("a", header(idA) + D)] });
    const revised = out.records.find((r) => r.team_id === idA) as PrivateTeamRecord;
    expect(revised).toMatchObject({ revision: 2, supersedes_record_hash: recordHash(record(idA)), status: "active" });
    expect(revised.canonical_team_hash).toBe(canonicalTeamHash(importPaste(engine, D)));
    expect(readPrivateTeamRecord(engine, revised)).toEqual(revised);
    expect(out.summary).toEqual({ new: 0, revised: 1, unchanged: 2, retired: 0 });
  });

  it("revises on a tag change alone, and keeps the parent's tags when the header names none", () => {
    const tagged = next({ staged: [file("b", header(idB, "other,keep") + B)] });
    expect(tagged.records.find((r) => r.team_id === idB)).toMatchObject({ revision: 2, tags: ["keep", "other"] });
    const kept = next({ staged: [file("b", header(idB) + D)] });
    expect(kept.records.find((r) => r.team_id === idB)).toMatchObject({ revision: 2, tags: ["keep"] });
  });

  it("revises when only the Pokemon order changed (a different team_hash, the same canonical hash)", () => {
    const blocks = importPaste(engine, A);
    const reordered = [...blocks].reverse();
    const text = engine.sim.Teams.export(reordered as PokemonSet[]);
    const out = next({ staged: [file("a", header(idA) + text)] });
    const r = out.records.find((x) => x.team_id === idA) as PrivateTeamRecord;
    expect(r.revision).toBe(2);
    expect(r.canonical_team_hash).toBe(record(idA).canonical_team_hash);
    expect(r.team_hash).not.toBe(record(idA).team_hash);
  });

  it("retires explicitly: a retired revision with the same team and a fresh link", () => {
    const out = next({ retire: [idC] });
    const gone = out.records.find((r) => r.team_id === idC) as PrivateTeamRecord;
    expect(gone).toMatchObject({ status: "retired", revision: 2, supersedes_record_hash: recordHash(record(idC)), paste: record(idC).paste });
    expect(out.summary).toEqual({ new: 0, revised: 0, unchanged: 2, retired: 1 });
    expect(() => checkPrivateRecordSet(out.records)).not.toThrow();
  });

  it("lets a retired team's hash be staged again as a new team", () => {
    const retired = next({ retire: [idC] }).records;
    const out = plan({ parent: retired, staged: [file("c", C)], newTeamId: idSource(200) });
    expect(out.summary).toEqual({ new: 1, revised: 0, unchanged: 3, retired: 0 });
    expect(out.records.filter((r) => r.canonical_team_hash === record(idC).canonical_team_hash).map((r) => r.status).sort()).toEqual(["active", "retired"]);
  });

  it("makes a retired team active again when its file names its team_id", () => {
    const retired = next({ retire: [idC] }).records;
    const out = plan({ parent: retired, staged: [file("c", header(idC) + C)] });
    expect(out.records.find((r) => r.team_id === idC)).toMatchObject({ status: "active", revision: 3 });
    expect(out.summary).toEqual({ new: 0, revised: 1, unchanged: 2, retired: 0 });
  });

  it("allows a revision to take the hash an import-mate gives up", () => {
    const out = next({ staged: [file("a", header(idA) + B), file("b", header(idB) + D)] });
    expect(out.summary.revised).toBe(2);
  });

  it("mixes new, revised, retired and unchanged", () => {
    const out = next({ staged: [file("a", header(idA) + D), file("n", fixture("legal-synthetic.txt"))], retire: [idB] });
    expect(out.summary).toEqual({ new: 1, revised: 1, unchanged: 1, retired: 1 });
  });
});

describe("refusals", () => {
  const next = (over: Partial<ImportPlanInput>) => ({ parent, ...over });

  it.each<[ImportRefusal, Partial<ImportPlanInput>]>([
    ["header_malformed", { staged: [file("a", `#! tags=\n${A}`)] }],
    ["comment_line", { staged: [file("a", `${A}\n# note`)] }],
    ["import_refused", { staged: [file("a", "this is not a team")] }],
    ["import_refused", { staged: [file("a", "")] }],
    ["team_illegal", { staged: [file("a", fixture("illegal-item-clause.txt"))] }],
    ["team_illegal", { staged: [file("a", fixture("illegal-stat-points-over-32.txt"))] }],
    ["team_illegal", { staged: [file("a", fixture("illegal-five-pokemon.txt"))] }],
    ["team_id_unknown", next({ staged: [file("a", header(hex(999)) + D)] })],
    ["team_id_unknown", { staged: [file("a", header(hex(1)) + D)] }],
    ["team_id_repeated", next({ staged: [file("a", header(idA) + D), file("b", header(idA) + A)] })],
    ["duplicate_of_active", next({ staged: [file("a", A)] })],
    ["duplicate_of_active", { staged: [file("a", A), file("b", A)] }],
    ["duplicate_of_active", next({ staged: [file("a", header(idA) + D), file("b", D)], newTeamId: idSource(100) })],
    ["duplicate_canonical_hash", next({ staged: [file("a", header(idA) + B)] })],
    ["duplicate_canonical_hash", next({ staged: [file("a", header(idC) + A)] })],
    ["retire_id_malformed", next({ retire: ["ABC"] })],
    ["retire_repeated", next({ retire: [idA, idA] })],
    ["retire_unknown", next({ retire: [hex(999)] })],
    ["retire_unknown", { retire: [hex(1)], staged: [file("a", A)] }],
    ["retire_and_staged", next({ retire: [idA], staged: [file("a", header(idA) + D)] })],
    ["record_invalid", { staged: [file("a", A)], labels: () => [{ label_id: "!", value: "x" }] }],
    ["id_source_invalid", { staged: [file("a", A)], newTeamId: () => "not-an-id" }],
    ["id_source_invalid", next({ staged: [file("a", D)], newTeamId: () => idA })],
    ["id_source_invalid", { staged: [file("a", A), file("b", B)], newTeamId: () => hex(5) }],
  ])("%s", (expected, over) => {
    expect(code(over)).toBe(expected);
  });

  it("refuses to retire a record that is already retired", () => {
    const retired = plan({ parent, retire: [idC] }).records;
    expect(code({ parent: retired, retire: [idC] })).toBe("retire_not_active");
  });

  it("refuses an unreadable parent: a tampered record, a repeated team_id, a repeated active team", () => {
    const tampered = parent.map((r) => (r.team_id === idA ? { ...r, team_hash: "0".repeat(64) } : r));
    expect(code({ parent: tampered })).toBe("parent_invalid");
    expect(code({ parent: [...parent, record(idA)] })).toBe("parent_invalid");
    const twin = { ...record(idA), team_id: hex(77) };
    expect(code({ parent: [...parent, twin] })).toBe("parent_invalid");
  });

  it("names the file's position, not the file", () => {
    const e = refusal({ staged: [file("zzz", A), file("aaa", "nope")] });
    expect(e.position).toEqual({ subject: "file", index: 2 });
    expect(refusal(next({ retire: [idA, hex(999)] })).position).toEqual({ subject: "retire", index: 2 });
  });

  it("names the staged file when a carried-over record is the other half of a duplicate, whichever id sorts first", () => {
    const e = refusal(next({ staged: [file("a", header(idA) + C)] }));
    expect(e.code).toBe("duplicate_canonical_hash");
    expect(e.position).toEqual({ subject: "file", index: 1 });
    const low = parent.find((r) => r.team_id < idC && r.team_id !== idA) as PrivateTeamRecord;
    const back = refusal(next({ staged: [file("a", header(idC) + low.paste)] }));
    expect(back).toMatchObject({ code: "duplicate_canonical_hash", position: { subject: "file", index: 1 } });
  });
});

describe("the active-record invariant", () => {
  const retireSet = new Set([idC]);
  it("passes a result that keeps or retires every active parent record", () => {
    expect(() => assertNoActiveRecordLost(parent, parent, new Set())).not.toThrow();
    const retired = parent.map((r) => (r.team_id === idC ? { ...r, status: "retired" as const } : r));
    expect(() => assertNoActiveRecordLost(parent, retired, retireSet)).not.toThrow();
  });
  it("refuses a result with a missing record, or a retired one without --retire", () => {
    expect(() => assertNoActiveRecordLost(parent, parent.slice(1), new Set())).toThrow(ImportPlanError);
    const retired = parent.map((r) => (r.team_id === idC ? { ...r, status: "retired" as const } : r));
    expect(() => assertNoActiveRecordLost(parent, retired, new Set())).toThrow(ImportPlanError);
  });
  it("ignores parent records that were already retired", () => {
    const old = parent.map((r) => ({ ...r, status: "retired" as const }));
    expect(() => assertNoActiveRecordLost(old, [], new Set())).not.toThrow();
  });
});
