import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { DEFAULT_SHOWDOWN_DIR, type Engine, EngineError, loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import {
  exportRoleSets,
  readRoleSetTable,
  ROLE_SETS_FILE,
  ROLE_SETS_SUMMARY,
  type RoleSetTable,
  RoleSetError,
  roleSetSourcePath,
  roleSetSummary,
  roleSetTableHash,
} from "../src/role-sets.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE, type TeamVocabulary } from "../src/team-vocabulary.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary: TeamVocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const golden = JSON.parse(readFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")) as RoleSetTable;
const bytes = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

const SOURCE = "data/random-battles/champions/doubles-sets.json";
type Source = Record<string, { level: number; sets: Record<string, unknown>[] }>;
const pinnedSource = readFileSync(join(DEFAULT_SHOWDOWN_DIR, SOURCE), "utf8");

const scratch = mkdtempSync(join(tmpdir(), "role-sets-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
let n = 0;

/** A copy of the engine checkout's source file in a temp dir, doctored by `change`, exported with the real engine. */
function doctored(change: (source: Source) => void, using: Engine = engine): RoleSetTable {
  const source = JSON.parse(pinnedSource) as Source;
  change(source);
  return exportFrom(JSON.stringify(source), using);
}
function exportFrom(text: string, using: Engine = engine): RoleSetTable {
  const dir = join(scratch, String(n++));
  mkdirSync(join(dir, "data/random-battles/champions"), { recursive: true });
  writeFileSync(join(dir, SOURCE), text);
  return exportRoleSets(using, GOLDEN_FORMAT, vocabulary, table, dir);
}
const refused = (change: (source: Source) => void, message: RegExp | string, using: Engine = engine) => {
  expect(() => doctored(change, using)).toThrow(RoleSetError);
  expect(() => doctored(change, using)).toThrow(message);
};
const outsideRoster = table.species.find((x) => !vocabulary.roster.some((r) => r.id === x.id)) as { id: string };
const outsidePool = table.moves.find((m) => !(vocabulary.roster.find((r) => r.id === "venusaur") as { moves: string[] }).moves.includes(m.id)) as { name: string };
const firstSet = (source: Source, species = "venusaur") => (source[species] as Source[string]).sets[0] as Record<string, unknown>;

describe("the golden role-set table (ADR-020 section 3)", () => {
  it("re-exports to the same bytes and the same hash at the pin", () => {
    const fresh = exportRoleSets(engine, GOLDEN_FORMAT, vocabulary, table);
    expect(readFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")).toBe(bytes(fresh));
    expect(readFileSync(join(GOLDEN_DIR, ROLE_SETS_SUMMARY), "utf8")).toBe(bytes(roleSetSummary(fresh)));
    expect(roleSetTableHash(fresh)).toBe(roleSetTableHash(golden));
    expect(readRoleSetTable(golden, vocabulary)).toEqual(fresh);
  });

  it("records the pinned file's counts, its hash and the hashes of what it was resolved against", () => {
    const summary = roleSetSummary(golden);
    expect(summary.counts.species).toBe(342);
    expect(summary.counts.sets).toBe(495);
    expect(golden.source_path).toBe(SOURCE);
    expect(golden.source_sha256).toBe(createHash("sha256").update(pinnedSource).digest("hex"));
    expect(golden.showdown_commit).toBe(engine.commit);
    expect(golden.vocabulary_hash).toBe(JSON.parse(readFileSync(join(GOLDEN_DIR, "team-vocabulary.summary.json"), "utf8")).vocabulary_hash);
    expect(golden.engine_data_hash).toBe(vocabulary.engine_data_hash);
  });

  it("is hashed as the SHA-256 of its JCS bytes, and any change changes the hash", () => {
    const hash = roleSetTableHash(golden);
    expect(hash).toBe(createHash("sha256").update(jcs(golden as unknown as Json), "utf8").digest("hex"));
    expect(roleSetTableHash({ ...golden, showdown_commit: "0".repeat(40) })).not.toBe(hash);
    const edited = structuredClone(golden);
    (edited.species[0]?.sets[0]?.moves as string[]).pop();
    expect(roleSetTableHash(edited)).not.toBe(hash);
  });

  it("keeps the engine's own roles and drops level and the type lists", () => {
    const venusaur = golden.species.find((s) => s.id === "venusaur");
    expect(venusaur?.sets).toEqual([
      { role: "Offensive Protect", moves: ["earthpower", "gigadrain", "knockoff", "leafstorm", "protect", "sludgebomb"], abilities: ["chlorophyll", "overgrow"] },
    ]);
    expect(JSON.stringify(golden)).not.toMatch(/level|preferredTypes|teraTypes|paste|evs/);
  });

  it("gives a Mega forme the abilities of the forme it comes from", () => {
    expect(golden.species.find((s) => s.id === "venusaurmega")?.sets[0]?.abilities).toEqual(["chlorophyll"]);
    expect(vocabulary.roster.find((r) => r.id === "venusaurmega")?.abilities).toEqual(["thickfat"]);
    expect(golden.species.find((s) => s.id === "floettemega")?.sets[0]?.abilities).toEqual(["flowerveil"]);
  });
});

describe("exportRoleSets fails closed on a doctored source", () => {
  it("accepts an unchanged copy and drops a well-formed teraTypes and preferredTypes", () => {
    expect(exportFrom(pinnedSource)).toEqual(golden);
    const fresh = doctored((s) => Object.assign(firstSet(s), { teraTypes: ["Fire"], preferredTypes: ["Grass"] }));
    expect({ ...fresh, source_sha256: golden.source_sha256 }).toEqual(golden);
  });

  it("refuses a species the engine-data table does not know", () => {
    refused((s) => ((s as Record<string, unknown>).notaspecies = s.venusaur), "species notaspecies is not in the engine-data table");
  });

  it("refuses a species outside the roster", () => {
    refused((s) => ((s as Record<string, unknown>)[outsideRoster.id] = s.venusaur), `species ${outsideRoster.id} is not in the team vocabulary roster`);
  });

  it("refuses a move name the table does not know, and a move the validator does not accept on the species", () => {
    refused((s) => ((firstSet(s).movepool as string[])[0] = "Not A Move"), "move Not A Move is not in the engine-data table");
    refused((s) => ((firstSet(s).movepool as string[])[0] = outsidePool.name), `move ${outsidePool.name} is not accepted by the validator on venusaur`);
  });

  it("refuses an ability name the table does not know, and an ability the validator does not accept on the species", () => {
    refused((s) => ((firstSet(s).abilities as string[])[0] = "Not An Ability"), "ability Not An Ability is not in the engine-data table");
    refused((s) => ((firstSet(s).abilities as string[])[0] = "Blaze"), "ability Blaze is not accepted by the validator on venusaur");
    refused((s) => ((firstSet(s, "venusaurmega").abilities as string[])[0] = "Blaze"), "ability Blaze is not accepted by the validator on venusaurmega");
  });

  it("refuses a shape it does not know", () => {
    refused((s) => Object.assign(firstSet(s), { item: "Leftovers" }), "unknown field item");
    refused((s) => Object.assign(s.venusaur as object, { evs: {} }), "unknown field evs");
    refused((s) => ((s.venusaur as { level: unknown }).level = 50.5), "level is not an integer");
    refused((s) => delete (firstSet(s) as { role?: string }).role, "role is not a non-empty string");
    refused((s) => (firstSet(s).role = ""), "role is not a non-empty string");
    refused((s) => (firstSet(s).movepool = []), "movepool: not a non-empty list of names");
    refused((s) => (firstSet(s).abilities = [7]), "abilities: not a non-empty list of names");
    refused((s) => (firstSet(s).teraTypes = "Fire"), "teraTypes: not a list of strings");
    refused((s) => (firstSet(s).preferredTypes = [1]), "preferredTypes: not a list of strings");
    refused((s) => ((s.venusaur as { sets: unknown }).sets = []), "species venusaur: no sets");
    refused((s) => ((s as Record<string, unknown>).venusaur = [1]), "species venusaur: not an object");
    refused((s) => ((s.venusaur as { sets: unknown[] }).sets = [3]), "species venusaur set 0: not an object");
  });

  it("refuses a role name the contract would not carry", () => {
    refused((s) => (firstSet(s).role = "Support|Lead"), /invalid role-set table/);
  });

  it("refuses a source that is missing, not JSON, or not an object", () => {
    expect(() => exportRoleSets(engine, GOLDEN_FORMAT, vocabulary, table, join(scratch, "nowhere"))).toThrow("cannot read");
    expect(() => exportFrom("{")).toThrow("not valid JSON");
    expect(() => exportFrom("[]")).toThrow(": not an object");
  });

  it("refuses an unknown format, a vocabulary of another format, and a format that is not doubles", () => {
    expect(() => exportRoleSets(engine, "gen9nosuchformat", vocabulary, table)).toThrow(EngineError);
    expect(() => exportRoleSets(engine, "gen9ou", vocabulary, table)).toThrow(RoleSetError);
    expect(() => roleSetSourcePath({ gameType: "singles", id: "x", mod: "champions" } as never)).toThrow(/not doubles/);
    expect(roleSetSourcePath({ gameType: "doubles", id: "x", mod: "champions" } as never)).toBe(SOURCE);
  });
});

describe("exportRoleSets and a Mega forme's item data", () => {
  const sim = engine.sim as unknown as { Dex: { formats: unknown; mod(name: string): { items: { get(id: string): unknown } } } };
  // venusaurite's map is replaced; every other item keeps the engine's own.
  const withStone = (stone: () => unknown): Engine =>
    ({
      commit: engine.commit,
      sim: {
        Dex: {
          ...sim.Dex,
          formats: sim.Dex.formats,
          mod: (name: string) => ({ items: { get: (id: string) => (id === "venusaurite" ? { megaStone: stone() } : sim.Dex.mod(name).items.get(id)) } }),
        },
      },
    }) as unknown as Engine;

  it("refuses an item that names no map of Mega formes", () => {
    for (const stone of [undefined, "Venusaur-Mega", ["Venusaur-Mega"], null]) {
      refused(() => undefined, "names no Mega forme map", withStone(() => stone));
    }
  });

  it("accepts a stone that maps the Mega forme from the forme the ability belongs to, and refuses one that does not", () => {
    // venusaurmega's roster ability is Thick Fat; the sets name Chlorophyll, Venusaur's.
    expect(doctored(() => undefined, withStone(() => ({ Venusaur: "Venusaur-Mega", Other: "Other-Mega" })), ).species.length).toBe(342);
    refused(() => undefined, "ability Chlorophyll is not accepted by the validator on venusaurmega", withStone(() => ({ Charizard: "Venusaur-Mega" })));
  });
});

describe("readRoleSetTable", () => {
  const edit = (change: (t: RoleSetTable) => void): RoleSetTable => {
    const copy = structuredClone(golden);
    change(copy);
    return copy;
  };
  const setsOf = (t: RoleSetTable, id: string) => (t.species.find((s) => s.id === id) as RoleSetTable["species"][number]).sets;

  it("accepts the golden table with or without its vocabulary", () => {
    expect(readRoleSetTable(golden)).toEqual(golden);
    expect(readRoleSetTable(golden, vocabulary)).toEqual(golden);
  });

  it.each([
    ["a schema violation", () => ({ ...golden, schema_version: 2 }), /invalid role-set table/],
    ["unsorted species", () => edit((t) => t.species.reverse()), /species: .* repeated or out of order/],
    ["repeated moves", () => edit((t) => ((setsOf(t, "venusaur")[0] as { moves: string[] }).moves = ["protect", "protect"])), /moves: .* repeated or out of order/],
    ["unsorted abilities", () => edit((t) => ((setsOf(t, "venusaur")[0] as { abilities: string[] }).abilities = ["overgrow", "chlorophyll"])), /abilities: .* repeated or out of order/],
    ["a repeated set", () => edit((t) => setsOf(t, "kingambit").push(structuredClone(setsOf(t, "kingambit")[0]) as never)), /set \d is repeated or out of order/],
    ["unsorted sets", () => edit((t) => setsOf(t, "venusaurmega").reverse()), /set 1 is repeated or out of order/],
  ])("refuses %s", (_label, make, message) => {
    expect(() => readRoleSetTable(make())).toThrow(RoleSetError);
    expect(() => readRoleSetTable(make())).toThrow(message);
  });

  it.each([
    ["another vocabulary", () => edit((t) => (t.vocabulary_hash = "f".repeat(64))), /vocabulary_hash is not the hash/],
    ["another commit", () => edit((t) => (t.showdown_commit = "0".repeat(40))), /showdown_commit differs/],
    ["another format", () => edit((t) => (t.format_id = "gen9other")), /format_id differs/],
    ["another engine-data hash", () => edit((t) => (t.engine_data_hash = "0".repeat(64))), /engine_data_hash differs/],
    ["a species outside the roster", () => edit((t) => ((t.species[0] as { id: string }).id = "aaaaaaa")), /not in the vocabulary roster/],
    ["a move outside the pool", () => edit((t) => ((setsOf(t, "venusaur")[0] as { moves: string[] }).moves = ["splash"])), /not in its roster move pool/],
    ["an ability the roster does not list", () => edit((t) => ((setsOf(t, "venusaur")[0] as { abilities: string[] }).abilities = ["blaze"])), /not in its roster abilities/],
    ["a Mega forme's ability no base forme has", () => edit((t) => ((setsOf(t, "venusaurmega")[0] as { abilities: string[] }).abilities = ["zzzzzzzz"])), /not in its roster abilities/],
  ])("with the vocabulary, refuses %s", (_label, make, message) => {
    const attempt = () => readRoleSetTable(make(), vocabulary);
    expect(attempt).toThrow(RoleSetError);
    expect(attempt).toThrow(message);
  });
});
