import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { type EngineData, engineDataHash, exportEngineData } from "../src/engine-data.ts";
import { type Engine, EngineError, loadEngine, type SimRuleTable } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import {
  exportTeamVocabulary,
  readTeamVocabulary,
  TEAM_VOCABULARY_FILE,
  TEAM_VOCABULARY_SUMMARY,
  type TeamVocabulary,
  TeamVocabularyError,
  teamVocabularyHash,
  teamVocabularySummary,
} from "../src/team-vocabulary.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const golden = JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")) as TeamVocabulary;
const bytes = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

type Raw = Record<string, unknown>;
interface Dex {
  species: { all(): Raw[] };
  natures: { all(): Raw[] };
  moves: unknown;
  items: unknown;
}
interface Validator {
  validateSet(set: Raw, teamHas: Raw): string[] | null;
  checkItem(set: Raw, item: unknown, setHas: Raw): string | null;
}
interface Sim {
  Dex: {
    formats: { get(id: string): unknown; getRuleTable(format: unknown): SimRuleTable };
    mod(name: string): Dex;
  };
  TeamValidator: { get(id: string): Validator };
}
const real = engine.sim as unknown as Sim;
const realDex = (): Dex => real.Dex.mod((real.Dex.formats.get(GOLDEN_FORMAT) as { mod: string }).mod);
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const named = (list: Raw[], ...names: string[]): Raw[] => list.filter((e) => names.includes(e.name as string)).map(clone);

interface Changes {
  species?: (all: Raw[]) => Raw[];
  natures?: (all: Raw[]) => Raw[];
  rules?: (table: SimRuleTable) => SimRuleTable;
  validator?: (real: Validator) => Validator;
}

/** The real engine, with a short species list and the named parts changed, to reach the fail-closed paths quickly. */
function patched(changes: Changes = {}): Engine {
  const dex = realDex();
  const species = (changes.species ?? ((all) => named(all, "Garchomp")))(dex.species.all());
  const natures = (changes.natures ?? ((all) => all))(dex.natures.all());
  const sim = {
    Dex: {
      formats: {
        get: (id: string) => real.Dex.formats.get(id),
        getRuleTable: (format: unknown) => {
          const rules = real.Dex.formats.getRuleTable(format);
          return changes.rules ? changes.rules(rules) : rules;
        },
      },
      mod: () => ({ ...dex, species: { all: () => species }, natures: { all: () => natures } }),
    },
    TeamValidator: {
      get: (id: string) => {
        const validator = real.TeamValidator.get(id);
        return changes.validator ? changes.validator(validator) : validator;
      },
    },
  };
  return { commit: engine.commit, sim } as unknown as Engine;
}

const failing = (engineUnderTest: Engine): (() => unknown) => () => exportTeamVocabulary(engineUnderTest, GOLDEN_FORMAT, table);

describe("the golden team vocabulary (ADR-013 section 2)", () => {
  it("re-exports to the same bytes and the same hash at the pin", () => {
    const fresh = exportTeamVocabulary(engine, GOLDEN_FORMAT);
    expect(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")).toBe(bytes(fresh));
    expect(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_SUMMARY), "utf8")).toBe(bytes(teamVocabularySummary(fresh)));
    expect(teamVocabularyHash(fresh)).toBe(teamVocabularyHash(golden));
    expect(readTeamVocabulary(golden, table)).toEqual(fresh);
  }, 120_000);

  it("refuses an unknown format", () => {
    expect(() => exportTeamVocabulary(engine, "gen9nosuchformat", table)).toThrow(EngineError);
  });

  it("is hashed as the SHA-256 of its JCS bytes, and a pin bump or any entry change changes the hash", () => {
    const hash = teamVocabularyHash(golden);
    expect(hash).toBe(createHash("sha256").update(jcs(golden as unknown as Json), "utf8").digest("hex"));
    expect(teamVocabularyHash({ ...golden, showdown_commit: "0".repeat(40) })).not.toBe(hash);
    const edited = structuredClone(golden);
    (edited.roster[0] as { moves: string[] }).moves.pop();
    expect(teamVocabularyHash(edited)).not.toBe(hash);
  });

  it("carries the rule table's values: no constant in the code", () => {
    const format = real.Dex.formats.get(GOLDEN_FORMAT);
    const rules = real.Dex.formats.getRuleTable(format);
    expect(golden.rules).toEqual({
      adjust_level: rules.adjustLevel,
      ev_limit: rules.evLimit,
      item_clause: Number(rules.valueRules.get("itemclause")),
      min_team_size: rules.minTeamSize,
      max_team_size: rules.maxTeamSize,
      picked_team_size: rules.pickedTeamSize,
    });
  });

  it("lists the species the validator accepts and leaves out the ones it bans", () => {
    const ids = new Set(golden.roster.map((r) => r.id));
    expect(ids.has("garchomp")).toBe(true);
    expect(ids.has("mewtwo")).toBe(false);
    const validator = real.TeamValidator.get(GOLDEN_FORMAT);
    const probe = (species: string, move: string) => ({
      name: "",
      species,
      item: "",
      ability: "Pressure",
      moves: [move],
      nature: "Serious",
      gender: "",
      evs: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
    });
    expect((validator.validateSet(probe("Mewtwo", "Psychic"), {}) ?? []).join(" ")).toMatch(/banned/);
    // Every pool move of a few species is accepted by the engine alone, and a move outside the pool is not.
    for (const id of ["garchomp", "charizardmegax", "pikachu"]) {
      const entry = golden.roster.find((r) => r.id === id) as TeamVocabulary["roster"][number];
      const species = table.species.find((s) => s.id === id) as { name: string; abilities: string[] };
      const ability = table.abilities.find((a) => a.id === entry.abilities[0]) as { name: string };
      const item = entry.required_item === undefined ? "" : (table.items.find((i) => i.id === entry.required_item) as { name: string }).name;
      for (const move of entry.moves.slice(0, 5)) {
        const set = { ...probe(species.name, table.moves.find((m) => m.id === move)?.name as string), ability: ability.name, item };
        expect(validator.validateSet(set, {}), `${id} ${move}`).toBeNull();
      }
      const outside = table.moves.find((m) => !entry.moves.includes(m.id)) as { name: string };
      expect(validator.validateSet({ ...probe(species.name, outside.name), ability: ability.name, item }, {}), `${id} outside ${outside.name}`).not.toBeNull();
    }
  });

  it("names a forme's required item, and lists only what the engine-data table knows", () => {
    expect(golden.roster.find((r) => r.id === "charizardmegax")?.required_item).toBe("charizarditex");
    expect(golden.roster.find((r) => r.id === "garchomp")?.required_item).toBeUndefined();
    expect(golden.natures.find((n) => n.id === "adamant")).toEqual({ id: "adamant", name: "Adamant", plus: "atk", minus: "spa" });
    expect(golden.natures.find((n) => n.id === "serious")).toEqual({ id: "serious", name: "Serious", plus: "", minus: "" });
    expect(golden.engine_data_hash).toBe(engineDataHash(table));
    expect(teamVocabularySummary(golden).counts).toEqual({
      roster: golden.roster.length,
      natures: golden.natures.length,
      items: golden.items.length,
      pool_moves: golden.roster.reduce((sum, r) => sum + r.moves.length, 0),
    });
  });
});

describe("exportTeamVocabulary fails closed", () => {
  it("judges a short roster with the real validator, and a second ability when the first is refused", () => {
    const two = (all: Raw[]) => named(all, "Garchomp", "Charizard-Mega-X", "Mewtwo");
    const refuseRoughSkin = (v: Validator): Validator => ({
      validateSet: (set, has) => (set.ability === "Rough Skin" ? ["refused"] : v.validateSet(set, has)),
      checkItem: (set, item, has) => v.checkItem(set, item, has),
    });
    const result = exportTeamVocabulary(patched({ species: two, validator: refuseRoughSkin }), GOLDEN_FORMAT, table);
    expect(result.roster.map((r) => r.id)).toEqual(["charizardmegax", "garchomp"]);
    expect(result.roster.find((r) => r.id === "garchomp")?.abilities).toEqual(["sandveil"]);
    expect(result.roster.find((r) => r.id === "charizardmegax")?.required_item).toBe("charizarditex");
  });

  it("reads a forme's item from requiredItems when requiredItem is absent", () => {
    const listed = (all: Raw[]) => named(all, "Charizard-Mega-X").map((s) => ({ ...s, requiredItem: undefined, requiredItems: ["Charizardite X"] }));
    expect(exportTeamVocabulary(patched({ species: listed }), GOLDEN_FORMAT, table).roster[0]?.required_item).toBe("charizarditex");
    const empty = (all: Raw[]) => named(all, "Garchomp").map((s) => ({ ...s, requiredItems: [] }));
    expect(exportTeamVocabulary(patched({ species: empty }), GOLDEN_FORMAT, table).roster[0]?.required_item).toBeUndefined();
  });

  it("refuses a species the validator takes no move for, and a validator that takes none", () => {
    expect(failing(patched({ species: (all) => named(all, "Mewtwo") }))).toThrow("the validator accepts no species");
    const refuseAll = (v: Validator): Validator => ({ validateSet: () => ["refused"], checkItem: (set, item, has) => v.checkItem(set, item, has) });
    expect(failing(patched({ validator: refuseAll }))).toThrow(TeamVocabularyError);
  });

  it("refuses an unset or malformed rule value", () => {
    const set = (key: string, value: unknown) => (rules: SimRuleTable) => Object.assign(Object.create(rules) as SimRuleTable, { [key]: value });
    expect(failing(patched({ rules: set("adjustLevel", null) }))).toThrow(/adjustLevel: null/);
    expect(failing(patched({ rules: set("evLimit", 0) }))).toThrow(/evLimit/);
    expect(failing(patched({ rules: set("pickedTeamSize", null) }))).toThrow(/pickedTeamSize/);
    const noClause = (rules: SimRuleTable) => Object.assign(Object.create(rules) as SimRuleTable, { valueRules: new Map() });
    expect(failing(patched({ rules: noClause }))).toThrow(/itemclause: null/);
  });

  it("refuses a nature of a shape it does not know, and an engine with no neutral nature", () => {
    const withStat = (plus: unknown) => (all: Raw[]) => all.map((n) => ({ ...clone(n), plus }));
    expect(failing(patched({ natures: withStat("hp") }))).toThrow(/nature plus: hp is not a stat/);
    expect(failing(patched({ natures: (all) => all.filter((n) => n.plus !== undefined) }))).toThrow("no neutral nature");
    expect(failing(patched({ natures: (all) => [{ ...clone(all[0] as Raw), name: "" }] }))).toThrow(/nature name/);
    expect(failing(patched({ natures: () => [7 as unknown as Raw] }))).toThrow(/nature: not an object/);
  });

  it("drops a nature the validator refuses", () => {
    const refuseBold = (v: Validator): Validator => ({
      validateSet: (set, has) => (set.nature === "Bold" ? ["refused"] : v.validateSet(set, has)),
      checkItem: (set, item, has) => v.checkItem(set, item, has),
    });
    const natures = exportTeamVocabulary(patched({ validator: refuseBold }), GOLDEN_FORMAT, table).natures.map((n) => n.id);
    expect(natures).not.toContain("bold");
    expect(natures).toContain("adamant");
  });

  it("drops an item the validator bans", () => {
    const banLeftovers = (v: Validator): Validator => ({
      validateSet: (set, has) => v.validateSet(set, has),
      checkItem: (set, item, has) => ((item as { id: string }).id === "leftovers" ? "banned" : v.checkItem(set, item, has)),
    });
    const items = exportTeamVocabulary(patched({ validator: banLeftovers }), GOLDEN_FORMAT, table).items;
    expect(items).not.toContain("leftovers");
    expect(items).toContain("sitrusberry");
  });
});

describe("readTeamVocabulary", () => {
  const edit = (change: (v: TeamVocabulary) => void): TeamVocabulary => {
    const copy = structuredClone(golden);
    change(copy);
    return copy;
  };
  const withTable = (change: (t: EngineData) => void): EngineData => {
    const copy = structuredClone(table);
    change(copy);
    return copy;
  };

  it("accepts the golden vocabulary with or without its table", () => {
    expect(readTeamVocabulary(golden)).toEqual(golden);
    expect(readTeamVocabulary(golden, table)).toEqual(golden);
  });

  it.each([
    ["a schema violation", () => ({ ...golden, schema_version: 2 }), /invalid team vocabulary/],
    ["a minimum team size above the maximum", () => edit((v) => (v.rules.min_team_size = v.rules.max_team_size + 1)), /min_team_size exceeds/],
    ["a brought team larger than the team", () => edit((v) => (v.rules.picked_team_size = v.rules.max_team_size + 1)), /picked_team_size exceeds/],
    ["unsorted natures", () => edit((v) => v.natures.reverse()), /natures: .* repeated or out of order/],
    ["repeated items", () => edit((v) => (v.items = [v.items[0] as string, v.items[0] as string])), /items: .* repeated/],
    ["an unsorted roster", () => edit((v) => v.roster.reverse()), /roster: .* repeated or out of order/],
    ["unsorted abilities", () => edit((v) => ((v.roster[1] as { abilities: string[] }).abilities = ["zzz", "aaa"])), /abilities: .* repeated or out of order/],
    ["repeated moves", () => edit((v) => ((v.roster[0] as { moves: string[] }).moves = ["pound", "pound"])), /moves: .* repeated/],
  ])("refuses %s", (_label, make, message) => {
    expect(() => readTeamVocabulary(make())).toThrow(TeamVocabularyError);
    expect(() => readTeamVocabulary(make())).toThrow(message);
  });

  it.each([
    ["another table", () => ({ vocabulary: golden, table: withTable((t) => (t.species[0] = { ...t.species[0], num: 99999 } as never)) }), /engine_data_hash is not the hash/],
    ["another commit", () => ({ vocabulary: edit((v) => (v.showdown_commit = "0".repeat(40))), table }), /showdown_commit differs/],
    ["another format", () => ({ vocabulary: edit((v) => (v.format_id = "gen9other")), table }), /format_id differs/],
    ["another mod", () => ({ vocabulary: edit((v) => (v.mod = "gen9")), table }), /mod differs/],
    ["an unknown item", () => ({ vocabulary: edit((v) => (v.items = ["aaaaaaa"])), table }), /item aaaaaaa is not in/],
    ["an unknown species", () => ({ vocabulary: edit((v) => ((v.roster[0] as { id: string }).id = "aaaaaaa")), table }), /roster species aaaaaaa is not in/],
    ["a foreign ability", () => ({ vocabulary: edit((v) => ((v.roster[0] as { abilities: string[] }).abilities = ["zzzzzz"])), table }), /not one of its engine-data abilities/],
    ["an unknown move", () => ({ vocabulary: edit((v) => ((v.roster[0] as { moves: string[] }).moves = ["zzzzzz"])), table }), /move zzzzzz is not in/],
    ["an unknown required item", () => ({ vocabulary: edit((v) => ((v.roster.find((r) => r.required_item) as { required_item: string }).required_item = "zzzzzz")), table }), /required item zzzzzz is not in/],
  ])("with the table, refuses %s", (_label, make, message) => {
    const { vocabulary, table: given } = make();
    const attempt = () => readTeamVocabulary(vocabulary, given);
    expect(attempt).toThrow(TeamVocabularyError);
    expect(attempt).toThrow(message);
  });

  it("with the table, an ability not in the table fails too, and the table itself must be valid", () => {
    expect(() => readTeamVocabulary(golden, { ...table, schema_version: 2 } as never)).toThrow();
  });
});
