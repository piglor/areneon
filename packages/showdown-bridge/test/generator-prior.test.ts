import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type Json, jcs } from "../src/canonical-json.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { type Engine, loadEngine } from "../src/engine.ts";
import { deriveGeneratorPrior, GENERATOR_PRIOR_SCRIPT, GeneratorPriorError, generatorConfigHash, megaHolderAbilities, megaPreviewSpecies } from "../src/generator-prior.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { readTeamGeneratorConfig } from "../src/team-records.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE, type TeamVocabulary, teamVocabularyHash } from "../src/team-vocabulary.ts";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const config = readTeamGeneratorConfig(JSON.parse(readFileSync(join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json"), "utf8")), table);
const COMMIT = "b7cda7af2f1744ad4934d38fb379b1d3cfc3ffc2";
const COMMITTED = join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, `generator-matched.${config.config_id}.json`);
const shows = megaPreviewSpecies(engine, vocabulary, table);
const holders = megaHolderAbilities(engine, vocabulary, shows);
const derive = (over: Partial<Parameters<typeof deriveGeneratorPrior>[0]> = {}) => deriveGeneratorPrior({ vocabulary, config, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, ...over });
const sha = (value: unknown): string => createHash("sha256").update(jcs(value as Json), "utf8").digest("hex");

/** A small vocabulary: `roster` and `items` of the golden one are replaced (the builder reads nothing else). */
const mini = (roster: TeamVocabulary["roster"], items: string[]): TeamVocabulary => ({ ...vocabulary, roster, items });
const entry = (id: string, abilities: string[], moves: string[], required_item?: string): TeamVocabulary["roster"][number] => ({ id, abilities, moves, ...(required_item === undefined ? {} : { required_item }) }) as never;

describe("the generator-matched prior of the committed config (ADR-025 section 10)", () => {
  const prior = derive();

  it("is a valid usage prior of the golden vocabulary that names the config and the commit", () => {
    expect(readUsagePrior(structuredClone(prior), vocabulary)).toEqual(prior);
    expect(prior.source).toEqual({ kind: "generator-matched", generator_config_hash: sha(config), generator_commit: COMMIT });
    expect(generatorConfigHash(config)).toBe(sha(config));
    expect(prior.vocabulary_hash).toBe(teamVocabularyHash(vocabulary));
    expect(prior.derivation.script_version).toBe(GENERATOR_PRIOR_SCRIPT);
  });

  it("is committed as reviewed data and rebuilt from the config byte for byte", () => {
    const bytes = readFileSync(COMMITTED, "utf8");
    expect(bytes).toBe(`${jcs(prior as unknown as Json)}\n`);
    expect(usagePriorHash(JSON.parse(bytes))).toBe(usagePriorHash(prior));
  });

  it("groups every Mega forme under the species it shows as, and every roster entry once", () => {
    const formes = prior.species.flatMap((s) => s.formes.map((f) => f.id));
    expect(formes.sort()).toEqual(vocabulary.roster.map((r) => r.id).sort());
    for (const s of prior.species) {
      expect(s.denominator).toBe(s.formes.length);
      expect(s.formes.every((f) => f.weight === 1)).toBe(true);
      expect(s.formes.some((f) => f.id === s.id)).toBe(true);
    }
    const abomasnow = prior.species.find((s) => s.id === "abomasnow");
    expect(abomasnow?.formes.map((f) => f.id)).toEqual(["abomasnow", "abomasnowmega"]);
    expect(shows.get("abomasnowmega")).toBe("abomasnow");
    expect(prior.species.length).toBe(vocabulary.roster.length - shows.size);
  });

  it("has the proposal's marginals: uniform abilities, the config's item weights, k of n moves", () => {
    const weight = new Map(config.item_weights.map((w) => [w.item, w.weight]));
    const positive = vocabulary.items.filter((i) => (weight.get(i) ?? config.default_item_weight) > 0).sort();
    for (const species of prior.species) {
      for (const f of species.formes) {
        const roster = vocabulary.roster.find((r) => r.id === f.id) as TeamVocabulary["roster"][number];
        expect(f.abilities.entries.map((e) => e.id)).toEqual([...new Set(roster.abilities)].sort());
        expect(new Set(f.abilities.entries.map((e) => e.weight)).size).toBe(1);
        expect(f.abilities.residual + f.abilities.unmapped).toBe(0);
        if (roster.required_item !== undefined) expect(f.items.entries.map((e) => e.id)).toEqual([roster.required_item]);
        else {
          expect(f.items.entries.map((e) => e.id)).toEqual(positive);
          expect(new Set(f.items.entries.map((e) => e.weight)).size).toBe(1);
        }
        expect(f.items.no_item + f.items.residual + f.items.unmapped).toBe(0);
        const n = roster.moves.length;
        const k = Math.min(4, n);
        expect(f.moves.denominator).toBe(f.items.denominator);
        expect(f.moves.slots).toBe(4);
        expect(f.moves.entries.map((e) => e.id)).toEqual([...roster.moves].sort());
        for (const e of f.moves.entries) expect(e.inclusion * n).toBe(f.moves.denominator * k);
        expect(f.moves.empty_slots).toBe((4 - k) * f.moves.denominator);
        expect(f.spreads).toEqual({ denominator: 1, entries: [], residual: 1, unmapped: 0 });
      }
    }
  });

  it("changes its hash with the config, with the commit and with the vocabulary", () => {
    const base = usagePriorHash(prior);
    expect(usagePriorHash(derive({ generator_commit: "0".repeat(40) }))).not.toBe(base);
    expect(usagePriorHash(derive({ config: { ...config, default_item_weight: 2 } }))).not.toBe(base);
    expect(usagePriorHash(derive({ config: { ...config, moves_per_set: 3 } }))).not.toBe(base);
    expect(usagePriorHash(derive({ config: { ...config, iv: 30 } }))).not.toBe(base);
    expect(usagePriorHash(derive({ vocabulary: { ...vocabulary, items: vocabulary.items.slice(0, -1) } }))).not.toBe(base);
    expect(usagePriorHash(derive())).toBe(base);
  });
});

describe("the stone holders' pre-Evolution abilities (#1167)", () => {
  const prior = derive();
  const pick = (mega: boolean) => {
    for (const s of prior.species) for (const f of s.formes) if ((f.id !== s.id) === mega) return { s, f: structuredClone(f), id: f.id };
    throw new Error("none");
  };

  it("every Mega forme holds the single ability the pinned validator leaves its holder, the base forme none", () => {
    const holders = megaHolderAbilities(engine, vocabulary, shows);
    expect([...holders.keys()].sort()).toEqual([...shows.keys()].sort());
    for (const s of prior.species) {
      for (const f of s.formes) {
        if (f.id === s.id) {
          expect(f.holder_abilities, f.id).toBeUndefined();
          continue;
        }
        const counts = holders.get(f.id) as ReadonlyMap<string, number>;
        expect(f.holder_abilities?.entries, f.id).toEqual([...counts].sort(([a], [b]) => (a < b ? -1 : 1)).map(([id, weight]) => ({ id, weight })));
        expect(f.holder_abilities?.denominator).toBe(entryAbilities(f.id));
        // The post-Evolution list is the Mega forme's own; the holder ability is one the preview species lists.
        expect(f.abilities.entries.map((e) => e.id)).toEqual([...new Set(vocabulary.roster.find((r) => r.id === f.id)?.abilities)].sort());
        for (const e of f.holder_abilities?.entries ?? []) expect(vocabulary.roster.find((r) => r.id === s.id)?.abilities).toContain(e.id);
      }
    }
    // Absol: the validator leaves the base forme's first listed ability whichever Mega ability was written.
    expect(holders.get("absolmega")).toEqual(new Map([["pressure", 1]]));
  });

  const entryAbilities = (id: string): number => new Set(vocabulary.roster.find((r) => r.id === id)?.abilities).size;

  it("the reader refuses a holder list on a forme that is not a Mega forme, an ability the species does not list and a bad sum", () => {
    const withHolder = (mega: boolean, change: (h: NonNullable<ReturnType<typeof pick>["f"]["holder_abilities"]>) => void) => {
      const target = pick(mega);
      const p = structuredClone(prior);
      const forme = p.species.find((s) => s.id === target.s.id)?.formes.find((f) => f.id === target.id) as (typeof target)["f"];
      forme.holder_abilities = { denominator: 1, entries: [{ id: vocabulary.roster.find((r) => r.id === target.s.id)?.abilities[0] as string, weight: 1 }], residual: 0, unmapped: 0 };
      change(forme.holder_abilities);
      return p;
    };
    expect(() => readUsagePrior(withHolder(false, () => undefined), vocabulary)).toThrow(/not a Mega forme/);
    expect(() => readUsagePrior(withHolder(true, (h) => (h.entries[0] = { id: "aaaaaaaaa", weight: 1 })), vocabulary)).toThrow(/holder ability aaaaaaaaa/);
    expect(() => readUsagePrior(withHolder(true, (h) => (h.denominator = 2)), vocabulary)).toThrow(/holder abilities: the weights/);
    expect(() => readUsagePrior(withHolder(true, () => undefined), vocabulary)).not.toThrow();
  });

  it("refuses a validator that moves the stone or the species, and a Mega forme with no result", () => {
    expect(() => deriveGeneratorPrior({ vocabulary, config, generator_commit: COMMIT, shows_as: shows, holder_abilities: new Map() })).toThrow(/no holder abilities/);
    expect(() => megaHolderAbilities(engine, vocabulary, new Map([["abomasnowmega", "snover"]]))).toThrow(/leaves the species/);
  });
});

describe("the derivation on small vocabularies", () => {
  const cfg = (change: Record<string, unknown> = {}) => ({ ...config, item_weights: [], default_item_weight: 1, moves_per_set: 4, ...change }) as typeof config;

  it("scales unequal item weights and a short pool to integer inclusions", () => {
    const v = mini([entry("aaa", ["x"], ["m1", "m2", "m3"]), entry("bbb", ["x", "y"], ["m1", "m2", "m3", "m4", "m5", "m6", "m7"])], ["i1", "i2", "i3"]);
    const prior = deriveGeneratorPrior({ vocabulary: v, config: cfg({ item_weights: [{ item: "i2", weight: 3 }, { item: "i3", weight: 0 }] }), generator_commit: COMMIT, shows_as: new Map(), holder_abilities: new Map() });
    const [a, b] = prior.species.map((s) => s.formes[0]);
    // aaa: 3 moves, 4 slots: a forced move each, one empty slot per set; items 1:3 over a pool of 3.
    expect(a?.items).toMatchObject({ denominator: 12, entries: [{ id: "i1", weight: 3 }, { id: "i2", weight: 9 }] });
    expect(a?.moves).toMatchObject({ denominator: 12, empty_slots: 12, entries: [{ id: "m1", inclusion: 12 }, { id: "m2", inclusion: 12 }, { id: "m3", inclusion: 12 }] });
    // bbb: 4 of 7 moves, each in 4/7 of the sets.
    expect(b?.items.denominator).toBe(28);
    expect(b?.moves.entries.every((e) => e.inclusion === 16)).toBe(true);
    expect(b?.moves.empty_slots).toBe(0);
    readUsagePrior(prior, v);
  });

  it("gives all the mass to no item when every weight is zero", () => {
    const v = mini([entry("aaa", ["x"], ["m1", "m2"])], ["i1"]);
    const prior = deriveGeneratorPrior({ vocabulary: v, config: cfg({ default_item_weight: 0 }), generator_commit: COMMIT, shows_as: new Map(), holder_abilities: new Map() });
    expect(prior.species[0]?.formes[0]?.items).toEqual({ denominator: 2, entries: [], no_item: 2, residual: 0, unmapped: 0 });
    readUsagePrior(prior, v);
  });

  it("holds a Mega forme's stone and shows it as the pre-Mega species", () => {
    const v = mini([entry("aaa", ["x"], ["m1"]), entry("aaamega", ["z"], ["m1"], "i1")], ["i1", "i2"]);
    const prior = deriveGeneratorPrior({ vocabulary: v, config: cfg({ item_weights: [{ item: "i1", weight: 0 }] }), generator_commit: COMMIT, shows_as: new Map([["aaamega", "aaa"]]), holder_abilities: new Map([["aaamega", new Map([["x", 1]])]]) });
    expect(prior.species.map((s) => s.id)).toEqual(["aaa"]);
    const [base, mega] = (prior.species[0] as { formes: { items: { entries: unknown[] } }[] }).formes;
    expect(base?.items.entries).toEqual([{ id: "i2", weight: 1 }]);
    expect(mega?.items.entries).toEqual([{ id: "i1", weight: 1 }]);
    readUsagePrior(prior, v);
  });

  it("refuses what it cannot derive", () => {
    const v = mini([entry("aaa", ["x"], ["m1"]), entry("aaamega", ["z"], ["m1"], "i1")], ["i1"]);
    const base = { vocabulary: v, config: cfg(), generator_commit: COMMIT, shows_as: new Map<string, string>(), holder_abilities: new Map<string, Map<string, number>>() };
    expect(() => deriveGeneratorPrior({ ...base, generator_commit: "abc" })).toThrow(GeneratorPriorError);
    const { moves_per_set: _, ...without } = cfg();
    expect(() => deriveGeneratorPrior({ ...base, config: without as never })).toThrow(/no moves_per_set/);
    expect(() => deriveGeneratorPrior({ ...base, config: cfg({ item_weights: [{ item: "nope", weight: 1 }] }) })).toThrow(/does not list/);
    expect(() => deriveGeneratorPrior({ ...base, shows_as: new Map([["aaamega", "ghost"]]) })).toThrow(/not in the roster/);
    expect(() => deriveGeneratorPrior({ ...base, shows_as: new Map([["aaamega", "aaa"], ["aaa", "aaamega"]]) })).toThrow(/itself shows as another/);
    const huge = mini([entry("aaa", ["x"], Array.from({ length: 7 }, (_, k) => `m${k}`))], ["i1"]);
    expect(() => deriveGeneratorPrior({ ...base, vocabulary: huge, config: cfg({ default_item_weight: 200_000_000_000 }) })).toThrow(/too large/);
  });
});

describe("megaPreviewSpecies reads the pinned engine's stone data", () => {
  it("maps every Mega forme of the roster to its pre-Mega forme", () => {
    const megas = vocabulary.roster.filter((r) => r.required_item !== undefined);
    expect(shows.size).toBeGreaterThan(50);
    expect(shows.size).toBeLessThanOrEqual(megas.length);
    for (const [mega, pre] of shows) {
      expect(vocabulary.roster.some((r) => r.id === pre && r.required_item === undefined)).toBe(true);
      expect(mega.startsWith(pre.slice(0, 4))).toBe(true);
    }
  });

  const fake = (stone: unknown): Engine => ({ sim: { Dex: { mod: () => ({ items: { get: () => ({ megaStone: stone }) } }) } } }) as never;
  const v = mini([entry("aaa", ["x"], ["m1"]), entry("bbb", ["x"], ["m1"]), entry("aaamega", ["z"], ["m1"], "i1"), entry("mask", ["z"], ["m1"], "i2")], ["i1", "i2"]);
  const names = { ...table, species: [{ id: "aaa", name: "Aaa" }, { id: "bbb", name: "Bbb" }, { id: "aaamega", name: "Aaa-Mega" }, { id: "mask", name: "Mask" }] } as never;

  it("takes one pre-Mega forme, leaves an entry the stone map does not name as its own species", () => {
    expect(megaPreviewSpecies(fake({ Aaa: "Aaa-Mega" }), v, names)).toEqual(new Map([["aaamega", "aaa"]]));
    expect(megaPreviewSpecies(fake(undefined), v, names).size).toBe(0);
    expect(megaPreviewSpecies(fake(["Aaa"]), v, names).size).toBe(0);
    expect(megaPreviewSpecies(fake({ Other: "Else" }), v, names).size).toBe(0);
  });

  it("refuses a stone that names two forms or a form outside the roster", () => {
    expect(() => megaPreviewSpecies(fake({ Aaa: "Aaa-Mega", Bbb: "Aaa-Mega" }), v, names)).toThrow(/exactly one/);
    expect(() => megaPreviewSpecies(fake({ Ghost: "Aaa-Mega" }), v, names)).toThrow(/exactly one/);
  });
});
