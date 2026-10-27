import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type Json, jcs } from "../src/canonical-json.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { deriveGeneratorPrior, megaPreviewSpecies } from "../src/generator-prior.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { deriveRoleSetsPrior, ROLE_SETS_PRIOR_SCRIPT, RoleSetsPriorError } from "../src/role-sets-prior.ts";
import { readRoleSetTable, ROLE_SETS_FILE, roleSetTableHash, type RoleSetTable } from "../src/role-sets.ts";
import { readTeamGeneratorConfig } from "../src/team-records.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE, type TeamVocabulary, teamVocabularyHash } from "../src/team-vocabulary.ts";
import { readUsagePrior, usagePriorHash } from "../src/usage-prior.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const OPPONENT = join(ROOT, "config", "opponent-model", GOLDEN_FORMAT);
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const roleSets = readRoleSetTable(JSON.parse(readFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")), vocabulary);
const fillBytes = readFileSync(join(OPPONENT, "generator-matched.uniform-v1.json"), "utf8");
const fill = readUsagePrior(JSON.parse(fillBytes), vocabulary);
const shows = megaPreviewSpecies(engine, vocabulary, table);
const derive = (over: Partial<Parameters<typeof deriveRoleSetsPrior>[0]> = {}) => deriveRoleSetsPrior({ vocabulary, table: roleSets, fill, shows_as: shows, ...over });
const prior = derive();
const entryOf = (id: string) => vocabulary.roster.find((r) => r.id === id) as TeamVocabulary["roster"][number];
const setsOf = (id: string) => (roleSets.species.find((s) => s.id === id) as RoleSetTable["species"][number]).sets;
const formeOf = (id: string) => prior.species.flatMap((s) => s.formes).find((f) => f.id === id) as (typeof prior.species)[number]["formes"][number];

describe("the role-sets prior of the committed table (ADR-025 section 2)", () => {
  it("is a valid usage prior that names the table and the fill prior by hash", () => {
    expect(readUsagePrior(structuredClone(prior), vocabulary)).toEqual(prior);
    expect(prior.source).toEqual({ kind: "role-sets", role_set_table_hash: roleSetTableHash(roleSets), fill_prior_hash: usagePriorHash(fill) });
    expect(prior.vocabulary_hash).toBe(teamVocabularyHash(vocabulary));
    expect(prior.derivation.script_version).toBe(ROLE_SETS_PRIOR_SCRIPT);
  });

  it("is committed as reviewed data and rebuilt from the table and the fill byte for byte", () => {
    const bytes = readFileSync(join(OPPONENT, "role-sets.fill-uniform-v1.json"), "utf8");
    expect(bytes).toBe(`${jcs(prior as unknown as Json)}\n`);
    expect(usagePriorHash(JSON.parse(bytes))).toBe(usagePriorHash(prior));
    // The fill is the committed generator-matched record, itself rebuilt by its own test.
    expect(fillBytes).toBe(`${jcs(fill as unknown as Json)}\n`);
  });

  it("holds every table entry once, grouped under the species it shows as, weighted by its roles", () => {
    const formes = prior.species.flatMap((s) => s.formes.map((f) => f.id));
    expect(formes.sort()).toEqual(roleSets.species.map((s) => s.id).sort());
    for (const s of prior.species) {
      expect(s.denominator).toBe(s.formes.reduce((t, f) => t + setsOf(f.id).length, 0));
      for (const f of s.formes) expect(f.weight).toBe(setsOf(f.id).length);
    }
    const venusaur = prior.species.find((s) => s.id === "venusaur");
    expect(venusaur?.formes.map((f) => [f.id, f.weight])).toEqual([["venusaur", 1], ["venusaurmega", 2]]);
  });

  it("draws a role's moves uniformly: the inclusion of a move is the exact mean of k / n over the roles that hold it", () => {
    for (const s of prior.species) {
      for (const f of s.formes) {
        const roles = setsOf(f.id);
        const slots = f.moves.slots;
        const ids = new Set(roles.flatMap((r) => r.moves));
        expect(f.moves.entries.map((e) => e.id)).toEqual([...ids].sort());
        expect(f.moves.residual + f.moves.unmapped).toBe(0);
        // inclusion / denominator = sum over roles of k_r / (W n_r), compared by cross multiplication.
        for (const e of f.moves.entries) {
          let num = 0n;
          let den = 1n;
          for (const r of roles) {
            const k = BigInt(Math.min(slots, r.moves.length));
            const n = BigInt(r.moves.length);
            const w = BigInt(roles.length);
            const hit = r.moves.includes(e.id) ? k : 0n;
            // num/den + hit/(w n)
            num = num * (w * n) + hit * den;
            den *= w * n;
          }
          expect(BigInt(e.inclusion) * den, `${f.id} ${e.id}`).toBe(num * BigInt(f.moves.denominator));
        }
        const empty = roles.reduce((t, r) => t + (slots - Math.min(slots, r.moves.length)), 0);
        expect(f.moves.empty_slots * roles.length).toBe(empty * f.moves.denominator);
      }
    }
  });

  it("draws a role's abilities uniformly, roles equal; a Mega forme's pre-Evolution reading is in the base forme's list", () => {
    const share = (f: ReturnType<typeof formeOf>, id: string): number => (f.abilities.entries.find((e) => e.id === id)?.weight ?? 0) / f.abilities.denominator;
    // Venusaur: one base role {chlorophyll, overgrow} and two Mega roles {chlorophyll}: 3 roles in the base forme's list.
    expect(share(formeOf("venusaur"), "chlorophyll")).toBeCloseTo((1 / 2 + 2) / 3, 12);
    expect(share(formeOf("venusaur"), "overgrow")).toBeCloseTo(1 / 6, 12);
    // The Mega forme's own list is the roster's post-Evolution ability.
    expect(formeOf("venusaurmega").abilities.entries.map((e) => e.id)).toEqual(entryOf("venusaurmega").abilities);
    // Aggron's Mega role names Sturdy, which the base forme's role does not: the base list has it, so a Sturdy holder is not excluded before the Evolution.
    expect(formeOf("aggron").abilities.entries.map((e) => e.id)).toEqual(["rockhead", "sturdy"]);
    // Every ability of the base forme's list is one the vocabulary lists for it, and all mass is mapped.
    for (const s of prior.species) {
      for (const f of s.formes) {
        expect(f.abilities.entries.every((e) => entryOf(f.id).abilities.includes(e.id))).toBe(true);
        expect(f.abilities.residual + f.abilities.unmapped).toBe(0);
        expect(f.abilities.entries.reduce((t, e) => t + e.weight, 0)).toBe(f.abilities.denominator);
      }
    }
  });

  it("takes items, natures and Stat Points from the fill prior: its item shares, its stone, its other-spread entry", () => {
    const fillOf = (id: string) => fill.species.flatMap((s) => s.formes).find((f) => f.id === id) as (typeof fill.species)[number]["formes"][number];
    for (const f of prior.species.flatMap((s) => s.formes)) {
      const theirs = fillOf(f.id);
      expect(f.spreads).toEqual(theirs.spreads);
      expect(f.items.entries.map((e) => e.id)).toEqual(theirs.items.entries.map((e) => e.id));
      for (let k = 0; k < f.items.entries.length; k++) expect((f.items.entries[k]?.weight as number) * theirs.items.denominator).toBe((theirs.items.entries[k]?.weight as number) * f.items.denominator);
      expect(f.items.no_item * theirs.items.denominator).toBe(theirs.items.no_item * f.items.denominator);
      const stone = entryOf(f.id).required_item;
      if (stone !== undefined) expect(f.items.entries.map((e) => e.id)).toEqual([stone]);
    }
    expect(prior.derivation.k_spreads).toBe(fill.derivation.k_spreads);
  });

  it("changes its hash with the table, with the fill and with the vocabulary it names", () => {
    const base = usagePriorHash(prior);
    expect(usagePriorHash(derive())).toBe(base);
    const other = deriveGeneratorPrior({ vocabulary, config: readTeamGeneratorConfig(JSON.parse(readFileSync(join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json"), "utf8")), table), generator_commit: "0".repeat(40), shows_as: shows });
    const changed = derive({ fill: other });
    expect(changed.source).toMatchObject({ kind: "role-sets", fill_prior_hash: usagePriorHash(other) });
    expect(usagePriorHash(changed)).not.toBe(base);
    const fewer = { ...roleSets, species: roleSets.species.filter((s) => s.id !== "abomasnow") } as RoleSetTable;
    expect(usagePriorHash(derive({ table: fewer }))).not.toBe(base);
  });
});

describe("the derivation on small vocabularies", () => {
  const entry = (id: string, abilities: string[], moves: string[], required_item?: string): TeamVocabulary["roster"][number] => ({ id, abilities, moves, ...(required_item === undefined ? {} : { required_item }) }) as never;
  const config = { ...readTeamGeneratorConfig(JSON.parse(readFileSync(join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json"), "utf8")), table), item_weights: [], default_item_weight: 1, moves_per_set: 4 };
  const mini = (roster: TeamVocabulary["roster"], items: string[]) => {
    const v: TeamVocabulary = { ...vocabulary, roster, items };
    const f = deriveGeneratorPrior({ vocabulary: v, config, generator_commit: "1".repeat(40), shows_as: new Map([["aaamega", "aaa"]]) });
    const t = (species: RoleSetTable["species"]): RoleSetTable => ({ ...roleSets, vocabulary_hash: teamVocabularyHash(v), species });
    return { v, f, t };
  };
  const role = (moves: [string, ...string[]], abilities: [string, ...string[]]) => ({ role: "r", moves, abilities });

  it("forces the moves of a short role, counts its empty slots and scales to integer inclusions", () => {
    const { v, f, t } = mini([entry("aaa", ["x", "y"], ["m1", "m2", "m3", "m4", "m5", "m6"]), entry("aaamega", ["z"], ["m1", "m2", "m3", "m4", "m5", "m6"], "i1")], ["i1", "i2", "i3"]);
    const p = deriveRoleSetsPrior({ vocabulary: v, table: t([{ id: "aaa", sets: [role(["m1", "m2", "m3"], ["x"]), role(["m1", "m2", "m3", "m4", "m5", "m6"], ["x", "y"])] }]), fill: f, shows_as: new Map([["aaamega", "aaa"]]) });
    const base = p.species[0]?.formes[0];
    // Two roles: a short one (3 moves, a slot empty) and a long one (4 of 6): m1 is in both, m4 in the long one only.
    const d = base?.moves.denominator as number;
    const inc = (id: string) => (base?.moves.entries.find((e) => e.id === id)?.inclusion as number) / d;
    expect(inc("m1")).toBeCloseTo((1 + 4 / 6) / 2, 12);
    expect(inc("m4")).toBeCloseTo(4 / 6 / 2, 12);
    expect(base?.moves.empty_slots).toBe(d / 2);
    expect(base?.abilities.entries.map((e) => [e.id, e.weight / base.abilities.denominator])).toEqual([["x", 0.75], ["y", 0.25]]);
    expect(p.species[0]?.denominator).toBe(2);
    readUsagePrior(p, v);
  });

  it("keeps a species whose table entries are all stone holders without a base forme", () => {
    const { v, f, t } = mini([entry("aaa", ["x", "y"], ["m1", "m2"]), entry("aaamega", ["z"], ["m1", "m2"], "i1")], ["i1", "i2"]);
    const p = deriveRoleSetsPrior({ vocabulary: v, table: t([{ id: "aaamega", sets: [role(["m1", "m2"], ["x"])] }]), fill: f, shows_as: new Map([["aaamega", "aaa"]]) });
    expect(p.species.map((s) => [s.id, s.formes.map((x) => x.id)])).toEqual([["aaa", ["aaamega"]]]);
    readUsagePrior(p, v);
  });

  it("refuses a table, a fill or a map that does not fit", () => {
    const { v, f, t } = mini([entry("aaa", ["x"], ["m1"]), entry("aaamega", ["z"], ["m1"], "i1")], ["i1"]);
    const ok = { vocabulary: v, table: t([{ id: "aaa", sets: [role(["m1"], ["x"])] }]), fill: f, shows_as: new Map<string, string>() };
    expect(deriveRoleSetsPrior(ok).species).toHaveLength(1);
    expect(() => deriveRoleSetsPrior({ ...ok, table: roleSets })).toThrow(RoleSetsPriorError);
    expect(() => deriveRoleSetsPrior({ ...ok, fill: prior })).toThrow(/not a generator-matched/);
    expect(() => deriveRoleSetsPrior({ ...ok, fill: { ...f, vocabulary_hash: "0".repeat(64) } })).toThrow(/not derived for this vocabulary/);
    expect(() => deriveRoleSetsPrior({ ...ok, shows_as: new Map([["aaa", "ghost"]]) })).toThrow(/not in the roster/);
    expect(() => deriveRoleSetsPrior({ ...ok, table: t([{ id: "aaa", sets: [role(["m1"], ["x"])] }, { id: "aaamega", sets: [role(["m1"], ["z"])] }]), shows_as: new Map([["aaa", "aaamega"], ["aaamega", "aaa"]]) })).toThrow(/itself shows as another/);
    const noForme = { ...f, species: [] } as never;
    expect(() => deriveRoleSetsPrior({ ...ok, fill: noForme })).toThrow(/no forme aaa/);
  });
});
