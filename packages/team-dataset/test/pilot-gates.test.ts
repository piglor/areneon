// The pilot gates P1 to P4 and the per-component report of a mixture dataset (ADR-020 section 6, E-13.5, E-13.6):
// P3 against hand-worked toys, the gate arithmetic, the cross-split near-duplicate counts on a hand-built pair of
// splits, and one small mixture generation. Generation tests are small: the validator is slow.

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { recordHash, type PokemonSet, type TeamGeneratorConfig, teamVocabularyHash } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { Drawer } from "../src/draw.ts";
import { generate, GenerateError } from "../src/generate.ts";
import { crossSplitNear } from "../src/near-index.ts";
import { type Pilot, readPilot, runPilot } from "../src/pilot.ts";
import { bitsOf, GateError, populationOf, proposalCoverage, roleSetsBound, roleSetsMapping, uniformBound, mixtureBound } from "../src/pilot-gates.ts";
import { planQuotas } from "../src/plan.ts";
import { Proposal } from "../src/proposal.ts";
import { radices } from "../src/labeller.ts";
import { roleSetsSource } from "../src/role-set-proposal.ts";
import { ASSIGNMENT_KEY, engine, format, FORMAT, KEY, light, small, table, vocabulary } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const source = roleSetsSource(engine, vocabulary, table, JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json"), "utf8")));
const frac = (num: number, den: number) => ({ num: BigInt(num), den: BigInt(den) });

describe("bitsOf", () => {
  it("is the largest b with value <= 2^-b", () => {
    expect(bitsOf(frac(1, 1))).toBe(0);
    expect(bitsOf(frac(3, 2))).toBe(0);
    expect(bitsOf(frac(1, 2))).toBe(1);
    expect(bitsOf(frac(1, 3))).toBe(1);
    expect(bitsOf(frac(1, 900))).toBe(9); // 1/512 >= 1/900 > 1/1024
    expect(bitsOf(frac(1, 1024))).toBe(10);
    expect(bitsOf(frac(7, 1800))).toBe(8);
    expect(bitsOf({ num: 1n, den: 2n ** 100n })).toBe(100);
  });
  it("refuses a bound of zero", () => {
    expect(() => bitsOf(frac(0, 1))).toThrow(RangeError);
  });
});

// Toy vocabulary: teams of 2.
const toyRoster = (rows: { id: string; abilities: number; moves: number; required?: string }[]) =>
  rows.map((r) => ({
    id: r.id,
    abilities: Array.from({ length: r.abilities }, (_, k) => `ab${k}`),
    moves: Array.from({ length: r.moves }, (_, k) => `mv${k}`),
    ...(r.required === undefined ? {} : { required_item: r.required }),
  }));
const toyConfig: Parameters<typeof uniformBound>[1] = { item_weights: [], default_item_weight: 1, gender_policy: "unset", moves_per_set: 2 };
const toy = (roster: ReturnType<typeof toyRoster>) => ({ roster, items: ["i1", "i2", "i3", "i4"], natures: Array.from({ length: 5 }, (_, k) => ({ name: `n${k}` })) as never, rules: { max_team_size: 2, item_clause: 1 } }) as never;

describe("P3, the bound on the most likely team, against a toy worked by hand", () => {
  // Uniform: four entries. Each keeps a 1/5 nature factor; items are four of weight 1, Item Clause 1, so the free
  // item factor is 1 / (4 - 1) = 1/3 (the largest weight over the total less the largest weight one earlier member can hold).
  //   A: 1 ability, 2 moves (C(2,2) = 1), required item      -> 1 * 1 * 1/5          = 1/5
  //   B: 2 abilities, 3 moves (C(3,2) = 3), required item    -> 1/6 * 1/5            = 1/30
  //   C: 1 ability, 4 moves (C(4,2) = 6), free item          -> 1/6 * 1/3 * 1/5      = 1/90
  //   D: 4 abilities, 2 moves (1), free item                 -> 1/4 * 1/3 * 1/5      = 1/60
  // The two largest are A and B: 1/150. Orderings of a pair out of 4: 2! / (4 * 3) = 1/6. Bound 1/900: 2^-10 < 1/900 <= 2^-9.
  const roster = toyRoster([
    { id: "a", abilities: 1, moves: 2, required: "i1" },
    { id: "b", abilities: 2, moves: 3, required: "i2" },
    { id: "c", abilities: 1, moves: 4 },
    { id: "d", abilities: 4, moves: 2 },
  ]);
  it("uniform: 1/900, nine bits", () => {
    const bound = uniformBound(toy(roster), toyConfig);
    expect(bound.num * 900n).toBe(bound.den);
    expect(bitsOf(bound)).toBe(9);
  });

  it("role sets: an entry's bound is the mean of its roles' bounds", () => {
    // Three table entries, every roster entry holds a required item (free item factor 1), nature 1/5, orderings 2!/(3*2) = 1/3.
    //   E1: roles (1 ability, 2 moves: 1) and (2 abilities, 3 moves: 1/6) -> (1 + 1/6) / 2 = 7/12, times 1/5 = 7/60
    //   E2: one role (1 ability, 4 moves: 1/6)                             -> 1/6 * 1/5 = 1/30
    //   E3: one role (2 abilities, 2 moves: 1/2)                           -> 1/2 * 1/5 = 1/10
    // Largest two are E1 and E3: 7/600; times 1/3 = 7/1800 (= 0.00389), above 2^-9 and below 2^-8.
    const names = (n: number, p: string) => Array.from({ length: n }, (_, k) => `${p}${k}`);
    const role = (abilities: number, moves: number) => ({ role: "r", abilities: names(abilities, "ab"), moves: names(moves, "mv") });
    const roster3 = toyRoster([
      { id: "e1", abilities: 1, moves: 3, required: "i1" },
      { id: "e2", abilities: 1, moves: 4, required: "i2" },
      { id: "e3", abilities: 2, moves: 2, required: "i3" },
    ]);
    const tableOf = { species: ([{ id: "e1", sets: [role(1, 2), role(2, 3)] }, { id: "e2", sets: [role(1, 4)] }, { id: "e3", sets: [role(2, 2)] }] as unknown) as Parameters<typeof roleSetsBound>[2]["species"] };
    const bound = roleSetsBound(toy(roster3), toyConfig, tableOf);
    expect(bound.num * 1800n).toBe(7n * bound.den);
    expect(bitsOf(bound)).toBe(8);
  });

  it("the mixture's bound is the share-weighted sum", () => {
    const sum = mixtureBound([{ share_permille: 300, bound: frac(1, 900) }, { share_permille: 700, bound: frac(7, 1800) }]);
    // 0.3/900 + 0.7 * 7/1800 = 1/3000 + 49/18000 = 6/18000 + 49/18000 = 55/18000
    expect(sum.num * 18000n).toBe(55n * sum.den);
    expect(bitsOf(sum)).toBe(8);
  });

  it("random gender halves a member's probability and no Item Clause removes nothing", () => {
    const withGender = uniformBound(toy(roster), { ...toyConfig, gender_policy: "random" });
    const base = uniformBound(toy(roster), toyConfig);
    expect(withGender.num * base.den * 4n).toBe(base.num * withGender.den); // two members, each 1/2
    const noClause = uniformBound({ roster, items: ["i1", "i2", "i3", "i4"], natures: Array.from({ length: 5 }, () => ({ name: "n" })), rules: { max_team_size: 2 } } as never, toyConfig);
    // the free item factor is now 1/4, smaller: C and D shrink, A and B (required items) are unchanged, so the bound is too
    expect(noClause.num * base.den).toBe(base.num * noClause.den);
  });

  it("on the real vocabulary, no component comes close to the 2^-40 default of the ADR", () => {
    const config = small();
    expect(bitsOf(uniformBound(vocabulary, config))).toBeGreaterThanOrEqual(40);
    expect(bitsOf(roleSetsBound(vocabulary, config, source.table))).toBeGreaterThanOrEqual(40);
  });
});

describe("P1 for the role-set component", () => {
  it("counts the table's ids that resolve in the vocabulary", () => {
    const real = roleSetsMapping(vocabulary, source.table);
    expect(real.mapped).toBe(real.total);
    const broken = { species: [{ id: "nosuchmon", sets: [{ role: "r", moves: ["nosuchmove"], abilities: ["nosuchability"] }] }, ...source.table.species.slice(0, 1)] as unknown as typeof source.table.species };
    const bad = roleSetsMapping(vocabulary, broken);
    expect(bad.total - bad.mapped).toBe(3); // the species, the move and the ability
  });
});

const set = (species: string, over: Partial<PokemonSet> = {}): PokemonSet => ({
  name: "", species, item: "Leftovers", ability: "Pressure", moves: ["Tackle", "Protect"], nature: "Adamant", gender: "", evs: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50, ...over,
});

const far = (species: string): PokemonSet => set(species, { item: "Sitrus Berry", ability: "Static", nature: "Jolly", moves: ["Surf"], evs: { hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 0 } });

const other = (species: string): PokemonSet => set(species, { item: "Choice Scarf", ability: "Intimidate", nature: "Bold", moves: ["Fire Blast"], evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 252 } });

describe("cross-split near-duplicate counts", () => {
  // Teams of two. dev has T1 = (a, b) and T2 = (c, d); training has T3 = (e, f) built from `far` sets. The distance counts differing fields (of six) over the best pairing.
  const dev = [[set("a"), set("b")], [set("c"), set("d")]];
  const training = [[far("e"), far("f")]];
  const validation = [
    [set("a"), set("b", { nature: "Jolly" })], // 1 from T1
    [set("a", { item: "Sitrus Berry", ability: "Static" }), set("b")], // 2 from T1
    [other("g"), other("h")], // far from everything (6 fields in each of two sets)
    [far("e"), { ...far("f"), moves: ["Surf", "Protect"] }], // 1 from T3
  ];
  it("counts validation and dev teams within d_team of the other splits, counts only", () => {
    const one = crossSplitNear({ dev, validation, training }, 1);
    expect(one).toEqual({
      d_team: 1,
      pairs: [
        { of: "validation", against: "dev", teams: 4, near: 1 },
        { of: "validation", against: "training", teams: 4, near: 1 },
        { of: "dev", against: "training", teams: 2, near: 0 },
      ],
    });
    const two = crossSplitNear({ dev, validation, training }, 2);
    expect(two.pairs.map((p) => p.near)).toEqual([2, 1, 0]);
    expect(crossSplitNear({ dev, validation, training }, 0).pairs.map((p) => p.near)).toEqual([0, 0, 0]);
    // No team of any split appears in the output.
    expect(JSON.stringify(two)).not.toMatch(/Adamant|Leftovers|Tackle/);
  });
  it("skips a pair of which a split is missing", () => {
    expect(crossSplitNear({ dev, validation }, 1).pairs).toEqual([{ of: "validation", against: "dev", teams: 4, near: 1 }]);
    expect(crossSplitNear({ validation }, 1).pairs).toEqual([]);
  });
});

describe("a small mixture generation", () => {
  const scratch = mkdtempSync(join(tmpdir(), "pilot-gates-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));
  const mixed = (gates?: object): TeamGeneratorConfig =>
    small({
      config_id: "gates-test",
      proposal: { kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: source.hash, share_permille: 700 }, { kind: "uniform", share_permille: 300 }], ...(gates === undefined ? {} : { gates }) } as never,
    });
  const identityOf = (cfg: TeamGeneratorConfig) => ({ config_hash: recordHash(cfg), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) });
  const pilotOf = (cfg: TeamGeneratorConfig): Pilot =>
    runPilot(new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, cfg, source), labelSet: light, generationKey: KEY }), identityOf(cfg), light, cfg.pilot_draws);
  const loose = { p2_min_accept_permille: 1, p3_min_bits: 1, p4_min_strata_permille: 1 };
  let n = 0;
  const run = (cfg: TeamGeneratorConfig, over: object = {}) => {
    const logDir = join(scratch, `log-${n++}`);
    mkdirSync(logDir);
    return generate({
      engine, format, table, vocabulary, labelSet: light, config: cfg, roleSets: source,
      generationKey: KEY, assignmentKey: ASSIGNMENT_KEY, datasetId: "gates-1", planId: "plan-1",
      sizes: { dev: 2, validation: 2, training: 5 }, generatorCommit: "a".repeat(40),
      reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, logDir, ...over,
    });
  };

  const cfg = mixed(loose);
  const plain = structuredClone(cfg);
  delete plain.proposal;
  const pilot = pilotOf(cfg);
  const uniformPilot = runPilot(new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, plain), labelSet: light, generationKey: KEY }), identityOf(plain), light, plain.pilot_draws);
  const result = run(cfg, { pilot, uniformPilot, dTeam: 2 });
  const coverage = result.coverage;

  it("records the pilot per component, adding up to the pilot", () => {
    const parts = pilot.components ?? [];
    expect(parts.map((p) => p.kind)).toEqual(["role_sets", "uniform"]);
    expect(parts.reduce((s, p) => s + p.draws, 0)).toBe(pilot.draws);
    expect(parts.reduce((s, p) => s + p.rejected, 0)).toBe(pilot.rejected);
    expect(readPilot(JSON.parse(JSON.stringify(pilot)), identityOf(cfg), light)).toEqual(pilot);
  });

  it("reports P1 to P4 per component with integers only", () => {
    const report = coverage.proposal;
    expect(report?.components.map((c) => c.kind)).toEqual(["role_sets", "uniform"]);
    const [r, u] = report?.components ?? [];
    expect(r?.p1).toMatchObject({ value: 1000, threshold: 1000, pass: true });
    expect(u?.p1).toBeNull();
    const part = (kind: string) => pilot.components?.find((p) => p.kind === kind) as NonNullable<Pilot["components"]>[number];
    for (const c of [r, u]) {
      const p = part(c?.kind as string);
      expect(c?.pilot).toEqual({ draws: p.draws, rejected: p.rejected, accepted: p.draws - p.rejected });
      expect(c?.p2.value).toBe(Math.floor(((p.draws - p.rejected) * 1000) / p.draws));
      expect(c?.p2.pass).toBe(true);
      expect(c?.pilot_cell_hits).toEqual(p.cell_hits);
    }
    expect(r?.p3.value).toBe(bitsOf(roleSetsBound(vocabulary, cfg, source.table)));
    expect(u?.p3.value).toBe(bitsOf(uniformBound(vocabulary, cfg)));
    // P4: the reference plan is the uniform pilot's, at the same d_max.
    const reference = planQuotas({ cellHits: uniformPilot.cell_hits, radices: radices(light), pilotDraws: uniformPilot.draws, pilotMinHits: cfg.pilot_min_hits, dMax: cfg.d_max, nStratumMin: cfg.n_stratum_min, sizes: { dev: 2, validation: 2, training: 5 }, forced: new Set() });
    expect(report?.p4).toMatchObject({ strata: result.plan.strata.length, reference_strata: reference.strata.length, threshold: 1, pass: true });
    expect(report?.publishable).toBe(true);
    expect(report?.failed).toEqual([]);
    expect(JSON.stringify(report).includes(".")).toBe(false); // no fraction anywhere
  });

  it("counts the delivered teams per component and cell", () => {
    const delivered = coverage.proposal?.components.map((c) => c.delivered_cells.reduce((a, b) => a + b, 0)) ?? [];
    const total = (["dev", "validation", "training"] as const).reduce((s, split) => s + (result.records[split]?.length ?? 0), 0);
    expect(delivered.reduce((a, b) => a + b, 0)).toBe(total);
    for (const c of coverage.proposal?.components ?? []) expect(Object.values(c.delivered).reduce((a, b) => a + (b ?? 0), 0)).toBe(c.delivered_cells.reduce((a, b) => a + b, 0));
  });

  it("states the population and the cross-split counts", () => {
    expect(coverage.population).toBe("proposal mixture gates-test (role_sets 700 per mille, uniform 300 per mille) conditioned on the merged cells, weighted by the published quotas");
    expect(coverage.near_duplicates?.d_team).toBe(2);
    expect(coverage.near_duplicates?.pairs.map((p) => `${p.of}:${p.against}`)).toEqual(["validation:dev", "validation:training", "dev:training"]);
  });

  it("is byte-identical on rerun", () => {
    const again = run(cfg, { pilot, uniformPilot, dTeam: 2 });
    expect(JSON.stringify(again.coverage)).toBe(JSON.stringify(coverage));
  });

  it("does not publish without configured gates, and reports why", () => {
    const bare = proposalCoverage({ config: mixed(), vocabulary, roleSets: source, pilot, plan: result.plan, referenceStrata: 1, delivered: new Map() });
    expect(bare.publishable).toBe(false);
    expect(bare.failed).toContain("gates:not_configured");
    expect(bare.components.every((c) => c.p2.pass === null && c.p3.pass === null && c.p2.threshold === null && c.p3.threshold === null)).toBe(true);
  });

  it("fails each gate on its own threshold, and reports P4 as not evaluated without the reference pilot", () => {
    const base = { vocabulary, roleSets: source, pilot, plan: result.plan, delivered: new Map() };
    const strict = (over: object) => proposalCoverage({ ...base, config: mixed({ ...loose, ...over }), referenceStrata: result.plan.strata.length });
    const [first, second] = pilot.components as NonNullable<Pilot["components"]>;
    const pilotBase: Pilot = pilot;
    const allRejected = { ...pilotBase, components: [{ ...(first as NonNullable<typeof first>), rejected: (first as NonNullable<typeof first>).draws, cell_hits: (first as NonNullable<typeof first>).cell_hits.map(() => 0) }, second as NonNullable<typeof second>] };
    expect(proposalCoverage({ ...base, pilot: allRejected, config: mixed({ ...loose, p2_min_accept_permille: 500 }), referenceStrata: result.plan.strata.length }).failed).toEqual(["P2:role_sets"]);
    expect(strict({ p3_min_bits: 1000 }).failed).toEqual(["P3:role_sets", "P3:uniform"]);
    expect(strict({ p4_min_strata_permille: 1000 }).failed).toEqual([]); // equal strata meet 1000
    expect(proposalCoverage({ ...base, config: mixed({ ...loose, p4_min_strata_permille: 1000 }), referenceStrata: result.plan.strata.length * 2 }).failed).toEqual(["P4"]);
    const none = proposalCoverage({ ...base, config: mixed(loose), referenceStrata: null });
    expect(none.failed).toEqual(["P4"]);
    expect(none.p4).toMatchObject({ reference_strata: null, value: null, pass: null });
  });

  it("refuses a recorded pilot without component counts, a bad uniform reference, and a config with no proposal", () => {
    const { components: _components, ...old } = pilot;
    expect(() => run(cfg, { pilot: old })).toThrow(GenerateError);
    expect(() => run(cfg, { pilot, uniformPilot: { ...uniformPilot, config_hash: "0".repeat(64) } })).toThrow(/uniform reference pilot/);
    expect(() => proposalCoverage({ config: small(), vocabulary, pilot, plan: result.plan, referenceStrata: null, delivered: new Map() })).toThrow(GateError);
    expect(() => proposalCoverage({ config: cfg, vocabulary, roleSets: source, pilot: old, plan: result.plan, referenceStrata: null, delivered: new Map() })).toThrow(/no per-component counts/);
  });

  it("reports a component the pilot never drew as 0 per mille, and a missing role-set table as an error", () => {
    const [first, second] = pilot.components as NonNullable<Pilot["components"]>;
    const none = { ...(first as NonNullable<typeof first>), draws: 0, rejected: 0, cell_hits: (first as NonNullable<typeof first>).cell_hits.map(() => 0) };
    const base = { vocabulary, roleSets: source, plan: result.plan, delivered: new Map(), referenceStrata: 1 };
    const report = proposalCoverage({ ...base, config: mixed(loose), pilot: { ...pilot, components: [none, second as NonNullable<typeof second>] } });
    expect(report.components[0]?.p2).toMatchObject({ value: 0, pass: false });
    expect(report.failed).toContain("P2:role_sets");
    expect(() => proposalCoverage({ ...base, roleSets: undefined, config: mixed(loose), pilot })).toThrow(/no role-set table/);
    expect(roleSetsMapping(vocabulary, { species: [] })).toEqual({ mapped: 0, total: 0 });
    expect(proposalCoverage({ ...base, config: mixed(loose), pilot, roleSets: { table: { species: [] } } }).components[0]?.p1).toMatchObject({ value: 0, pass: false });
  });

  it("a uniform dataset's coverage has none of the new fields", () => {
    const uniform = run(small(), { pilot: runPilot(new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, small()), labelSet: light, generationKey: KEY }), identityOf(small()), light, small().pilot_draws) });
    expect(Object.keys(uniform.coverage)).not.toEqual(expect.arrayContaining(["proposal"]));
    expect(uniform.coverage.population).toBeUndefined();
    expect(uniform.coverage.near_duplicates).toBeUndefined();
    expect(uniform.pilot.components).toBeUndefined();
    expect(uniform.manifest.evaluation_only).toBeUndefined();
  });

  it("states a uniform population too", () => {
    expect(populationOf(small())).toContain("proposal uniform");
    expect(populationOf(small(), "note")).toMatch(/; note$/);
  });
}, 600_000); // the pilots and runs draw a few hundred teams each through the pinned validator
