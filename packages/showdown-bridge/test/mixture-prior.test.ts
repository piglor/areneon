import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { megaHolderAbilities, megaPreviewSpecies } from "../src/generator-prior.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { apportion, deriveMixturePrior, MIXTURE_DENOMINATOR, MixturePriorError, successiveInclusion } from "../src/mixture-prior.ts";
import { readRoleSetTable, ROLE_SETS_FILE, roleSetTableHash } from "../src/role-sets.ts";
import { readTeamGeneratorConfig, type TeamGeneratorConfig } from "../src/team-records.ts";
import { readTeamVocabulary, TEAM_VOCABULARY_FILE } from "../src/team-vocabulary.ts";
import { readUsagePrior } from "../src/usage-prior.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const vocabulary = readTeamVocabulary(JSON.parse(readFileSync(join(GOLDEN_DIR, TEAM_VOCABULARY_FILE), "utf8")), table);
const config = readTeamGeneratorConfig(JSON.parse(readFileSync(join(ROOT, "config", "teams", GOLDEN_FORMAT, "generator.json"), "utf8")), table);
const roles = readRoleSetTable(JSON.parse(readFileSync(join(GOLDEN_DIR, ROLE_SETS_FILE), "utf8")), vocabulary);
const shows = megaPreviewSpecies(engine, vocabulary, table);
const holders = megaHolderAbilities(engine, vocabulary, shows);
const COMMIT = "a".repeat(40);
const mixture = (components: unknown[]): TeamGeneratorConfig => ({ ...config, proposal: { kind: "mixture", components } }) as unknown as TeamGeneratorConfig;
const derive = (cfg: TeamGeneratorConfig, over: object = {}) => deriveMixturePrior({ vocabulary, config: cfg, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, ...over });

describe("apportion", () => {
  it("fixes masses to an exact integer total by largest remainder, ties to the lower index", () => {
    expect(apportion([1, 1, 1], 100)).toEqual([34, 33, 33]);
    expect(apportion([0.5, 0.25, 0.25], 10)).toEqual([5, 3, 2]);
    expect(apportion([0, 0], 10)).toEqual([0, 0]);
  });
});

describe("successiveInclusion", () => {
  it("is k/n for equal weights and any floor, and refuses a mask past the limit", () => {
    const pool = ["a", "b", "c", "d"];
    const got = successiveInclusion(pool, pool.map((id) => ({ id, w: 1 })), 0.3, 2);
    for (const id of pool) expect(got.get(id) as number).toBeCloseTo(0.5, 12);
    const many = Array.from({ length: 41 }, (_, k) => `m${k}`);
    expect(() => successiveInclusion(many, many.map((id) => ({ id, w: 1 })), 0, 2)).toThrow(MixturePriorError);
  });
  it("puts the weight on the listed moves when the floor is 0 and spreads the rest over the unlisted", () => {
    const got = successiveInclusion(["a", "b", "c"], [{ id: "a", w: 1 }], 0, 2);
    expect(got.get("a")).toBeCloseTo(1, 12);
    expect(got.get("b")).toBeCloseTo(0.5, 12);
  });
});

describe("deriveMixturePrior", () => {
  it("a mixture of the uniform proposal alone is the uniform prior's distribution", () => {
    const prior = deriveMixturePrior({ vocabulary, config: mixture([{ kind: "uniform", share_permille: 1000 }]), generator_commit: COMMIT, shows_as: shows, holder_abilities: holders });
    expect(readUsagePrior(structuredClone(prior), vocabulary)).toEqual(prior);
    const base = prior.species.flatMap((s) => s.formes).find((f) => f.id === "gallade") as (typeof prior.species)[number]["formes"][number];
    const pool = vocabulary.roster.find((r) => r.id === "gallade")?.moves.length as number;
    expect(base.moves.entries).toHaveLength(pool);
    expect(prior.derivation.k_spreads).toBe(1);
    expect(prior.species.flatMap((s) => s.formes).every((f) => f.spreads.entries.length === 0 && f.spreads.residual === MIXTURE_DENOMINATOR)).toBe(true);
  });

  it("a role_sets-only mixture draws only the table's entries, and a move outside every role has no mass", () => {
    const prior = derive(mixture([{ kind: "role_sets", role_set_table_hash: roleSetTableHash(roles), share_permille: 1000 }]), { role_sets: roles });
    const ids = new Set(prior.species.flatMap((s) => s.formes.map((f) => f.id)));
    expect(ids).toEqual(new Set(roles.species.map((s) => s.id)));
    const entry = roles.species[0] as (typeof roles.species)[number];
    const forme = prior.species.flatMap((s) => s.formes).find((f) => f.id === entry.id) as (typeof prior.species)[number]["formes"][number];
    const inRoles = new Set(entry.sets.flatMap((s) => s.moves));
    expect(forme.moves.entries.every((m) => inRoles.has(m.id))).toBe(true);
  });

  it("refuses what the builder does not know or cannot check", () => {
    const u = { kind: "uniform", share_permille: 1000 };
    expect(() => derive({ ...config })).toThrow(/no mixture/);
    expect(() => derive({ ...config, proposal: { kind: "other", components: [u] } } as never)).toThrow(/not a mixture/);
    expect(() => derive(mixture([u, u]))).toThrow(/twice/);
    expect(() => derive(mixture([{ kind: "uniform", share_permille: 900 }]))).toThrow(/sum to 1000/);
    expect(() => derive(mixture([{ kind: "mystery", share_permille: 1000 }]))).toThrow(/does not know/);
    expect(() => derive(mixture([{ kind: "role_sets", role_set_table_hash: "0".repeat(64), share_permille: 1000 }]), { role_sets: roles })).toThrow(/not the one/);
    expect(() => derive(mixture([{ kind: "usage", usage_table_hash: "0".repeat(64), species_root: 2, set_root: 3, floor_permille: 100, share_permille: 1000 }]))).toThrow(/no usage table/);
    expect(() => derive(mixture([u]), { role_sets: roles })).toThrow(/no role_sets component/);
    expect(() => deriveMixturePrior({ vocabulary, config: mixture([u]), generator_commit: "x", shows_as: shows, holder_abilities: holders })).toThrow(/40/);
    expect(() => derive({ ...mixture([u]), moves_per_set: undefined } as never)).toThrow(/moves_per_set/);
  });
});
