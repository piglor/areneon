// The generator-matched prior of an ADR-020 mixture config (ADR-025 slice 3b, #1134) against the mixture proposal it
// describes: the committed records are rebuilt byte for byte, the exact pieces (usage move inclusion by the
// successive-draw programme) are held to the real `UsageSampler` within binomial bounds, and the proposal's own
// per-entry frequencies are compared with the record, with the deviation of the approximated piece (the usage
// component's species inclusion) measured and bounded.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  deriveMixturePrior,
  jcs,
  megaHolderAbilities,
  megaPreviewSpecies,
  readRoleSetTable,
  readTeamGeneratorConfig,
  readUsagePrior,
  readUsageWeightTable,
  rootedWeight,
  usageTableHash,
  successiveInclusion,
  type PokemonSet,
  type UsagePrior,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { Proposal } from "../src/proposal.ts";
import { roleSetsSource } from "../src/role-set-proposal.ts";
import { type SpreadChoice, UsageSampler, usageSource } from "../src/usage-proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { engine, FORMAT, KEY, table, vocabulary } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const COMMIT = "26a1256302bdf1fecbac68b322624e7608b72352";
const json = (...parts: string[]): unknown => JSON.parse(readFileSync(join(ROOT, ...parts), "utf8")) as unknown;
const configU = readTeamGeneratorConfig(json("config", "teams", FORMAT, "generator.mixture-u.json"), table);
const configR = readTeamGeneratorConfig(json("config", "teams", FORMAT, "generator.mixture-r.json"), table);
const roleTable = readRoleSetTable(json("fixtures", "showdown", FORMAT, "role-sets.json"), vocabulary);
const usage = readUsageWeightTable(json("config", "teams", FORMAT, "usage", "2026-09-1630.json"), vocabulary);
const shows = megaPreviewSpecies(engine, vocabulary, table);
const holders = megaHolderAbilities(engine, vocabulary, shows);
const derive = (config: typeof configU): UsagePrior => deriveMixturePrior({ vocabulary, config, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, role_sets: roleTable, ...(config.proposal?.components.some((c) => c.kind === "usage") ? { usage } : {}) });
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const Z = 5.5;
const within = (hits: number, n: number, p: number, what: string): void => {
  const sd = Math.sqrt(n * p * (1 - p));
  expect(Math.abs(hits - n * p), `${what}: ${hits} of ${n} against p=${p}`).toBeLessThanOrEqual(Z * Math.max(sd, 1));
};

describe("the committed records", () => {
  for (const [name, config] of [["mixture-r-v1", configR], ["mixture-u-v1", configU]] as const) {
    it(`${name} is rebuilt byte for byte and passes the reader`, () => {
      const committed = readFileSync(join(ROOT, "config", "opponent-model", FORMAT, `generator-matched.${name}.json`), "utf8");
      const prior = derive(config);
      expect(`${jcs(prior as never)}\n`).toBe(committed);
      expect(readUsagePrior(JSON.parse(committed), vocabulary)).toEqual(prior);
    });
  }
});

describe("successiveInclusion", () => {
  // The exact programme against a direct enumeration of ordered draws on a toy pool.
  const enumerate = (pool: string[], listed: { id: string; w: number }[], floor: number, slots: number): Map<string, number> => {
    const out = new Map(pool.map((id) => [id, 0]));
    const go = (taken: string[], p: number): void => {
      if (taken.length === slots) {
        for (const id of taken) out.set(id, (out.get(id) as number) + p);
        return;
      }
      const left = pool.filter((id) => !taken.includes(id));
      const wl = listed.filter((l) => !taken.includes(l.id));
      const wSum = wl.reduce((t, l) => t + l.w, 0);
      for (const id of left) {
        const q = wSum > 0 ? floor / left.length + ((1 - floor) * (listed.find((l) => l.id === id)?.w ?? 0)) / wSum : 1 / left.length;
        go([...taken, id], p * q);
      }
    };
    go([], 1);
    return out;
  };
  it("equals the enumeration with a floor, unlisted moves and a short pool", () => {
    const pool = ["a", "b", "c", "d", "e", "f"];
    const listed = [{ id: "a", w: 5 }, { id: "b", w: 3 }, { id: "c", w: 1 }];
    for (const floor of [0, 0.1, 0.5, 1]) {
      const got = successiveInclusion(pool, listed, floor, 3);
      const want = enumerate(pool, listed, floor, 3);
      for (const id of pool) expect(got.get(id) as number).toBeCloseTo(want.get(id) as number, 12);
      expect([...got.values()].reduce((a, b) => a + b, 0)).toBeCloseTo(3, 12);
    }
    const short = successiveInclusion(["a", "b"], [{ id: "a", w: 1 }], 0.2, 4);
    expect(short.get("a")).toBeCloseTo(1, 12);
    expect(short.get("b")).toBeCloseTo(1, 12);
  });
});

describe("the usage component's moves against the real sampler", () => {
  it("the programme's inclusion equals the UsageSampler's move frequencies", () => {
    const sampler = new UsageSampler(vocabulary, usageSource(vocabulary, usage), 3, 100, 4);
    const row = [...usage.species].sort((a, b) => b.usage_ppm - a.usage_ppm)[0] as (typeof usage.species)[number];
    const entry = vocabulary.roster.find((r) => r.id === row.id) as (typeof vocabulary.roster)[number];
    const pool = [...new Set(entry.moves)].sort();
    const listed = row.moves.entries.map((e) => ({ id: e.id, w: rootedWeight(e.ppm, 3) })).filter((e) => e.w > 0);
    const want = successiveInclusion(pool, listed, 0.1, 4);
    const uniform = { item: () => "", spread: (): SpreadChoice => ({ nature: "hardy", stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }) };
    const n = 40_000;
    const counts = new Map<string, number>();
    for (let k = 0; k < n; k++) {
      const rng = engineRng(engine, drawSeed(KEY, { config_hash: "e".repeat(64), plan_id: "mixture-prior-moves" }, k));
      for (const m of sampler.details(rng, entry, uniform).moves) counts.set(m, (counts.get(m) ?? 0) + 1);
    }
    for (const id of pool) within(counts.get(id) ?? 0, n, want.get(id) as number, `${row.id} ${id}`);
  });
});

describe("the mixture-u record against the proposal's frequencies", () => {
  const prior = derive(configU);
  const proposal = new Proposal(vocabulary, table, configU, roleSetsSource(engine, vocabulary, table, JSON.parse(JSON.stringify(roleTable))), usageSource(vocabulary, usage));
  const TEAMS = 30_000;
  const speciesOf = new Map(table.species.map((s) => [s.name, s.id]));
  const generated = new Map<string, PokemonSet[]>();
  for (let k = 0; k < TEAMS; k++) {
    for (const set of proposal.propose(engineRng(engine, drawSeed(KEY, { config_hash: "f".repeat(64), plan_id: "mixture-prior" }, k)))) {
      // The role-set component plays a Mega entry as its preview species holding the stone: that set is the stone holder (the Mega forme entry).
      const base = speciesOf.get(set.species) as string;
      const holder = vocabulary.roster.find((r) => r.required_item !== undefined && r.required_item === toId(set.item) && shows.get(r.id) === base);
      const id = holder?.id ?? base;
      generated.set(id, [...(generated.get(id) ?? []), set]);
    }
  }
  const formes = prior.species.flatMap((s) => s.formes.map((f) => ({ preview: s.id, f, species: s })));

  it("the within-species forme shares are the record's within 0.03 (measured 0.012 on a fixed seed)", () => {
    let worst = 0;
    let compared = 0;
    for (const s of prior.species) {
      if (s.formes.length < 2) continue;
      const counts = s.formes.map((f) => generated.get(f.id)?.length ?? 0);
      const total = counts.reduce((a, b) => a + b, 0);
      if (total < 2000) continue;
      s.formes.forEach((f, i) => {
        worst = Math.max(worst, Math.abs((counts[i] as number) / total - f.weight / s.denominator));
        compared += 1;
      });
    }
    expect(compared).toBeGreaterThan(10);
    expect(worst).toBeLessThanOrEqual(0.03);
  });

  it("per-entry ability, item, move and spread marginals are the record's within 0.06 (measured 0.033)", () => {
    const worst = { ability: 0, item: 0, move: 0, spread: 0 };
    let entries = 0;
    for (const { f } of formes) {
      const sets = generated.get(f.id) ?? [];
      const n = sets.length;
      if (n < 1500) continue;
      entries += 1;
      const entry = vocabulary.roster.find((r) => r.id === f.id) as (typeof vocabulary.roster)[number];
      if (entry.required_item === undefined) {
        for (const a of f.abilities.entries) worst.ability = Math.max(worst.ability, Math.abs(sets.filter((s) => toId(s.ability) === a.id).length / n - a.weight / f.abilities.denominator));
        for (const i of f.items.entries) worst.item = Math.max(worst.item, Math.abs(sets.filter((s) => toId(s.item) === i.id).length / n - i.weight / f.items.denominator));
      }
      for (const m of f.moves.entries) worst.move = Math.max(worst.move, Math.abs(sets.filter((s) => s.moves.some((x) => toId(x) === m.id)).length / n - m.inclusion / f.moves.denominator));
      for (const sp of f.spreads.entries) {
        const hit = sets.filter((s) => s.nature === vocabulary.natures.find((x) => x.id === sp.nature)?.name && (Object.keys(sp.stat_points) as (keyof typeof sp.stat_points)[]).every((k) => (s.evs as Record<string, number>)[k] === sp.stat_points[k])).length;
        worst.spread = Math.max(worst.spread, Math.abs(hit / n - sp.weight / f.spreads.denominator));
      }
    }
    expect(entries).toBeGreaterThan(20);
    expect(worst.ability).toBeLessThanOrEqual(0.06);
    expect(worst.item).toBeLessThanOrEqual(0.06);
    expect(worst.move).toBeLessThanOrEqual(0.06);
    expect(worst.spread).toBeLessThanOrEqual(0.06);
  });
});

describe("the usage-shaped evaluation config of the later month", () => {
  it("names the committed 2026-10 table, which is another month than the prior's and reads against the vocabulary", () => {
    const evalConfig = readTeamGeneratorConfig(json("config", "teams", FORMAT, "generator.usage-eval-2026-10.json"), table);
    const later = readUsageWeightTable(json("config", "teams", FORMAT, "usage", "2026-10-1630.json"), vocabulary);
    expect(later.month).toBe("2026-10");
    expect(usage.month).toBe("2026-09");
    expect(evalConfig.proposal?.components).toEqual([{ kind: "usage", share_permille: 1000, usage_table_hash: usageTableHash(later), species_root: later.derivation.species_root, set_root: 3, floor_permille: 100 }]);
    expect(usageTableHash(later)).not.toBe(usageTableHash(usage));
  });
});

describe("refusals", () => {
  it("refuses a config with no mixture, a missing or foreign table, a lone table without its component", () => {
    const { proposal: _drop, ...rest } = configU;
    const plain = rest as typeof configU;
    expect(() => deriveMixturePrior({ vocabulary, config: plain, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders })).toThrow(/no mixture/);
    expect(() => deriveMixturePrior({ vocabulary, config: configU, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, usage })).toThrow(/no role-set table/);
    expect(() => deriveMixturePrior({ vocabulary, config: configU, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, role_sets: roleTable })).toThrow(/no usage table/);
    expect(() => deriveMixturePrior({ vocabulary, config: configR, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, role_sets: roleTable, usage })).toThrow(/no usage component|has no usage/);
    expect(() => deriveMixturePrior({ vocabulary, config: configR, generator_commit: "abc", shows_as: shows, holder_abilities: holders, role_sets: roleTable })).toThrow(/40/);
    const unknown = { ...configR, proposal: { ...configR.proposal, components: [{ kind: "gamble", share_permille: 1000 }] } } as unknown as typeof configR;
    expect(() => deriveMixturePrior({ vocabulary, config: unknown, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders })).toThrow(/does not know/);
    const other = { ...roleTable, species: roleTable.species.slice(1) };
    expect(() => deriveMixturePrior({ vocabulary, config: configR, generator_commit: COMMIT, shows_as: shows, holder_abilities: holders, role_sets: other })).toThrow(/not the one/);
  });
});
