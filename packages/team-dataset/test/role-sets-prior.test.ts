// The role-sets prior against the table it reads and the fills it names (ADR-025 section 2, slice 1): worlds
// sampled from a belief over the committed record, with the generator's SpreadSampler as the fill, vary in
// item, nature and Stat Points, keep a role's moves and abilities at the table's frequencies, give a move
// outside every role of its species probability zero, and pass the pinned validator.
// `packages/opponent-model` is imported by path only, as in generator-prior.test.ts.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createSetLegality, type RoleSetTable, readRoleSetTable, readUsagePrior, teamVocabularyHash } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { interpretChannel } from "../../public-state/src/index.ts";
import { BeliefError, belief, createSampler, type OpponentBelief, type WorldSet } from "../../opponent-model/src/index.ts";
import { generatorSpreadSampler } from "../src/spread-sampler.ts";
import { config, engine, FORMAT, table, vocabulary } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const prior = readUsagePrior(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", FORMAT, "role-sets.fill-uniform-v1.json"), "utf8")), vocabulary);
const roleSets: RoleSetTable = readRoleSetTable(JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json"), "utf8")), vocabulary);
const view = { ...vocabulary, vocabulary_hash: teamVocabularyHash(vocabulary) };
const beliefConfig = { schema_version: 1 as const, identity_hiding_abilities: [], max_legality_redraws: 32, max_disguise_assignments: 64 };

// A Mega-bearing species with one base role and two Mega roles (venusaur), one with one Mega role (gallade, two base roles),
// a single-role species with a five-move pool (charizard), one with two roles of 5 and 6 moves (aerodactyl) and two more.
const PREVIEW = ["venusaur", "gallade", "charizard", "aerodactyl", "garchomp", "abomasnow"] as const;
const names = new Map(table.species.map((s) => [s.id, s.name]));
const header = ["|gametype|doubles", "|player|p1|Alice|", "|player|p2|Bob|", "|clearpoke", ...PREVIEW.map((id) => `|poke|p2|${names.get(id)}, L50, M|`)];
const opponentOf = (extra: string[] = []) => {
  const run = interpretChannel("p1", [...header, ...extra]);
  return { opponent: (run.frames[run.frames.length - 1] as NonNullable<(typeof run.frames)[number]>).state.opponent, flags: run.flags };
};
const base = opponentOf();
const beliefOf = (o = base): OpponentBelief => belief({ prior, vocabulary: view, config: beliefConfig, opponent: o.opponent, flags: o.flags });

const Z = 5.5;
const within = (hits: number, n: number, p: number, what: string): void => {
  const sd = Math.sqrt(n * p * (1 - p));
  expect(Math.abs(hits - n * p), `${what}: ${hits} of ${n} against p=${p}`).toBeLessThanOrEqual(Z * Math.max(sd, 1));
};
const entryOf = (id: string) => vocabulary.roster.find((r) => r.id === id) as (typeof vocabulary.roster)[number];
const rolesOf = (id: string) => roleSets.species.find((s) => s.id === id)?.sets ?? [];
const weights = new Map(config.item_weights.map((w) => [w.item, w.weight]));
const freeItems = vocabulary.items.filter((i) => (weights.get(i) ?? config.default_item_weight) > 0);
const stream = (a: number, b: number) => Uint8Array.from({ length: 32 }, (_, i) => (i * a + b) & 255);

describe("sampled worlds of the role-sets prior", () => {
  const sampler = createSampler({ legality: { legal: () => true }, spreads: generatorSpreadSampler(vocabulary, config) });
  const b = beliefOf();
  const WORLDS = 6000;
  const out = sampler.sampleWorlds(b, stream(7, 1), WORLDS);
  const sets: WorldSet[] = out.worlds.flatMap((w) => w.world.slots.flatMap((s) => (s.set === null ? [] : [s.set])));

  it("hold each forme at its share of the species' roles, and every move inside one of the forme's roles", () => {
    const bySpecies = new Map<string, WorldSet[]>();
    for (const s of sets) bySpecies.set(s.species, [...(bySpecies.get(s.species) ?? []), s]);
    expect([...bySpecies.keys()].sort()).toEqual([...PREVIEW].sort());
    for (const [species, list] of bySpecies) {
      const entry = prior.species.find((s) => s.id === species) as (typeof prior.species)[number];
      for (const f of entry.formes) {
        const mine = list.filter((s) => s.forme === f.id);
        within(mine.length, list.length, f.weight / entry.denominator, `${f.id} share of ${species}`);
        const pool = new Set(rolesOf(f.id).flatMap((r) => r.moves));
        for (const s of mine) {
          expect(s.moves.every((m) => pool.has(m)), `${f.id} ${s.moves}`).toBe(true);
          expect(new Set(s.moves).size).toBe(s.moves.length);
          expect(s.moves.length).toBeGreaterThan(0);
          expect(s.moves.length).toBeLessThanOrEqual(f.moves.slots);
        }
      }
    }
  });

  it("have a single role's moves at k / n and its abilities uniform", () => {
    const charizard = sets.filter((s) => s.forme === "charizard");
    const role = rolesOf("charizard");
    expect(role).toHaveLength(1);
    const pool = (role[0] as (typeof role)[number]).moves;
    for (const m of pool) within(charizard.filter((s) => s.moves.includes(m)).length, charizard.length, 4 / pool.length, `charizard ${m}`);
    for (const a of (role[0] as (typeof role)[number]).abilities) within(charizard.filter((s) => s.ability === a).length, charizard.length, 1 / (role[0] as (typeof role)[number]).abilities.length, `charizard ${a}`);
  });

  it("show a stone holder's pre-Evolution ability from the species' roles, and its stone", () => {
    const mega = sets.filter((s) => s.forme === "venusaurmega");
    expect(mega.length).toBeGreaterThan(1000);
    expect(mega.every((s) => s.item === entryOf("venusaurmega").required_item)).toBe(true);
    // The base forme's list mixes the three roles of the species: Chlorophyll in 2.5 of 3, Overgrow in 0.5 of 3.
    within(mega.filter((s) => s.ability === "chlorophyll").length, mega.length, 5 / 6, "venusaurmega chlorophyll");
    expect(mega.every((s) => s.ability === "chlorophyll" || s.ability === "overgrow")).toBe(true);
  });

  it("take items, natures and Stat Points from the fill: uniform items and natures, the format's Stat Point total", () => {
    const free = sets.filter((s) => entryOf(s.forme).required_item === undefined);
    for (const item of freeItems) within(free.filter((s) => s.item === item).length, free.length, 1 / freeItems.length, `item ${item}`);
    for (const n of vocabulary.natures) within(sets.filter((s) => s.nature === n.id).length, sets.length, 1 / vocabulary.natures.length, `nature ${n.id}`);
    for (const s of sets) {
      const points = Object.values(s.stat_points);
      expect(points.reduce((t, v) => t + v, 0)).toBe(vocabulary.rules.ev_limit);
      expect(Math.max(...points)).toBeLessThanOrEqual(config.stat_point_max_per_stat);
    }
  });

  it("differ between streams in item, nature and Stat Points", () => {
    const draw = (a: number, c: number) => sampler.sampleWorlds(b, stream(a, c), 40).worlds.flatMap((w) => w.world.slots.flatMap((s) => (s.set === null ? [] : [s.set])));
    const one = draw(3, 5);
    const two = draw(11, 9);
    expect(one.map((s) => s.item)).not.toEqual(two.map((s) => s.item));
    expect(one.map((s) => s.nature)).not.toEqual(two.map((s) => s.nature));
    expect(one.map((s) => JSON.stringify(s.stat_points))).not.toEqual(two.map((s) => JSON.stringify(s.stat_points)));
    // Within one stream the fills vary too (the worlds are not one set repeated).
    expect(new Set(one.map((s) => s.nature)).size).toBeGreaterThan(5);
    expect(new Set(one.map((s) => JSON.stringify(s.stat_points))).size).toBeGreaterThan(20);
    expect(JSON.stringify(draw(3, 5))).toBe(JSON.stringify(one));
  });
});

describe("a move outside a species' role sets", () => {
  // The known mismatch with the uniform-team generator, whose sets draw from the whole roster pool: the prior gives such a move no mass.
  const outside = (species: string, formes: string[]): string => {
    const inRoles = new Set(formes.flatMap((f) => rolesOf(f).flatMap((r) => r.moves)));
    return entryOf(species).moves.filter((m) => !inRoles.has(m)).sort()[0] as string;
  };
  const gallade = outside("gallade", ["gallade", "gallademega"]);
  const MOVE = (id: string) => table.moves.find((m) => m.id === id)?.name as string;
  const reveal = (move: string) =>
    opponentOf(["|poke|p1|Garchomp, L50, M|", "|teampreview|4", "|start", "|switch|p1a: Garchomp|Garchomp, L50, M|100/100", "|switch|p2a: Gallade|Gallade, L50, M|100/100", "|turn|1", `|move|p2a: Gallade|${MOVE(move)}|p1a: Garchomp`]);

  it("has probability zero: the belief has no world that holds it", () => {
    expect(gallade).toBeDefined();
    expect(entryOf("gallade").moves).toContain(gallade);
    let caught: unknown;
    try {
      beliefOf(reveal(gallade));
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(BeliefError);
    expect((caught as BeliefError).code).toBe("evidence_has_no_support");
  });

  it("is a refusal only for the move outside the roles: a role's own move conditions the belief", () => {
    const inside = rolesOf("gallade")[0]?.moves.find((m) => m !== "protect") as string;
    const b = beliefOf(reveal(inside));
    const sampler = createSampler({ legality: { legal: () => true }, spreads: generatorSpreadSampler(vocabulary, config) });
    const gal = sampler.sampleWorlds(b, stream(5, 2), 200).worlds.flatMap((w) => w.world.slots.flatMap((s) => (s.set === null ? [] : [s.set]))).filter((s) => s.species === "gallade");
    expect(gal.length).toBeGreaterThan(100);
    expect(gal.every((s) => s.moves.includes(inside))).toBe(true);
  });
});

describe("sampled worlds pass the pinned validator", () => {
  it("every brought set is legal under the target format, with Item Clause across the team", () => {
    const legality = createSetLegality({ engine, vocabulary });
    const sampler = createSampler({ legality, spreads: generatorSpreadSampler(vocabulary, config) });
    const out = sampler.sampleWorlds(beliefOf(), stream(5, 3), 150);
    expect(out.redraws).toBe(0);
    let count = 0;
    for (const { world } of out.worlds) {
      const worldSets = world.slots.flatMap((s) => (s.set === null ? [] : [s.set]));
      const items = worldSets.flatMap((s) => (s.item === null ? [] : [s.item]));
      expect(new Set(items).size).toBe(items.length);
      for (const set of worldSets) {
        expect(legality.legal(set), `${set.forme} ${JSON.stringify(set)}`).toBe(true);
        count += 1;
      }
    }
    expect(count).toBe(150 * vocabulary.rules.picked_team_size);
  });
});
