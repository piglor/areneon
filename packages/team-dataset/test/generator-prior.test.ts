// The generator-matched prior against the generator it describes (ADR-025 section 10, slice 3): worlds
// sampled from a belief over the committed prior, with the generator's SpreadSampler, have the item,
// ability, nature, move and Stat Point frequencies of the proposal itself, within stated binomial bounds,
// and pass the pinned validator. `packages/opponent-model` is imported by path only: the package never
// imports this one and this one does not depend on it.

import { createSetLegality, deriveGeneratorPrior, megaHolderAbilities, megaPreviewSpecies, type PokemonSet } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { interpretChannel } from "../../public-state/src/index.ts";
import { belief, createSampler, type OpponentBelief, type WorldSet } from "../../opponent-model/src/index.ts";
import { Proposal } from "../src/proposal.ts";
import { generatorSpreadSampler } from "../src/spread-sampler.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { config, engine, KEY, table, vocabulary } from "./support.ts";
import { teamVocabularyHash } from "@areneon/showdown-bridge";

const COMMIT = "b7cda7af2f1744ad4934d38fb379b1d3cfc3ffc2";
const prior = deriveGeneratorPrior({ vocabulary, config, generator_commit: COMMIT, shows_as: megaPreviewSpecies(engine, vocabulary, table), holder_abilities: megaHolderAbilities(engine, vocabulary, megaPreviewSpecies(engine, vocabulary, table)) });
const view = { ...vocabulary, vocabulary_hash: teamVocabularyHash(vocabulary) };
const beliefConfig = { schema_version: 1 as const, identity_hiding_abilities: [], max_legality_redraws: 32, max_disguise_assignments: 64 };

// The preview: a Mega-bearing species with two abilities, one with two Megas, the smallest move pool (Ditto,
// one move: three empty slots), the largest (Gallade, 106), and a species with three abilities.
const PREVIEW = ["abomasnow", "charizard", "ditto", "gallade", "aerodactyl", "garchomp"] as const;
const names = new Map(table.species.map((s) => [s.id, s.name]));
const lines = ["|gametype|doubles", "|player|p1|Alice|", "|player|p2|Bob|", "|clearpoke", ...PREVIEW.map((id) => `|poke|p2|${names.get(id)}, L50, M|`)];
const run = interpretChannel("p1", lines);
const opponent = (run.frames[run.frames.length - 1] as NonNullable<(typeof run.frames)[number]>).state.opponent;

/** z bound for every comparison: two-sided 5.5 sigma, so a fixed-seed pass is not luck (declared in the ticket as binomial bounds). */
const Z = 5.5;
const within = (hits: number, n: number, p: number, what: string): void => {
  const sd = Math.sqrt(n * p * (1 - p));
  expect(Math.abs(hits - n * p), `${what}: ${hits} of ${n} against p=${p}`).toBeLessThanOrEqual(Z * Math.max(sd, 1));
};

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const entryOf = (id: string) => vocabulary.roster.find((r) => r.id === id) as (typeof vocabulary.roster)[number];
const weights = new Map(config.item_weights.map((w) => [w.item, w.weight]));
const freeItems = vocabulary.items.filter((i) => (weights.get(i) ?? config.default_item_weight) > 0);
const slots = config.moves_per_set as number;
const shown = megaPreviewSpecies(engine, vocabulary, table);

describe("sampled worlds have the generator's frequencies", () => {
  const spreads = generatorSpreadSampler(vocabulary, config);
  const sampler = createSampler({ legality: { legal: () => true }, spreads });
  const b: OpponentBelief = belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags: run.flags });
  const WORLDS = 6000;
  const stream = Uint8Array.from({ length: 32 }, (_, i) => (i * 7 + 1) & 255);
  const worlds = sampler.sampleWorlds(b, stream, WORLDS);
  const sets: WorldSet[] = worlds.worlds.flatMap((w) => w.world.slots.flatMap((s) => (s.set === null ? [] : [s.set])));

  it("draws four brought sets per world with no redraw", () => {
    expect(sets).toHaveLength(WORLDS * vocabulary.rules.picked_team_size);
    expect(worlds.redraws).toBe(0);
  });

  it("matches the proposal's per-species marginals: forme, ability, item, nature and moves", () => {
    const bySpecies = new Map<string, WorldSet[]>();
    for (const s of sets) bySpecies.set(s.species, [...(bySpecies.get(s.species) ?? []), s]);
    expect([...bySpecies.keys()].sort()).toEqual([...PREVIEW].sort());
    let checks = 0;
    for (const [species, list] of bySpecies) {
      const formes = prior.species.find((s) => s.id === species)?.formes.map((f) => f.id) as string[];
      for (const forme of formes) {
        const mine = list.filter((s) => s.forme === forme);
        const entry = entryOf(forme);
        within(mine.length, list.length, 1 / formes.length, `${forme} share of ${species}`);
        const n = mine.length;
        const k = Math.min(slots, entry.moves.length);
        // An unevolved Mega forme shows what the validator leaves a stone holder: the preview species' first ability (`holder_abilities`, #1167).
        const holder = (prior.species.find((s) => s.id === species) as { formes: { id: string; holder_abilities?: { denominator: number; entries: { id: string; weight: number }[] } }[] }).formes.find((f) => f.id === forme)?.holder_abilities;
        const abilities = new Map<string, number>(holder === undefined ? [...new Set(entry.abilities)].map((a) => [a, 1]) : holder.entries.map((e) => [e.id, e.weight]));
        const total = [...abilities.values()].reduce((x, y) => x + y, 0);
        if (holder === undefined) expect(entry.required_item, `${forme} has no holder abilities`).toBeUndefined();
        for (const [ability, weight] of abilities) within(mine.filter((s) => s.ability === ability).length, n, weight / total, `${forme} ${ability}`);
        expect(mine.every((s) => abilities.has(s.ability)), `${forme} shows only its holder abilities`).toBe(true);
        for (const nature of vocabulary.natures) within(mine.filter((s) => s.nature === nature.id).length, n, 1 / vocabulary.natures.length, `${forme} ${nature.id}`);
        for (const move of entry.moves) within(mine.filter((s) => s.moves.includes(move)).length, n, k / entry.moves.length, `${forme} ${move}`);
        expect(mine.every((s) => s.moves.length === k), `${forme} holds exactly ${k} moves`).toBe(true);
        if (entry.required_item !== undefined) expect(mine.every((s) => s.item === entry.required_item)).toBe(true);
        else for (const item of freeItems) within(mine.filter((s) => s.item === item).length, n, 1 / freeItems.length, `${forme} ${item}`);
        checks += 1;
      }
    }
    expect(checks).toBeGreaterThanOrEqual(PREVIEW.length);
  });

  // The proposal itself: 60,000 teams from fixed seeds (about 360,000 sets); its frequencies against the same exact marginals,
  // so the worlds above and the generator are both held to the prior, and the Stat Points of the two against each other.
  const proposal = new Proposal(vocabulary, table, config);
  const TEAMS = 60_000;
  const generated: { species: string; set: PokemonSet }[] = [];
  const speciesOf = new Map(table.species.map((s) => [s.name, s.id]));
  for (let k = 0; k < TEAMS; k++) {
    for (const set of proposal.propose(engineRng(engine, drawSeed(KEY, { config_hash: "d".repeat(64), plan_id: "prior-frequencies" }, k)))) {
      generated.push({ species: speciesOf.get(set.species) as string, set });
    }
  }

  it("the proposal's item, ability and nature frequencies are the prior's, pooled over formes without a required item", () => {
    const free = generated.filter((g) => entryOf(g.species).required_item === undefined);
    for (const item of freeItems) within(free.filter((g) => toId(g.set.item) === item).length, free.length, 1 / freeItems.length, `proposal ${item}`);
    for (const nature of vocabulary.natures) within(generated.filter((g) => g.set.nature === nature.name).length, generated.length, 1 / vocabulary.natures.length, `proposal ${nature.id}`);
    // Abilities and moves per roster entry, as standardised scores: no entry strays beyond the bound.
    const perEntry = new Map<string, PokemonSet[]>();
    for (const g of generated) perEntry.set(g.species, [...(perEntry.get(g.species) ?? []), g.set]);
    let worst = 0;
    let tests = 0;
    for (const [id, list] of perEntry) {
      const entry = entryOf(id);
      // A Mega-forme entry's ability is replaced by the validator on the way to the played team (above), so only its moves are compared.
      const abilityChecked = entry.required_item === undefined;
      const k = Math.min(slots, entry.moves.length);
      const z = (hits: number, p: number): number => (p >= 1 ? (hits === list.length ? 0 : Infinity) : (hits - list.length * p) / Math.sqrt(list.length * p * (1 - p)));
      if (abilityChecked) for (const ability of new Set(entry.abilities)) worst = Math.max(worst, Math.abs(z(list.filter((s) => toId(s.ability) === ability).length, 1 / new Set(entry.abilities).size)));
      for (const move of entry.moves) {
        worst = Math.max(worst, Math.abs(z(list.filter((s) => s.moves.some((m) => toId(m) === move)).length, k / entry.moves.length)));
        tests += 1;
      }
    }
    expect(tests).toBeGreaterThan(10_000);
    expect(worst).toBeLessThanOrEqual(Z);
  });

  // The stone holders (#1167): the proposal writes the Mega forme with one of its own abilities; the pinned validator, as the pool runs it,
  // leaves the preview species holding the stone with a single ability, which is exactly the prior's `holder_abilities`.
  it("an unevolved stone holder of the proposal, after the validator, shows the prior's holder ability", () => {
    const validator = (engine.sim as unknown as { TeamValidator: { get(id: string): { validateSet(set: unknown, teamHas: unknown): string[] | null } } }).TeamValidator.get(vocabulary.format_id);
    const holders = generated.filter((g) => entryOf(g.species).required_item !== undefined).slice(0, 8000);
    expect(holders).toHaveLength(8000);
    const seen = new Set<string>();
    let accepted = 0;
    for (const { species, set } of holders) {
      const copy = structuredClone(set);
      // a set the validator refuses for another reason (a move combination) is a rejected draw, not a played team
      if ((validator.validateSet(copy, {}) ?? []).length > 0) continue;
      accepted += 1;
      const shows = shown.get(species) as string;
      expect(toId(copy.species), `${species} shows as ${shows}`).toBe(shows);
      const holder = (prior.species.find((s) => s.id === shows) as { formes: { id: string; holder_abilities?: { entries: { id: string }[] } }[] }).formes.find((f) => f.id === species)?.holder_abilities;
      expect(holder?.entries.map((e) => e.id), species).toEqual([toId(copy.ability)]);
      seen.add(species);
    }
    expect(accepted).toBeGreaterThan(7000);
    expect(seen.size).toBeGreaterThan(shown.size / 2);
  });

  it("the Stat Points of sampled worlds and of the proposal have the same per-stat distribution", () => {
    const stats = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
    const a = sets.map((s) => s.stat_points);
    const g = generated.map((x) => x.set.evs as Record<(typeof stats)[number], number>);
    for (const stat of stats) {
      // The mean, by a two-sample z; and the share at zero and at the per-stat maximum, by two-proportion z.
      const mean = (xs: readonly number[]): number => xs.reduce((t, v) => t + v, 0) / xs.length;
      const variance = (xs: readonly number[]): number => {
        const m = mean(xs);
        return xs.reduce((t, v) => t + (v - m) ** 2, 0) / (xs.length - 1);
      };
      const xa = a.map((p) => p[stat]);
      const xg = g.map((p) => p[stat]);
      const se = Math.sqrt(variance(xa) / xa.length + variance(xg) / xg.length);
      expect(Math.abs(mean(xa) - mean(xg)) / se, `${stat} mean`).toBeLessThanOrEqual(Z);
      for (const edge of [0, config.stat_point_max_per_stat]) {
        const pa = xa.filter((v) => v === edge).length / xa.length;
        const pg = xg.filter((v) => v === edge).length / xg.length;
        const pooled = (pa * xa.length + pg * xg.length) / (xa.length + xg.length);
        const sep = Math.sqrt(pooled * (1 - pooled) * (1 / xa.length + 1 / xg.length));
        expect(Math.abs(pa - pg) / sep, `${stat} share at ${edge}`).toBeLessThanOrEqual(Z);
      }
    }
    for (const p of a) expect(Object.values(p).reduce((t, v) => t + v, 0)).toBe(vocabulary.rules.ev_limit);
  });
});

describe("sampled worlds pass the pinned validator", () => {
  it("every brought set is legal and the engine's canonical paste, with no redraw", () => {
    const legality = createSetLegality({ engine, vocabulary });
    const sampler = createSampler({ legality, spreads: generatorSpreadSampler(vocabulary, config) });
    const b = belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags: run.flags });
    const out = sampler.sampleWorlds(b, Uint8Array.from({ length: 32 }, (_, i) => (i * 5 + 3) & 255), 150);
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
