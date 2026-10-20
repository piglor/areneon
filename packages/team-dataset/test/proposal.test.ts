import { type PokemonSet, type TeamGeneratorConfig, type TeamVocabulary } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { Proposal, ProposalError } from "../src/proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { engine, KEY, small, table, vocabulary } from "./support.ts";

const stream = { config_hash: "c".repeat(64), plan_id: "proposal-test" };
const propose = (cfg: TeamGeneratorConfig, k: number, vocab: TeamVocabulary = vocabulary): PokemonSet[] =>
  new Proposal(vocab, table, cfg).propose(engineRng(engine, drawSeed(KEY, stream, k)));
const slots = (sets: PokemonSet[]): number[] => sets.flatMap((s) => Object.values(s.evs));

describe("the proposal draws from the vocabulary and the rule table (ADR-013 section 2)", () => {
  const cfg = small();
  const teams = Array.from({ length: 150 }, (_, k) => propose(cfg, k));

  it("makes a team of the rule table's size, level and Stat Point total, with the config's IVs", () => {
    const { rules } = vocabulary;
    for (const team of teams) {
      expect(team).toHaveLength(rules.max_team_size);
      for (const set of team) {
        expect(set.level).toBe(rules.adjust_level);
        expect(Object.values(set.evs).reduce((a, b) => a + b, 0)).toBe(rules.ev_limit);
        expect(Math.max(...Object.values(set.evs))).toBeLessThanOrEqual(cfg.stat_point_max_per_stat);
        expect(Object.values(set.ivs).every((v) => v === cfg.iv)).toBe(true);
        expect(set.moves.length).toBeGreaterThanOrEqual(1);
        expect(set.moves.length).toBeLessThanOrEqual(4);
        expect(new Set(set.moves).size).toBe(set.moves.length);
        expect(set.name).toBe("");
        expect(set.gender).toBe("");
      }
    }
    expect(slots(teams[0] as PokemonSet[])).toHaveLength(36);
  });

  it("draws distinct species, abilities and moves of the species' own pool, and a nature of the vocabulary", () => {
    const names = new Map(table.species.map((s) => [s.name, s.id]));
    for (const team of teams) {
      const ids = team.map((s) => names.get(s.species) as string);
      expect(new Set(ids).size).toBe(ids.length);
      team.forEach((set, k) => {
        const entry = vocabulary.roster.find((r) => r.id === ids[k]) as TeamVocabulary["roster"][number];
        const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
        expect(entry.abilities).toContain(id(set.ability));
        for (const move of set.moves) expect(entry.moves).toContain(id(move));
        expect(vocabulary.natures.map((n) => n.name)).toContain(set.nature);
      });
    }
  });

  it("gives a species its required item, and respects the item weights and the Item Clause", () => {
    const stones = new Set(vocabulary.roster.flatMap((r) => (r.required_item === undefined ? [] : [r.required_item])));
    const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
    let required = 0;
    for (const team of teams) {
      const held = team.map((s) => id(s.item));
      for (const item of held.filter((i) => i !== "")) expect(held.filter((i) => i === item)).toHaveLength(vocabulary.rules.item_clause);
      for (const set of team) {
        const entry = vocabulary.roster.find((r) => r.id === id(set.species)) as TeamVocabulary["roster"][number];
        if (entry.required_item !== undefined) {
          expect(id(set.item)).toBe(entry.required_item);
          required++;
        } else {
          // the committed config gives every Mega Stone weight 0: only the Mega species hold one
          expect(stones.has(id(set.item))).toBe(false);
          expect(set.item).not.toBe("");
        }
      }
    }
    expect(required).toBeGreaterThan(30);
  });

  it("is deterministic in its draw, and different draws differ", () => {
    expect(propose(cfg, 5)).toEqual(propose(cfg, 5));
    expect(propose(cfg, 5)).not.toEqual(propose(cfg, 6));
  });

  it("follows the nickname and gender policies, and an item weight of zero for everything leaves no item", () => {
    const named = propose(small({ nickname_policy: "species", gender_policy: "random" }), 0);
    expect(named.every((s) => s.name === s.species)).toBe(true);
    expect(named.every((s) => s.gender === "M" || s.gender === "F")).toBe(true);
    const bare = propose(small({ default_item_weight: 0 }), 1);
    const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
    for (const set of bare) {
      const entry = vocabulary.roster.find((r) => r.id === id(set.species)) as TeamVocabulary["roster"][number];
      expect(id(set.item)).toBe(entry.required_item ?? "");
    }
  });

  it("draws an item in proportion to its weight", () => {
    const heavy = vocabulary.items.find((i) => !vocabulary.roster.some((r) => r.required_item === i)) as string;
    const weighted = small({ default_item_weight: 1, item_weights: [{ item: heavy, weight: 400 }] });
    const proposal = new Proposal(vocabulary, table, weighted);
    const name = (table.items.find((i) => i.id === heavy) as { name: string }).name;
    let hits = 0;
    let slotsDrawn = 0;
    for (let k = 0; k < 300; k++) {
      const team = proposal.propose(engineRng(engine, drawSeed(KEY, stream, k)));
      for (const set of team) {
        slotsDrawn++;
        if (set.item === name) hits++;
      }
    }
    // weight 400 of about 400 + 166: a share near 0.7 of the free slots, but the Item Clause caps it at one per team
    expect(hits).toBeGreaterThan(200);
    expect(hits).toBeLessThanOrEqual(300);
    expect(slotsDrawn).toBe(1800);
  });
});

describe("the proposal refuses a vocabulary or config it cannot draw from", () => {
  it("refuses a roster smaller than a team", () => {
    expect(() => new Proposal({ ...vocabulary, roster: vocabulary.roster.slice(0, 3) as never }, table, small())).toThrow(/smaller than a team/);
  });
  it("refuses a config without moves_per_set, and gives a set no more moves than it says", () => {
    const { moves_per_set: _omitted, ...without } = small();
    expect(() => new Proposal(vocabulary, table, without as TeamGeneratorConfig)).toThrow(/no moves_per_set/);
    const two = propose(small({ moves_per_set: 2 }), 3);
    expect(two.every((s) => s.moves.length <= 2)).toBe(true);
    expect(propose(small({ moves_per_set: 4 }), 3).some((s) => s.moves.length > 2)).toBe(true);
  });
  it("refuses no natures", () => {
    expect(() => new Proposal({ ...vocabulary, natures: [] }, table, small())).toThrow(/no nature/);
  });
  it("refuses a per-stat maximum that cannot hold the Stat Point total", () => {
    expect(() => new Proposal(vocabulary, table, small({ stat_point_max_per_stat: 10 }))).toThrow(/cannot hold 66 Stat Points/);
  });
  it("refuses item weights that overflow a 32-bit pick", () => {
    expect(() => new Proposal(vocabulary, table, small({ default_item_weight: 1_000_000, item_weights: [] }))).not.toThrow();
    const many = { ...vocabulary, items: Array.from({ length: 4300 }, (_, k) => `item${String(k).padStart(5, "0")}`) };
    expect(() => new Proposal(many, table, small({ default_item_weight: 1_000_000, item_weights: [] }))).toThrow(ProposalError);
  });
  it("refuses a roster species the engine-data table does not name", () => {
    const entry = { ...(vocabulary.roster[0] as TeamVocabulary["roster"][number]), id: "nosuchspecies" };
    expect(() => new Proposal({ ...vocabulary, roster: [entry, ...vocabulary.roster.slice(1)] }, table, small())).toThrow(/no name/);
  });
});
