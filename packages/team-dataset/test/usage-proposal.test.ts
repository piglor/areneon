// Component U of the mixture proposal (ADR-020 section 4, slice 7): the committed example config, the draw rule on a toy
// table (species pairs by the teammate rule, the per-slot floor, items under the Item Clause), the refusals, P1 and P3 for
// U against a hand-worked toy and the committed table, and a small target-share generation. Generation tests are small:
// the validator is slow.

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readUsageWeightTable, recordHash, rootedWeight, type TeamGeneratorConfig, teamVocabularyHash, usageTableHash, type UsageWeightTable } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { Drawer } from "../src/draw.ts";
import { generate } from "../src/generate.ts";
import { PilotError, readPilot, runPilot } from "../src/pilot.ts";
import { bitsOf, usageBound, usageBounds, usageMapping } from "../src/pilot-gates.ts";
import { Proposal } from "../src/proposal.ts";
import { roleSetsSource } from "../src/role-set-proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { pickWeighted, UsageSampler, type UsageSource, usageSource } from "../src/usage-proposal.ts";
import { ASSIGNMENT_KEY, engine, format, FORMAT, KEY, light, small, table, vocabulary } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const committed = usageSource(vocabulary, JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT, "usage", "2026-09-1630.json"), "utf8")));
const roleSource = roleSetsSource(engine, vocabulary, table, JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json"), "utf8")));
const stream = { config_hash: "e".repeat(64), plan_id: "usage-test" };

type Component = Extract<NonNullable<TeamGeneratorConfig["proposal"]>["components"][number], { kind: "usage" }>;
const usageComponent = (source: UsageSource, share = 1000, floor = 100, setRoot = 1): Component => ({ kind: "usage", share_permille: share, usage_table_hash: source.hash, species_root: 2, set_root: setRoot, floor_permille: floor });
const pure = (source: UsageSource, floor = 100, setRoot = 1): TeamGeneratorConfig =>
  small({ config_id: "usage-test", proposal: { kind: "mixture", components: [usageComponent(source, 1000, floor, setRoot)] } });

// A toy table: eight roster species with invented numbers, read through the real reader against the real vocabulary.
const NS = 8;
const toyIds = vocabulary.roster.filter((e) => e.required_item === undefined && e.abilities.length >= 1 && e.moves.length >= 6).slice(0, NS).map((e) => e.id).sort();
const entryOf = (id: string) => vocabulary.roster.find((e) => e.id === id) as (typeof vocabulary.roster)[number];
const natureIds = vocabulary.natures.map((n) => n.id);
/** ppm shares from raw positive numbers; the remainder goes to the first (or `rest`) so the list sums to `total`. */
function shares(raw: number[], total = 1_000_000): number[] {
  const sum = raw.reduce((a, b) => a + b, 0);
  const out = raw.map((r) => Math.floor((total * r) / sum));
  out[0] = (out[0] as number) + (total - out.reduce((a, b) => a + b, 0));
  return out;
}
const rawUsage = [9, 7, 5, 4, 3, 2, 2, 1];
const rawTeammate = (i: number, j: number): number => 1 + ((i * 7 + j * 3) % 11);
function toyTable(): UsageWeightTable {
  const usage = shares(rawUsage);
  const species = toyIds.map((id, i) => {
    const entry = entryOf(id);
    const items = [vocabulary.items[2 * i] as string, vocabulary.items[2 * i + 1] as string].sort();
    const [i0, i1] = shares([6, 4], 900_000) as [number, number];
    const moves = entry.moves.slice(0, 5).sort();
    const movePpm = shares([40, 30, 20, 5, 5], 950_000);
    const others = toyIds.map((t, j) => ({ t, j })).filter((x) => x.j !== i);
    const tm = shares(others.map((x) => rawTeammate(i, x.j)), 900_000);
    return {
      id,
      usage_ppm: usage[i] as number,
      w_species: rootedWeight(usage[i] as number, 2),
      abilities: { entries: [{ id: entry.abilities[0] as string, ppm: 1_000_000 }], unmapped_ppm: 0 },
      items: { entries: [{ id: items[0] as string, ppm: i0 }, { id: items[1] as string, ppm: i1 }], truncated_ppm: 50_000, unmapped_ppm: 0, no_item_ppm: 50_000 },
      moves: { entries: moves.map((m, k) => ({ id: m, ppm: movePpm[k] as number })), truncated_ppm: 30_000, unmapped_ppm: 0, empty_ppm: 20_000 },
      spreads: { entries: [{ nature: natureIds[i] as string, stat_points: { hp: 32, atk: 32, def: 2, spa: 0, spd: 0, spe: 0 }, ppm: 900_000 }], truncated_ppm: 100_000, unmapped_ppm: 0 },
      teammates: {
        entries: others.map((x, k) => ({ id: x.t, ppm: tm[k] as number, weight: rootedWeight(tm[k] as number, 2) })),
        truncated_ppm: 100_000,
        unmapped_ppm: 0,
      },
    };
  });
  return {
    schema_version: 1,
    format_id: FORMAT,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: teamVocabularyHash(vocabulary),
    month: "2026-09",
    cutoff: 1630,
    number_of_battles: 1000,
    derivation: { script_version: "toy", k_items: 16, k_moves: 32, k_spreads: 32, k_teammates: 48, species_root: 2, teammate_root: 2 },
    sources: [{ source_id: "toy", url: "https://example.org/toy", retrieved_at: "2026-10-01T00:00:00Z", size: 1, sha256: "a".repeat(64), attribution: "invented numbers", basis: "owner-accepted-risk", basis_reference: "ADR-009 erratum 1" }],
    unmapped_species_ppm: 0,
    species,
  } as unknown as UsageWeightTable;
}
const toyRaw = toyTable();
const toy = usageSource(vocabulary, toyRaw);

/** The five-and-a-half standard deviation bound used throughout: for a count of n trials at probability p. */
const within = (observed: number, n: number, p: number): boolean => Math.abs(observed - n * p) <= 5.5 * Math.sqrt(n * p * (1 - p)) + 1;
const toyUniform = {
  item: () => "",
  spread: () => ({ nature: natureIds[0] as string, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }),
};

describe("the committed example config", () => {
  const cfg = JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT, "generator.mixture-u.json"), "utf8")) as TeamGeneratorConfig;
  it("is the target mixture uniform 300, role_sets 200, usage 500 over the committed tables", () => {
    expect(cfg.proposal?.components).toEqual([
      { kind: "role_sets", role_set_table_hash: roleSource.hash, share_permille: 200 },
      { kind: "uniform", share_permille: 300 },
      { kind: "usage", share_permille: 500, usage_table_hash: committed.hash, species_root: 2, set_root: 3, floor_permille: 100 },
    ]);
    expect(committed.hash).toBe(usageTableHash(readUsageWeightTable(JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT, "usage", "2026-09-1630.json"), "utf8")), vocabulary)));
  });
  it("is a valid proposal for the pinned engine and the committed tables", () => {
    expect(() => new Proposal(vocabulary, table, small({ proposal: cfg.proposal as NonNullable<TeamGeneratorConfig["proposal"]> }), roleSource, committed)).not.toThrow();
  });
});

describe("the draw rule on a toy table", () => {
  const sampler = new UsageSampler(vocabulary, toy, 1, 0, 4);
  const N = 20_000;
  const rngAt = (k: number) => engineRng(engine, drawSeed(KEY, stream, k));

  it("draws species 1 by w_species and species 2 by the first species' teammate weights, within a binomial bound", () => {
    const first = new Map<string, number>();
    const pair = new Map<string, number>();
    for (let k = 0; k < N; k++) {
      const team = sampler.species(rngAt(k)).map((e) => e.id);
      expect(new Set(team).size).toBe(team.length);
      for (const id of team) expect(toyIds).toContain(id);
      first.set(team[0] as string, (first.get(team[0] as string) ?? 0) + 1);
      const key = `${team[0]}>${team[1]}`;
      pair.set(key, (pair.get(key) ?? 0) + 1);
    }
    const wSum = toy.table.species.reduce((a, s) => a + s.w_species, 0);
    for (const s of toy.table.species) expect(within(first.get(s.id) ?? 0, N, s.w_species / wSum), `first ${s.id}`).toBe(true);
    // The second species: weights over the unchosen entries, the first species' teammate list alone.
    for (const a of toy.table.species) {
      const tw = new Map(a.teammates.entries.map((t) => [t.id, t.weight]));
      const total = [...tw.values()].reduce((x, y) => x + y, 0);
      for (const [id, w] of tw) {
        const p = (a.w_species / wSum) * (w / total);
        expect(within(pair.get(`${a.id}>${id}`) ?? 0, N, p), `${a.id} then ${id}`).toBe(true);
      }
    }
  }, 120_000);

  it("sums the teammate weights of every chosen species for later picks", () => {
    // Third species given the first two: weights tw_a[j] + tw_b[j] over the unchosen entries; checked against the draws of species(),
    // by conditioning on the first two (the pair with the most draws).
    const counts = new Map<string, number>();
    let base = 0;
    const [a, b] = [toy.table.species[0], toy.table.species[1]] as [(typeof toy.table.species)[number], (typeof toy.table.species)[number]];
    for (let k = 0; k < 60_000; k++) {
      const team = sampler.species(rngAt(k)).map((e) => e.id);
      if (team[0] !== a.id || team[1] !== b.id) continue;
      base++;
      counts.set(team[2] as string, (counts.get(team[2] as string) ?? 0) + 1);
    }
    expect(base).toBeGreaterThan(300);
    const tw = (s: typeof a, id: string): number => s.teammates.entries.find((t) => t.id === id)?.weight ?? 0;
    const open = toyIds.filter((id) => id !== a.id && id !== b.id);
    const total = open.reduce((x, id) => x + tw(a, id) + tw(b, id), 0);
    for (const id of open) expect(within(counts.get(id) ?? 0, base, (tw(a, id) + tw(b, id)) / total), `third ${id}`).toBe(true);
  }, 120_000);

  it("draws items by weight, abilities, moves without replacement and spreads from the member's own lists", () => {
    const entry = entryOf(toyIds[0] as string);
    const member = toy.table.species[0] as (typeof toy.table.species)[number];
    const itemCount = new Map<string, number>();
    for (let k = 0; k < N; k++) {
      const rng = rngAt(k);
      const item = sampler.item(rng, entry, new Map(), toyUniform);
      itemCount.set(item, (itemCount.get(item) ?? 0) + 1);
      const d = sampler.details(rng, entry, toyUniform);
      expect(d.ability).toBe(member.abilities.entries[0]?.id);
      expect(new Set(d.moves).size).toBe(4);
      for (const m of d.moves) expect(member.moves.entries.map((e) => e.id)).toContain(m);
      expect(d.spread.nature).toBe(member.spreads.entries[0]?.nature);
    }
    const listed = member.items.entries.map((e) => e.ppm).reduce((x, y) => x + y, 0) + member.items.no_item_ppm;
    for (const e of member.items.entries) expect(within(itemCount.get(e.id) ?? 0, N, e.ppm / listed), e.id).toBe(true);
    expect(within(itemCount.get("") ?? 0, N, member.items.no_item_ppm / listed)).toBe(true);
  }, 120_000);

  it("respects the Item Clause: an item already held is not drawn again", () => {
    const entry = entryOf(toyIds[0] as string);
    const member = toy.table.species[0] as (typeof toy.table.species)[number];
    const held = member.items.entries[0]?.id as string;
    for (let k = 0; k < 500; k++) expect(sampler.item(rngAt(k), entry, new Map([[held, vocabulary.rules.item_clause]]), toyUniform)).not.toBe(held);
  });

  it("holds a Mega forme to its required item and uses the uniform draw when a species has no list", () => {
    const mega = vocabulary.roster.find((e) => e.required_item !== undefined) as (typeof vocabulary.roster)[number];
    expect(sampler.item(rngAt(1), mega, new Map(), toyUniform)).toBe(mega.required_item);
    const unlisted = vocabulary.roster.find((e) => e.required_item === undefined && !toyIds.includes(e.id)) as (typeof vocabulary.roster)[number];
    expect(sampler.item(rngAt(2), unlisted, new Map(), { ...toyUniform, item: () => "uniform-item" })).toBe("uniform-item");
    const d = sampler.details(rngAt(3), unlisted, { ...toyUniform, spread: () => ({ nature: "uniform-nature", stat_points: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }) });
    expect(unlisted.abilities).toContain(d.ability);
    expect(d.spread.nature).toBe("uniform-nature");
    expect(d.moves).toHaveLength(4);
  });
});

describe("the per-slot floor", () => {
  const rosterSize = vocabulary.roster.length;
  it("takes the first species uniformly over the roster with probability floor_permille", () => {
    const floor = new UsageSampler(vocabulary, toy, 1, 300, 4);
    const N = 20_000;
    let outside = 0;
    for (let k = 0; k < N; k++) if (!toyIds.includes(floor.species(engineRng(engine, drawSeed(KEY, stream, k)))[0]?.id as string)) outside++;
    // A floor draw lands outside the table's eight species with probability (roster - 8) / roster; a weighted draw never does.
    expect(within(outside, N, (0.3 * (rosterSize - NS)) / rosterSize)).toBe(true);
  }, 120_000);

  it("draws nothing for the floor at 0 and everything uniformly at 1000", () => {
    const rng = { below: (n: number): number => (n === 1000 ? (() => { throw new Error("floor coin drawn"); })() : 0) };
    expect(() => new UsageSampler(vocabulary, toy, 1, 0, 4).species(rng)).not.toThrow();
    const all = new UsageSampler(vocabulary, toy, 1, 1000, 4);
    const allUniform = { below: (n: number): number => (n === 1000 ? 0 : n - 1) };
    expect(all.species(allUniform)[0]?.id).toBe(vocabulary.roster.at(-1)?.id);
  });

  it("applies the set root to the set weights only", () => {
    const squared = new UsageSampler(vocabulary, toy, 2, 0, 4);
    const m = (squared.members.get(toyIds[0] as string) as NonNullable<ReturnType<typeof squared.members.get>>).items;
    const ppm = (toy.table.species[0] as (typeof toy.table.species)[number]).items.entries.map((e) => e.ppm);
    const listed = m.weights.slice(0, 2);
    expect(listed.sort()).toEqual(ppm.map((p) => rootedWeight(p, 2)).sort());
  });
});

describe("the floor and the Item Clause, statistically", () => {
  const N = 20_000;
  const entry = entryOf(toyIds[0] as string);
  const member = toy.table.species[0] as (typeof toy.table.species)[number];
  const rngAt = (k: number) => engineRng(engine, drawSeed(KEY, stream, k));

  it("takes the item and the spread uniformly with probability floor_permille (counted at the uniform routes, 300 per mille)", () => {
    const floor = new UsageSampler(vocabulary, toy, 1, 300, 4);
    let uniformItems = 0;
    let uniformSpreads = 0;
    const counting = {
      item: () => (uniformItems++, ""),
      spread: () => (uniformSpreads++, { nature: natureIds[0] as string, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }),
    };
    for (let k = 0; k < N; k++) {
      const rng = rngAt(k);
      floor.item(rng, entry, new Map(), counting);
      floor.details(rng, entry, counting);
    }
    // The toy member has one spread and lists for every field, so each uniform route is taken exactly when its floor coin says so.
    expect(within(uniformItems, N, 0.3), `item floor ${uniformItems}`).toBe(true);
    expect(within(uniformSpreads, N, 0.3), `spread floor ${uniformSpreads}`).toBe(true);
  }, 120_000);

  it("draws the items the Item Clause still allows in proportion to their weights", () => {
    const sampler = new UsageSampler(vocabulary, toy, 1, 0, 4);
    const [held, other] = member.items.entries as [(typeof member.items.entries)[number], (typeof member.items.entries)[number]];
    const counts = new Map<string, number>();
    for (let k = 0; k < N; k++) {
      const item = sampler.item(rngAt(k), entry, new Map([[held.id, vocabulary.rules.item_clause]]), toyUniform);
      counts.set(item, (counts.get(item) ?? 0) + 1);
    }
    expect(counts.get(held.id)).toBeUndefined();
    // Allowed: the other listed item and "no item"; the held item's weight is out of the total, not redistributed by anything else.
    const total = other.ppm + member.items.no_item_ppm;
    expect(within(counts.get(other.id) ?? 0, N, other.ppm / total), other.id).toBe(true);
    expect(within(counts.get("") ?? 0, N, member.items.no_item_ppm / total)).toBe(true);
  }, 120_000);
});

describe("pickWeighted", () => {
  it("is one integer pick over a positive total and refuses a zero total", () => {
    expect(pickWeighted({ below: () => 0 }, [1, 2])).toBe(0);
    expect(pickWeighted({ below: () => 1 }, [1, 2])).toBe(1);
    expect(pickWeighted({ below: () => 2 }, [1, 2])).toBe(1);
    expect(() => pickWeighted({ below: () => 0 }, [])).toThrow(/no weight/);
    expect(() => pickWeighted({ below: () => 99 }, [1, 2])).toThrow(/ran past/);
  });
});

describe("refusals", () => {
  it("refuses a usage component with no table, another hash, or another species_root", () => {
    const cfg = pure(toy);
    expect(() => new Proposal(vocabulary, table, cfg)).toThrow(/no usage weight table/);
    expect(() => new Proposal(vocabulary, table, cfg, undefined, { ...toy, hash: "0".repeat(64) })).toThrow(/hash mismatch/);
    const root = small({ proposal: { kind: "mixture", components: [{ ...usageComponent(toy), species_root: 3 }] } });
    expect(() => new Proposal(vocabulary, table, root, undefined, toy)).toThrow(/species_root/);
  });
  it("refuses a table that does not read against the vocabulary", () => {
    expect(() => usageSource(vocabulary, { ...toyRaw, vocabulary_hash: "0".repeat(64) })).toThrow(/vocabulary_hash/);
  });
});

describe("a usage team", () => {
  it("is made of the table's species, items, abilities, moves and spreads when the floor is 0, and keeps the config's fixed fields", () => {
    const proposal = new Proposal(vocabulary, table, pure(toy, 0), undefined, toy);
    const ids = new Map(table.species.map((s) => [s.name, s.id]));
    const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
    for (let k = 0; k < 200; k++) {
      const team = proposal.propose(engineRng(engine, drawSeed(KEY, stream, k)));
      expect(team).toHaveLength(vocabulary.rules.max_team_size);
      for (const set of team) {
        const member = toy.table.species.find((s) => s.id === ids.get(set.species));
        expect(member, set.species).toBeDefined();
        expect(set.moves.map(id).every((m) => member?.moves.entries.some((e) => e.id === m))).toBe(true);
        expect(id(set.ability)).toBe(member?.abilities.entries[0]?.id);
        expect(set.level).toBe(vocabulary.rules.adjust_level);
        expect(Object.values(set.evs).reduce((a, b) => a + b, 0)).toBe(66);
      }
    }
  });

  it("draws the same team from the same seed", () => {
    const a = new Proposal(vocabulary, table, pure(toy), undefined, toy);
    const b = new Proposal(vocabulary, table, pure(toy), undefined, toy);
    for (let k = 0; k < 10; k++) expect(a.propose(engineRng(engine, drawSeed(KEY, stream, k)))).toEqual(b.propose(engineRng(engine, drawSeed(KEY, stream, k))));
  });
});

describe("P3 for component U", () => {
  it("is worked by hand on a toy: 2 x (4/7) x (1/9)^2", () => {
    // Three species, teams of two, no floor, Item Clause 1. w_species 4, 2, 1: the first pick is at most 4/7. A later pick is
    // at most 1 (a teammate list's top weight over what it leaves is above 1). Each member: one ability (1), a spread (1),
    // moves (clamped to 1) and an item list of ten equal weights, of which the Item Clause may have taken one: (1/10) / (9/10) = 1/9.
    const list = (ids: string[], ppm: number) => ids.map((id) => ({ id, ppm }));
    const species = ["a", "b", "c"].map((id, k) => ({
      id,
      w_species: [4, 2, 1][k],
      abilities: { entries: [{ id: "x", ppm: 1_000_000 }], unmapped_ppm: 0 },
      items: { entries: list(["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"], 100_000), no_item_ppm: 0 },
      moves: { entries: [{ id: "m0", ppm: 600_000 }, { id: "m1", ppm: 400_000 }] },
      spreads: { entries: [{ nature: "n", stat_points: {}, ppm: 1_000_000 }] },
      teammates: { entries: [{ id: "b", ppm: 1, weight: 3 }, { id: "c", ppm: 1, weight: 1 }] },
    }));
    const source = { table: { species }, hash: "x" } as never;
    const vocab = { roster: ["a", "b", "c"].map((id) => ({ id, abilities: ["x"], moves: ["m0", "m1"] })), items: ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"], natures: [{}], rules: { max_team_size: 2, item_clause: 1 } } as never;
    const cfg = { item_weights: [], default_item_weight: 1, gender_policy: "unset", moves_per_set: 2 } as never;
    const bound = usageBound(vocab, cfg, usageComponent(source, 1000, 0), source);
    expect(bound.num * 567n).toBe(8n * bound.den); // 8 / 567
    expect(bitsOf(bound)).toBe(bitsOf({ num: 8n, den: 567n }));
  });

  it("is 12 bits (set-level, set_root 1) and 49 bits (ordered sequence) on the committed table at floor 100; at set_root 2 they are 35 and 69", () => {
    // The bound is loose by construction (it multiplies the largest conditional probabilities and the orderings of a
    // team); it is a bound on the generator, not the probability of any real team. The figure is pinned so a change of
    // the bound or of the table is seen.
    const both = usageBounds(vocabulary, small(), usageComponent(committed), committed);
    const bound = both.set;
    process.stderr.write(`usage P3 on the committed table: set-level ${bitsOf(both.set)} bits, ordered-sequence ${bitsOf(both.sequence)} bits\n`);
    expect(bitsOf(bound)).toBe(12); // 18 bits before the 2^m factor for the Mega entries a team can hold (m = 6)
    expect(bitsOf(both.sequence)).toBe(49);
    const squared = usageBounds(vocabulary, small(), usageComponent(committed, 1000, 100, 2), committed);
    expect(bitsOf(squared.set)).toBe(35);
    expect(bitsOf(squared.sequence)).toBe(69);
  });
});

describe("P3 on degenerate tables", () => {
  const emptied = (change: (s: (typeof toyRaw.species)[number]) => void): UsageSource => {
    const t = structuredClone(toyRaw);
    for (const s of t.species) change(s);
    return { table: t, hash: "x" };
  };
  const cases: [string, UsageSource][] = [
    ["no teammates", emptied((s) => (s.teammates.entries = [] as never))],
    ["no moves", emptied((s) => (s.moves.entries = [] as never))],
    ["one move", emptied((s) => (s.moves.entries = s.moves.entries.slice(0, 1) as never))],
    ["no items", emptied((s) => (s.items = { ...s.items, entries: [], no_item_ppm: 0 } as never))],
    ["no abilities", emptied((s) => (s.abilities.entries = [] as never))],
    ["no spreads", emptied((s) => (s.spreads.entries = [] as never))],
    ["no usage", emptied((s) => (s.w_species = 0))],
  ];
  for (const [name, source] of cases) {
    for (const floor of [0, 300, 1000]) {
      it(`gives a positive bound of at most 1 (${name}, floor ${floor})`, () => {
        const bound = usageBound(vocabulary, { ...small(), gender_policy: "random" }, usageComponent(source, 1000, floor, 2), source);
        expect(bound.num).toBeGreaterThan(0n);
        expect(bound.num).toBeLessThanOrEqual(bound.den * 720n); // 6! orderings times probabilities that may each be 1 in a degenerate table
      });
    }
  }
  it("is looser at a higher floor only when the floor is uniform over fewer options than the weights favour", () => {
    expect(bitsOf(usageBound(vocabulary, small(), usageComponent(toy, 1000, 1000), toy))).toBeGreaterThan(bitsOf(usageBound(vocabulary, small(), usageComponent(toy, 1000, 0), toy)));
  });
});

describe("P3 against an exhaustive enumeration of a tiny table", () => {
  // Four roster entries, teams of two, Item Clause 2 (never binding for two members), two natures, two abilities, four moves
  // with two per set, items i1..i4. Every team is enumerated exactly with bigint rationals by a separate reading of the draw
  // rule, and the bound must be at least the probability of the most likely canonical set.
  type Q = { n: bigint; d: bigint };
  const q = (n: number | bigint, d: number | bigint = 1): Q => ({ n: BigInt(n), d: BigInt(d) });
  const add = (a: Q, b: Q): Q => q(a.n * b.d + b.n * a.d, a.d * b.d);
  const mul = (a: Q, b: Q): Q => q(a.n * b.n, a.d * b.d);
  const ge = (a: Q, b: Q): boolean => a.n * b.d >= b.n * a.d;
  const zero = q(0);
  const ids = ["a", "b", "c", "d"];
  const wSpecies = [4, 3, 2, 1];
  const teammate: Record<string, Record<string, number>> = { a: { b: 5, c: 1, d: 1 }, b: { a: 4, c: 4, d: 1 }, c: { a: 1, b: 1, d: 6 }, d: { a: 2, b: 2, c: 2 } };
  const itemList = [["i1", 600_000], ["i2", 300_000], ["", 100_000]] as const;
  const abilityList = [["x", 700_000], ["y", 300_000]] as const;
  const moveList = [["m0", 500_000], ["m1", 300_000], ["m2", 200_000]] as const;
  const spreadList = [["n1", 800_000], ["n2", 200_000]] as const;
  const members = ids.map((id, k) => ({
    id,
    w_species: wSpecies[k],
    abilities: { entries: abilityList.map(([i, ppm]) => ({ id: i, ppm })), unmapped_ppm: 0 },
    items: { entries: itemList.filter(([i]) => i !== "").map(([i, ppm]) => ({ id: i, ppm })), no_item_ppm: 100_000, truncated_ppm: 0, unmapped_ppm: 0 },
    moves: { entries: moveList.map(([i, ppm]) => ({ id: i, ppm })) },
    spreads: { entries: spreadList.map(([i, ppm]) => ({ nature: i, stat_points: {}, ppm })) },
    teammates: { entries: Object.entries(teammate[id] as Record<string, number>).map(([t, weight]) => ({ id: t, ppm: 1, weight })) },
  }));
  const source = { table: { species: members }, hash: "x" } as never;
  const vocab = { roster: ids.map((id) => ({ id, abilities: ["x", "y"], moves: ["m0", "m1", "m2", "m3"] })), items: ["i1", "i2", "i3", "i4"], natures: [{}, {}], rules: { max_team_size: 2, item_clause: 2 } } as never;
  const cfg = { item_weights: [], default_item_weight: 1, gender_policy: "unset", moves_per_set: 2 } as never;

  /** P(one specific pick) = (1 - f) * weighted + f * uniform. */
  const mixP = (f: number, weighted: Q, uniform: Q): Q => add(mul(q(1000 - f, 1000), weighted), mul(q(f, 1000), uniform));

  /** The most likely single member (item, ability, move pair, spread) of a species: enumerated, not a formula. */
  function bestMember(f: number): Q {
    const list = <T,>(entries: readonly (readonly [T, number])[]): Map<T, Q> => new Map(entries.map(([k, ppm]) => [k, q(ppm, entries.reduce((s, e) => s + e[1], 0))]));
    const items = list(itemList);
    const abilities = list(abilityList);
    const spreads = list(spreadList);
    const best = (options: Q[]): Q => options.reduce((a, b) => (ge(a, b) ? a : b));
    const itemP = [...items.entries()].map(([, w]) => mixP(f, w, q(1, 4)));
    const abilityP = [...abilities.entries()].map(([, w]) => mixP(f, w, q(1, 2)));
    const spreadP = [...spreads.entries()].map(([, w]) => mixP(f, w, q(1, 2)));
    // An unordered pair of the four pool moves: both orders, the second pick from what the list still holds (uniform once it holds nothing).
    const pool = ["m0", "m1", "m2", "m3"];
    const weight = new Map<string, number>(moveList.map(([m, ppm]) => [m, ppm]));
    const pick = (taken: string[], m: string): Q => {
      const left = [...weight.entries()].filter(([k]) => !taken.includes(k));
      const total = left.reduce((s, [, w]) => s + w, 0);
      const uniform = q(1, pool.length - taken.length);
      return left.length === 0 ? uniform : mixP(f, q(weight.get(m) ?? 0, total), uniform);
    };
    const moveP: Q[] = [];
    for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) moveP.push(add(mul(pick([], pool[i] as string), pick([pool[i] as string], pool[j] as string)), mul(pick([], pool[j] as string), pick([pool[j] as string], pool[i] as string))));
    return mul(mul(best(itemP), best(abilityP)), mul(best(moveP), best(spreadP)));
  }

  /** P(species a then b) under the teammate rule with the floor (all four species are listed). */
  function sequence(f: number, a: number, b: number): Q {
    const total = wSpecies.reduce((s, w) => s + w, 0);
    const first = mixP(f, q(wSpecies[a] as number, total), q(1, 4));
    const tm = teammate[ids[a] as string] as Record<string, number>;
    const sum = Object.values(tm).reduce((s, w) => s + w, 0);
    return mul(first, mixP(f, q(tm[ids[b] as string] as number, sum), q(1, 3)));
  }

  for (const f of [0, 300, 1000]) {
    it(`bounds the exact probability of the most likely team (floor ${f})`, () => {
      const member = bestMember(f);
      let exact = zero;
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          const set = add(sequence(f, a, b), sequence(f, b, a));
          const p = mul(set, mul(member, member));
          if (ge(p, exact)) exact = p;
        }
      }
      const both = usageBounds(vocab, cfg, usageComponent(source, 1000, f), source);
      expect(ge(q(both.set.num, both.set.den), exact), `bound ${both.set.num}/${both.set.den} vs exact ${exact.n}/${exact.d}`).toBe(true);
      // The ordered-sequence figure bounds one draw order of the most likely team, which is the exact set probability only up to its orderings.
      expect(ge(q(both.sequence.num, both.sequence.den), q(0))).toBe(true);
      expect(ge(q(both.set.num, both.set.den), q(both.sequence.num, both.sequence.den))).toBe(true);
    });
  }
});

describe("P3 when a canonical team has two routes (a Mega entry and its base forme plus stone)", () => {
  // Roster a, c, d and dm; dm requires the item "st" and d's only listed item is "st", so the canonical team {x, d + st} is
  // reached through dm and through d. All weights are equal and every member list is degenerate (one ability, one item, the
  // whole two-move pool, one spread), so the exact probability is the species part alone: P(a team) = sum of the ordered
  // species pairs that reach it, each 1/4 * 1/3. The set-level bound before the Mega factor is 1/4 and is BELOW that.
  type Q = { n: bigint; d: bigint };
  const q = (n: number | bigint, d: number | bigint = 1): Q => ({ n: BigInt(n), d: BigInt(d) });
  const ge = (a: Q, b: Q): boolean => a.n * b.d >= b.n * a.d;
  const ids = ["a", "c", "d", "dm"];
  const member = (id: string, item: string) => ({
    id,
    w_species: 1,
    abilities: { entries: [{ id: "x", ppm: 1_000_000 }], unmapped_ppm: 0 },
    items: { entries: [{ id: item, ppm: 1_000_000 }], no_item_ppm: 0, truncated_ppm: 0, unmapped_ppm: 0 },
    moves: { entries: [{ id: "m0", ppm: 600_000 }, { id: "m1", ppm: 400_000 }] },
    spreads: { entries: [{ nature: "n", stat_points: {}, ppm: 1_000_000 }] },
    teammates: { entries: ids.filter((t) => t !== id).map((t) => ({ id: t, ppm: 1, weight: 1 })) },
  });
  const source = { table: { species: ids.map((id) => member(id, id === "d" ? "st" : "i1")) }, hash: "x" } as never;
  const vocab = {
    roster: ids.map((id) => ({ id, abilities: ["x"], moves: ["m0", "m1"], ...(id === "dm" ? { required_item: "st" } : {}) })),
    items: ["i1", "st"],
    natures: [{}],
    rules: { max_team_size: 2, item_clause: 2 },
  } as never;
  const cfg = { item_weights: [], default_item_weight: 1, gender_policy: "unset", moves_per_set: 2 } as never;
  const canonical = (id: string): string => (id === "dm" ? "d" : id);

  it("is at least the exact probability of the most likely canonical team, and the bound without the Mega factor is not", () => {
    const teams = new Map<string, Q>();
    for (const x of ids) {
      for (const y of ids) {
        if (x === y) continue;
        const key = [canonical(x), canonical(y)].sort().join();
        const old = teams.get(key) ?? q(0);
        teams.set(key, q(old.n * 12n + old.d, old.d * 12n));
      }
    }
    const exact = [...teams.values()].reduce((a, b) => (ge(a, b) ? a : b));
    expect(ge(exact, q(4, 12)) && ge(q(4, 12), exact)).toBe(true); // {a,d}: (a,dm) (dm,a) (a,d) (d,a), each 1/12
    const both = usageBounds(vocab, cfg, usageComponent(source, 1000, 0), source);
    const bound = q(both.set.num, both.set.den);
    expect(ge(bound, q(exact.n, exact.d))).toBe(true);
    expect(ge(q(bound.n, bound.d * 2n), q(exact.n, exact.d))).toBe(false); // the bound before the 2^m factor (m = 1) was too small
  });
});

describe("the pilot's largest canonical-team count (usage only)", () => {
  const identity = { config_hash: "a".repeat(64), label_set_hash: recordHash(light), vocabulary_hash: "c".repeat(64) };
  const hits = [8, 0, 0, 0, 0, 0, 0, 0];
  const withPart = (change: Record<string, unknown>) => ({
    format_version: 1, ...identity, draws: 10, rejected: 2, cell_hits: hits,
    components: [{ kind: "usage", draws: 10, rejected: 2, cell_hits: hits, max_team_count: 3, ...change }],
  });
  it("is read, and refused on another component, above the draws or not a count", () => {
    expect(readPilot(withPart({}), identity, light).components?.[0]?.max_team_count).toBe(3);
    expect(() => readPilot(withPart({ kind: "uniform" }), identity, light)).toThrow(PilotError);
    expect(() => readPilot(withPart({ max_team_count: 11 }), identity, light)).toThrow(PilotError);
    expect(() => readPilot(withPart({ max_team_count: 1.5 }), identity, light)).toThrow(PilotError);
  });
});

describe("P1 for component U", () => {
  const all1000 = { species: 1000, abilities: 1000, items: 1000, moves: 1000, spreads: 1000, teammates: 1000 };
  it("counts the table's unmapped mass, usage-weighted, per list kind and in aggregate", () => {
    expect(usageMapping(toyRaw)).toEqual({ mapped: 6e12, total: 6e12, kinds: all1000 });
    const lossy = structuredClone(toyRaw);
    (lossy.species[0] as (typeof lossy.species)[number]).teammates.unmapped_ppm = 1000;
    const loss = (lossy.species[0] as (typeof lossy.species)[number]).usage_ppm * 1000;
    const result = usageMapping(lossy);
    expect(result).toMatchObject({ mapped: 6e12 - loss, total: 6e12 });
    expect(result.kinds.teammates).toBe(Math.floor(((1e12 - loss) * 1000) / 1e12));
    expect({ ...result.kinds, teammates: 1000 }).toEqual(all1000);
  });
  it("counts an unmapped species in all six kinds", () => {
    const t = structuredClone(toyRaw);
    t.unmapped_species_ppm = 100_000;
    expect(Object.values(usageMapping(t).kinds)).toEqual([900, 900, 900, 900, 900, 900]);
  });
  it("is at least 950 per mille in every kind on the committed table", () => {
    const m = usageMapping(committed.table);
    process.stderr.write(`usage P1 per kind (permille): ${JSON.stringify(m.kinds)}; aggregate ${Math.floor((m.mapped * 1000) / m.total)}\n`);
    for (const v of Object.values(m.kinds)) expect(v).toBeGreaterThanOrEqual(950);
  });
});

describe("a small target-share generation", () => {
  const scratch = mkdtempSync(join(tmpdir(), "usage-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));
  const committedConfig = JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT, "generator.mixture-u.json"), "utf8")) as TeamGeneratorConfig;
  const cfg = small({ config_id: "usage-small", proposal: committedConfig.proposal as NonNullable<TeamGeneratorConfig["proposal"]> });
  // The uniform reference pilot of gate P4: the same config without its proposal.
  const plain: TeamGeneratorConfig = { ...cfg };
  delete plain.proposal;
  const uniformPilot = runPilot(
    new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, plain), labelSet: light, generationKey: KEY }),
    { config_hash: recordHash(plain), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) },
    light,
    plain.pilot_draws,
  );
  const run = (n: number) => {
    mkdirSync(join(scratch, `log-${n}`));
    return generate({
      engine, format, table, vocabulary, labelSet: light, config: cfg, roleSets: roleSource, usage: committed, uniformPilot,
      generationKey: KEY, assignmentKey: ASSIGNMENT_KEY, datasetId: "usage-1", planId: "plan-1",
      sizes: { dev: 2, validation: 2, training: 5 }, generatorCommit: "a".repeat(40),
      reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, logDir: join(scratch, `log-${n}`),
    });
  };

  it("is byte-identical on rerun and reports P1, P2 and P3 for every component, usage included", () => {
    const a = run(0);
    const b = run(1);
    expect(JSON.stringify(a.manifest)).toBe(JSON.stringify(b.manifest));
    expect(JSON.stringify(a.records)).toBe(JSON.stringify(b.records));
    expect(a.pilot.config_hash).toBe(recordHash(cfg));
    const usage = a.coverage.proposal?.components.find((c) => c.kind === "usage");
    expect(usage?.share_permille).toBe(500);
    expect(usage?.p1?.pass).toBe(true);
    expect(usage?.p2.value).toBeGreaterThan(0);
    // The committed example config meets every gate on this small run: P1 to P3 per component (usage P3 at the same 40 bits as the others) and P4.
    expect(usage?.p3.threshold).toBe(40);
    expect(usage?.p3.value).toBe(48);
    expect(usage?.p3_sequence_bits).toBe(75);
    expect(a.coverage.proposal?.failed).toEqual([]);
    expect(a.coverage.proposal?.publishable).toBe(true);
    expect(usage?.pilot.max_team_count).toBeGreaterThanOrEqual(1);
    expect(usage?.pilot.max_team_count).toBeLessThanOrEqual(usage?.pilot.draws ?? 0);
    expect(a.coverage.proposal?.components.filter((c) => c.kind !== "usage").every((c) => c.pilot.max_team_count === undefined)).toBe(true);
    expect(Object.keys(usage?.p1?.kinds ?? {})).toEqual(["species", "abilities", "items", "moves", "spreads", "teammates"]);
    expect(usage?.p1?.value).toBe(Math.min(...Object.values(usage?.p1?.kinds ?? {})));
    expect(a.coverage.proposal?.components.filter((c) => c.kind !== "usage").every((c) => c.p3_sequence_bits === undefined)).toBe(true);
    expect(a.coverage.proposal?.components.map((c) => c.kind)).toEqual(["role_sets", "uniform", "usage"]);
  }, 300_000); // each run draws a few hundred teams through the pinned validator

  it("accepts at least the proposed 500 per mille of the usage component's pilot draws on the committed table (P2)", () => {
    const pilotCfg = pure(committed);
    const drawer = new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, pilotCfg, undefined, committed), labelSet: light, generationKey: KEY });
    const pilot = runPilot(drawer, { config_hash: recordHash(pilotCfg), label_set_hash: "b".repeat(64), vocabulary_hash: "c".repeat(64) }, light, 400);
    expect(((pilot.draws - pilot.rejected) * 1000) / pilot.draws).toBeGreaterThanOrEqual(500);
    process.stderr.write(`usage P2 (400 pilot draws): ${pilot.draws - pilot.rejected} of ${pilot.draws} accepted\n`);
  }, 300_000);
});
