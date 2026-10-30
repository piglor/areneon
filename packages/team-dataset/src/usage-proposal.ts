// Component U of the mixture proposal (ADR-020 section 4): teams drawn from the usage weight table. The module
// chooses WHAT a team is made of from integer weights; the proposal turns the choice into sets. It holds no game
// rule, id or weight: every number comes from the committed `UsageWeightTable` (derived offline from one recorded
// Smogon chaos file) and the component's config fields.
//
// What one U draw does, in order:
//   1. species 1 by `w_species`; species k+1 by the summed teammate weights of the chosen species over the
//      unchosen roster entries, falling back to `w_species` when that sum is zero (and to uniform when both are);
//   2. per member a Mega forme holds its required item; otherwise an item by weight among those the Item Clause
//      still allows (the table's "no item" is an option; truncated and unmapped mass is not);
//   3. an ability by weight, up to `moves_per_set` distinct moves by weight without replacement (the rest of the
//      pool uniformly once the table's entries run out), one spread (nature and Stat Points) by weight;
//   4. with probability `floor_permille`, each categorical draw (every species pick, the item, the ability, each
//      move, the spread) is taken as the uniform proposal takes it instead: a species uniformly among the
//      unchosen roster entries, the item by the config's `item_weights`, the ability and moves uniformly from the
//      roster entry, the nature uniformly and the Stat Points by `drawStatPoints`.
// Set weights are the table's ppm under the integer root `set_root` (1 leaves the ppm); the species and teammate
// weights are the table's own, so `species_root` must equal the table's. Only integer picks of `rng.below`.

import {
  readUsageWeightTable,
  rootedWeight,
  type StatTable,
  type TeamVocabulary,
  type UsageWeightTable,
  usageTableHash,
} from "@areneon/showdown-bridge";
import type { Rng } from "./stream.ts";

export class UsageProposalError extends Error {
  override name = "UsageProposalError";
}

type RosterEntry = TeamVocabulary["roster"][number];

/** The table and its hash: the input of component U (the config's component must name the hash). */
export interface UsageSource {
  table: UsageWeightTable;
  hash: string;
}

/** Read a table against the vocabulary (the reader's checks). */
export function usageSource(vocabulary: TeamVocabulary, table: unknown): UsageSource {
  const read = readUsageWeightTable(table, vocabulary);
  return { table: read, hash: usageTableHash(read) };
}

/** A weighted list: ids parallel to positive integer weights. */
interface Weighted<T> {
  items: T[];
  weights: number[];
}

export interface SpreadChoice {
  nature: string;
  stat_points: StatTable;
}

/** One species' lists with the set root applied. */
export interface UsageMember {
  w_species: number;
  teammates: Map<string, number>;
  abilities: Weighted<string>;
  /** "" is no item. */
  items: Weighted<string>;
  moves: Weighted<string>;
  spreads: Weighted<SpreadChoice>;
}

const weighted = <T>(entries: readonly { value: T; ppm: number }[], root: number): Weighted<T> => {
  const out: Weighted<T> = { items: [], weights: [] };
  for (const e of entries) {
    const w = rootedWeight(e.ppm, root);
    if (w > 0) {
      out.items.push(e.value);
      out.weights.push(w);
    }
  }
  return out;
};

/** The members of a table with the set root applied, by species id. */
export function usageMembers(table: UsageWeightTable, setRoot: number): Map<string, UsageMember> {
  return new Map(
    table.species.map((s): [string, UsageMember] => [
      s.id,
      {
        w_species: s.w_species,
        teammates: new Map(s.teammates.entries.map((t) => [t.id, t.weight])),
        abilities: weighted(s.abilities.entries.map((e) => ({ value: e.id, ppm: e.ppm })), setRoot),
        items: weighted([...s.items.entries.map((e) => ({ value: e.id, ppm: e.ppm })), ...(s.items.no_item_ppm > 0 ? [{ value: "", ppm: s.items.no_item_ppm }] : [])], setRoot),
        moves: weighted(s.moves.entries.map((e) => ({ value: e.id, ppm: e.ppm })), setRoot),
        spreads: weighted(s.spreads.entries.map((e) => ({ value: { nature: e.nature, stat_points: e.stat_points as StatTable }, ppm: e.ppm })), setRoot),
      },
    ]),
  );
}

const sum = (weights: readonly number[]): number => weights.reduce((a, b) => a + b, 0);

/** One integer weighted pick; `weights` hold a positive total of at most 2^32. */
export function pickWeighted(rng: Rng, weights: readonly number[]): number {
  const total = sum(weights);
  if (total < 1) throw new UsageProposalError("a weighted pick with no weight");
  let u = rng.below(total);
  for (let k = 0; k < weights.length; k++) {
    if (u < (weights[k] as number)) return k;
    u -= weights[k] as number;
  }
  throw new UsageProposalError("a weighted pick ran past its total");
}

/** What the proposal supplies: the uniform proposal's own draws, used for the floor and when a list is empty. */
export interface UniformDraws {
  /** An item the Item Clause still allows, by the config's weights ("" when none). */
  item(holders: ReadonlyMap<string, number>): string;
  /** A nature id and Stat Points as the uniform proposal draws them. */
  spread(): SpreadChoice;
}

export interface UsageMemberDraw {
  entry: RosterEntry;
  item: string;
  ability: string;
  moves: string[];
  spread: SpreadChoice;
}

export class UsageSampler {
  readonly #roster: readonly RosterEntry[];
  readonly #members: Map<string, UsageMember>;
  readonly #floor: number;
  readonly #teamSize: number;
  readonly #movesPerSet: number;
  readonly #itemLimit: number;

  constructor(vocabulary: TeamVocabulary, source: UsageSource, setRoot: number, floorPermille: number, movesPerSet: number) {
    this.#roster = vocabulary.roster;
    this.#members = usageMembers(source.table, setRoot);
    this.#floor = floorPermille;
    this.#teamSize = vocabulary.rules.max_team_size;
    this.#movesPerSet = movesPerSet;
    this.#itemLimit = vocabulary.rules.item_clause;
  }

  get members(): ReadonlyMap<string, UsageMember> {
    return this.#members;
  }

  /** One categorical draw is uniform instead, with probability floor_permille (nothing is drawn at 0). */
  #uniform(rng: Rng): boolean {
    return this.#floor > 0 && rng.below(1000) < this.#floor;
  }

  /** The roster entries of one team, in the order drawn. */
  species(rng: Rng): RosterEntry[] {
    const chosen: RosterEntry[] = [];
    const taken = new Set<string>();
    while (chosen.length < this.#teamSize) {
      const open = this.#roster.filter((e) => !taken.has(e.id));
      let pick: RosterEntry;
      if (this.#uniform(rng)) {
        pick = open[rng.below(open.length)] as RosterEntry;
      } else {
        // Teammate correlation from the second species on; the first (and a zero sum) falls back to w_species.
        const byTeammates = open.map((e) => chosen.reduce((s, c) => s + ((this.#members.get(c.id)?.teammates.get(e.id)) ?? 0), 0));
        const byUsage = open.map((e) => this.#members.get(e.id)?.w_species ?? 0);
        const weights = chosen.length > 0 && sum(byTeammates) > 0 ? byTeammates : sum(byUsage) > 0 ? byUsage : undefined;
        pick = open[weights === undefined ? rng.below(open.length) : pickWeighted(rng, weights)] as RosterEntry;
      }
      taken.add(pick.id);
      chosen.push(pick);
    }
    return chosen;
  }

  /** The item of one member: its required item, else a weighted or uniform draw the Item Clause allows. */
  item(rng: Rng, entry: RosterEntry, holders: ReadonlyMap<string, number>, uniform: UniformDraws): string {
    if (entry.required_item !== undefined) return entry.required_item;
    const list = this.#members.get(entry.id)?.items;
    if (list === undefined || this.#uniform(rng)) return uniform.item(holders);
    const allowed = list.weights.map((w, k) => ((holders.get(list.items[k] as string) ?? 0) < this.#itemLimit || list.items[k] === "" ? w : 0));
    return sum(allowed) > 0 ? (list.items[pickWeighted(rng, allowed)] as string) : uniform.item(holders);
  }

  /** The member's ability, moves and spread (the item is drawn first, by the caller's order of draws). */
  details(rng: Rng, entry: RosterEntry, uniform: UniformDraws): Pick<UsageMemberDraw, "ability" | "moves" | "spread"> {
    const member = this.#members.get(entry.id);
    const abilities = member?.abilities;
    const ability =
      abilities === undefined || abilities.items.length === 0 || this.#uniform(rng) ? (entry.abilities[rng.below(entry.abilities.length)] as string) : (abilities.items[pickWeighted(rng, abilities.weights)] as string);
    const pool = [...entry.moves];
    const count = Math.min(this.#movesPerSet, pool.length);
    const moves: string[] = [];
    const listed = member?.moves;
    const left = listed === undefined ? [] : listed.items.map((id, k) => ({ id, weight: listed.weights[k] as number }));
    for (let k = 0; k < count; k++) {
      let move: string;
      if (left.length === 0 || this.#uniform(rng)) {
        move = pool[rng.below(pool.length)] as string;
      } else {
        move = (left[pickWeighted(rng, left.map((m) => m.weight))] as { id: string }).id;
      }
      moves.push(move);
      pool.splice(pool.indexOf(move), 1);
      const at = left.findIndex((m) => m.id === move);
      if (at >= 0) left.splice(at, 1);
    }
    const spreads = member?.spreads;
    const spread = spreads === undefined || spreads.items.length === 0 || this.#uniform(rng) ? uniform.spread() : (spreads.items[pickWeighted(rng, spreads.weights)] as SpreadChoice);
    return { ability, moves, spread };
  }
}
