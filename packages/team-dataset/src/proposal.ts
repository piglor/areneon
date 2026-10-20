// The generator's proposal (ADR-013 section 2): one team drawn uniformly from
// the pinned engine's vocabulary, with integer item weights. It writes no game
// rule. Roster, move pools, abilities, items and natures come from the
// vocabulary the bridge exported from the engine; level, team size, the Stat
// Point total and the Item Clause come from the format's rule table; IVs, the
// per-stat draw maximum, gender, nicknames and the item weights are named
// config fields. A proposal may be illegal: `validateTeam` judges it, and a
// wrong named value only raises the rejection rate.
//
// What one draw does, in order:
//   1. choose `max_team_size` species, uniformly and without repeats;
//   2. a species that must hold an item (a Mega forme) holds it;
//   3. every other member draws an item by weight among the items the Item
//      Clause still allows;
//   4. each member draws an ability, up to `moves_per_set` distinct moves from its pool
//      and a nature, all uniformly;
//   5. Stat Points: until the total is spent, pick a stat that still has room
//      and give it a uniform amount from one up to what room and total allow.

import type { EngineData, PokemonSet, StatTable, TeamGeneratorConfig, TeamVocabulary } from "@areneon/showdown-bridge";
import type { Rng } from "./stream.ts";

export class ProposalError extends Error {
  override name = "ProposalError";
}

const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;

type RosterEntry = TeamVocabulary["roster"][number];

export class Proposal {
  readonly #vocabulary: TeamVocabulary;
  readonly #config: TeamGeneratorConfig;
  readonly #names: { species: Map<string, string>; moves: Map<string, string>; items: Map<string, string>; abilities: Map<string, string> };
  readonly #weights: number[];
  readonly #teamSize: number;
  readonly #movesPerSet: number;

  constructor(vocabulary: TeamVocabulary, table: EngineData, config: TeamGeneratorConfig) {
    this.#vocabulary = vocabulary;
    this.#config = config;
    const names = (entries: readonly { id: string; name: string }[]): Map<string, string> => new Map(entries.map((e) => [e.id, e.name]));
    this.#names = { species: names(table.species), moves: names(table.moves), items: names(table.items), abilities: names(table.abilities) };
    const { rules } = vocabulary;
    this.#teamSize = rules.max_team_size;
    if (config.moves_per_set === undefined) throw new ProposalError("the generator config has no moves_per_set");
    this.#movesPerSet = config.moves_per_set;
    if (vocabulary.roster.length < this.#teamSize) throw new ProposalError("the roster is smaller than a team");
    if (vocabulary.natures.length === 0) throw new ProposalError("the vocabulary lists no nature");
    if (config.stat_point_max_per_stat * STATS.length < rules.ev_limit) {
      throw new ProposalError(`${STATS.length} stats of at most ${config.stat_point_max_per_stat} cannot hold ${rules.ev_limit} Stat Points`);
    }
    const override = new Map(config.item_weights.map((w) => [w.item, w.weight]));
    this.#weights = vocabulary.items.map((item) => override.get(item) ?? config.default_item_weight);
    // A weight total must fit one 32-bit pick.
    if (this.#weights.reduce((a, b) => a + b, 0) > 2 ** 32) throw new ProposalError("the item weights add up to more than 2^32");
    for (const entry of vocabulary.roster) {
      if (!this.#names.species.has(entry.id)) throw new ProposalError(`species ${entry.id} has no name in the engine-data table`);
    }
  }

  /** One proposed team, in the order the species were drawn. */
  propose(rng: Rng): PokemonSet[] {
    const roster = this.#vocabulary.roster;
    const chosen: RosterEntry[] = [];
    const used = new Set<number>();
    while (chosen.length < this.#teamSize) {
      const at = rng.below(roster.length);
      if (used.has(at)) continue;
      used.add(at);
      chosen.push(roster[at] as RosterEntry);
    }
    const holders = new Map<string, number>();
    for (const entry of chosen) if (entry.required_item !== undefined) holders.set(entry.required_item, (holders.get(entry.required_item) ?? 0) + 1);
    return chosen.map((entry) => {
      const item = entry.required_item ?? this.#drawItem(rng, holders);
      if (entry.required_item === undefined && item !== "") holders.set(item, (holders.get(item) ?? 0) + 1);
      return this.#set(rng, entry, item);
    });
  }

  /** An item by weight among those the Item Clause still allows, or "" when none has weight. */
  #drawItem(rng: Rng, holders: ReadonlyMap<string, number>): string {
    const items = this.#vocabulary.items;
    const limit = this.#vocabulary.rules.item_clause;
    let total = 0;
    const cumulative = items.map((item, k) => {
      if ((holders.get(item) ?? 0) < limit) total += this.#weights[k] as number;
      return total;
    });
    if (total === 0) return "";
    const u = rng.below(total);
    return items[cumulative.findIndex((c) => u < c)] as string;
  }

  #set(rng: Rng, entry: RosterEntry, item: string): PokemonSet {
    const { natures, rules } = this.#vocabulary;
    const ability = entry.abilities[rng.below(entry.abilities.length)] as string;
    const pool = [...entry.moves];
    const count = Math.min(this.#movesPerSet, pool.length);
    for (let k = 0; k < count; k++) {
      const swap = k + rng.below(pool.length - k);
      [pool[k], pool[swap]] = [pool[swap] as string, pool[k] as string];
    }
    const nature = natures[rng.below(natures.length)] as TeamVocabulary["natures"][number];
    const gender = this.#config.gender_policy === "random" ? (rng.below(2) === 0 ? "M" : "F") : "";
    const species = this.#names.species.get(entry.id) as string;
    return {
      name: this.#config.nickname_policy === "species" ? species : "",
      species,
      item: item === "" ? "" : (this.#names.items.get(item) as string),
      ability: this.#names.abilities.get(ability) as string,
      moves: pool.slice(0, count).map((m) => this.#names.moves.get(m) as string),
      nature: nature.name,
      gender,
      evs: this.#statPoints(rng, rules.ev_limit),
      ivs: Object.fromEntries(STATS.map((s) => [s, this.#config.iv])) as StatTable,
      level: rules.adjust_level,
    };
  }

  #statPoints(rng: Rng, total: number): StatTable {
    const points: Record<string, number> = Object.fromEntries(STATS.map((s) => [s, 0]));
    const cap = this.#config.stat_point_max_per_stat;
    let left = total;
    while (left > 0) {
      const open = STATS.filter((s) => (points[s] as number) < cap);
      const stat = open[rng.below(open.length)] as string;
      const room = cap - (points[stat] as number);
      const amount = 1 + rng.below(Math.min(room, left));
      points[stat] = (points[stat] as number) + amount;
      left -= amount;
    }
    return points as StatTable;
  }
}
