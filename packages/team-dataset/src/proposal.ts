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
import { drawRolePicks, type RoleSetsSource } from "./role-set-proposal.ts";
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
  readonly #roleSets: RoleSetsSource | undefined;
  readonly #roster: Map<string, RosterEntry>;

  /**
   * With `config.proposal` (ADR-020), every draw first picks its component by `rng.below(1000)`; a
   * `role_sets` component needs `roleSets`, whose hash must be the one the config names. Without
   * `proposal` nothing below changes: no extra draw, the uniform path alone.
   */
  constructor(vocabulary: TeamVocabulary, table: EngineData, config: TeamGeneratorConfig, roleSets?: RoleSetsSource) {
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
    this.#roster = new Map(vocabulary.roster.map((e) => [e.id, e]));
    const named = config.proposal?.components.find((c) => c.kind === "role_sets");
    if (named !== undefined) {
      if (roleSets === undefined) throw new ProposalError("the config has a role_sets component and no role-set table was given");
      if (roleSets.hash !== named.role_set_table_hash) throw new ProposalError("the role-set table is not the one the config names (hash mismatch)");
      for (const species of roleSets.table.species) if (!this.#roster.has(species.id) || !this.#names.species.has(roleSets.shows_as.get(species.id) ?? species.id)) throw new ProposalError(`role-set species ${species.id} is not in the roster`);
    }
    this.#roleSets = roleSets;
  }

  /** The component a draw takes, read from its first `rng.below(1000)`; "uniform" when no mixture is configured (nothing is drawn). */
  component(rng: Rng): "uniform" | "role_sets" {
    const components = this.#config.proposal?.components;
    if (components === undefined) return "uniform";
    const u = rng.below(1000);
    let edge = 0;
    for (const c of components) {
      edge += c.share_permille;
      if (u < edge) return c.kind;
    }
    throw new ProposalError("the component shares do not add up to 1000");
  }

  /** The kinds of the configured components, in config order; undefined when no mixture is configured. */
  componentKinds(): ("uniform" | "role_sets")[] | undefined {
    return this.#config.proposal?.components.map((c) => c.kind);
  }

  /** One proposed team, in the order the species were drawn. */
  propose(rng: Rng): PokemonSet[] {
    if (this.component(rng) === "role_sets") return this.#proposeRoleSets(rng);
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
      return this.#set(rng, item, entry.id, entry.abilities, entry.moves);
    });
  }

  /** Component R: the members come from the role sets; item, nature and Stat Points are the uniform draws. */
  #proposeRoleSets(rng: Rng): PokemonSet[] {
    const picks = drawRolePicks(rng, this.#roleSets as RoleSetsSource, this.#teamSize);
    const entries = picks.map((pick) => this.#roster.get(pick.entry) as RosterEntry);
    const holders = new Map<string, number>();
    for (const entry of entries) if (entry.required_item !== undefined) holders.set(entry.required_item, (holders.get(entry.required_item) ?? 0) + 1);
    return picks.map((pick, k) => {
      const entry = entries[k] as RosterEntry;
      const item = entry.required_item ?? this.#drawItem(rng, holders);
      if (entry.required_item === undefined && item !== "") holders.set(item, (holders.get(item) ?? 0) + 1);
      return this.#set(rng, item, pick.species, pick.abilities, pick.moves);
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

  #set(rng: Rng, item: string, speciesId: string, abilities: readonly string[], moves: readonly string[]): PokemonSet {
    const { natures, rules } = this.#vocabulary;
    const ability = abilities[rng.below(abilities.length)] as string;
    const pool = [...moves];
    const count = Math.min(this.#movesPerSet, pool.length);
    for (let k = 0; k < count; k++) {
      const swap = k + rng.below(pool.length - k);
      [pool[k], pool[swap]] = [pool[swap] as string, pool[k] as string];
    }
    const nature = natures[rng.below(natures.length)] as TeamVocabulary["natures"][number];
    const gender = this.#config.gender_policy === "random" ? (rng.below(2) === 0 ? "M" : "F") : "";
    const species = this.#names.species.get(speciesId) as string;
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
    return drawStatPoints((n) => rng.below(n), total, this.#config.stat_point_max_per_stat);
  }
}

/**
 * The generator's Stat Point routine: until the total is spent, pick a stat that still has room
 * (uniformly among those below `cap`) and give it a uniform amount from one up to what room and total
 * allow. `below(n)` is a uniform integer in [0, n). It is the one definition: the proposal calls it, and
 * so does the opponent model's `SpreadSampler` (`spread-sampler.ts`), so a sampled "other spread" is
 * drawn by the generator's own rule.
 */
export function drawStatPoints(below: (n: number) => number, total: number, cap: number): StatTable {
  const points: Record<string, number> = Object.fromEntries(STATS.map((s) => [s, 0]));
  let left = total;
  while (left > 0) {
    const open = STATS.filter((s) => (points[s] as number) < cap);
    if (open.length === 0) throw new ProposalError(`${STATS.length} stats of at most ${cap} cannot hold ${total} Stat Points`);
    const stat = open[below(open.length)] as string;
    const room = cap - (points[stat] as number);
    const amount = 1 + below(Math.min(room, left));
    points[stat] = (points[stat] as number) + amount;
    left -= amount;
  }
  return points as StatTable;
}
