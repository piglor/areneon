// The `generator-matched` usage prior of an ADR-020 MIXTURE proposal (ADR-025 section 10, slice 3b): the
// distribution a team member is drawn from when the generator config has a `proposal` of `uniform`,
// `role_sets` and `usage` components. `deriveGeneratorPrior` (generator-prior.ts) is the uniform proposal
// alone and is unchanged; this is a separate builder and a separate record, never an edit of the committed one.
//
// How the mixture is read. A team's component is drawn first (its share), then every member of the team comes
// from that component. So, given a roster entry (a forme), the set lists are the shares-weighted mix of the
// components' lists, each component weighted by `share * inclusion(entry)`: the posterior of the component
// given that the entry is on the team. The entry's weight within its preview species is that same sum, so a
// Mega forme's share of its species follows the components that can draw it.
//
//   uniform    inclusion team_size / N; abilities, moves, items as `deriveGeneratorPrior`;
//   role_sets  inclusion team_size / E (E table entries, an entry not in the table has none); a role uniform
//              within the entry; abilities uniform over the role's list, moves `min(slots, pool)` distinct of
//              the role's pool (inclusion k / n of a role, averaged over the entry's roles); item as uniform;
//   usage      species 1 by `w_species`, later species by teammate sums, each pick uniform with the floor
//              probability. The inclusion used here is ESTIMATED by simulating the species draw with the
//              component's own rule (`usageInclusions`: 200,000 teams on a fixed-seed generator, standard
//              error of at most sqrt(p / 200,000), so 7% relative at an inclusion of 0.001 and 22% at 0.0001,
//              less after the Rao-Blackwell step); the records are reproducible byte for byte.
//              Abilities, items and spreads are the table's lists under `set_root`, each
//              draw uniform with probability `floor`; the moves are drawn without replacement by weight
//              (uniform with probability `floor`, uniform once the listed moves run out), and their
//              INCLUSION is computed exactly by a dynamic programme over the listed moves taken so far.
//
// Not modelled (as in the uniform prior): the Item Clause couples the slots and is the belief's subset DP;
// the validator's rejections and the stratum quotas change the real team distribution a little. A Mega
// forme's `holder_abilities` are the validator's rewrite of any proposed Mega-forme set and so the same for
// every component.
//
// Arithmetic. The lists are computed in IEEE-754 doubles (sums in a fixed order, only + - * / and `Math.round`),
// which every platform computes identically, and each list is then fixed to an exact integer total by the
// largest-remainder rule (ties by id), so the record is exact integers under the contract's denominators and is
// reproducible byte for byte. The package's no-float rule is the opponent model's, not this builder's.

import type { TeamGeneratorConfig } from "@areneon/schema/types/v1/team-generator-config.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { generatorConfigHash } from "./generator-prior.ts";
import { type RoleSetTable, roleSetTableHash } from "./role-sets.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";
import { type UsageWeightTable, rootedWeight, usageTableHash } from "./usage-weight-table.ts";

export class MixturePriorError extends Error {
  override name = "MixturePriorError";
}

function bad(message: string): never {
  throw new MixturePriorError(message);
}

/** Bumped when the derivation changes how a number is computed. */
export const MIXTURE_PRIOR_SCRIPT = "generator-prior-mixture.v1";

/** The denominator of every list of the record: inclusions and shares are fixed to millionths. */
export const MIXTURE_DENOMINATOR = 1_000_000;

type Forme = UsagePrior["species"][number]["formes"][number];
type Spread = UsagePrior["species"][number]["formes"][number]["spreads"]["entries"][number];
type RosterEntry = TeamVocabulary["roster"][number];
type Component = NonNullable<TeamGeneratorConfig["proposal"]>["components"][number];
type StatPoints = Spread["stat_points"];

const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const STATS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
const spreadKey = (nature: string, sp: StatPoints): string => `${nature}|${STATS.map((s) => sp[s]).join(",")}`;

/** A component's lists for one roster entry, as probabilities. */
interface Dist {
  abilities: Map<string, number>;
  /** "" is no item. */
  items: Map<string, number>;
  /** The inclusion probability of each move, and the expected number of moves a set holds. */
  moves: Map<string, number>;
  count: number;
  /** Named spreads (key to nature, points, probability); the rest is the generator's other-spread draw. */
  spreads: Map<string, { nature: string; stat_points: StatPoints; p: number }>;
}

/** Fix `masses` (non-negative, parallel to `ids`) to integers summing to exactly `total`: floors, then one unit per largest remainder (ties by lower index). */
export function apportion(masses: readonly number[], total: number): number[] {
  const sum = masses.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) return masses.map(() => 0);
  const exact = masses.map((m) => (m * total) / sum);
  const out = exact.map((x) => Math.floor(x));
  let left = total - out.reduce((a, b) => a + b, 0);
  const order = exact.map((x, k) => ({ k, r: x - Math.floor(x) })).sort((a, b) => b.r - a.r || a.k - b.k);
  for (let i = 0; left > 0; i = (i + 1) % order.length, left--) out[(order[i] as { k: number }).k] = (out[(order[i] as { k: number }).k] as number) + 1;
  return out;
}

/** Inclusion probabilities of `slots` successive draws without replacement from `pool`: weighted over `listed` (uniform with probability `floor`, uniform once none is left). */
export function successiveInclusion(pool: readonly string[], listed: readonly { id: string; w: number }[], floor: number, slots: number): Map<string, number> {
  const count = Math.min(slots, pool.length);
  const ids = listed.map((l) => l.id);
  // The listed moves are the bits of a mask held in a double (exact below 2^53); the table lists at most k_moves = 32, so 40 is a limit, not a rule.
  if (ids.length > 40) bad("more listed moves than the programme's mask holds");
  const pow = ids.map((_, j) => 2 ** j);
  const has = (mask: number, j: number): boolean => Math.floor(mask / (pow[j] as number)) % 2 === 1;
  const unlisted = pool.length - ids.length;
  const incl = new Array<number>(ids.length).fill(0);
  let unlistedMass = 0;
  let level = new Map<number, number>([[0, 1]]);
  for (let t = 0; t < count; t++) {
    const remaining = pool.length - t;
    const next = new Map<number, number>();
    for (const [key, p] of level) {
      const mask = Math.floor(key / 64);
      const u = key % 64;
      let wLeft = 0;
      for (let j = 0; j < ids.length; j++) if (!has(mask, j)) wLeft += (listed[j] as { w: number }).w;
      const weighted = wLeft > 0;
      const uniformEach = weighted ? floor / remaining : 1 / remaining;
      for (let j = 0; j < ids.length; j++) {
        if (has(mask, j)) continue;
        const q = uniformEach + (weighted ? ((1 - floor) * (listed[j] as { w: number }).w) / wLeft : 0);
        incl[j] = (incl[j] as number) + p * q;
        const to = (mask + (pow[j] as number)) * 64 + u;
        next.set(to, (next.get(to) ?? 0) + p * q);
      }
      const unlistedLeft = unlisted - u;
      if (unlistedLeft > 0) {
        const q = unlistedLeft * uniformEach;
        unlistedMass += p * q;
        const to = mask * 64 + u + 1;
        next.set(to, (next.get(to) ?? 0) + p * q);
      }
    }
    level = next;
  }
  const out = new Map<string, number>();
  ids.forEach((id, j) => out.set(id, incl[j] as number));
  const each = unlisted > 0 ? unlistedMass / unlisted : 0;
  for (const id of pool) if (!out.has(id)) out.set(id, each);
  return out;
}

const uniformOver = (ids: readonly string[]): Map<string, number> => new Map<string, number>([...new Set(ids)].sort(cmp).map((id, _, all): [string, number] => [id, 1 / all.length]));

/** Teams simulated to estimate the usage component's species inclusion. */
export const USAGE_INCLUSION_TEAMS = 200_000;
/** The seed of that simulation (fixed: the record is reproducible byte for byte). */
export const USAGE_INCLUSION_SEED = 0x5eed1134;

/**
 * The probability that a roster entry is on a team drawn by the usage component, estimated by simulating
 * `USAGE_INCLUSION_TEAMS` species draws with the component's own rule (species 1 by `w_species`, later ones by
 * the summed teammate weights of the chosen species over the unchosen entries, falling back to `w_species` and
 * then to uniform, each pick uniform over the unchosen entries with probability `floor`) on a fixed-seed 32-bit
 * generator (Mulberry32). Only the species draw is simulated: the sets follow from the lists, exactly. The
 * estimate is Rao-Blackwellized (each pick adds the probability, given the picks so far, that each open entry
 * is the next one, instead of counting the chosen entry), so its standard error is at most the plain count's,
 * sqrt(p / teams) per entry: 7% relative at p = 0.001, 22% at p = 0.0001. The floor must be positive (an entry
 * the table lacks is then still drawn).
 */
function usageInclusions(roster: readonly RosterEntry[], rows: ReadonlyMap<string, UsageWeightTable["species"][number]>, floor: number, teamSize: number): Map<string, number> {
  const n = roster.length;
  const index = new Map<string, number>(roster.map((r, k): [string, number] => [r.id, k]));
  const w = roster.map((r) => rows.get(r.id)?.w_species ?? 0);
  const wTotal = w.reduce((a, b) => a + b, 0);
  const mates: Float64Array[] = roster.map((r) => {
    const row = new Float64Array(n);
    for (const t of rows.get(r.id)?.teammates.entries ?? []) {
      const at = index.get(t.id);
      if (at !== undefined) row[at] = t.weight;
    }
    return row;
  });
  let state = USAGE_INCLUSION_SEED >>> 0;
  const unit = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const hits = new Float64Array(n);
  const taken = new Uint8Array(n);
  const sums = new Float64Array(n);
  const weights = new Float64Array(n);
  for (let team = 0; team < USAGE_INCLUSION_TEAMS; team++) {
    taken.fill(0);
    sums.fill(0);
    for (let pick = 0; pick < teamSize; pick++) {
      const open = n - pick;
      // The weighted branch's weights over the open entries (also the Rao-Blackwell contribution below).
      let total = 0;
      let byTeam = 0;
      for (let e = 0; e < n; e++) byTeam += taken[e] === 1 ? 0 : (sums[e] as number);
      const useTeam = pick > 0 && byTeam > 0;
      for (let e = 0; e < n; e++) {
        const v = taken[e] === 1 ? 0 : useTeam ? (sums[e] as number) : wTotal > 0 ? (w[e] as number) : 1;
        weights[e] = v;
        total += v;
      }
      if (total <= 0) {
        total = 0;
        for (let e = 0; e < n; e++) {
          weights[e] = taken[e] === 1 ? 0 : 1;
          total += weights[e] as number;
        }
      }
      // Rao-Blackwellized count: the probability, given the picks so far, that each open entry is the next pick.
      for (let e = 0; e < n; e++) if (taken[e] === 0) hits[e] = (hits[e] as number) + floor / open + ((1 - floor) * (weights[e] as number)) / total;
      let chosen = -1;
      if (unit() < floor) {
        let skip = Math.floor(unit() * open);
        for (let e = 0; e < n; e++) {
          if (taken[e] === 1) continue;
          if (skip === 0) {
            chosen = e;
            break;
          }
          skip--;
        }
      } else {
        let u = unit() * total;
        for (let e = 0; e < n; e++) {
          if ((weights[e] as number) <= 0) continue;
          chosen = e;
          if (u < (weights[e] as number)) break;
          u -= weights[e] as number;
        }
      }
      taken[chosen] = 1;
      const row = mates[chosen] as Float64Array;
      for (let e = 0; e < n; e++) sums[e] = (sums[e] as number) + (row[e] as number);
    }
  }
  return new Map<string, number>(roster.map((r, k): [string, number] => [r.id, (hits[k] as number) / USAGE_INCLUSION_TEAMS]));
}

const inclusionMemo = new WeakMap<UsageWeightTable, Map<string, Map<string, number>>>();

/** `usageInclusions`, kept per table object and (floor, team size, roster) for the life of the process: the simulation is a pure function of them, and a test derives the same record several times. */
function memoInclusions(table: UsageWeightTable, roster: readonly RosterEntry[], rows: ReadonlyMap<string, UsageWeightTable["species"][number]>, floor: number, teamSize: number): Map<string, number> {
  const key = `${floor}|${teamSize}|${roster.map((r) => r.id).join(",")}`;
  const byTable = inclusionMemo.get(table) ?? new Map<string, Map<string, number>>();
  inclusionMemo.set(table, byTable);
  const known = byTable.get(key);
  if (known !== undefined) return known;
  const computed = usageInclusions(roster, rows, floor, teamSize);
  byTable.set(key, computed);
  return computed;
}

export interface MixturePriorInput {
  readonly vocabulary: TeamVocabulary;
  readonly config: TeamGeneratorConfig;
  /** The commit of the generator code whose draw rules (uniform, role-set and usage proposals) this reproduces. */
  readonly generator_commit: string;
  /** `megaPreviewSpecies`. */
  readonly shows_as: ReadonlyMap<string, string>;
  /** `megaHolderAbilities`. */
  readonly holder_abilities: ReadonlyMap<string, ReadonlyMap<string, number>>;
  /** The role-set table, when the config has a `role_sets` component (its hash must be the component's). */
  readonly role_sets?: RoleSetTable;
  /** The usage weight table, when the config has a `usage` component (its hash must be the component's). */
  readonly usage?: UsageWeightTable;
}

/** The components of a mixture config, each checked against the table it names; refuses anything the builder does not know. */
function componentsOf(input: MixturePriorInput): Component[] {
  const { config, role_sets, usage } = input;
  if (config.proposal === undefined) bad("the generator config has no mixture proposal: use deriveGeneratorPrior");
  if ((config.proposal as { kind: string }).kind !== "mixture") bad("the proposal is not a mixture");
  const components = config.proposal.components;
  const kinds = new Set<string>();
  for (const c of components) {
    if (c.kind !== "uniform" && c.kind !== "role_sets" && c.kind !== "usage") bad(`the mixture has a component of a kind the builder does not know: ${String((c as { kind: unknown }).kind)}`);
    if (kinds.has(c.kind)) bad(`the mixture lists ${c.kind} twice`);
    kinds.add(c.kind);
    if (c.kind === "role_sets") {
      if (role_sets === undefined) bad("the mixture has a role_sets component and no role-set table was given");
      if (roleSetTableHash(role_sets) !== c.role_set_table_hash) bad("the role-set table is not the one the role_sets component names");
    }
    if (c.kind === "usage") {
      if (usage === undefined) bad("the mixture has a usage component and no usage table was given");
      if (usageTableHash(usage) !== c.usage_table_hash) bad("the usage table is not the one the usage component names");
      if (usage.derivation.species_root !== c.species_root) bad("the usage component's species_root is not the table's own");
    }
  }
  if (components.reduce((t, c) => t + c.share_permille, 0) !== 1000) bad("the component shares do not sum to 1000");
  if (role_sets !== undefined && !kinds.has("role_sets")) bad("a role-set table was given but the mixture has no role_sets component");
  if (usage !== undefined && !kinds.has("usage")) bad("a usage table was given but the mixture has no usage component");
  return [...components];
}

interface Parts {
  readonly weight: number;
  readonly dist: Dist;
}

/** Derive the `generator-matched` prior of a mixture `config` on `vocabulary`. */
export function deriveMixturePrior(input: MixturePriorInput): UsagePrior {
  const { vocabulary, config, generator_commit, shows_as, holder_abilities, role_sets, usage } = input;
  if (!/^[0-9a-f]{40}$/.test(generator_commit)) bad("generator_commit is not 40 lower-case hex digits");
  const components = componentsOf(input);
  const slots = config.moves_per_set ?? bad("the generator config has no moves_per_set");
  const teamSize = vocabulary.rules.max_team_size;
  const roster = vocabulary.roster;
  const rosterIds = new Set(roster.map((r) => r.id));
  const N = roster.length;

  // The uniform proposal's item weights (the floor and the role-set and uniform components draw items by them).
  const items = new Set(vocabulary.items);
  const itemWeights = new Map<string, number>();
  const override = new Map(config.item_weights.map((w) => [w.item, w.weight]));
  for (const id of override.keys()) if (!items.has(id)) bad(`the config weighs item ${id}, which the vocabulary does not list`);
  for (const id of vocabulary.items) itemWeights.set(id, override.get(id) ?? config.default_item_weight);

  const uniformItems = (entry: RosterEntry): Map<string, number> => {
    if (entry.required_item !== undefined) return new Map<string, number>([[entry.required_item, 1]]);
    const named = [...itemWeights.keys()].sort(cmp).filter((id) => (itemWeights.get(id) as number) > 0);
    const sum = named.reduce((t, id) => t + (itemWeights.get(id) as number), 0);
    return sum === 0 ? new Map<string, number>([["", 1]]) : new Map<string, number>(named.map((id): [string, number] => [id, (itemWeights.get(id) as number) / sum]));
  };
  const uniformMoves = (entry: RosterEntry): { moves: Map<string, number>; count: number } => {
    const pool = [...new Set(entry.moves)].sort(cmp);
    const k = Math.min(slots, pool.length);
    return { moves: new Map<string, number>(pool.map((id): [string, number] => [id, k / pool.length])), count: k };
  };
  const uniformDist = (entry: RosterEntry): Dist => ({
    abilities: uniformOver(entry.abilities),
    items: uniformItems(entry),
    ...(() => {
      const m = uniformMoves(entry);
      return { moves: m.moves, count: m.count };
    })(),
    spreads: new Map(),
  });

  const roleBy = new Map<string, RoleSetTable["species"][number]>((role_sets?.species ?? []).map((s): [string, RoleSetTable["species"][number]] => [s.id, s]));
  const roleDist = (entry: RosterEntry): Dist | null => {
    const table = roleBy.get(entry.id);
    if (table === undefined) return null;
    const moves = new Map<string, number>();
    let count = 0;
    const abilities = new Map<string, number>();
    for (const set of table.sets) {
      const pool = [...new Set(set.moves)];
      const k = Math.min(slots, pool.length);
      count += k / table.sets.length;
      for (const id of pool) moves.set(id, (moves.get(id) ?? 0) + k / pool.length / table.sets.length);
      const own = [...new Set(set.abilities)];
      for (const id of own) abilities.set(id, (abilities.get(id) ?? 0) + 1 / own.length / table.sets.length);
    }
    const u = uniformDist(entry);
    // A Mega forme's own list is the roster's (the post-Evolution ability); the table names the pre-Evolution one.
    return { abilities: shows_as.has(entry.id) ? u.abilities : abilities, items: u.items, moves, count, spreads: new Map() };
  };

  const usageBy = new Map<string, UsageWeightTable["species"][number]>((usage?.species ?? []).map((s): [string, UsageWeightTable["species"][number]] => [s.id, s]));
  const usageComponent = components.find((c) => c.kind === "usage");
  if (usageComponent?.kind === "usage" && usageComponent.floor_permille === 0) bad("a usage component with floor_permille 0 is refused: an entry the table lacks would have no inclusion, and the belief would refuse a legal opponent");
  const floor = usageComponent?.kind === "usage" ? usageComponent.floor_permille / 1000 : 0;
  const setRoot = usageComponent?.kind === "usage" ? usageComponent.set_root : 1;
  const mixWith = (a: Map<string, number>, wa: number, b: Map<string, number>, wb: number): Map<string, number> => {
    const out = new Map<string, number>();
    for (const [id, p] of a) out.set(id, (out.get(id) ?? 0) + wa * p);
    for (const [id, p] of b) out.set(id, (out.get(id) ?? 0) + wb * p);
    return out;
  };
  const rooted = (entries: readonly { id: string; ppm: number }[]): Map<string, number> => {
    const raw = entries.map((e) => ({ id: e.id, w: rootedWeight(e.ppm, setRoot) })).filter((e) => e.w > 0);
    const sum = raw.reduce((t, e) => t + e.w, 0);
    return new Map(raw.map((e) => [e.id, e.w / sum]));
  };
  const usageDist = (entry: RosterEntry): Dist => {
    const u = uniformDist(entry);
    const row = usageBy.get(entry.id);
    if (row === undefined) return u;
    const mega = shows_as.has(entry.id);
    const tableAbilities = rooted(row.abilities.entries);
    const abilities = mega || tableAbilities.size === 0 ? u.abilities : mixWith(tableAbilities, 1 - floor, u.abilities, floor);
    const tableItems = rooted([...row.items.entries, ...(row.items.no_item_ppm > 0 ? [{ id: "", ppm: row.items.no_item_ppm }] : [])]);
    const itemsOut = entry.required_item !== undefined || tableItems.size === 0 ? u.items : mixWith(tableItems, 1 - floor, u.items, floor);
    const pool = [...new Set(entry.moves)].sort(cmp);
    const listed = row.moves.entries.map((e) => ({ id: e.id, w: rootedWeight(e.ppm, setRoot) })).filter((e) => e.w > 0 && pool.includes(e.id));
    const moves = successiveInclusion(pool, listed, floor, slots);
    const spreads = new Map<string, { nature: string; stat_points: StatPoints; p: number }>();
    const rawSpreads = row.spreads.entries.map((e) => ({ e, w: rootedWeight(e.ppm, setRoot) })).filter((s) => s.w > 0);
    const spreadSum = rawSpreads.reduce((t, s) => t + s.w, 0);
    for (const { e, w } of rawSpreads) spreads.set(spreadKey(e.nature, e.stat_points as StatPoints), { nature: e.nature, stat_points: e.stat_points as StatPoints, p: ((1 - floor) * w) / spreadSum });
    return { abilities, items: itemsOut, moves, count: Math.min(slots, pool.length), spreads };
  };
  // The simulation runs only for a config that has a usage component (a floor of 0 is refused above).
  const usageInclusionOf = usageComponent === undefined || usage === undefined ? new Map<string, number>() : memoInclusions(usage, roster, usageBy, floor, teamSize);
  const usageInclusion = (entry: RosterEntry): number => usageInclusionOf.get(entry.id) ?? 0;

  const partsOf = (entry: RosterEntry): Parts[] => {
    const out: Parts[] = [];
    for (const c of components) {
      const share = c.share_permille / 1000;
      if (c.kind === "uniform") out.push({ weight: share * Math.min(1, teamSize / N), dist: uniformDist(entry) });
      else if (c.kind === "role_sets") {
        const d = roleDist(entry);
        if (d !== null) out.push({ weight: share * Math.min(1, teamSize / (role_sets as RoleSetTable).species.length), dist: d });
      } else out.push({ weight: share * usageInclusion(entry), dist: usageDist(entry) });
    }
    return out.filter((p) => p.weight > 0);
  };

  const formeOf = (entry: RosterEntry, parts: readonly Parts[]): Forme => {
    const D = MIXTURE_DENOMINATOR;
    const total = parts.reduce((t, p) => t + p.weight, 0);
    const mix = <T>(pick: (d: Dist) => Map<string, T>, value: (t: T) => number): Map<string, number> => {
      const out = new Map<string, number>();
      for (const p of parts) for (const [id, v] of pick(p.dist)) out.set(id, (out.get(id) ?? 0) + (p.weight / total) * value(v));
      return out;
    };
    const ids = (m: Map<string, number>): string[] => [...m.keys()].sort(cmp);
    const abilityMass = mix((d) => d.abilities, (v) => v);
    const abilityIds = ids(abilityMass).filter((id) => (abilityMass.get(id) as number) > 0);
    const abilityW = apportion(abilityIds.map((id) => abilityMass.get(id) as number), D);
    const itemMass = mix((d) => d.items, (v) => v);
    const itemIds = ids(itemMass).filter((id) => (itemMass.get(id) as number) > 0);
    const itemW = apportion(itemIds.map((id) => itemMass.get(id) as number), D);
    const moveMass = mix((d) => d.moves, (v) => v);
    const count = parts.reduce((t, p) => t + (p.weight / total) * p.dist.count, 0);
    const moveIds = ids(moveMass).filter((id) => (moveMass.get(id) as number) > 0);
    const target = Math.round(count * D);
    const moveW = apportion(moveIds.map((id) => moveMass.get(id) as number), target).map((w) => Math.min(w, D));
    const named = new Map<string, { nature: string; stat_points: StatPoints; p: number }>();
    for (const p of parts) for (const [key, s] of p.dist.spreads) named.set(key, { ...s, p: (named.get(key)?.p ?? 0) + (p.weight / total) * s.p });
    const spreadKeys = [...named.keys()].sort((a, b) => {
      const x = named.get(a) as { nature: string; stat_points: StatPoints };
      const y = named.get(b) as { nature: string; stat_points: StatPoints };
      if (x.nature !== y.nature) return cmp(x.nature, y.nature);
      for (const s of STATS) if (x.stat_points[s] !== y.stat_points[s]) return x.stat_points[s] - y.stat_points[s];
      return 0;
    });
    const namedP = spreadKeys.reduce((t, k) => t + (named.get(k) as { p: number }).p, 0);
    const spreadW = apportion([...spreadKeys.map((k) => (named.get(k) as { p: number }).p), Math.max(0, 1 - namedP)], D);
    const holder = shows_as.has(entry.id) ? (holder_abilities.get(entry.id) ?? bad(`roster entry ${entry.id} has no holder abilities`)) : undefined;
    const noItem = itemIds.indexOf("");
    const inclusion = moveIds.map((id, k) => ({ id, inclusion: moveW[k] as number })).filter((e) => e.inclusion > 0);
    return {
      id: entry.id,
      // At least 1: the contract wants a positive integer weight, and an entry kept in the record (total > 0) must not round to 0.
      weight: Math.max(1, Math.round(total * D)),
      abilities: { denominator: D, entries: abilityIds.map((id, k) => ({ id, weight: abilityW[k] as number })).filter((e) => e.weight > 0), residual: 0, unmapped: 0 },
      ...(holder === undefined
        ? {}
        : {
            holder_abilities: {
              denominator: [...holder.values()].reduce((a, b) => a + b, 0),
              entries: [...holder.keys()].sort(cmp).map((id) => ({ id, weight: holder.get(id) as number })),
              residual: 0,
              unmapped: 0,
            },
          }),
      items: {
        denominator: D,
        entries: itemIds.flatMap((id, k) => (id === "" || (itemW[k] as number) === 0 ? [] : [{ id, weight: itemW[k] as number }])),
        no_item: noItem < 0 ? 0 : (itemW[noItem] as number),
        residual: 0,
        unmapped: 0,
      },
      spreads: {
        denominator: D,
        entries: spreadKeys.flatMap((k, i) => ((spreadW[i] as number) > 0 ? [{ nature: (named.get(k) as { nature: string }).nature, stat_points: (named.get(k) as { stat_points: StatPoints }).stat_points, weight: spreadW[i] as number }] : [])),
        residual: spreadW[spreadKeys.length] as number,
        unmapped: 0,
      },
      moves: { denominator: D, slots, entries: inclusion, empty_slots: slots * D - inclusion.reduce((t, e) => t + e.inclusion, 0), residual: 0, unmapped: 0 },
    };
  };

  const groups = new Map<string, Forme[]>();
  for (const entry of roster) {
    const shown = shows_as.get(entry.id) ?? entry.id;
    if (!rosterIds.has(shown)) bad(`roster entry ${entry.id} shows as ${shown}, which is not in the roster`);
    if (shows_as.has(shown)) bad(`roster entry ${entry.id} shows as ${shown}, which itself shows as another species`);
    const parts = partsOf(entry);
    if (parts.length === 0) continue;
    groups.set(shown, [...(groups.get(shown) ?? []), formeOf(entry, parts)]);
  }
  if (groups.size === 0) bad("the mixture draws no roster entry");
  const species = [...groups.keys()].sort(cmp).map((id) => {
    const formes = (groups.get(id) as Forme[]).sort((a, b) => cmp(a.id, b.id));
    return { id, denominator: formes.reduce((t, f) => t + f.weight, 0), formes: formes as [Forme, ...Forme[]] };
  });
  const all = species.flatMap((s) => s.formes);
  return {
    schema_version: 1,
    format_id: vocabulary.format_id,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: teamVocabularyHash(vocabulary),
    source: { kind: "generator-matched", generator_config_hash: generatorConfigHash(config), generator_commit },
    derivation: {
      script_version: MIXTURE_PRIOR_SCRIPT,
      k_items: Math.max(1, ...all.map((f) => f.items.entries.length)),
      k_moves: Math.max(1, ...all.map((f) => f.moves.entries.length)),
      k_spreads: Math.max(1, ...all.map((f) => f.spreads.entries.length)),
    },
    species: species as UsagePrior["species"],
  };
}
