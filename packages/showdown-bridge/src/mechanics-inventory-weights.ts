// Weights and coverage curves over the mechanics inventory (ADR-030 groundwork
// G5, #1465). The inventory says WHAT the roster can reach; this says how often,
// so that a backend's build order follows the content real teams and games hold.
//
// Three weight sources feed one ranking: the teams of a dataset (the committed
// fixtures-v1 dev split, or a generated dataset's dev and training splits; never
// validation), the Smogon usage weight table (aggregate public data, ADR-020) and,
// in the run module, the Episodes of a run. Every number is an integer (parts per
// million or a count).
//
// "Damage-relevant" entities (`damage` below) are those that define a handler
// whose name is in the inventory's `damage_handler_names`; a condition counts
// when the mod's condition table defines it, a condition that lives in a move,
// ability or item is that entity's own handlers. A backend that implements the
// top N entities of the order covers a team when every damage-relevant entity of
// the team is among them (`curve`); two teams make a game, so the share of games
// covered is about the square of the share of teams. The usage curve has no
// teams to count: it is an independence estimate (each entity held by a team
// with the probability its usage gives, entities independent), stated as such.

import type { UsageWeightTable } from "@areneon/schema/types/v1/usage-weight-table.ts";
import { inventoryHash, type InventoryEntity, KINDS, type Kind, type MechanicsInventory, toId } from "./mechanics-inventory.ts";
import { usageTableHash } from "./usage-weight-table.ts";

export const WEIGHTS_FORMAT = "mechanics-inventory-weights/1";
/** The committed weights report, beside the inventory. */
export const WEIGHTS_FILE = "mechanics-inventory.weights.json";

const PPM = 1_000_000;
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
export const entityKey = (kind: Kind, id: string): string => `${kind}:${id}`;

/** The part of a team member the weights read. */
export interface TeamMember {
  item: string;
  ability: string;
  moves: string[];
}

/** What a weight pass needs from the inventory, indexed once. */
export interface InventoryIndex {
  /** `kind:id` of every entity, by kind. */
  keys: Record<Kind, Set<string>>;
  /** The damage-relevant entities and, for each, the names of its damage handlers (last name only), sorted. */
  damage: Map<string, string[]>;
  /** A starter's `kind:id` to the condition keys it starts. */
  starts: Map<string, string[]>;
}

const lastName = (handler: string): string => handler.slice(handler.lastIndexOf(".") + 1);

function listOf(inv: MechanicsInventory, kind: Kind): InventoryEntity[] {
  return { move: inv.moves, ability: inv.abilities, item: inv.items, condition: inv.conditions }[kind];
}

export function buildIndex(inv: MechanicsInventory): InventoryIndex {
  const keys = { move: new Set<string>(), ability: new Set<string>(), item: new Set<string>(), condition: new Set<string>() };
  const damage = new Map<string, string[]>();
  const starts = new Map<string, string[]>();
  for (const kind of KINDS) {
    for (const entity of listOf(inv, kind)) {
      const key = entityKey(kind, entity.id);
      keys[kind].add(key);
      const own = entity.defined_in === undefined || entity.defined_in === "table";
      if (entity.damage_handlers.length > 0 && own) damage.set(key, [...new Set(entity.damage_handlers.map(lastName))].sort(cmp));
      for (const starter of entity.started_by ?? []) starts.set(starter, [...(starts.get(starter) ?? []), key]);
    }
  }
  return { keys, damage, starts };
}

/** The entity keys a team holds, the conditions its entities start included, and how many ids the inventory does not have. */
export function teamKeys(index: InventoryIndex, members: TeamMember[]): { keys: Set<string>; unknown: number } {
  const keys = new Set<string>();
  let unknown = 0;
  const add = (kind: Kind, raw: string): void => {
    if (raw === "") return;
    const key = entityKey(kind, toId(raw));
    if (!index.keys[kind].has(key)) unknown++;
    else {
      keys.add(key);
      for (const started of index.starts.get(key) ?? []) keys.add(started);
    }
  };
  for (const member of members) {
    add("item", member.item);
    add("ability", member.ability);
    for (const move of member.moves) add("move", move);
  }
  return { keys, unknown };
}

// ---- coverage curves -------------------------------------------------------------------------

export interface Curve {
  /** How many units (teams, games, turns) were counted; 0 for an estimate. */
  units: number;
  /** The entities in build order, most frequent first (ties by key), with the units that hold each. */
  order: { key: string; count: number }[];
  /** `covered_ppm[n]`: the share of units fully covered when the top `n` entities are implemented (n = 0 to the order's length). */
  covered_ppm: number[];
}

/** The order and coverage of `units`, each a list of the keys it needs. */
export function curve(units: string[][]): Curve {
  const counts = new Map<string, number>();
  for (const unit of units) for (const key of new Set(unit)) counts.set(key, (counts.get(key) ?? 0) + 1);
  const order = [...counts].map(([key, count]) => ({ key, count })).sort((a, b) => b.count - a.count || cmp(a.key, b.key));
  const rank = new Map(order.map((e, n) => [e.key, n]));
  const at = new Array<number>(order.length + 1).fill(0);
  for (const unit of units) {
    const needed = Math.max(-1, ...unit.map((key) => rank.get(key) as number)) + 1;
    at[needed] = (at[needed] as number) + 1;
  }
  let covered = 0;
  const covered_ppm = at.map((n) => Math.floor(((covered += n) * PPM) / Math.max(units.length, 1)));
  return { units: units.length, order, covered_ppm };
}

/** The independence estimate: unit `i` is needed with probability `p_ppm`, entities independent. */
export function independenceCurve(probabilities: { key: string; p_ppm: number }[]): Curve {
  const order = [...probabilities].sort((a, b) => b.p_ppm - a.p_ppm || cmp(a.key, b.key));
  const covered_ppm = new Array<number>(order.length + 1).fill(PPM);
  for (let n = order.length - 1; n >= 0; n--) covered_ppm[n] = Math.floor(((covered_ppm[n + 1] as number) * (PPM - (order[n] as { p_ppm: number }).p_ppm)) / PPM);
  return { units: 0, order: order.map((e) => ({ key: e.key, count: e.p_ppm })), covered_ppm };
}

/** The smallest number of entities that covers at least `share_ppm` of the units, or null when the curve never does. */
export function entitiesFor(c: Curve, share_ppm: number): number | null {
  const n = c.covered_ppm.findIndex((covered) => covered >= share_ppm);
  return n < 0 ? null : n;
}

// ---- usage ----------------------------------------------------------------------------------

/** The probability, in ppm, that a team of `team_size` holds an entity that one slot holds with `slot_ppm`. */
export function teamPresence(slot_ppm: number, team_size: number): number {
  const keep = BigInt(PPM - Math.min(slot_ppm, PPM));
  return PPM - Number((keep ** BigInt(team_size) * BigInt(PPM)) / BigInt(PPM) ** BigInt(team_size));
}

/**
 * Per entity, the share of team slots (ppm) the usage table gives it: the species' usage times its share
 * of the species' item, ability or move list (a move's share is of the list's weight, so it is scaled by
 * the moves a set holds). A condition gets the sum over the roster entities that start it, capped at one
 * (an upper bound; the starters can overlap).
 */
export function usageSlots(inv: MechanicsInventory, index: InventoryIndex, table: UsageWeightTable, moves_per_set: number): Map<string, number> {
  const slots = new Map<string, number>();
  const add = (kind: Kind, id: string, ppm: number): void => void slots.set(entityKey(kind, id), (slots.get(entityKey(kind, id)) ?? 0) + ppm);
  for (const species of table.species) {
    for (const e of species.items.entries) add("item", e.id, Math.floor((species.usage_ppm * e.ppm) / PPM));
    for (const e of species.abilities.entries) add("ability", e.id, Math.floor((species.usage_ppm * e.ppm) / PPM));
    for (const e of species.moves.entries) add("move", e.id, Math.floor((species.usage_ppm * e.ppm * moves_per_set) / PPM));
  }
  for (const key of [...slots.keys()]) if (!index.keys[key.slice(0, key.indexOf(":")) as Kind].has(key)) slots.delete(key);
  for (const [starter, conditions] of index.starts) for (const condition of conditions) slots.set(condition, (slots.get(condition) ?? 0) + (slots.get(starter) ?? 0));
  for (const [key, ppm] of slots) slots.set(key, Math.min(ppm, PPM));
  return slots;
}

// ---- the report ------------------------------------------------------------------------------

export interface WeightInputs {
  /** One team per entry: the members' items, abilities and moves (names or ids). */
  teams: TeamMember[][];
  dataset: { id: string; file_sha256: string };
  usage: { file: string; table: UsageWeightTable };
  team_size: number;
}

export interface PairSummary {
  /** Damage-relevant moves, abilities and items. */
  entities: number;
  teams: number;
  /** Pairs of them held together by at least one dataset team. */
  pairs_seen: number;
  /** The most frequent pairs in the dataset. */
  top: { a: string; b: string; teams: number }[];
  /** Pairs of damage-relevant entities whose independence estimate of sharing a team reaches 10^-4 (ADR-028 section 5), and all such pairs. */
  independent_ge_1e4: number;
  independent_pairs: number;
}

export interface WeightsReport {
  schema_version: 1;
  weights_format: string;
  inventory_hash: string;
  showdown_commit: string;
  format_id: string;
  dataset: { id: string; file_sha256: string; teams: number; moves_per_set: number; unknown_ids: number };
  usage: { file: string; usage_table_hash: string; month: string; cutoff: number };
  /** Damage-relevant entities and their damage handler names. */
  damage: { key: string; handlers: string[] }[];
  /** For every damage-relevant entity: dataset teams holding it and its usage share of team slots (ppm). */
  weights: { key: string; teams: number; usage_slot_ppm: number; usage_team_ppm: number }[];
  coverage: { teams: Curve; handler_names: Curve; usage_independence: Curve };
  pairs: PairSummary;
}

const TOP_PAIRS = 40;
const PAIR_THRESHOLD_PPM = 100;

/** The report for the inventory under a dataset's teams and a usage table. */
export function weightsReport(inv: MechanicsInventory, input: WeightInputs): WeightsReport {
  const index = buildIndex(inv);
  const held = input.teams.map((members) => teamKeys(index, members));
  const moves_per_set = Math.max(...input.teams.flat().map((m) => m.moves.length));
  const damage = [...index.damage.keys()].sort(cmp);
  const unitsOf = (held_keys: Set<string>): string[] => damage.filter((key) => held_keys.has(key));
  const units = held.map((h) => unitsOf(h.keys));
  const teamCount = new Map<string, number>();
  for (const unit of units) for (const key of unit) teamCount.set(key, (teamCount.get(key) ?? 0) + 1);

  const slots = usageSlots(inv, index, input.usage.table, moves_per_set);
  const presence = new Map(damage.map((key) => [key, teamPresence(slots.get(key) ?? 0, input.team_size)]));

  // Pairs among damage-relevant moves, abilities and items.
  const plain = damage.filter((key) => !key.startsWith("condition:"));
  const together = new Map<string, number>();
  for (const unit of units) {
    const own = unit.filter((key) => !key.startsWith("condition:"));
    for (const [i, a] of own.entries()) for (const b of own.slice(i + 1)) together.set(`${a} ${b}`, (together.get(`${a} ${b}`) ?? 0) + 1);
  }
  let independent = 0;
  for (const [i, a] of plain.entries()) for (const b of plain.slice(i + 1)) independent += Math.floor(((presence.get(a) as number) * (presence.get(b) as number)) / PPM) >= PAIR_THRESHOLD_PPM ? 1 : 0;
  const top = [...together]
    .sort((x, y) => y[1] - x[1] || cmp(x[0], y[0]))
    .slice(0, TOP_PAIRS)
    .map(([pair, teams]) => ({ a: pair.slice(0, pair.indexOf(" ")), b: pair.slice(pair.indexOf(" ") + 1), teams }));

  const table = input.usage.table;
  return {
    schema_version: 1,
    weights_format: WEIGHTS_FORMAT,
    inventory_hash: inventoryHash(inv),
    showdown_commit: inv.showdown_commit,
    format_id: inv.format_id,
    dataset: { id: input.dataset.id, file_sha256: input.dataset.file_sha256, teams: input.teams.length, moves_per_set, unknown_ids: held.reduce((n, h) => n + h.unknown, 0) },
    usage: { file: input.usage.file, usage_table_hash: usageTableHash(table), month: table.month, cutoff: table.cutoff },
    damage: damage.map((key) => ({ key, handlers: index.damage.get(key) as string[] })),
    weights: damage.map((key) => ({ key, teams: teamCount.get(key) ?? 0, usage_slot_ppm: slots.get(key) ?? 0, usage_team_ppm: presence.get(key) as number })),
    coverage: {
      teams: curve(units),
      handler_names: curve(units.map((unit) => unit.flatMap((key) => index.damage.get(key) as string[]))),
      usage_independence: independenceCurve(damage.map((key) => ({ key, p_ppm: presence.get(key) as number }))),
    },
    pairs: { entities: plain.length, teams: input.teams.length, pairs_seen: together.size, top, independent_ge_1e4: independent, independent_pairs: (plain.length * (plain.length - 1)) / 2 },
  };
}

/** The committed weights file's bytes: a list or the coverage block is spread over lines, anything else is one line. */
export function weightsBytes(report: WeightsReport): string {
  const lines = (items: string[]): string => `[\n${items.map((v) => `    ${v}`).join(",\n")}\n  ]`;
  const body = Object.entries(report).map(([key, value]) => {
    const shown = Array.isArray(value)
      ? lines(value.map((v) => JSON.stringify(v)))
      : key === "coverage"
        ? `{\n${Object.entries(value).map(([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n")}\n  }`
        : JSON.stringify(value);
    return `  ${JSON.stringify(key)}: ${shown}`;
  });
  return `{\n${body.join(",\n")}\n}\n`;
}
